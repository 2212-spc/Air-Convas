# -*- coding: utf-8 -*-
"""AirCanvas - 隔空绘手：基于手势识别的虚拟演示系统

核心逻辑：
- 统一使用捏合 (Pinch) 作为主要交互手势
- 通过左侧工具栏 (Tool) 切换功能：画笔、橡皮、激光笔
- [System] 集成 ReplaySystem (录制 + 回放 + 文件管理)
"""

import sys
import io
import time
import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

# 修复 Windows 终端中文输出编码
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
cv2.setUseOptimized(True)

try:
    import pyautogui
except Exception:
    pyautogui = None

import config
from core.coordinate_mapper import CoordinateMapper
from core.gesture_recognizer import GestureRecognizer
from core.hand_detector import HandDetector, Hand, INDEX_TIP, THUMB_TIP, MIDDLE_TIP, WRIST
from core.async_detector import SyncAsyncHandDetector
from modules.canvas import Canvas
from modules.eraser import Eraser
from modules.shape_recognizer import ShapeRecognizer
from modules.virtual_pen import VirtualPen
from modules.particle_system import ParticleSystem
from modules.laser_pointer import LaserPointer
from modules.palm_hud import PalmHUD
from modules.brush_manager import BrushManager
from modules.gesture_ui import GestureUI
from modules.ppt_gesture_controller import PPTGestureController
from modules.temporary_ink import TemporaryInkManager
from modules.visual_effects import EffectManager
from modules.interactive_effects import InteractiveEffectsManager
from modules.tutorial_manager import TutorialManager
from modules.replay_system import ReplayRecorder, ReplayPlayer


@dataclass
class LandmarkAdapter:
    """
    数据适配器 (Adapter Pattern)
    
    将 HandDetector 的原始关键点数据转换为 PPT 控制器所需的格式。
    """
    x: float
    y: float
    z: float = 0.0


def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    将透明画布图层叠加到摄像头画面上。
    
    使用 cv2.add 进行加法混合，相比 addWeighted 能保留更好的色彩亮度，
    解决黑色背景下线条“看不清”的问题。

    Args:
        frame (np.ndarray): 摄像头原始帧 (BGR)。
        canvas (np.ndarray): 绘图画布层 (BGR, 黑色背景)。

    Returns:
        np.ndarray: 混合后的图像。
    """
    # 创建掩码：找出画布上非黑色的像素区域
    mask = np.any(canvas != 0, axis=2)
    
    if np.any(mask):
        # 仅对有内容的区域进行混合计算，优化性能
        frame_roi = frame[mask]
        canvas_roi = canvas[mask]
        blended = cv2.add(frame_roi, canvas_roi)
        frame[mask] = blended
        
    return frame


def palm_center(hand: Hand) -> Tuple[float, float]:
    """
    计算手掌几何中心（质心）。
    
    取手腕(WRIST)、食指根部(5)、小指根部(17)构成的三角形重心，
    比单纯取手腕坐标更稳定，适合用于 HUD 跟随。

    Args:
        hand (Hand): 检测到的手部对象。

    Returns:
        Tuple[float, float]: 归一化的中心坐标 (x, y)。
    """
    # 0: Wrist, 5: Index MCP, 17: Pinky MCP
    pts = [hand.landmarks_norm[i] for i in (WRIST, 5, 17)]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {config.CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    
    # 摄像头优化设置
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    
    for _ in range(3):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.read()

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(config.WINDOW_NAME, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    FULLSCREEN = False
    
    mouse_clicked = False
    mouse_click_pos: Optional[Tuple[int, int]] = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_clicked, mouse_click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True
            mouse_click_pos = (x, y)
    
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

    def map_mouse_to_frame(pos: Optional[Tuple[int, int]], frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
        """将窗口鼠标坐标映射回摄像头帧坐标系"""
        if pos is None: return None
        fx_w = frame_shape[1]
        fx_h = frame_shape[0]
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(config.WINDOW_NAME)
        except Exception:
            win_w, win_h = fx_w, fx_h
        sx = fx_w / max(1, win_w)
        sy = fx_h / max(1, win_h)
        x = int(pos[0] * sx)
        y = int(pos[1] * sy)
        return (max(0, min(fx_w - 1, x)), max(0, min(fx_h - 1, y)))

    # 初始化检测器
    detector = SyncAsyncHandDetector(
        async_mode=getattr(config, 'ASYNC_INFERENCE', True),
        max_num_hands=1,
        infer_width=getattr(config, 'INFER_WIDTH', 640),
        infer_height=getattr(config, 'INFER_HEIGHT', 360),
    )
    detector.start()

    if pyautogui:
        try:
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False
        except Exception: pass

    gesture = GestureRecognizer(
        pinch_threshold=config.PINCH_THRESHOLD,
        pinch_release_threshold=config.PINCH_RELEASE_THRESHOLD,
        swipe_threshold=config.SWIPE_THRESHOLD,
        swipe_velocity_threshold=getattr(config, 'SWIPE_VELOCITY_THRESHOLD', 0.015),
        swipe_cooldown_frames=config.SWIPE_COOLDOWN_FRAMES,
        pinch_confirm_frames=getattr(config, 'PINCH_CONFIRM_FRAMES', 3),
        pinch_release_confirm_frames=getattr(config, 'PINCH_RELEASE_CONFIRM_FRAMES', 1),
        pinch_velocity_boost=getattr(config, 'PINCH_VELOCITY_BOOST', 0.02),
    )

    # 坐标映射器初始化
    one_euro_min_cutoff = getattr(config, 'ONE_EURO_MIN_CUTOFF', 1.2)
    one_euro_beta = getattr(config, 'ONE_EURO_BETA', 0.03)
    
    draw_mapper = CoordinateMapper(
        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        getattr(config, 'ACTIVE_REGION_DRAW', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'DRAW_SMOOTHING_FACTOR', 0.3),
        smoothing_mode='one_euro',
        one_euro_min_cutoff=one_euro_min_cutoff,
        one_euro_beta=one_euro_beta,
    )

    ink_mapper = CoordinateMapper(
        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        getattr(config, 'ACTIVE_REGION_DRAW', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'DRAW_SMOOTHING_FACTOR', 0.3),
        smoothing_mode='one_euro',
        one_euro_min_cutoff=max(1.2, one_euro_min_cutoff),
        one_euro_beta=max(0.008, one_euro_beta),
    )

    if pyautogui:
        SCREEN_W, SCREEN_H = pyautogui.size()
    else:
        SCREEN_W, SCREEN_H = getattr(config, 'SCREEN_WIDTH', 1920), getattr(config, 'SCREEN_HEIGHT', 1080)

    cursor_mapper = CoordinateMapper(
        (SCREEN_W, SCREEN_H),
        getattr(config, 'ACTIVE_REGION_CURSOR', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'CURSOR_SMOOTHING_FACTOR', 0.15),
        smoothing_mode='one_euro',
        one_euro_min_cutoff=one_euro_min_cutoff,
    )
    
    ui_mapper = CoordinateMapper(
        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        getattr(config, 'ACTIVE_REGION_DRAW', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=0.0,
        smoothing_mode='ema',
        one_euro_min_cutoff=one_euro_min_cutoff,
        one_euro_beta=one_euro_beta,
    )

    # 模块初始化
    ppt_controller = PPTGestureController(external_mp=True, cursor_mapper=cursor_mapper)
    canvas = Canvas(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, max_history=getattr(config, 'MAX_HISTORY', 50))
    brush_manager = BrushManager()
    
    # [System] 录像与播放模块初始化
    recorder = ReplayRecorder()
    player = ReplayPlayer(canvas) # 播放器需要持有 canvas 引用
    
    # [System] 录像文件列表管理
    recording_files: List[str] = []
    current_file_index: int = -1
    
    # [辅助函数] 刷新文件列表
    def refresh_recordings():
        nonlocal recording_files, current_file_index
        folder = "recordings"
        if not os.path.exists(folder): return
        # 获取所有 json 文件
        files = glob.glob(os.path.join(folder, '*.json'))
        # 按创建时间排序（旧 -> 新）
        recording_files = sorted(files, key=os.path.getctime)
        # 默认选中最新的
        if recording_files:
            current_file_index = len(recording_files) - 1
            
    # 启动时先刷新一次
    refresh_recordings()
    
    prev_record_pt: Optional[Tuple[int, int]] = None

    pen = VirtualPen(
        canvas=canvas,
        brush_manager=brush_manager,
        smoothing=None,
        jump_threshold=getattr(config, 'STROKE_JUMP_THRESHOLD', 80),
        enable_bezier=getattr(config, 'BEZIER_ENABLED', True),
        bezier_segments=getattr(config, 'BEZIER_SEGMENTS', 8),
        enable_pen_effect=getattr(config, 'PEN_EFFECT_ENABLED', True),
        min_thickness_ratio=getattr(config, 'PEN_MIN_THICKNESS_RATIO', 0.4),
        max_thickness_ratio=getattr(config, 'PEN_MAX_THICKNESS_RATIO', 1.2),
        speed_threshold=getattr(config, 'PEN_SPEED_THRESHOLD', 25.0),
        thickness_smoothing=getattr(config, 'PEN_THICKNESS_SMOOTHING', 0.25),
    )
    eraser = Eraser(canvas, size=config.ERASER_SIZE)
    shape_recognizer = ShapeRecognizer(
        enable_line_assist=getattr(config, 'LINE_ASSIST_ENABLED', True),
        line_variance_thresh=getattr(config, 'LINE_VARIANCE_THRESH', 0.015),
        min_line_length=getattr(config, 'MIN_LINE_LENGTH', 50),
    )

    particle_system = ParticleSystem(max_particles=config.MAX_PARTICLES, emit_count=config.PARTICLE_EMIT_COUNT)
    laser_pointer = LaserPointer()
    palm_hud = PalmHUD()
    temp_ink_manager = TemporaryInkManager(default_lifetime=1.5)
    effect_manager = EffectManager()
    interactive_effects = InteractiveEffectsManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    
    gesture_ui = GestureUI(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    tutorial_manager = TutorialManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # 全局状态
    APP_MODE = "DRAW" # DRAW, PPT, REPLAY
    fps = 0.0
    last_time = time.time()
    frame_count = 0
    save_counter = 0
    draw_lock = 0
    DRAW_LOCK_FRAMES = getattr(config, 'DRAW_LOCK_FRAMES', 5)
    is_drawing = False

    ENABLE_PARTICLES = False
    ENABLE_LASER = True
    ENABLE_PALM_HUD = True
    ENABLE_LINE_ASSIST = getattr(config, 'LINE_ASSIST_ENABLED', True)
    ENABLE_PEN_EFFECT = getattr(config, 'PEN_EFFECT_ENABLED', True)
    ENABLE_INTERACTIVE_EFFECTS = False
    
    SHAPE_DWELL_FRAMES = 15
    SHAPE_DWELL_THRESHOLD = 20
    stroke_end_positions = []

    SHOW_HELP = False
    undo_redo_hint = ""
    undo_redo_hint_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 教程逻辑 (优先级最高)
        if tutorial_manager.is_active:
            hands = detector.detect(frame)
            cursor_pos = None
            if hands:
                detector.draw_hand(frame, hands[0])
                cursor_pos = draw_mapper.map(hands[0].landmarks_norm[INDEX_TIP])
            tutorial_manager.render(frame, cursor_pos)
            if hands and gesture.classify(hands[0])["pinch_start"] and cursor_pos:
                tutorial_manager.handle_click(cursor_pos)
            if mouse_clicked:
                mapped = map_mouse_to_frame(mouse_click_pos, frame.shape)
                if mapped: tutorial_manager.handle_click(mapped)
                mouse_clicked = False
            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key != 255: tutorial_manager.handle_key(key)
            continue

        if draw_lock > 0: draw_lock -= 1
        if undo_redo_hint_frames > 0: undo_redo_hint_frames -= 1
        else: undo_redo_hint = ""
        
        # [System] 核心逻辑分流: 回放模式 vs 实时模式
        if APP_MODE == "REPLAY":
            # --- 回放模式逻辑 ---
            # 1. 停止更新手势，只运行播放器
            player.update()
            
            # 2. 如果播放结束，自动切回 DRAW
            if not player.is_playing:
                APP_MODE = "DRAW"
                undo_redo_hint = "播放结束"
                undo_redo_hint_frames = 40
            
            # 3. 渲染
            frame = overlay_canvas(frame, canvas.get_canvas())
            
            # 4. 显示回放进度 HUD
            progress = player.get_progress()
            hud_data = {
                "fps": fps,
                "mode": "REPLAY",
                "history": canvas.get_history_info(), # 恢复显示真实的历史步数
                "message": f"回放进度: {int(progress*100)}%  (按 P 停止)" # 进度移到这里显示
            }
            # 回放时不显示动作按钮，只显示HUD
            gesture_ui.render(frame, brush_manager, hud_data=hud_data)

        elif APP_MODE == "PPT":
            # --- PPT 模式逻辑 ---
            hands = detector.detect(frame)
            if hands:
                detector.draw_hand(frame, hands[0])
                hand = hands[0]
                landmarks = [LandmarkAdapter(x=lm[0], y=lm[1], z=0.0) for lm in hand.landmarks_norm]
                ppt_controller.process_hand_data(landmarks, frame)
            
            hud_data = {"fps": fps, "mode": "PPT", "message": "Tab 键切换回画板"}
            gesture_ui.render(frame, brush_manager, hud_data=hud_data)

        else:
            # --- DRAW 模式 (AirCanvas 核心) ---
            hands = detector.detect(frame)
            
            current_mode = "idle"
            ui_draw_pt = None
            ui_erase_pt = None
            ui_pinching = False
            palm_pos_for_hud = None
            index_tip_pt = None
            palm_pos_pixel = None
            g = None

            if hands:
                detector.draw_hand(frame, hands[0])
                hand = hands[0]
                g = gesture.classify(hand)
                current_mode = g["mode"]

                index_norm = hand.landmarks_norm[INDEX_TIP]
                thumb_norm = hand.landmarks_norm[THUMB_TIP]
                middle_norm = hand.landmarks_norm[MIDDLE_TIP]
                palm_norm = palm_center(hand)

                tip_norm = ((index_norm[0] + thumb_norm[0]) / 2.0, (index_norm[1] + thumb_norm[1]) / 2.0) if g["pinching"] else index_norm

                draw_pt = draw_mapper.map(tip_norm)
                index_tip_pt = ui_mapper.map(index_norm)
                ink_pt = ink_mapper.map(tip_norm)

                palm_pos_for_hud = palm_norm
                palm_pos_pixel = draw_mapper.map(palm_norm)

                ui_draw_pt = ink_pt
                ui_pinching = g["pinching"]

                # 视觉反馈
                if g["pinch_start"]: effect_manager.add_ripple(draw_pt, color=(0, 255, 255))
                if g["index_middle"]:
                    click_pt = draw_mapper.map(index_norm) # Simplified click pos
                    if frame_count % 10 == 0: effect_manager.add_ripple(click_pt, color=(0, 255, 0))

                # 撤销/重做 (集成录像)
                if g["swipe"] and g["three_fingers"]:
                    if g["swipe"] == "SWIPE_LEFT":
                        if canvas.undo():
                            undo_redo_hint = "已撤销"; undo_redo_hint_frames = 30
                            effect_manager.add_ripple(draw_pt, color=(0, 0, 255))
                            if recorder.is_recording: recorder.record_event("undo")
                    elif g["swipe"] == "SWIPE_RIGHT":
                        if canvas.redo():
                            undo_redo_hint = "已重做"; undo_redo_hint_frames = 30
                            effect_manager.add_ripple(draw_pt, color=(0, 255, 0))
                            if recorder.is_recording: recorder.record_event("redo")
                
                # 工具切换
                if g["swipe"] and g["index_only"]:
                    if g["swipe"] == "SWIPE_UP":
                        brush_manager.current_tool_index = (brush_manager.current_tool_index - 1) % len(brush_manager.TOOLS)
                        undo_redo_hint = f"工具: {brush_manager.tool.upper()}"; undo_redo_hint_frames = 30
                        effect_manager.add_ripple(draw_pt, color=(255, 255, 0))
                    elif g["swipe"] == "SWIPE_DOWN":
                        brush_manager.current_tool_index = (brush_manager.current_tool_index + 1) % len(brush_manager.TOOLS)
                        undo_redo_hint = f"工具: {brush_manager.tool.upper()}"; undo_redo_hint_frames = 30
                        effect_manager.add_ripple(draw_pt, color=(255, 255, 0))

                # UI 交互
                if gesture_ui.visible and index_tip_pt:
                    if not g["pinching"]:
                        gesture_ui.update_hover(index_tip_pt, brush_manager)

                if gesture_ui.visible:
                    ui_action = None
                    dwell_selected = False
                    if g["pinch_start"] and gesture_ui.hover_item:
                        res = gesture_ui.select_hover_item(brush_manager)
                        ui_action = res["action"]
                    elif not g["pinching"]:
                        res = gesture_ui.consume_pending_selection(brush_manager)
                        if res["selected"]:
                            dwell_selected = True
                            ui_action = res["action"]

                    if ui_action or (gesture_ui.hover_item and (g["pinch_start"] or dwell_selected)):
                        if gesture_ui.hover_item[0] == "tool":
                            undo_redo_hint = f"工具: {brush_manager.tool.upper()}"; undo_redo_hint_frames = 30
                        elif gesture_ui.hover_item[0] == "action":
                            if ui_action == "clear":
                                canvas.clear(); temp_ink_manager.clear(); particle_system.clear()
                                undo_redo_hint = "画布已清空"; undo_redo_hint_frames = 30
                                if recorder.is_recording: recorder.record_event("clear")
                            elif ui_action == "particles":
                                ENABLE_PARTICLES = not ENABLE_PARTICLES
                                if not ENABLE_PARTICLES: particle_system.clear()
                            elif ui_action == "effects":
                                ENABLE_INTERACTIVE_EFFECTS = interactive_effects.toggle()
                                undo_redo_hint = f"互动特效: {interactive_effects.get_effect_label()}"
                                undo_redo_hint_frames = 45
                        draw_lock = DRAW_LOCK_FRAMES
                        effect_manager.add_ripple(draw_pt, color=(255, 255, 255))

                # ========== 绘图逻辑 (含录像) ==========
                if g["pinch_start"]:
                    if not gesture_ui.is_in_dead_zone(draw_pt, brush_manager):
                        if brush_manager.tool == "pen":
                            pen.start_stroke(); is_drawing = True; prev_record_pt = ink_pt
                        elif brush_manager.tool == "eraser":
                            is_drawing = True; prev_record_pt = ink_pt
                        elif brush_manager.tool == "laser":
                            temp_ink_manager.start_stroke(color=(0,0,255), thickness=4)
                            is_drawing = True; prev_record_pt = ink_pt
                    draw_lock = 0

                if g["pinching"] and draw_lock == 0:
                    if not gesture_ui.is_in_dead_zone(draw_pt, brush_manager):
                        if brush_manager.tool == "pen":
                            smoothed_pt = pen.draw(ink_pt)
                            if ENABLE_PARTICLES: particle_system.emit(ink_pt, brush_manager.color)
                            
                            # [Recorder] 记录
                            if recorder.is_recording and prev_record_pt:
                                recorder.record_stroke_segment(prev_record_pt, smoothed_pt, brush_manager.color, brush_manager.thickness, "pen")
                            prev_record_pt = smoothed_pt
                            
                            stroke_end_positions.append(ink_pt)
                            if len(stroke_end_positions) > SHAPE_DWELL_FRAMES: stroke_end_positions.pop(0)
                        
                        elif brush_manager.tool == "eraser":
                            eraser.erase(ink_pt)
                            if recorder.is_recording and prev_record_pt:
                                recorder.record_stroke_segment(prev_record_pt, ink_pt, (0,0,0), config.ERASER_SIZE, "eraser")
                            prev_record_pt = ink_pt
                            
                        elif brush_manager.tool == "laser":
                            temp_ink_manager.add_point(ink_pt)
                
                if g["pinch_end"]:
                    is_drawing = False; prev_record_pt = None
                    if brush_manager.tool == "pen":
                        finished_points = pen.end_stroke()
                        draw_lock = DRAW_LOCK_FRAMES
                        if finished_points:
                            should_beautify = False
                            if len(stroke_end_positions) >= SHAPE_DWELL_FRAMES:
                                movement = 0
                                for i in range(1, len(stroke_end_positions)):
                                    p1 = stroke_end_positions[i-1]; p2 = stroke_end_positions[i]
                                    movement += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                                if movement < SHAPE_DWELL_THRESHOLD: should_beautify = True
                            
                            if should_beautify:
                                beautified = shape_recognizer.beautify(finished_points, canvas.get_canvas(), brush_manager.color, brush_manager.thickness)
                                if beautified: effect_manager.add_ripple(tuple(np.mean(finished_points, axis=0).astype(int)), color=(0, 255, 0))
                            
                            canvas.save_stroke()
                            # [关键修复] 记录 save_stroke 事件
                            if recorder.is_recording: recorder.record_event("save_stroke")
                            
                            stroke_end_positions.clear()
                    elif brush_manager.tool == "laser":
                        temp_ink_manager.end_stroke()

            # 鼠标逻辑
            if mouse_clicked:
                mapped = map_mouse_to_frame(mouse_click_pos, frame.shape)
                consumed = False
                if mapped and gesture_ui.visible:
                    if gesture_ui.handle_mouse_click(mapped, brush_manager):
                        consumed = True
                        if gesture_ui.hover_item and gesture_ui.hover_item[0] == "action":
                            # Action logic
                            typ, idx = gesture_ui.hover_item
                            key_act = gesture_ui.action_items[idx][0]
                            if key_act == "clear":
                                canvas.clear(); temp_ink_manager.clear(); particle_system.clear()
                                undo_redo_hint = "画布已清空"; undo_redo_hint_frames = 30
                                if recorder.is_recording: recorder.record_event("clear")
                            elif key_act == "particles":
                                ENABLE_PARTICLES = not ENABLE_PARTICLES
                                if not ENABLE_PARTICLES: particle_system.clear()
                            elif key_act == "effects":
                                ENABLE_INTERACTIVE_EFFECTS = interactive_effects.toggle()
                                undo_redo_hint = f"互动特效: {interactive_effects.get_effect_label()}"
                                undo_redo_hint_frames = 45
                        draw_lock = DRAW_LOCK_FRAMES
                if mapped and not consumed: tutorial_manager.handle_click(mapped)
                mouse_clicked = False

            # 特效与渲染
            temp_ink_manager.update(); effect_manager.update()
            if ENABLE_PARTICLES: particle_system.update()
            if ENABLE_INTERACTIVE_EFFECTS:
                is_open = (sum(g["fingers"]) >= 4) if g else False
                is_pinch = g["pinching"] if g else False
                interactive_effects.update(ui_draw_pt, is_open, is_pinch)

            frame = overlay_canvas(frame, canvas.get_canvas())

            if ENABLE_PARTICLES: particle_system.render(frame)
            if ENABLE_LASER:
                temp_ink_manager.render(frame)
                if brush_manager.tool == "laser" and ui_draw_pt: laser_pointer.render(frame, ui_draw_pt)
            effect_manager.render(frame)
            if ENABLE_INTERACTIVE_EFFECTS: interactive_effects.render(frame, ui_draw_pt, is_open, is_pinch)
            if ENABLE_PALM_HUD and palm_pos_for_hud and palm_pos_pixel:
                palm_hud.update(palm_pos_for_hud)
                if palm_hud.is_still and current_mode != "erase": palm_hud.render(frame, palm_pos_pixel)

            # 绘制光标
            if ui_draw_pt:
                if brush_manager.tool == "pen":
                    cv2.circle(frame, ui_draw_pt, 6, (0, 255, 255), 2, cv2.LINE_AA)
                    if g and g["pinching"]: cv2.circle(frame, ui_draw_pt, 3, (0, 200, 200), -1, cv2.LINE_AA)
                elif brush_manager.tool == "eraser":
                    cv2.circle(frame, ui_draw_pt, config.ERASER_SIZE, (0, 0, 255), 2, cv2.LINE_AA)
            if index_tip_pt and gesture_ui.visible:
                cv2.circle(frame, index_tip_pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

            # FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                frame_count = 0; last_time = time.time()

            # [UI Update] HUD 逻辑优化
            rec_msg = "REC ●" if recorder.is_recording else ""
            
            final_msg = ""
            if undo_redo_hint:
                final_msg = undo_redo_hint
            elif rec_msg:
                final_msg = rec_msg
            elif APP_MODE == "REPLAY":
                pass 
            else:
                final_msg = ""

            hud_data = {
                "fps": fps,
                "mode": "DRAW",
                "history": canvas.get_history_info(),
                "message": final_msg
            }
            action_state = {
                "particles": ENABLE_PARTICLES,
                "line_assist": ENABLE_LINE_ASSIST,
                "pen_effect": getattr(config, 'PEN_EFFECT_ENABLED', False),
                "effects": ENABLE_INTERACTIVE_EFFECTS
            }
            gesture_ui.render(frame, brush_manager, action_state=action_state, hud_data=hud_data)

            if ui_draw_pt: tutorial_manager.render(frame, ui_draw_pt)
            if g and g["pinch_start"] and ui_draw_pt: tutorial_manager.handle_click(ui_draw_pt)

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        # ========== 键盘控制 ==========
        if key == ord("q"): break
        if key == 9: # Tab
            APP_MODE = "PPT" if APP_MODE == "DRAW" else "DRAW"
            print(f"Switched to {APP_MODE}")
            if APP_MODE == "PPT": pen.end_stroke()
            else: ppt_controller.gesture_history.clear()

        # [新增] 录像文件切换 (N/B)
        if key == ord("n"): # Next
            refresh_recordings()
            if recording_files:
                current_file_index = (current_file_index + 1) % len(recording_files)
                fname = os.path.basename(recording_files[current_file_index])
                undo_redo_hint = f"切换: {fname[-15:]}"; undo_redo_hint_frames = 60
        
        if key == ord("b"): # Back
            refresh_recordings()
            if recording_files:
                current_file_index = (current_file_index - 1 + len(recording_files)) % len(recording_files)
                fname = os.path.basename(recording_files[current_file_index])
                undo_redo_hint = f"切换: {fname[-15:]}"; undo_redo_hint_frames = 60

        # [新增] 录像 (R) 与 回放 (P)
        if key == ord("r"):
            if recorder.is_recording:
                saved = recorder.stop_recording()
                undo_redo_hint = "录制已保存"; undo_redo_hint_frames = 60
                print(f"Saved: {saved}")
                refresh_recordings() # 保存后刷新列表
            else:
                recorder.start_recording()
                undo_redo_hint = "开始录制..."; undo_redo_hint_frames = 30
                print("Recording...")
        
        if key == ord("p"):
            if APP_MODE == "REPLAY":
                player.stop()
                APP_MODE = "DRAW"
                undo_redo_hint = "回放停止"; undo_redo_hint_frames = 30
            else:
                refresh_recordings()
                if recording_files and current_file_index >= 0:
                    target = recording_files[current_file_index]
                    fname = os.path.basename(target)
                    if player.load_file(target):
                        APP_MODE = "REPLAY"
                        player.play(speed=1.5)
                        undo_redo_hint = f"回放: {fname[-15:]}"; undo_redo_hint_frames = 60
                    else:
                        undo_redo_hint = "文件损坏"; undo_redo_hint_frames = 30
                else:
                    undo_redo_hint = "无录像文件"; undo_redo_hint_frames = 30

        if key == ord("c"):
            canvas.clear(); temp_ink_manager.clear()
            if recorder.is_recording: recorder.record_event("clear")
        if key == ord("s"):
            out_path = Path("captures") / f"canvas_{save_counter}.png"
            out_path.parent.mkdir(exist_ok=True)
            canvas.save(str(out_path)); save_counter += 1
        if key == ord("z"):
            if canvas.undo():
                undo_redo_hint = "已撤销"; undo_redo_hint_frames = 30
                if recorder.is_recording: recorder.record_event("undo")
        if key == ord("y"):
            if canvas.redo():
                undo_redo_hint = "已重做"; undo_redo_hint_frames = 30
                if recorder.is_recording: recorder.record_event("redo")
        
        if key == ord('1'):
            ENABLE_PARTICLES = not ENABLE_PARTICLES
            if not ENABLE_PARTICLES: particle_system.clear()
        if key == ord('2'): ENABLE_LASER = not ENABLE_LASER
        if key == ord('3'): ENABLE_PALM_HUD = not ENABLE_PALM_HUD; palm_hud.reset()
        if key == ord('4'): ENABLE_INTERACTIVE_EFFECTS = interactive_effects.toggle()
        if key == ord('5') and ENABLE_INTERACTIVE_EFFECTS: interactive_effects.next_effect()
        if key == ord('l'):
            ENABLE_LINE_ASSIST = not ENABLE_LINE_ASSIST
            shape_recognizer.set_line_assist(ENABLE_LINE_ASSIST)
        if key == ord('u'): gesture_ui.toggle_visibility()
        if key == ord('h'): SHOW_HELP = not SHOW_HELP
        if key == ord('t'): brush_manager.next_tool()
        if key == ord('['): brush_manager.prev_color()
        if key == ord(']'): brush_manager.next_color()
        if key == ord('b'): brush_manager.next_brush_type()
        if key == ord('w'):
            FULLSCREEN = not FULLSCREEN
            if FULLSCREEN: cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else: cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()