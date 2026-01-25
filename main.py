# -*- coding: utf-8 -*-
"""AirCanvas - 隔空绘手：基于手势识别的虚拟演示系统

核心逻辑：
- 统一使用捏合 (Pinch) 作为主要交互手势
- 通过左侧工具栏 (Tool) 切换功能：画笔、橡皮、激光笔
"""

import sys
import io
import time
from pathlib import Path

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
from core.hand_detector import HandDetector, INDEX_TIP, THUMB_TIP, MIDDLE_TIP, WRIST
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
from modules.temporary_ink import TemporaryInkManager
from modules.visual_effects import EffectManager


def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    Composite non-black canvas pixels onto the frame.
    使用加法混合增强亮度，解决"看不清"的问题
    """
    # 找出画布上非黑色的像素
    mask = np.any(canvas != 0, axis=2)
    
    # 加法混合（类似于滤色/发光效果，字迹更亮，不会被视频"遮挡"）
    if np.any(mask):
        frame_roi = frame[mask]
        canvas_roi = canvas[mask]
        blended = cv2.add(frame_roi, canvas_roi)
        frame[mask] = blended
        
    return frame


def palm_center(hand) -> tuple:
    """计算掌心中心点（归一化坐标）"""
    pts = [hand.landmarks_norm[i] for i in (WRIST, 5, 17)]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {config.CAMERA_ID}")
        print("请检查：")
        print("1. 摄像头是否已连接")
        print("2. 摄像头是否被其他程序占用")
        print("3. config.py 中的 CAMERA_ID 是否正确")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # 创建可调整大小的窗口
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(config.WINDOW_NAME, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    FULLSCREEN = False

    # 推理分辨率
    INFER_W = getattr(config, 'INFER_WIDTH', 640)
    INFER_H = getattr(config, 'INFER_HEIGHT', 360)
    ASYNC_MODE = getattr(config, 'ASYNC_INFERENCE', True)
    
    # 使用同步/异步混合检测器
    detector = SyncAsyncHandDetector(
        async_mode=ASYNC_MODE,
        max_num_hands=1,
        infer_width=INFER_W,
        infer_height=INFER_H,
    )
    detector.start()

    # 降低 pyautogui 调用的系统性延迟
    if pyautogui:
        try:
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False
        except Exception:
            pass

    # 手势识别器
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

    # 1€ Filter 参数
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
        one_euro_beta=one_euro_beta,
    )

    # 画布
    canvas = Canvas(
        config.CAMERA_WIDTH, 
        config.CAMERA_HEIGHT,
        max_history=getattr(config, 'MAX_HISTORY', 50)
    )

    # 笔刷管理器
    brush_manager = BrushManager()

    # 虚拟钢笔
    pen = VirtualPen(
        canvas=canvas,
        brush_manager=brush_manager,
        smoothing=None,  # mapper already smooths movement
        jump_threshold=getattr(config, 'STROKE_JUMP_THRESHOLD', 80),
        enable_bezier=getattr(config, 'BEZIER_ENABLED', True),
        bezier_segments=getattr(config, 'BEZIER_SEGMENTS', 8),
        # 钢笔效果参数
        enable_pen_effect=getattr(config, 'PEN_EFFECT_ENABLED', True),
        min_thickness_ratio=getattr(config, 'PEN_MIN_THICKNESS_RATIO', 0.4),
        max_thickness_ratio=getattr(config, 'PEN_MAX_THICKNESS_RATIO', 1.2),
        speed_threshold=getattr(config, 'PEN_SPEED_THRESHOLD', 25.0),
        thickness_smoothing=getattr(config, 'PEN_THICKNESS_SMOOTHING', 0.25),
    )
    eraser = Eraser(canvas, size=config.ERASER_SIZE)
    
    # 图形识别器
    shape_recognizer = ShapeRecognizer(
        enable_line_assist=getattr(config, 'LINE_ASSIST_ENABLED', True),
        line_variance_thresh=getattr(config, 'LINE_VARIANCE_THRESH', 0.015),
        min_line_length=getattr(config, 'MIN_LINE_LENGTH', 50),
    )

    # AR增强效果
    particle_system = ParticleSystem(
        max_particles=config.MAX_PARTICLES,
        emit_count=config.PARTICLE_EMIT_COUNT
    )
    laser_pointer = LaserPointer()
    palm_hud = PalmHUD()
    temp_ink_manager = TemporaryInkManager(default_lifetime=1.5)  # 激光笔笔迹持续1.5秒
    effect_manager = EffectManager()

    # 手势UI界面
    gesture_ui = GestureUI(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    fps = 0
    last_time = time.time()
    frame_count = 0
    save_counter = 0

    # 模式控制
    draw_lock = 0
    DRAW_LOCK_FRAMES = getattr(config, 'DRAW_LOCK_FRAMES', 5)

    # AR效果控制开关
    ENABLE_PARTICLES = False  # 默认关闭粒子效果以减少延迟
    ENABLE_LASER = True
    ENABLE_PALM_HUD = True

    # 直线辅助开关
    ENABLE_LINE_ASSIST = getattr(config, 'LINE_ASSIST_ENABLED', True)

    # 钢笔效果开关
    ENABLE_PEN_EFFECT = getattr(config, 'PEN_EFFECT_ENABLED', True)

    # 显示帮助信息
    SHOW_HELP = False

    # 撤销/重做提示显示
    undo_redo_hint = ""
    undo_redo_hint_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧")
            break

        if draw_lock > 0:
            draw_lock -= 1

        # 撤销/重做提示淡出
        if undo_redo_hint_frames > 0:
            undo_redo_hint_frames -= 1
        else:
            undo_redo_hint = ""

        frame = cv2.flip(frame, 1)
        
        hands = detector.detect(frame)
        current_mode = "idle"
        ui_draw_pt = None
        ui_erase_pt = None
        ui_pinching = False
        ui_pinch_dist = 0.0
        palm_pos_for_hud = None
        palm_pos_pixel = None
        g = None  # 手势结果

        if hands:
            hand = hands[0]
            g = gesture.classify(hand)
            current_mode = g["mode"]

            index_norm = hand.landmarks_norm[INDEX_TIP]
            thumb_norm = hand.landmarks_norm[THUMB_TIP]
            middle_norm = hand.landmarks_norm[MIDDLE_TIP]
            palm_norm = palm_center(hand)

            # 笔尖位置：捏合时用拇指食指中心点，其他时候用食指尖
            if g["pinching"] or current_mode == "active" or brush_manager.tool == "laser":
                tip_norm = (
                    (index_norm[0] + thumb_norm[0]) / 2.0,
                    (index_norm[1] + thumb_norm[1]) / 2.0
                )
                if not g["pinching"]: # 如果不是捏合状态，就用食指尖
                     tip_norm = index_norm
            else:
                tip_norm = index_norm

            draw_pt = draw_mapper.map(tip_norm)
            erase_pt = draw_mapper.map(palm_norm)
            screen_pt = cursor_mapper.map(index_norm)

            # 更新掌心HUD位置
            palm_pos_for_hud = palm_norm
            palm_pos_pixel = draw_mapper.map(palm_norm)

            # 记录UI提示点位
            ui_draw_pt = draw_pt
            ui_erase_pt = erase_pt if brush_manager.tool == "eraser" else None
            ui_pinching = g["pinching"]
            ui_pinch_dist = g["pinch_distance"]

            # ========== 视觉反馈 (波纹特效) ==========
            # 捏合开始触发波纹
            if g["pinch_start"]:
                effect_manager.add_ripple(draw_pt, color=(0, 255, 255))
            
            # 点击手势触发波纹
            if g["index_middle"]:
                # 使用食指和中指的中间位置
                idx_pt = draw_mapper.map(hand.landmarks_norm[INDEX_TIP])
                mid_pt = draw_mapper.map(hand.landmarks_norm[MIDDLE_TIP])
                click_pt = ((idx_pt[0] + mid_pt[0]) // 2, (idx_pt[1] + mid_pt[1]) // 2)
                # 限制触发频率
                if frame_count % 10 == 0:
                    effect_manager.add_ripple(click_pt, color=(0, 255, 0))

            # ========== 撤销/重做手势检测 ==========
            # 三指滑动：左滑撤销，右滑重做
            if g["swipe"] and g["three_fingers"]:
                if g["swipe"] == "SWIPE_LEFT":
                    if canvas.undo():
                        undo_redo_hint = "Undo"
                        undo_redo_hint_frames = 30
                        effect_manager.add_ripple(draw_pt, color=(0, 0, 255))
                        print("撤销")
                elif g["swipe"] == "SWIPE_RIGHT":
                    if canvas.redo():
                        undo_redo_hint = "Redo"
                        undo_redo_hint_frames = 30
                        effect_manager.add_ripple(draw_pt, color=(0, 255, 0))
                        print("重做")
            
            # ========== 工具快捷切换手势 ==========
            # 食指单指：上滑切换上一个工具，下滑切换下一个工具
            if g["swipe"] and g["index_only"]:
                if g["swipe"] == "SWIPE_UP":
                    brush_manager.current_tool_index = (brush_manager.current_tool_index - 1) % len(brush_manager.TOOLS)
                    undo_redo_hint = f"Tool: {brush_manager.tool.upper()}"
                    undo_redo_hint_frames = 30
                    effect_manager.add_ripple(draw_pt, color=(255, 255, 0))
                    print(f"工具切换到: {brush_manager.tool}")
                elif g["swipe"] == "SWIPE_DOWN":
                    brush_manager.current_tool_index = (brush_manager.current_tool_index + 1) % len(brush_manager.TOOLS)
                    undo_redo_hint = f"Tool: {brush_manager.tool.upper()}"
                    undo_redo_hint_frames = 30
                    effect_manager.add_ripple(draw_pt, color=(255, 255, 0))
                    print(f"工具切换到: {brush_manager.tool}")

            # ========== 手势UI交互 ==========
            if g["index_middle"]:
                if not hasattr(gesture_ui, '_two_finger_triggered'):
                    gesture_ui._two_finger_triggered = False
                if not gesture_ui._two_finger_triggered:
                    gesture_ui.toggle_visibility()
                    gesture_ui._two_finger_triggered = True
                    print(f"UI {'显示' if gesture_ui.visible else '隐藏'}")
            else:
                if hasattr(gesture_ui, '_two_finger_triggered'):
                    gesture_ui._two_finger_triggered = False

            # 更新UI悬停状态（在非捏合状态下更新）
            if gesture_ui.visible:
                if not g["pinching"]:
                    hover_item = gesture_ui.update_hover(draw_pt, brush_manager)

            # 捏合选择UI项 / 停留自动选择（仅工具/动作）
            if gesture_ui.visible:
                ui_action = None
                dwell_item = None
                if g["pinch_start"] and gesture_ui.hover_item:
                    select_result = gesture_ui.select_hover_item(brush_manager)
                    ui_action = select_result["action"]
                elif not g["pinching"]:
                    dwell_item = gesture_ui.consume_dwell_item()
                    if dwell_item:
                        gesture_ui.hover_item = dwell_item
                        select_result = gesture_ui.select_hover_item(brush_manager)
                        ui_action = select_result["action"]

                if ui_action or (gesture_ui.hover_item and (g["pinch_start"] or dwell_item)):
                    item_type, _ = gesture_ui.hover_item
                    if item_type == "tool":
                        print(f"工具切换到: {brush_manager.tool}")
                    elif item_type == "color":
                        print(f"颜色切换到: {brush_manager.color_name}")
                    elif item_type == "thickness":
                        print(f"粗细切换到: {brush_manager.thickness}")
                    elif item_type == "brush":
                        print(f"笔刷切换到: {brush_manager.brush_type}")
                    elif item_type == "action":
                        if ui_action == "clear":
                            canvas.clear()
                            temp_ink_manager.clear()
                            particle_system.clear()
                            undo_redo_hint = "Clear"
                            undo_redo_hint_frames = 30
                            print("画布已清空")
                        elif ui_action == "particles":
                            ENABLE_PARTICLES = not ENABLE_PARTICLES
                            if not ENABLE_PARTICLES:
                                particle_system.clear()
                            print(f"Particle effects: {'ON' if ENABLE_PARTICLES else 'OFF'}")

                    draw_lock = DRAW_LOCK_FRAMES
                    effect_manager.add_ripple(draw_pt, color=(255, 255, 255))

            # ========== 工具逻辑 (基于当前选中的Tool执行) ==========
            
            # 1. 笔画开始/结束管理
            if g["pinch_start"]:
                if brush_manager.tool == "pen":
                    pen.start_stroke()
                elif brush_manager.tool == "laser":
                    if temp_ink_manager.current_stroke is None:
                        temp_ink_manager.start_stroke(color=(0, 0, 255), thickness=4)
                draw_lock = 0

            # 2. 执行工具动作 (捏合时)
            if g["pinching"] and draw_lock == 0:
                if brush_manager.tool == "pen":
                    # 画笔模式
                    smoothed_pt = pen.draw(draw_pt)
                    if ENABLE_PARTICLES:
                        particle_system.emit(draw_pt, brush_manager.color)
                
                elif brush_manager.tool == "eraser":
                    # 橡皮模式 (捏合时擦除)
                    eraser.erase(draw_pt)
                    
                elif brush_manager.tool == "laser":
                    # 激光笔模式 (捏合时画轨迹)
                    temp_ink_manager.add_point(draw_pt)
            
            # 3. 结束动作 (捏合结束)
            if g["pinch_end"]:
                if brush_manager.tool == "pen":
                    finished_points = pen.end_stroke()
                    draw_lock = DRAW_LOCK_FRAMES
                    if finished_points:
                        # 图形识别与美化
                        beautified = shape_recognizer.beautify(
                            finished_points,
                            canvas.get_canvas(),
                            brush_manager.color,
                            brush_manager.thickness,
                        )
                        canvas.save_stroke()
                        if beautified:
                            print(f"识别到图形: {beautified}")
                            center = np.mean(finished_points, axis=0).astype(int)
                            effect_manager.add_ripple(tuple(center), color=(0, 255, 0))
                
                elif brush_manager.tool == "laser":
                    temp_ink_manager.end_stroke()

            detector.draw_hand(frame, hand)
        else:
            # 无手时强制断笔并重置
            pen.end_stroke()
            temp_ink_manager.end_stroke()
            palm_hud.reset()
            gesture.reset_pinch_history()

        # ========== 特效更新与渲染 ==========
        
        temp_ink_manager.update()
        effect_manager.update()
        if ENABLE_PARTICLES:
            particle_system.update()

        frame = overlay_canvas(frame, canvas.get_canvas())

        if ENABLE_PARTICLES:
            particle_system.render(frame)
        
        # 激光笔渲染 (只要是Laser模式，始终显示光标)
        if ENABLE_LASER:
            temp_ink_manager.render(frame)
            # 如果当前工具是激光笔，且检测到手，就显示光标
            if brush_manager.tool == "laser" and ui_draw_pt:
                laser_pointer.render(frame, ui_draw_pt)

        effect_manager.render(frame)

        if ENABLE_PALM_HUD and palm_pos_for_hud and palm_pos_pixel:
            palm_hud.update(palm_pos_for_hud)
            if palm_hud.is_still and current_mode != "erase":
                palm_hud.render(frame, palm_pos_pixel)

        # ========== UI提示绘制 ==========
        if ui_draw_pt is not None:
            # 画笔模式：显示小圈
            if brush_manager.tool == "pen":
                cv2.circle(frame, ui_draw_pt, 6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                if g and g["pinching"]:
                    cv2.circle(frame, ui_draw_pt, 3, (0, 200, 200), -1, lineType=cv2.LINE_AA)
            # 橡皮模式：显示大圈
            elif brush_manager.tool == "eraser":
                cv2.circle(frame, ui_draw_pt, config.ERASER_SIZE, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                if g and g["pinching"]:
                    cv2.circle(frame, ui_draw_pt, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            
        if g and g["pinching"]:
            cv2.putText(frame, f"pinch: {ui_pinch_dist:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if undo_redo_hint:
            alpha = min(1.0, undo_redo_hint_frames / 15.0)
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.putText(frame, undo_redo_hint, (config.CAMERA_WIDTH // 2 - 50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, lineType=cv2.LINE_AA)

        # FPS计算
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        # 状态信息
        status_lines = [
            f"FPS: {fps:.1f}",
            brush_manager.get_status_text(),
            canvas.get_history_info(),
        ]

        # 效果状态
        effect_status = []
        if ENABLE_PEN_EFFECT:
            effect_status.append("Pen")
        if ENABLE_LINE_ASSIST:
            effect_status.append("Line")
        if ENABLE_PARTICLES:
            effect_status.append(f"Particles:{particle_system.get_count()}")
        if ENABLE_LASER:
            effect_status.append("Laser(Fade)")
        if ENABLE_PALM_HUD:
            effect_status.append("HUD")

        if effect_status:
            status_lines.append(f"Effects: {' | '.join(effect_status)}")

        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示帮助信息
        if SHOW_HELP:
            help_text = [
                "=== AirCanvas Controls ===",
                "Tool Selection (Left Panel):",
                "  - Pen: Draw (Pinch)",
                "  - Eraser: Erase (Pinch)",
                "  - Laser: Point & Fade (Pinch)",
                "Hand Gestures:",
                "  Pinch: Activate current tool",
                "  2 fingers: Toggle UI",
                "  3 fingers swipe: Undo/Redo",
                "Keyboard:",
                "  q: Quit  c: Clear  s: Save",
                "  z: Undo  y: Redo  h: Help",
                "  t: Cycle Tools",
            ]
            overlay = frame.copy()
            y_offset = 100
            for i, text in enumerate(help_text):
                cv2.putText(overlay, text, (50, y_offset + i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # 渲染手势UI界面
        gesture_ui.render(frame, brush_manager, action_state={"particles": ENABLE_PARTICLES})

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        # ========== 键盘控制 ==========
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas.clear()
            temp_ink_manager.clear()
            print("画布已清空")
        if key == ord("s"):
            out_path = Path("captures") / f"canvas_{save_counter}.png"
            out_path.parent.mkdir(exist_ok=True)
            canvas.save(str(out_path))
            print(f"Saved canvas to {out_path}")
            save_counter += 1
        if key == ord("z"):
            if canvas.undo():
                undo_redo_hint = "Undo"
                undo_redo_hint_frames = 30
                print("撤销")
        if key == ord("y"):
            if canvas.redo():
                undo_redo_hint = "Redo"
                undo_redo_hint_frames = 30
                print("重做")
        if key == ord('1'):
            ENABLE_PARTICLES = not ENABLE_PARTICLES
            if not ENABLE_PARTICLES:
                particle_system.clear()
            print(f"Particle effects: {'ON' if ENABLE_PARTICLES else 'OFF'}")
        if key == ord('2'):
            ENABLE_LASER = not ENABLE_LASER
            temp_ink_manager.clear()
            print(f"Laser pointer: {'ON' if ENABLE_LASER else 'OFF'}")
        if key == ord('3'):
            ENABLE_PALM_HUD = not ENABLE_PALM_HUD
            palm_hud.reset()
            print(f"Palm HUD: {'ON' if ENABLE_PALM_HUD else 'OFF'}")
        if key == ord('l'):
            ENABLE_LINE_ASSIST = not ENABLE_LINE_ASSIST
            shape_recognizer.set_line_assist(ENABLE_LINE_ASSIST)
            print(f"Line assist: {'ON' if ENABLE_LINE_ASSIST else 'OFF'}")
        if key == ord('r'):
            palm_hud.reset_timer()
            print("Timer reset")
        if key == ord('h'):
            SHOW_HELP = not SHOW_HELP
        if key == ord('t'):
            brush_manager.next_tool()
            print(f"Tool switched to: {brush_manager.tool}")
        if key == ord('['):
            brush_manager.prev_color()
            print(f"Color: {brush_manager.color_name}")
        if key == ord(']'):
            brush_manager.next_color()
            print(f"Color: {brush_manager.color_name}")
        if key == ord('-') or key == ord('_'):
            brush_manager.prev_thickness()
            print(f"Thickness: {brush_manager.thickness}")
        if key == ord('=') or key == ord('+'):
            brush_manager.next_thickness()
            print(f"Thickness: {brush_manager.thickness}")
        if key == ord('b'):
            brush_manager.next_brush_type()
            print(f"Brush type: {brush_manager.brush_type}")
        if key == ord('p'):
            ENABLE_PEN_EFFECT = not ENABLE_PEN_EFFECT
            pen.enable_pen_effect = ENABLE_PEN_EFFECT
            print(f"Pen effect: {'ON' if ENABLE_PEN_EFFECT else 'OFF'}")
        if key == ord('w'):
            FULLSCREEN = not FULLSCREEN
            if FULLSCREEN:
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print(f"Fullscreen: {'ON' if FULLSCREEN else 'OFF'}")

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
