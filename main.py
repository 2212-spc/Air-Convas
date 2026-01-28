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
from dataclasses import dataclass

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
from modules.ppt_gesture_controller import PPTGestureController
from modules.temporary_ink import TemporaryInkManager
from modules.visual_effects import EffectManager
from modules.interactive_effects import InteractiveEffectsManager
from modules.tutorial_manager import TutorialManager
from modules.particle_mode_manager import ParticleModeManager


@dataclass
class LandmarkAdapter:
    """Adapter to make HandDetector landmarks compatible with PPTGestureController"""
    x: float
    y: float
    z: float = 0.0


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
    
    # 摄像头优化设置 - 提高清晰度
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 关闭自动对焦
    cap.set(cv2.CAP_PROP_FOCUS, 0)       # 手动对焦 (0=无限远, 根据需要调整)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟

    # 创建可调整大小的窗口
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(config.WINDOW_NAME, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    FULLSCREEN = False
    
    # 鼠标点击处理的全局变量
    mouse_clicked = False
    mouse_click_pos = None
    
    def mouse_callback(event, x, y, flags, param):
        """鼠标回调函数"""
        nonlocal mouse_clicked, mouse_click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True
            mouse_click_pos = (x, y)
    
    # 设置鼠标回调
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

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
    )
    
    # UI选择专用mapper（无平滑，直接跟踪指尖）
    ui_mapper = CoordinateMapper(
        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        getattr(config, 'ACTIVE_REGION_DRAW', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=0.0,  # 无平滑，直接响应
        smoothing_mode='ema',
        one_euro_min_cutoff=one_euro_min_cutoff,
        one_euro_beta=one_euro_beta,
    )

    # --- PPT 控制器初始化 ---
    # 注入 cursor_mapper，确保 PPT 模式下的光标平滑度与绘图模式一致
    ppt_controller = PPTGestureController(external_mp=True, cursor_mapper=cursor_mapper)
    APP_MODE = "DRAW"  # "DRAW" or "PPT"

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
    
    # 互动特效管理器
    interactive_effects = InteractiveEffectsManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    ENABLE_INTERACTIVE_EFFECTS = False  # 特效模式开关

    # 手势UI界面
    gesture_ui = GestureUI(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # 教程管理器
    tutorial_manager = TutorialManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    
    # 粒子特效管理器
    particle_mode_manager = ParticleModeManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    fps = 0
    last_time = time.time()
    frame_count = 0
    save_counter = 0

    # 模式控制
    draw_lock = 0
    DRAW_LOCK_FRAMES = getattr(config, 'DRAW_LOCK_FRAMES', 5)
    
    # 画画状态追踪（用于死区检测）
    is_drawing = False  # 是否正在画画

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

        frame = cv2.flip(frame, 1)

        # 如果教程激活，只渲染教程界面和基本视频
        if tutorial_manager.is_active:
            # 获取手部位置用于点击检测
            hands = detector.detect(frame)
            cursor_pos = None
            if hands:
                hand = hands[0]
                index_norm = hand.landmarks_norm[INDEX_TIP]
                cursor_pos = draw_mapper.map(index_norm)
                detector.draw_hand(frame, hand)
            
            # 渲染教程
            tutorial_manager.render(frame, cursor_pos)
            
            # 检测手势点击（捏合手势作为点击）
            if hands:
                g = gesture.classify(hand)
                if g["pinch_start"]:
                    if cursor_pos:
                        tutorial_manager.handle_click(cursor_pos)
            
            # 检测鼠标点击
            if mouse_clicked:
                if mouse_click_pos:
                    tutorial_manager.handle_click(mouse_click_pos)
                mouse_clicked = False
                mouse_click_pos = None
            
            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            
            # 教程中的键盘处理
            if key == ord("q"):
                break
            elif key != 255:  # 任意其他按键
                tutorial_manager.handle_key(key)
            
            continue  # 跳过正常游戏逻辑
        
        # ========== 粒子特效模式 ==========
        if particle_mode_manager.is_active:
            # 检测手部
            hands = detector.detect(frame)
            
            # 如果在UI选择阶段
            if particle_mode_manager.is_ui_selecting:
                # 保存原始摄像头画面
                display_frame = frame.copy()
                
                # 渲染UI
                particle_mode_manager.render(display_frame, hands)
                
                # 处理手势点击 - 只使用食指尖作为控制点
                cursor_pos = None
                if hands:
                    hand = hands[0]
                    g = gesture.classify(hand)
                    
                    # 获取食指尖位置（唯一控制点）
                    index_norm = hand.landmarks_norm[INDEX_TIP]
                    cursor_pos = draw_mapper.map(index_norm)
                    
                    # 在UI区域只显示一个控制点（绿色圆点）
                    cv2.circle(display_frame, cursor_pos, 8, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(display_frame, cursor_pos, 10, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    
                    # 更新UI悬停状态（使用食指尖位置）
                    particle_mode_manager.ui.handle_mouse(cv2.EVENT_MOUSEMOVE, cursor_pos[0], cursor_pos[1], 0, None)
                    
                    # 捏合作为点击
                    if g["pinch_start"] and cursor_pos:
                        particle_mode_manager.handle_click(cursor_pos)
                
                # 处理鼠标点击
                if mouse_clicked and mouse_click_pos:
                    particle_mode_manager.handle_click(mouse_click_pos)
                    mouse_clicked = False
                    mouse_click_pos = None
                
                cv2.imshow(config.WINDOW_NAME, display_frame)
            else:
                # 粒子特效显示阶段
                # 创建纯黑背景
                black_frame = np.zeros_like(frame)
                
                # 更新粒子系统
                particle_mode_manager.update(hands)
                
                # 渲染粒子特效到黑色背景
                particle_mode_manager.render(black_frame, hands)
                
                # 绘制摄像头画面到左下角
                camera_with_hand = frame.copy()
                if hands:
                    detector.draw_hand(camera_with_hand, hands[0])
                particle_mode_manager.render_camera_feed(black_frame, camera_with_hand)
                
                cv2.imshow(config.WINDOW_NAME, black_frame)
            
            # 键盘处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            particle_mode_manager.handle_keyboard(key)
            
            continue  # 跳过正常绘图逻辑

        if draw_lock > 0:
            draw_lock -= 1

        # 撤销/重做提示淡出
        if undo_redo_hint_frames > 0:
            undo_redo_hint_frames -= 1
        else:
            undo_redo_hint = ""
        
        hands = detector.detect(frame)
        
        current_mode = "idle"
        ui_draw_pt = None
        ui_erase_pt = None
        ui_pinching = False
        ui_pinch_dist = 0.0
        palm_pos_for_hud = None
        index_tip_pt = None  # UI选择专用点（无平滑）
        palm_pos_pixel = None
        g = None  # 手势结果

        if hands:
            detector.draw_hand(frame, hands[0])

        if APP_MODE == "PPT":
            # --- PPT 演示模式逻辑 ---
            if hands:
                hand = hands[0]
                # 适配器转换：HandDetector -> PPTGestureController
                # 补充 z=0.0
                landmarks = [LandmarkAdapter(x=lm[0], y=lm[1], z=0.0) for lm in hand.landmarks_norm]
                ppt_controller.process_hand_data(landmarks, frame)
            
            # 显示 PPT 模式特定的 UI
            cv2.putText(frame, "MODE: PPT PRESENTATION (Tab to Switch)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 计算 FPS
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            # --- AirCanvas 绘图模式逻辑 ---
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
                
                # UI选择专用点（无平滑，直接跟踪食指尖）
                index_tip_pt = ui_mapper.map(index_norm)

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
                # 禁用双指切换UI（容易误触），改用键盘 'u' 键切换
                # if g["index_middle"]:
                #     ...
                pass  # UI切换已移至键盘快捷键

                # 更新UI悬停状态（在非捏合状态下更新）
                if gesture_ui.visible and index_tip_pt is not None:
                    if not g["pinching"]:
                        hover_item = gesture_ui.update_hover(index_tip_pt, brush_manager)

                # 捏合选择UI项 / 停留自动选择（仅工具/动作）
                if gesture_ui.visible:
                    ui_action = None
                    dwell_selected = False
                    if g["pinch_start"] and gesture_ui.hover_item:
                        select_result = gesture_ui.select_hover_item(brush_manager)
                        ui_action = select_result["action"]
                    elif not g["pinching"]:
                        # 使用新的悬停自动选择接口
                        dwell_result = gesture_ui.consume_pending_selection(brush_manager)
                        if dwell_result["selected"]:
                            dwell_selected = True
                            ui_action = dwell_result["action"]

                    if ui_action or (gesture_ui.hover_item and (g["pinch_start"] or dwell_selected)):
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
                                # 切换粒子特效模式
                                if particle_mode_manager.is_active:
                                    particle_mode_manager.deactivate()
                                else:
                                    particle_mode_manager.activate()
                                print(f"3D Particle Mode: {'ON' if particle_mode_manager.is_active else 'OFF'}")
                            elif ui_action == "effects":
                                ENABLE_INTERACTIVE_EFFECTS = interactive_effects.toggle()
                                undo_redo_hint = f"Effects: {interactive_effects.get_effect_label()}"
                                undo_redo_hint_frames = 45
                                print(f"Interactive effects: {'ON' if ENABLE_INTERACTIVE_EFFECTS else 'OFF'}")

                        draw_lock = DRAW_LOCK_FRAMES
                        effect_manager.add_ripple(draw_pt, color=(255, 255, 255))
                
                # 鼠标点击UI选择
                if mouse_clicked and mouse_click_pos:
                    # 如果UI可见，处理UI点击
                    if gesture_ui.visible:
                        if gesture_ui.handle_mouse_click(mouse_click_pos, brush_manager):
                            # 处理UI动作
                            if gesture_ui.hover_item:
                                item_type, _ = gesture_ui.hover_item
                                if item_type == "tool":
                                    print(f"工具切换到: {brush_manager.tool}")
                                    effect_manager.add_ripple(mouse_click_pos, color=(255, 255, 0))
                                elif item_type == "color":
                                    print(f"颜色切换到: {brush_manager.color_name}")
                                    effect_manager.add_ripple(mouse_click_pos, color=brush_manager.color)
                                elif item_type == "thickness":
                                    print(f"粗细切换到: {brush_manager.thickness}")
                                    effect_manager.add_ripple(mouse_click_pos, color=(0, 255, 255))
                                elif item_type == "brush":
                                    print(f"笔刷切换到: {brush_manager.brush_type}")
                                    effect_manager.add_ripple(mouse_click_pos, color=(255, 0, 255))
                                elif item_type == "action":
                                    # 执行动作
                                    action_key = gesture_ui.action_items[gesture_ui.hover_item[1]][0]
                                    if action_key == "clear":
                                        canvas.clear()
                                        temp_ink_manager.clear()
                                        particle_system.clear()
                                        undo_redo_hint = "Clear"
                                        undo_redo_hint_frames = 30
                                        effect_manager.add_ripple(mouse_click_pos, color=(255, 0, 0))
                                        print("画布已清空")
                                    elif action_key == "particles":
                                        # 切换粒子特效模式
                                        if particle_mode_manager.is_active:
                                            particle_mode_manager.deactivate()
                                        else:
                                            particle_mode_manager.activate()
                                        effect_manager.add_ripple(mouse_click_pos, color=(0, 255, 0))
                                        print(f"3D Particle Mode: {'ON' if particle_mode_manager.is_active else 'OFF'}")
                                    elif action_key == "effects":
                                        ENABLE_INTERACTIVE_EFFECTS = interactive_effects.toggle()
                                        undo_redo_hint = f"Effects: {interactive_effects.get_effect_label()}"
                                        undo_redo_hint_frames = 45
                                        effect_manager.add_ripple(mouse_click_pos, color=(255, 200, 0))
                                        print(f"Interactive effects: {'ON' if ENABLE_INTERACTIVE_EFFECTS else 'OFF'}")
                            
                            draw_lock = DRAW_LOCK_FRAMES
                    
                    # 无论是否点击到按钮，都清除鼠标点击标志
                    mouse_clicked = False
                    mouse_click_pos = None

                # ========== 工具逻辑 (基于当前选中的Tool执行) ==========
                
                # 1. 笔画开始/结束管理
                if g["pinch_start"]:
                    # 死区检测：在死区内完全不允许开始画画
                    in_dead_zone = gesture_ui.is_in_dead_zone(draw_pt, brush_manager)
                    
                    if in_dead_zone:
                        # 在死区内，不允许开始画画
                        print("在按钮区域，无法开始画画")
                    else:
                        # 不在死区，允许开始画画
                        if brush_manager.tool == "pen":
                            pen.start_stroke()
                            is_drawing = True  # 标记进入画画状态
                        elif brush_manager.tool == "eraser":
                            is_drawing = True  # 橡皮也需要标记
                        elif brush_manager.tool == "laser":
                            if temp_ink_manager.current_stroke is None:
                                temp_ink_manager.start_stroke(color=(0, 0, 255), thickness=4)
                            is_drawing = True  # 标记进入画画状态
                    draw_lock = 0

                # 2. 执行工具动作 (捏合时) - 死区检查
                if g["pinching"] and draw_lock == 0:
                    # 每一帧都检查是否在死区，在死区则不执行画画操作
                    in_dead_zone = gesture_ui.is_in_dead_zone(draw_pt, brush_manager)
                    
                    if in_dead_zone:
                        # 在死区内，完全禁止画画
                        if is_drawing:
                            # 如果之前在画，现在进入死区，暂停画画但不结束笔画
                            print("进入按钮死区，暂停画画")
                    else:
                        # 不在死区，正常画画
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
                    # 所有工具都清除画画状态
                    is_drawing = False  # 清除画画状态
                    
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
                    
                    elif brush_manager.tool == "eraser":
                        pass  # 橡皮不需要特殊结束处理

                detector.draw_hand(frame, hand)
            else:
                # 无手时强制断笔并重置
                pen.end_stroke()
                temp_ink_manager.end_stroke()
                palm_hud.reset()
                gesture.reset_pinch_history()
                is_drawing = False  # 清除画画状态

            # ========== 特效更新与渲染 ==========
            
            temp_ink_manager.update()
            effect_manager.update()
            if ENABLE_PARTICLES:
                particle_system.update()
            
            # 互动特效更新
            if ENABLE_INTERACTIVE_EFFECTS:
                is_open_palm = (sum(g["fingers"]) >= 4) if g else False  # 4+手指张开=张开手掌
                is_pinching = g["pinching"] if g else False
                interactive_effects.update(ui_draw_pt, is_open_palm, is_pinching)

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
            
            # 渲染互动特效
            if ENABLE_INTERACTIVE_EFFECTS:
                is_open_palm = (sum(g["fingers"]) >= 4) if g else False  # 4+手指张开=张开手掌
                is_pinching = g["pinching"] if g else False
                interactive_effects.render(frame, ui_draw_pt, is_open_palm, is_pinching)

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
                    
            # UI选择指示点（绿色小点，显示在食指尖位置）
            if index_tip_pt is not None and gesture_ui.visible:
                cv2.circle(frame, index_tip_pt, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                
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
            
            # 工具大标题提示（醒目显示当前工具）
            tool_display = {
                "pen": "TOOL: PEN (Draw)",
                "eraser": "TOOL: ERASER", 
                "laser": "TOOL: LASER (Fades in 1.5s)"
            }
            tool_text = tool_display.get(brush_manager.tool, brush_manager.tool.upper())
            tool_color = (0, 255, 255) if brush_manager.tool == "pen" else (0, 165, 255) if brush_manager.tool == "laser" else (0, 0, 255)
            cv2.putText(frame, tool_text, (config.CAMERA_WIDTH // 2 - 150, 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, tool_color, 2, lineType=cv2.LINE_AA)

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
            
            # 显示当前模式提示 (Tab切换)
            cv2.putText(frame, "Tab: Switch to PPT Mode", (10, config.CAMERA_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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
                    "  t: Cycle Tools  5: Particle FX",
                    "  Tab: Switch Mode (Draw/PPT)",
                ]
                overlay = frame.copy()
                y_offset = 100
                for i, text in enumerate(help_text):
                    cv2.putText(overlay, text, (50, y_offset + i * 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            # 渲染手势UI界面
            gesture_ui.render(frame, brush_manager, action_state={
                "particles": particle_mode_manager.is_active, 
                "effects": ENABLE_INTERACTIVE_EFFECTS
            })

            # 渲染Help按钮（如果教程已完成）
            help_cursor_pos = None
            if ui_draw_pt:
                help_cursor_pos = ui_draw_pt
            tutorial_manager.render(frame, help_cursor_pos)
            
            # 检测Help按钮点击（捏合手势）
            if g and g["pinch_start"] and help_cursor_pos:
                tutorial_manager.handle_click(help_cursor_pos)
            
            # 检测Help按钮鼠标点击
            if mouse_clicked:
                if mouse_click_pos:
                    tutorial_manager.handle_click(mouse_click_pos)
                mouse_clicked = False
                mouse_click_pos = None

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        # ========== 键盘控制 ==========
        if key == ord("q"):
            break
        
        # PPT 模式切换
        if key == 9:  # Tab key
            if APP_MODE == "DRAW":
                APP_MODE = "PPT"
                print(">>> 切换到 PPT 演示模式")
                draw_lock = 0
                pen.end_stroke()
            else:
                APP_MODE = "DRAW"
                print(">>> 切换到 AirCanvas 绘图模式")
                # 重置 PPT 控制器状态
                ppt_controller.gesture_history.clear()

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
        if key == ord('4'):
            ENABLE_INTERACTIVE_EFFECTS = interactive_effects.toggle()
            print(f"Interactive effects: {'ON' if ENABLE_INTERACTIVE_EFFECTS else 'OFF'} - {interactive_effects.get_effect_label()}")
        if key == ord('5'):
            # 切换粒子特效模式
            if particle_mode_manager.is_active:
                particle_mode_manager.deactivate()
            else:
                particle_mode_manager.activate()
            print(f"Particle Mode: {'ON' if particle_mode_manager.is_active else 'OFF'}")
        if key == ord('l'):
            ENABLE_LINE_ASSIST = not ENABLE_LINE_ASSIST
            shape_recognizer.set_line_assist(ENABLE_LINE_ASSIST)
            print(f"Line assist: {'ON' if ENABLE_LINE_ASSIST else 'OFF'}")
        if key == ord('u'):
            gesture_ui.toggle_visibility()
            print(f"UI: {'ON' if gesture_ui.visible else 'OFF'}")
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
