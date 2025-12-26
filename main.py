import time
from pathlib import Path

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
from modules.canvas import Canvas
from modules.eraser import Eraser
from modules.shape_recognizer import ShapeRecognizer
from modules.virtual_pen import VirtualPen
from modules.particle_system import ParticleSystem
from modules.laser_pointer import LaserPointer
from modules.palm_hud import PalmHUD
from modules.brush_manager import BrushManager
from modules.gesture_ui import GestureUI
from modules.particle_mode_manager import ParticleModeManager
from modules.particle_mode_ui import ParticleModeUI
from modules.particle_system_3d import ParticleSystem3D


def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Composite non-black canvas pixels onto the frame."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)


def palm_center(hand) -> tuple:
    # Use wrist, index mcp (5), pinky mcp (17) for a stable palm anchor
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

    # 创建可调整大小的窗口（解决窗口太小无法放大的问题）
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(config.WINDOW_NAME, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    FULLSCREEN = False  # 全屏状态标记

    detector = HandDetector(max_num_hands=1)

    # 降低 pyautogui 调用的系统性延迟
    if pyautogui:
        try:
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False
        except Exception:
            pass
    gesture = GestureRecognizer(
        pinch_threshold=config.PINCH_THRESHOLD,
        pinch_release_threshold=config.PINCH_RELEASE_THRESHOLD,
        swipe_threshold=config.SWIPE_THRESHOLD,
        palm_spread_threshold=getattr(config, 'PALM_SPREAD_THRESHOLD', 0.08),
        swipe_velocity_threshold=getattr(config, 'SWIPE_VELOCITY_THRESHOLD', 0.015),
        swipe_cooldown_frames=config.SWIPE_COOLDOWN_FRAMES,
        pinch_confirm_frames=getattr(config, 'PINCH_CONFIRM_FRAMES', 3),
        pinch_release_confirm_frames=getattr(config, 'PINCH_RELEASE_CONFIRM_FRAMES', 1),
    )

    # 画图映射器：使用原始配置
    draw_mapper = CoordinateMapper(
        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        getattr(config, 'ACTIVE_REGION_DRAW', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'DRAW_SMOOTHING_FACTOR', 0.3),
        use_one_euro=False,  # 关闭 One Euro，使用传统 EMA
        one_euro_preset="DRAWING",
    )

    if pyautogui:
        SCREEN_W, SCREEN_H = pyautogui.size()
    else:
        SCREEN_W, SCREEN_H = getattr(config, 'SCREEN_WIDTH', 1920), getattr(config, 'SCREEN_HEIGHT', 1080)

    # 光标映射器：使用原始配置
    cursor_mapper = CoordinateMapper(
        (SCREEN_W, SCREEN_H),
        getattr(config, 'ACTIVE_REGION_CURSOR', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'CURSOR_SMOOTHING_FACTOR', 0.15),
        use_one_euro=False,  # 关闭 One Euro，使用传统 EMA
        one_euro_preset="CURSOR",
    )

    canvas = Canvas(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # 笔刷管理器
    brush_manager = BrushManager()

    pen = VirtualPen(
        canvas=canvas,
        brush_manager=brush_manager,
        smoothing=None,  # mapper already smooths movement
        jump_threshold=getattr(config, 'STROKE_JUMP_THRESHOLD', 80),
    )
    eraser = Eraser(canvas, size=config.ERASER_SIZE)
    shape_recognizer = ShapeRecognizer()

    # 阶段四：AR增强效果
    particle_system = ParticleSystem(
        max_particles=config.MAX_PARTICLES,
        emit_count=config.PARTICLE_EMIT_COUNT
    )
    laser_pointer = LaserPointer()
    palm_hud = PalmHUD()

    # 手势UI界面
    gesture_ui = GestureUI(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    
    # 3D粒子系统
    particle_system_3d = ParticleSystem3D()
    particle_ui = ParticleModeUI(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    
    # 确定按钮回调函数
    def confirm_particle_mode():
        particle_system_3d.active = True
        if len(particle_system_3d.particles) == 0:
            print("正在初始化5000个3D粒子（性能优化）...")
            particle_system_3d.initialize_particles(5000)
        particle_ui.hide()
        print("3D粒子模式已激活！")
    
    # 取消按钮回调函数
    def cancel_particle_mode():
        particle_system_3d.active = False
        particle_system_3d.reset()
        particle_ui.hide()
        print("粒子模式已取消")
    
    # 设置UI回调
    particle_ui.on_model_change = lambda model: particle_system_3d.set_model(model)
    particle_ui.on_color_change = lambda color: particle_system_3d.set_color(color)
    particle_ui.on_confirm = confirm_particle_mode
    particle_ui.on_cancel = cancel_particle_mode
    
    # 鼠标回调
    def mouse_callback(event, x, y, flags, param):
        particle_ui.handle_mouse(event, x, y, flags, param)
    
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

    # 激光笔拖尾历史
    laser_trail = []
    max_trail_length = 10

    fps = 0
    last_time = time.time()
    frame_count = 0
    save_counter = 0

    # 模式控制
    draw_lock = 0  # 捏合结束后短暂冷却，防误连笔
    DRAW_LOCK_FRAMES = getattr(config, 'DRAW_LOCK_FRAMES', 5)

    # AR效果控制开关
    ENABLE_PARTICLES = True
    ENABLE_LASER = True
    ENABLE_PALM_HUD = True

    # 显示帮助信息
    SHOW_HELP = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧")
            break

        if draw_lock > 0:
            draw_lock -= 1

        frame = cv2.flip(frame, 1)
        hands = detector.detect(frame)
        current_mode = "idle"
        ui_draw_pt = None
        ui_erase_pt = None
        ui_pinching = False
        ui_pinch_dist = 0.0
        palm_pos_for_hud = None
        palm_pos_pixel = None  # 掌心的像素坐标（用于HUD显示）
        has_hand_this_frame = len(hands) > 0

        if hands:
            hand = hands[0]
            g = gesture.classify(hand)
            current_mode = g["mode"]

            index_norm = hand.landmarks_norm[INDEX_TIP]
            thumb_norm = hand.landmarks_norm[THUMB_TIP]
            middle_norm = hand.landmarks_norm[MIDDLE_TIP]
            palm_norm = palm_center(hand)

            # 作为笔尖：三指捏合时用三指中心点，否则用食指尖
            if g["pinching"] or current_mode == "draw":
                tip_norm = (
                    (index_norm[0] + thumb_norm[0] + middle_norm[0]) / 3.0,
                    (index_norm[1] + thumb_norm[1] + middle_norm[1]) / 3.0
                )
            else:
                tip_norm = index_norm

            draw_pt = draw_mapper.map(tip_norm)
            erase_pt = draw_mapper.map(palm_norm)
            screen_pt = cursor_mapper.map(index_norm)

            # 更新掌心HUD位置（归一化坐标和像素坐标）
            palm_pos_for_hud = palm_norm
            palm_pos_pixel = draw_mapper.map(palm_norm)

            # 记录UI提示点位
            ui_draw_pt = draw_pt
            ui_erase_pt = erase_pt if g["open_palm"] else None
            ui_pinching = g["pinching"]
            ui_pinch_dist = g["pinch_distance"]

            # ========== 手势UI交互 ==========
            # 两指（食指+中指）切换UI显示
            if g["index_middle"]:
                # 检测到两指且之前没有在两指状态（避免重复触发）
                if not hasattr(gesture_ui, '_two_finger_triggered'):
                    gesture_ui._two_finger_triggered = False

                if not gesture_ui._two_finger_triggered:
                    gesture_ui.toggle_visibility()
                    gesture_ui._two_finger_triggered = True
                    print(f"UI {'显示' if gesture_ui.visible else '隐藏'}")
            else:
                if hasattr(gesture_ui, '_two_finger_triggered'):
                    gesture_ui._two_finger_triggered = False

            # 更新UI悬停状态（在多种手势下都更新，包括食指指向、捏合准备和捏合中）
            # 关键修复：不再限制为 index_only，让悬停在捏合过程中也能保持
            if gesture_ui.visible:
                # 在非擦除、非拳头状态下都更新悬停检测
                if not g["open_palm"] and not g["fist"]:
                    hover_item = gesture_ui.update_hover(draw_pt, brush_manager)

            # 捏合选择UI项（只在UI可见且悬停在按钮上时）
            if g["pinch_start"] and gesture_ui.visible and gesture_ui.hover_item:
                selected = gesture_ui.select_hover_item(brush_manager)
                if selected:
                    item_type, _ = gesture_ui.hover_item
                    if item_type == "color":
                        print(f"颜色切换到: {brush_manager.color_name}")
                    elif item_type == "thickness":
                        print(f"粗细切换到: {brush_manager.thickness}")
                    elif item_type == "brush":
                        print(f"笔刷切换到: {brush_manager.brush_type}")
                    # 选择后短暂跳过绘图，避免选择时误画
                    draw_lock = DRAW_LOCK_FRAMES

            # 粒子模式控制
            if g["three_fingers"] and not particle_ui.visible and not particle_system_3d.active:
                # 三指竖起：呼出粒子模式选择面板
                if not hasattr(particle_ui, '_three_finger_triggered'):
                    particle_ui._three_finger_triggered = False
                
                if not particle_ui._three_finger_triggered:
                    particle_ui.show()
                    particle_ui._three_finger_triggered = True
                    print("粒子模式面板已打开")
            elif not g["three_fingers"]:
                if hasattr(particle_ui, '_three_finger_triggered'):
                    particle_ui._three_finger_triggered = False
            
            # 3D粒子模式激活时的手势控制
            if particle_system_3d.active:
                # 计算手掌开合程度（基于所有手指的平均距离）
                from core.hand_detector import distance as point_distance
                from core.hand_detector import RING_TIP, PINKY_TIP
                
                wrist = hand.landmarks_norm[WRIST]
                finger_tips = [
                    hand.landmarks_norm[THUMB_TIP],
                    hand.landmarks_norm[INDEX_TIP],
                    hand.landmarks_norm[MIDDLE_TIP],
                    hand.landmarks_norm[RING_TIP],
                    hand.landmarks_norm[PINKY_TIP],
                ]
                
                # 计算手指到手腕的平均距离（手掌张开程度）
                avg_distance = sum(point_distance(wrist, tip) for tip in finger_tips) / len(finger_tips)
                # 归一化：0.15（闭合）到 0.35（张开）-> 7倍爆发式缩放
                hand_open = min(1.0, max(0.0, (avg_distance - 0.15) / 0.20))
                
                particle_system_3d.update_hand_control(hand_open)
                
                # 初始化粒子（如果需要）
                if len(particle_system_3d.particles) == 0:
                    particle_system_3d.initialize_particles(5000)  # 5000个3D粒子（性能优化）
                
                # 更新粒子
                particle_system_3d.update(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, has_hand=True)
            
            # 本地Canvas模式
            if not particle_system_3d.active:  # 只在非粒子模式时画图
                if g["pinch_start"]:
                    pen.start_stroke()
                    draw_lock = 0

                if g["pinching"] and current_mode == "draw" and draw_lock == 0:
                    smoothed_pt = pen.draw(draw_pt)
                    # 绘图时不再自动发射粒子
                elif g["three_fingers"] and current_mode == "particle":
                    # 三指模式：只发射粒子特效，不绘图
                    if ENABLE_PARTICLES:
                        particle_system.emit(draw_pt, brush_manager.color)
                    pen.end_stroke()  # 确保不绘图
                elif g["open_palm"]:
                    eraser.erase(erase_pt)
                    pen.end_stroke()
                elif g["index_only"]:
                    pen.end_stroke()
                    # 更新激光笔拖尾
                    if ENABLE_LASER:
                        laser_trail.append(draw_pt)
                        if len(laser_trail) > max_trail_length:
                            laser_trail.pop(0)
                    if pyautogui:
                        try:
                            pyautogui.moveTo(*screen_pt, duration=0, _pause=False)
                        except TypeError:
                            pyautogui.moveTo(*screen_pt, duration=0)
                elif g["fist"]:
                    pen.end_stroke()

                # 关键修复：捏合结束时处理笔画
                if g["pinch_end"]:
                    finished_points = pen.end_stroke()
                    draw_lock = DRAW_LOCK_FRAMES  # 使用配置的冷却帧数
                    if finished_points:
                        shape_recognizer.beautify(
                            finished_points,
                            canvas.get_canvas(),
                            brush_manager.color,
                            brush_manager.thickness,
                        )


            # 绘制手部关键点（包括粒子模式）
            detector.draw_hand(frame, hand)
        else:
            # 无手时强制断笔
            pen.end_stroke()
            laser_trail.clear()
            palm_hud.reset()

        # ========== 阶段四：AR增强效果渲染 ==========

        # 1. 更新并渲染粒子系统
        if ENABLE_PARTICLES:
            particle_system.update()

        # 更新3D粒子系统（无手时）
        if particle_system_3d.active and not has_hand_this_frame:
            particle_system_3d.update(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, has_hand=False)
        
        # 渲染3D粒子模式或普通画布
        if particle_system_3d.active:
            # 保存摄像头画面
            camera_feed = frame.copy()
            
            # 背景变成纯黑
            frame = np.zeros_like(frame)
            
            # 渲染3D粒子
            particle_system_3d.render(frame)
            
            # 摄像头缩小到左下角
            small_w = config.CAMERA_WIDTH // 5
            small_h = config.CAMERA_HEIGHT // 5
            small_frame = cv2.resize(camera_feed, (small_w, small_h))
            
            # 添加边框
            cv2.rectangle(small_frame, (0, 0), (small_w-1, small_h-1), (60, 60, 60), 2)
            
            # 叠加到左下角
            y_offset = config.CAMERA_HEIGHT - small_h - 20
            x_offset = 20
            frame[y_offset:y_offset+small_h, x_offset:x_offset+small_w] = small_frame
        else:
            # 普通模式：渲染画布
            frame = overlay_canvas(frame, canvas.get_canvas())

            # 2. 渲染粒子（在画布之后，在UI之前）
            if ENABLE_PARTICLES:
                particle_system.render(frame)

        # 3. 渲染激光笔（食指指向模式）
        if ENABLE_LASER and current_mode == "move" and ui_draw_pt:
            if len(laser_trail) > 1:
                laser_pointer.render_with_trail(frame, ui_draw_pt, laser_trail)
            else:
                laser_pointer.render(frame, ui_draw_pt)

        # 4. 渲染掌心HUD（手掌静止且非擦除模式时）
        if ENABLE_PALM_HUD and palm_pos_for_hud and palm_pos_pixel:
            palm_hud.update(palm_pos_for_hud)
            # 只在非擦除模式时显示HUD，避免视觉冲突
            if palm_hud.is_still and current_mode != "erase":
                palm_hud.render(frame, palm_pos_pixel)

        # 画UI提示：笔尖、橡皮、捏合距离
        if ui_draw_pt is not None:
            cv2.circle(frame, ui_draw_pt, 6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            if ui_pinching:
                cv2.circle(frame, ui_draw_pt, 3, (0, 200, 200), -1, lineType=cv2.LINE_AA)
        if ui_erase_pt is not None:
            cv2.circle(frame, ui_erase_pt, config.ERASER_SIZE, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        if ui_pinching:
            cv2.putText(frame, f"pinch: {ui_pinch_dist:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        # 状态信息
        status_lines = [
            f"Mode: {current_mode} | FPS: {fps:.1f}",
            brush_manager.get_status_text(),  # 笔刷信息
        ]

        # AR效果状态
        ar_status = []
        if particle_system_3d.active:
            ar_status.append(f"3D Particles:{particle_system_3d.get_particle_count()}")
        elif ENABLE_PARTICLES:
            ar_status.append(f"Particles:{particle_system.get_count()}")
        if ENABLE_LASER:
            ar_status.append("Laser")
        if ENABLE_PALM_HUD:
            ar_status.append("HUD")

        if ar_status:
            status_lines.append(f"AR: {' | '.join(ar_status)}")
        
        # 平滑模式状态
        smooth_status = "OneEuro" if draw_mapper.use_one_euro else "EMA"
        if draw_mapper.use_one_euro:
            preset_name = getattr(draw_mapper, '_preset_name', "DRAWING")
            smooth_status = f"OneEuro({preset_name})"
        status_lines.append(f"Smooth: {smooth_status}")

        # 绘制状态文本
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示帮助信息
        if SHOW_HELP:
            help_text = [
                "=== AirCanvas Controls ===",
                "Hand Gestures:",
                "  3-finger pinch (thumb+index+middle): Draw",
                "  3 fingers up: Open Particle Mode UI",
                "  2 fingers (index+middle): Toggle UI",
                "  Open palm: Erase",
                "  Index only: Laser / UI hover",
                "  Fist: Pause",
                "Particle Mode:",
                "  3 fingers up: Open UI panel",
                "  Hand open/close: Zoom in/out",
                "  1: Confirm  2: Exit",
                "  Mouse: Click to select model/color",
                "Gesture UI:",
                "  2 fingers: Show/Hide UI",
                "  Index: Hover over buttons",
                "  Pinch: Select button",
                "Keyboard:",
                "  q: Quit  c: Clear  s: Save",
                "  w: Fullscreen  h: Help",
                "  [ / ]: Color  - / +: Size",
                "  b: Brush type",
                "  1/2/3: Toggle AR  r: Timer/Reset",
            ]
            overlay = frame.copy()
            y_offset = 100
            for i, text in enumerate(help_text):
                cv2.putText(overlay, text, (50, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            # 半透明背景
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # 渲染手势UI界面（只在非粒子模式时显示）
        if not particle_system_3d.active:
            gesture_ui.render(frame, brush_manager)
        
        # 渲染粒子模式UI
        if particle_ui.visible or particle_ui.entering or particle_ui.exiting:
            frame = particle_ui.render(frame)
            # 更新激活状态显示
            particle_ui.set_active_model(particle_system_3d.current_model)
        
        # 粒子模式激活时显示退出提示
        if particle_system_3d.active:
            tip_text = "Press '2' to Exit Particle Mode"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (100, 255, 255)  # 黄色
            
            # 计算文字大小
            text_size = cv2.getTextSize(tip_text, font, font_scale, thickness)[0]
            text_x = (config.CAMERA_WIDTH - text_size[0]) // 2
            text_y = config.CAMERA_HEIGHT - 30
            
            # 绘制半透明背景
            overlay = frame.copy()
            padding = 10
            cv2.rectangle(overlay, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 绘制文字
            cv2.putText(frame, tip_text, (text_x, text_y), font, font_scale, 
                       color, thickness, cv2.LINE_AA)

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas.clear()
        if key == ord("s"):
            out_path = Path("captures") / f"canvas_{save_counter}.png"
            out_path.parent.mkdir(exist_ok=True)
            canvas.save(str(out_path))
            print(f"Saved canvas to {out_path}")
            save_counter += 1
        if key == ord('1'):
            # 切换粒子效果
            ENABLE_PARTICLES = not ENABLE_PARTICLES
            if not ENABLE_PARTICLES:
                particle_system.clear()
            print(f"Particle effects: {'ON' if ENABLE_PARTICLES else 'OFF'}")
        if key == ord('2'):
            # 切换激光笔
            ENABLE_LASER = not ENABLE_LASER
            laser_trail.clear()
            print(f"Laser pointer: {'ON' if ENABLE_LASER else 'OFF'}")
        if key == ord('3'):
            # 切换掌心HUD
            ENABLE_PALM_HUD = not ENABLE_PALM_HUD
            palm_hud.reset()
            print(f"Palm HUD: {'ON' if ENABLE_PALM_HUD else 'OFF'}")
        if key == ord('r'):
            # 重置计时器
            palm_hud.reset_timer()
            print("Timer reset")
        if key == ord('h'):
            # 切换帮助显示
            SHOW_HELP = not SHOW_HELP
        if key == ord('['):
            # 上一个颜色
            brush_manager.prev_color()
            print(f"Color: {brush_manager.color_name}")
        if key == ord(']'):
            # 下一个颜色
            brush_manager.next_color()
            print(f"Color: {brush_manager.color_name}")
        if key == ord('-') or key == ord('_'):
            # 减少粗细
            brush_manager.prev_thickness()
            print(f"Thickness: {brush_manager.thickness}")
        if key == ord('=') or key == ord('+'):
            # 增加粗细
            brush_manager.next_thickness()
            print(f"Thickness: {brush_manager.thickness}")
        if key == ord('b'):
            # 切换笔刷类型
            brush_manager.next_brush_type()
            print(f"Brush type: {brush_manager.brush_type}")
        if key == ord('w'):
            # 切换全屏/窗口模式
            FULLSCREEN = not FULLSCREEN
            if FULLSCREEN:
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print(f"Fullscreen: {'ON' if FULLSCREEN else 'OFF'}")
        if key == ord('1') and particle_ui.visible:
            # 在粒子UI可见时，1键确定（直接调用回调函数）
            confirm_particle_mode()
        if key == ord('2') and (particle_ui.visible or particle_system_3d.active):
            # 2键退出粒子模式
            if particle_ui.visible:
                cancel_particle_mode()
            else:
                particle_system_3d.active = False
                particle_system_3d.reset()
                print("3D粒子模式已退出")
        if key == ord('r') and particle_system_3d.active:
            # r键重置粒子（调试用）
            particle_system_3d.reset()
            particle_system_3d.initialize_particles(5000)
            print("3D粒子系统已重置 (5000粒子)")
        if key == ord('t') and particle_system_3d.active:
            # t键切换自动旋转
            particle_system_3d.auto_rotate = not particle_system_3d.auto_rotate
            print(f"自动旋转: {'ON' if particle_system_3d.auto_rotate else 'OFF'}")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
