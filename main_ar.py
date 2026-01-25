# -*- coding: utf-8 -*-
"""
AirCanvas AR模式 V2 - 阿里云美效SDK风格的悬浮窗口书写系统

特点：
- ROI窗口在白板上移动（不是画板移动）
- 白板完整缩放显示
- 卡尔曼滤波 + 时域滤波提高准确性
- 笔迹映射到ROI对应的白板位置
- 边缘自动移动ROI窗口

使用方式：
    python main_ar.py
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

import config
from core.gesture_recognizer import GestureRecognizer
from core.hand_detector import INDEX_TIP, THUMB_TIP, WRIST
from core.async_detector import SyncAsyncHandDetector
from modules.brush_manager import BrushManager
from modules.viewport import FloatingROI, create_grid_background


def main() -> None:
    # ========== 初始化摄像头 ==========
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {config.CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # 创建窗口
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(config.WINDOW_NAME, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # ========== 初始化手势检测 ==========
    INFER_W = getattr(config, 'INFER_WIDTH', 640)
    INFER_H = getattr(config, 'INFER_HEIGHT', 360)
    ASYNC_MODE = getattr(config, 'ASYNC_INFERENCE', True)
    
    detector = SyncAsyncHandDetector(
        async_mode=ASYNC_MODE,
        max_num_hands=1,
        infer_width=INFER_W,
        infer_height=INFER_H,
    )
    detector.start()

    gesture = GestureRecognizer(
        pinch_threshold=config.PINCH_THRESHOLD,
        pinch_release_threshold=config.PINCH_RELEASE_THRESHOLD,
        swipe_threshold=config.SWIPE_THRESHOLD,
        swipe_velocity_threshold=getattr(config, 'SWIPE_VELOCITY_THRESHOLD', 0.015),
        swipe_cooldown_frames=config.SWIPE_COOLDOWN_FRAMES,
        pinch_confirm_frames=getattr(config, 'PINCH_CONFIRM_FRAMES', 3),
        pinch_release_confirm_frames=getattr(config, 'PINCH_RELEASE_CONFIRM_FRAMES', 1),
    )

    # ========== 初始化AR悬浮窗口系统（阿里云风格） ==========
    # 白板尺寸（比显示尺寸大，但不会太大以便完整显示）
    WHITEBOARD_W = getattr(config, 'CANVAS_WIDTH', 1920)
    WHITEBOARD_H = getattr(config, 'CANVAS_HEIGHT', 1080)
    
    # 创建网格背景白板
    grid_size = getattr(config, 'GRID_SIZE', 25)
    whiteboard = create_grid_background(WHITEBOARD_W, WHITEBOARD_H, grid_size=grid_size)
    
    # 创建绘图层
    draw_layer = np.zeros((WHITEBOARD_H, WHITEBOARD_W, 3), dtype=np.uint8)
    
    # 可移动的ROI窗口
    floating_roi = FloatingROI(
        whiteboard_size=(WHITEBOARD_W, WHITEBOARD_H),
        display_size=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        roi_scale=getattr(config, 'ROI_SCALE', 0.5),
        camera_padding=getattr(config, 'ROI_PADDING', 0.1),
        edge_margin=getattr(config, 'EDGE_MARGIN', 50),
        move_speed=getattr(config, 'SCROLL_SPEED', 8.0),
    )

    # ========== 初始化滤波器（极速响应配置） ==========
    from utils.smoothing import OneEuroFilter
    # 使用与 main.py 一致的配置，确保手感统一
    pen_filter = OneEuroFilter(
        min_cutoff=getattr(config, 'ONE_EURO_MIN_CUTOFF', 1.0),
        beta=getattr(config, 'ONE_EURO_BETA', 0.1),
        d_cutoff=1.0,
        freq=30.0
    )

    # ========== 初始化笔刷 ==========
    brush_manager = BrushManager()
    
    # 绘图状态
    prev_wb_pt = None  # 上一个白板坐标点
    is_drawing = False
    draw_lock = 0
    DRAW_LOCK_FRAMES = getattr(config, 'DRAW_LOCK_FRAMES', 5)

    # FPS 计算
    fps = 0
    last_time = time.time()
    frame_count = 0
    save_counter = 0

    # 显示帮助
    SHOW_HELP = False
    
    print("=== AirCanvas AR Mode V2 ===")
    print("Features:")
    print("  - ROI window moves on whiteboard")
    print("  - Kalman + Temporal filtering")
    print("Controls:")
    print("  Pinch: Draw | Palm: Erase")
    print("  Move to edge: ROI auto-moves")
    print("  Arrow keys: Manual move ROI")
    print("  0: Reset ROI | c: Clear | s: Save | q: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧")
            break

        if draw_lock > 0:
            draw_lock -= 1

        frame = cv2.flip(frame, 1)
        
        # ========== 手势检测 ==========
        hands = detector.detect(frame)
        current_mode = "idle"
        pen_display_pt = None  # 笔尖在显示坐标系中的位置
        pen_wb_pt = None       # 笔尖在白板坐标系中的位置

        if hands:
            hand = hands[0]
            g = gesture.classify(hand)
            current_mode = g["mode"]

            index_norm = hand.landmarks_norm[INDEX_TIP]
            thumb_norm = hand.landmarks_norm[THUMB_TIP]

            # 笔尖位置（两指中心）
            if g["pinching"] or current_mode == "draw":
                tip_norm = (
                    (index_norm[0] + thumb_norm[0]) / 2.0,
                    (index_norm[1] + thumb_norm[1]) / 2.0
                )
            else:
                tip_norm = index_norm

            # 使用卡尔曼+时域滤波平滑坐标
            filtered_norm = pen_filter.push(tip_norm)
            
            # 获取笔尖在ROI内的相对位置
            roi_norm = floating_roi.get_roi_norm_position(filtered_norm)
            
            # ========== 边缘自动移动ROI ==========
            # 当笔尖靠近ROI边缘时，ROI窗口在白板上移动
            move_dx, move_dy = floating_roi.check_edge_move(roi_norm)
            if move_dx != 0 or move_dy != 0:
                floating_roi.move(move_dx, move_dy)
            
            # 将归一化坐标映射到白板坐标
            pen_wb_pt = floating_roi.map_to_whiteboard(filtered_norm)
            
            # 将白板坐标转换为显示坐标（用于绘制指示器）
            pen_display_pt = floating_roi.whiteboard_to_display(pen_wb_pt)

            # ========== 绘图逻辑 ==========
            if g["pinch_start"]:
                is_drawing = True
                prev_wb_pt = pen_wb_pt
                draw_lock = 0

            if g["pinching"] and current_mode == "draw" and draw_lock == 0 and is_drawing:
                if prev_wb_pt is not None and pen_wb_pt is not None:
                    # 在白板上绘制
                    brush_manager.draw_line(draw_layer, prev_wb_pt, pen_wb_pt)
                prev_wb_pt = pen_wb_pt
            
            if g["pinch_end"]:
                is_drawing = False
                prev_wb_pt = None
                draw_lock = DRAW_LOCK_FRAMES

            # 橡皮擦模式
            if g["open_palm"] and pen_wb_pt is not None:
                is_drawing = False
                prev_wb_pt = None
                # 在白板上擦除
                cv2.circle(draw_layer, pen_wb_pt, config.ERASER_SIZE, (0, 0, 0), -1)

        else:
            # 无手时重置状态
            is_drawing = False
            prev_wb_pt = None
            pen_filter.reset()

        # ========== 渲染 ==========
        # 使用 FloatingROI 渲染完整画面
        display_frame = floating_roi.render(whiteboard, frame, draw_layer, opacity=0.7)
        
        # 绘制笔尖指示器
        if pen_display_pt is not None:
            color = (0, 255, 255) if is_drawing else (200, 200, 200)
            thickness = 3 if is_drawing else 2
            cv2.circle(display_frame, pen_display_pt, 8, color, thickness, lineType=cv2.LINE_AA)
            if is_drawing:
                cv2.circle(display_frame, pen_display_pt, 4, (0, 200, 200), -1, lineType=cv2.LINE_AA)

        # 绘制状态信息
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        status_lines = [
            f"AR V2 | FPS: {fps:.1f} | {floating_roi.get_position_info()}",
            brush_manager.get_status_text(),
        ]
        
        for i, line in enumerate(status_lines):
            cv2.putText(display_frame, line, (10, 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 0), 2)

        # 显示帮助
        if SHOW_HELP:
            help_text = [
                "=== AR Canvas V2 ===",
                "  Pinch: Draw",
                "  Open palm: Erase",
                "  Edge: Auto-move ROI",
                "",
                "  Arrows: Move ROI",
                "  0: Reset ROI",
                "  [/]: Color  b: Brush",
                "  c: Clear  s: Save  q: Quit",
            ]
            overlay = display_frame.copy()
            for i, text in enumerate(help_text):
                cv2.putText(overlay, text, (50, 100 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)

        # 显示
        cv2.imshow(config.WINDOW_NAME, display_frame)

        # ========== 键盘控制 ==========
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("c"):
            draw_layer[:] = 0
            print("Canvas cleared")
        elif key == ord("s"):
            out_path = Path("captures") / f"ar_canvas_{save_counter}.png"
            out_path.parent.mkdir(exist_ok=True)
            save_canvas = whiteboard.copy()
            mask = np.any(draw_layer != 0, axis=2)
            save_canvas[mask] = draw_layer[mask]
            cv2.imwrite(str(out_path), save_canvas)
            print(f"Saved canvas to {out_path}")
            save_counter += 1
        elif key == ord("h"):
            SHOW_HELP = not SHOW_HELP
        elif key == 82:  # Up arrow
            floating_roi.move(0, -20)
        elif key == 84:  # Down arrow
            floating_roi.move(0, 20)
        elif key == 81:  # Left arrow
            floating_roi.move(-20, 0)
        elif key == 83:  # Right arrow
            floating_roi.move(20, 0)
        elif key == ord("0"):
            floating_roi.reset()
            print("ROI reset to center")
        elif key == ord("["):
            brush_manager.prev_color()
            print(f"Color: {brush_manager.color_name}")
        elif key == ord("]"):
            brush_manager.next_color()
            print(f"Color: {brush_manager.color_name}")
        elif key == ord("b"):
            brush_manager.next_brush_type()
            print(f"Brush: {brush_manager.brush_type}")

    # 清理
    detector.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
