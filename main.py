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
from core.hand_detector import HandDetector, INDEX_TIP, THUMB_TIP, WRIST
from modules.canvas import Canvas
from modules.eraser import Eraser
from modules.ppt_controller import PPTController
from modules.shape_recognizer import ShapeRecognizer
from modules.virtual_pen import VirtualPen


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

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
        swipe_velocity_threshold=getattr(config, 'SWIPE_VELOCITY_THRESHOLD', 0.015),
        swipe_cooldown_frames=config.SWIPE_COOLDOWN_FRAMES,
        pinch_confirm_frames=getattr(config, 'PINCH_CONFIRM_FRAMES', 3),
    )

    draw_mapper = CoordinateMapper(
        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        getattr(config, 'ACTIVE_REGION_DRAW', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'DRAW_SMOOTHING_FACTOR', 0.3),
    )

    if pyautogui:
        SCREEN_W, SCREEN_H = pyautogui.size()
    else:
        SCREEN_W, SCREEN_H = getattr(config, 'SCREEN_WIDTH', 1920), getattr(config, 'SCREEN_HEIGHT', 1080)

    cursor_mapper = CoordinateMapper(
        (SCREEN_W, SCREEN_H),
        getattr(config, 'ACTIVE_REGION_CURSOR', (0.0, 0.0, 1.0, 1.0)),
        smoothing_factor=getattr(config, 'CURSOR_SMOOTHING_FACTOR', 0.15),
    )

    canvas = Canvas(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    pen = VirtualPen(
        canvas=canvas,
        color=config.PEN_COLOR,
        thickness=config.PEN_THICKNESS,
        smoothing=None,  # mapper already smooths movement
    )
    eraser = Eraser(canvas, size=config.ERASER_SIZE)
    ppt = PPTController()
    shape_recognizer = ShapeRecognizer()

    fps = 0
    last_time = time.time()
    frame_count = 0
    save_counter = 0
    slide_counter = 1

    # 模式控制
    PPT_MODE = False
    last_erase_sent = False
    draw_lock = 0  # 捏合结束后短暂冷却，防误连笔

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
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

        if hands:
            hand = hands[0]
            g = gesture.classify(hand)
            current_mode = g["mode"]

            index_norm = hand.landmarks_norm[INDEX_TIP]
            thumb_norm = hand.landmarks_norm[THUMB_TIP]
            palm_norm = palm_center(hand)

            # 作为笔尖：捏合时用拇指-食指中点，否则用食指尖
            if g["pinching"] or current_mode == "draw":
                tip_norm = ((index_norm[0] + thumb_norm[0]) / 2.0, (index_norm[1] + thumb_norm[1]) / 2.0)
            else:
                tip_norm = index_norm

            draw_pt = draw_mapper.map(tip_norm)
            erase_pt = draw_mapper.map(palm_norm)
            screen_pt = cursor_mapper.map(index_norm)

            # 记录UI提示点位
            ui_draw_pt = draw_pt
            ui_erase_pt = erase_pt if g["open_palm"] else None
            ui_pinching = g["pinching"]
            ui_pinch_dist = g["pinch_distance"]

            # PPT墨迹模式：用系统鼠标/键盘直接在PPT里画
            if PPT_MODE and pyautogui:
                if g["pinch_start"]:
                    try:
                        pyautogui.mouseDown()
                    except Exception:
                        pass
                if g["pinching"]:
                    try:
                        pyautogui.moveTo(*screen_pt, duration=0, _pause=False)
                    except TypeError:
                        pyautogui.moveTo(*screen_pt, duration=0)
                    except Exception:
                        pass
                if g["pinch_end"]:
                    try:
                        pyautogui.mouseUp()
                    except Exception:
                        pass

                # 食指指针移动（便于定位后再捏合画）
                if g["index_only"]:
                    try:
                        pyautogui.moveTo(*screen_pt, duration=0, _pause=False)
                    except TypeError:
                        pyautogui.moveTo(*screen_pt, duration=0)
                    except Exception:
                        pass

                # 张开手尝试切橡皮（不同版本行为不同）
                if g["open_palm"] and not last_erase_sent:
                    try:
                        pyautogui.hotkey('ctrl','e')
                    except Exception:
                        try:
                            pyautogui.press('e')
                        except Exception:
                            pass
                    last_erase_sent = True
                if not g["open_palm"]:
                    last_erase_sent = False

                # 不在本地Canvas绘图，避免冲突
                pen.end_stroke()

            else:
                # 本地Canvas模式
                if g["pinch_start"]:
                    pen.start_stroke()
                    draw_lock = 0
                if g["pinch_end"]:
                    finished_points = pen.end_stroke()
                    draw_lock = 6  # 冷却若干帧，防止勾连
                    if finished_points:
                        shape_recognizer.beautify(
                            finished_points,
                            canvas.get_canvas(),
                            config.PEN_COLOR,
                            config.PEN_THICKNESS,
                        )

                if g["pinching"] and current_mode == "draw" and draw_lock == 0:
                    pen.draw(draw_pt)
                elif g["open_palm"]:
                    eraser.erase(erase_pt)
                    pen.end_stroke()
                elif g["index_only"]:
                    pen.end_stroke()
                    if pyautogui:
                        try:
                            pyautogui.moveTo(*screen_pt, duration=0, _pause=False)
                        except TypeError:
                            pyautogui.moveTo(*screen_pt, duration=0)
                elif g["fist"]:
                    pen.end_stroke()

            # 手势翻页
            if g["swipe"]:
                if PPT_MODE and pyautogui:
                    try:
                        if g["swipe"] == "SWIPE_RIGHT":
                            pyautogui.press('right')
                            slide_counter += 1
                        elif g["swipe"] == "SWIPE_LEFT":
                            pyautogui.press('left')
                            slide_counter = max(1, slide_counter - 1)
                        elif g["swipe"] == "SWIPE_UP":
                            pyautogui.press('home')
                            slide_counter = 1
                        elif g["swipe"] == "SWIPE_DOWN":
                            pyautogui.press('end')
                    except Exception:
                        pass
                else:
                    if g["swipe"] == "SWIPE_RIGHT":
                        ppt.next_slide()
                        slide_counter += 1
                    elif g["swipe"] == "SWIPE_LEFT":
                        ppt.prev_slide()
                        slide_counter = max(1, slide_counter - 1)
                    elif g["swipe"] == "SWIPE_UP":
                        ppt.first_slide()
                        slide_counter = 1
                    elif g["swipe"] == "SWIPE_DOWN":
                        ppt.last_slide()
                        slide_counter = max(slide_counter, 1)

            detector.draw_hand(frame, hand)
        else:
            # 无手时强制断笔；PPT模式下释放鼠标
            pen.end_stroke()
            if PPT_MODE and pyautogui:
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass

        frame = overlay_canvas(frame, canvas.get_canvas())

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
        status_text = f"Mode: {current_mode} | PPT:{'ON' if PPT_MODE else 'OFF'} | FPS: {fps:.1f} | Slide: {slide_counter}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
        if key == ord('f'):
            # 切换PPT墨迹模式
            PPT_MODE = not PPT_MODE
            pen.end_stroke()
            last_erase_sent = False
            if pyautogui:
                try:
                    if PPT_MODE:
                        # 进入PPT画笔
                        pyautogui.hotkey('ctrl','p')
                    else:
                        # 回到箭头指针
                        pyautogui.hotkey('ctrl','a')
                except Exception:
                    pass

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
