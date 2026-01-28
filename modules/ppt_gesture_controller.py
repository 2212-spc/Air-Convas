# -*- coding: utf-8 -*-
"""
PPT手势控制模块 - 独立可运行的版本
基于手势识别控制PowerPoint演示文稿
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from collections import deque, Counter
from core.coordinate_mapper import CoordinateMapper
from utils.smoothing import catmull_rom_spline

# COM接口支持
try:
    import win32com.client
    COM_AVAILABLE = True
except ImportError:
    COM_AVAILABLE = False
    print("警告: win32com不可用，将使用模拟按键作为降级方案")

# 透明叠加层支持（当 COM 画线不可用时的备选方案）
try:
    from modules.transparent_overlay import get_overlay
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False
    print("警告: 透明叠加层不可用")

# --- ⚙️ 配置区域 (Configuration) ---
# 摄像头设置
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# 屏幕设置
try:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
except Exception:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # 默认值

# 平滑因子 (0.1~0.9): 越小越灵敏但抖动，越大越平滑但有延迟
SMOOTHING_FACTOR = 0.7

# 模式切换确认时间 (秒)
CONFIRM_DELAY = 0.3  # 减少到300ms，更快响应 (was 1.0)

# 书写后模式锁定冷却时间 (秒)
# 在写字(捏合)结束后的这段时间内，禁止切换模式，防止误触
PINCH_MODE_LOCK_COOLDOWN = 0.6

# 手指伸直检测容差 (归一化坐标，0.0~1.0)
# 允许指尖稍微在关节下方，仍然判定为伸直
FINGER_EXTEND_TOLERANCE = 0.1  # 约10%的容差

# 手指伸直距离阈值 (归一化坐标，0.0~1.0)
# 指尖到关节的距离必须大于此值才认为手指真正伸直
FINGER_EXTEND_DISTANCE_THRESHOLD = 0.03  # 至少3%的距离

# 相对捏合阈值 (捏合距离 / 手掌宽度) - 动态适应远近
PINCH_RATIO_THRESHOLD = 0.25  # 捏合距离小于手掌宽度的25%认为捏合

# 挥手判定阈值 (归一化坐标，0.0~1.0)
# 使用归一化坐标适配不同分辨率
SWIPE_THRESHOLD = 0.18  # 归一化位移阈值：越小越容易触发

# 挥手速度阈值（归一化坐标/秒）：避免“慢慢移动也翻页”
SWIPE_VELOCITY_THRESHOLD = 0.9


# 挥手冷却时间 (秒)
SWIPE_COOLDOWN = 0.35

# 空间复位逻辑参数（Neutral Zone）
NEUTRAL_ZONE_X_MIN = 0.2  # 屏幕中央安全区（归一化坐标）
NEUTRAL_ZONE_X_MAX = 0.8
NEUTRAL_ZONE_Y_MIN = 0
NEUTRAL_ZONE_Y_MAX = 1
NEUTRAL_STAY_FRAMES = 4  # 在安全区停留的帧数才认为归位（越小越容易解锁）

# OneEuroFilter 参数 (优化：更平滑的光标移动)
ONEEURO_MIN_CUTOFF = 1.0   # 恢复为 1.0 以提高响应速度
ONEEURO_BETA = 0.007       # 恢复为 0.007 以提高响应速度
ONEEURO_DCUTOFF = 1.0      # 速度平滑截止频率

# 状态机迟滞参数
GESTURE_CONFIRM_FRAMES = 5  # 建议使用更长窗口 + 多数投票，提高稳定性

# --- 🏷️ 状态常量定义 ---
MODE_NONE = 0
MODE_PEN = 1      # 手势1: 只有食指
MODE_ERASER = 2   # 手势2: 食指+中指
MODE_NAV = 3      # 手势3: 食指+中指+无名指

# 导航状态机状态
STATE_IDLE = 0
STATE_SWIPE = 1
STATE_COOLDOWN = 2
STATE_WAIT_NEUTRAL = 3  # 等待归位状态


class PPTController:
    """
    使用COM接口控制PowerPoint，实现确定性状态管理
    替代不可靠的模拟按键
    """
    def __init__(self):
        self.app = None
        self.last_slide_index = -1
        
        if COM_AVAILABLE:
            try:
                # 获取现有的 PPT 实例
                self.app = win32com.client.GetActiveObject("PowerPoint.Application")
                print("COM接口初始化成功 (动态模式)")
            except Exception as e:
                print(f"COM初始化失败: {e}，将使用模拟按键")
                self.app = None
    
    @property
    def active_view(self):
        """动态获取当前的放映视图，防止对象失效"""
        if not self.app:
            return None
        try:
            # 必须动态获取，不能缓存！
            if self.app.SlideShowWindows.Count > 0:
                # 获取当前活跃的放映窗口
                return self.app.SlideShowWindows(1).View
        except Exception:
            # 尝试重新连接 app
            self.reconnect()
        return None

    def set_pointer_type(self, pointer_type):
        """
        设置PPT指针类型（确定性）
        pointer_type: 1=箭头, 2=画笔, 3=激光笔, 5=橡皮
        """
        view = self.active_view
        if view:
            try:
                view.PointerType = pointer_type
                return True
            except Exception as e:
                # 某些时候设置失败是正常的（如切换瞬间），不打印刷屏日志
                pass
        return False
    
    def check_slide_changed(self):
        """
        检测是否翻页，如果翻页则返回True并自动切笔
        """
        view = self.active_view
        if view:
            try:
                current_index = view.CurrentSlide.SlideIndex
                if current_index != self.last_slide_index:
                    self.last_slide_index = current_index
                    # 翻页后自动切笔
                    self.set_pointer_type(2)  # 画笔
                    return True
            except Exception:
                pass
        return False
    
    def reconnect(self):
        """尝试重新连接PPT应用"""
        if COM_AVAILABLE:
            try:
                self.app = win32com.client.GetActiveObject("PowerPoint.Application")
                return True
            except Exception:
                pass
        return False

    def draw_line(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        使用 COM 接口在 PPT 放映窗口直接画线
        """
        view = self.active_view
        if view:
            try:
                # 强制确保是画笔模式
                if view.PointerType != 2:
                    view.PointerType = 2
                
                # DrawLine(BeginX, BeginY, EndX, EndY)
                view.DrawLine(int(x1), int(y1), int(x2), int(y2))
                return True
            except Exception:
                pass
        return False

    def erase_drawing(self) -> bool:
        """清除当前幻灯片的所有墨迹"""
        view = self.active_view
        if view:
            try:
                view.EraseDrawing()
                return True
            except Exception:
                pass
        return False

    def is_slideshow_active(self) -> bool:
        """检查 PPT 是否在放映模式"""
        return self.active_view is not None


class OneEuroFilter:
    """
    OneEuroFilter: 自适应低通滤波器
    根据信号变化速度动态调整截止频率，消除抖动同时保持响应速度
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, dcutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def __call__(self, x, t):
        """滤波函数"""
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        # 计算时间差
        dt = t - self.t_prev
        if dt < 1e-6:
            return self.x_prev
        
        # 计算速度
        dx = (x - self.x_prev) / dt
        
        # 平滑速度
        if self.dx_prev is None:
            self.dx_prev = dx
        else:
            alpha = self._smoothing_factor(dt, self.dcutoff)
            self.dx_prev = alpha * dx + (1 - alpha) * self.dx_prev
        
        # 动态截止频率：速度越快，截止频率越高
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        
        # 平滑位置
        alpha = self._smoothing_factor(dt, cutoff)
        x_filtered = alpha * x + (1 - alpha) * self.x_prev
        
        self.x_prev = x_filtered
        self.t_prev = t
        
        return x_filtered
    
    def _smoothing_factor(self, dt, cutoff):
        """计算平滑因子"""
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)
    
    def reset(self):
        """重置滤波器状态"""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class PPTGestureController:
    def __init__(
        self,
        external_mp: bool = False,
        cursor_mapper=None,
        confirm_delay: float = CONFIRM_DELAY,
        gesture_confirm_frames: int = GESTURE_CONFIRM_FRAMES,
        swipe_threshold: float = SWIPE_THRESHOLD,
        swipe_velocity_threshold: float = SWIPE_VELOCITY_THRESHOLD,
        swipe_cooldown: float = SWIPE_COOLDOWN,
        neutral_stay_frames: int = NEUTRAL_STAY_FRAMES,
        pinch_trigger_threshold: float = 0.33,
        pinch_release_threshold: float = 0.65,
        auto_pen_on_pinch: bool = True,
        auto_pen_on_slide_change: bool = True,
        debug_overlay: bool = True,
    ):
        # 1. 初始化 MediaPipe
        self.external_mp = external_mp
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        if not external_mp:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7  # 提升到0.7，确保高质量跟踪
            )
        else:
            self.hands = None

        # 2. 状态变量
        self.current_mode = MODE_NAV  # 默认模式
        self.last_gesture = MODE_NONE
        self.gesture_timer = 0
        self.last_time = time.time()
        self.confirm_delay = float(confirm_delay)

        # 行为/阈值参数（可由 main.py / config.py 注入）
        self.gesture_confirm_frames = max(3, int(gesture_confirm_frames))
        self.swipe_threshold = float(swipe_threshold)
        self.swipe_velocity_threshold = float(swipe_velocity_threshold)
        self.swipe_cooldown = float(swipe_cooldown)
        self.neutral_stay_frames = max(1, int(neutral_stay_frames))
        self.pinch_trigger_threshold = float(pinch_trigger_threshold)
        self.pinch_release_threshold = float(pinch_release_threshold)
        self.auto_pen_on_pinch = bool(auto_pen_on_pinch)
        self.auto_pen_on_slide_change = bool(auto_pen_on_slide_change)
        self.debug_overlay = bool(debug_overlay)

        # 3. 平滑算法变量
        self.prev_x, self.prev_y = 0, 0

        # 4. 导航模式变量 (用于计算挥手速度)
        self.prev_hand_x_norm = 0  # 归一化坐标（0.0-1.0）
        self.last_swipe_time = 0
        
        # 5. 状态机变量（空间复位逻辑）
        self.nav_state = STATE_IDLE
        self.neutral_stay_count = 0
        self._last_nav_eval_time = time.time()
        self.last_nav_delta_x_norm = 0.0
        self.last_nav_velocity_norm_s = 0.0
        self.last_in_neutral_zone = False
        
        # 6. PPT控制器（COM接口）
        self.ppt_controller = PPTController()

        # 7. 高级坐标映射器 (与绘图模式一致)
        if cursor_mapper:
            self.cursor_mapper = cursor_mapper
        else:
            # 如果没有注入，使用默认全屏区域
            self.cursor_mapper = CoordinateMapper(
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                (0.0, 0.0, 1.0, 1.0),
                smoothing_factor=0.15  # 与主程序绘图光标平滑度一致
            )

        # 鼠标状态追踪
        self.mouse_down = False
        self.prev_is_pinching = False
        
        # OneEuroFilter: 为21个关键点的x, y, z坐标创建滤波器
        self.landmark_filters = {}
        for i in range(21):
            for coord in ['x', 'y', 'z']:
                self.landmark_filters[(i, coord)] = OneEuroFilter(
                    min_cutoff=ONEEURO_MIN_CUTOFF,
                    beta=ONEEURO_BETA,
                    dcutoff=ONEEURO_DCUTOFF
                )
        
        # 状态机迟滞：滑动窗口确认机制（多数投票）
        self.gesture_history = deque(maxlen=self.gesture_confirm_frames)
        self.confirmed_gesture = MODE_NONE
        
        # 调试变量
        self.last_pinch_ratio = 0.0
        
        # 绘图轨迹平滑历史 (用于 Catmull-Rom Spline)
        self.point_history = deque(maxlen=3)
        
        # 捏合释放时间记录 (防止写字结束后立即切模式)
        self.last_pinch_release_time = 0.0

        # 透明叠加层（当 COM 画线不可用时的备选方案）
        self.overlay = None
        self.use_overlay = False  # 是否使用透明叠加层模式
        self._overlay_initialized = False

        # 手指状态迟滞（减少“角度阈值抖动”）
        # index/middle/ring/pinky: 1/2/3/4
        self._finger_extended_state = {1: False, 2: False, 3: False, 4: False}
        self._thumb_open_state = False

    def _majority_vote_gesture(self):
        """多数投票确认：避免“全一致才确认”导致模式永远不稳定。"""
        if not self.gesture_history:
            return MODE_NONE
        counts = Counter(self.gesture_history)
        gesture, top_count = counts.most_common(1)[0]
        # 过滤 NONE：避免没手/抖动把确认手势冲掉
        if gesture == MODE_NONE:
            return MODE_NONE
        ratio = top_count / max(1, len(self.gesture_history))
        return gesture if ratio >= 0.6 else MODE_NONE

    def _thumb_open_hysteresis(self, ratio: float) -> bool:
        """
        拇指开合迟滞：
        - open:  ratio > 0.65
        - close: ratio < 0.55
        """
        if self._thumb_open_state:
            if ratio < 0.55:
                self._thumb_open_state = False
        else:
            if ratio > 0.65:
                self._thumb_open_state = True
        return self._thumb_open_state

    def _finger_extended_hysteresis(self, finger_id: int, angle_deg: float, tip_pip_dist: float) -> bool:
        """
        角度 + 距离的迟滞判定：
        - 伸直进入：angle < 35 且 tip-pip 距离足够（避免远距离噪声）
        - 伸直保持：angle < 50 且 tip-pip 距离足够
        """
        dist_ok = tip_pip_dist >= FINGER_EXTEND_DISTANCE_THRESHOLD
        prev = self._finger_extended_state.get(finger_id, False)
        if prev:
            extended = dist_ok and (angle_deg < 50.0)
        else:
            extended = dist_ok and (angle_deg < 35.0)
        self._finger_extended_state[finger_id] = extended
        return extended

    def get_distance(self, p1, p2):
        """计算两点欧几里得距离"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def process_frame(self, frame):
        """
        核心处理循环 (独立运行模式)
        内部自行调用 MediaPipe 处理
        """
        if self.external_mp:
            raise RuntimeError("Instance initialized with external_mp=True, use process_hand_data instead.")
            
        # 镜像翻转，符合直觉
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制骨架 (调试用)
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # 转换格式适配 process_hand_data
                # MediaPipe 的 NormalizedLandmarkList 可以直接迭代，元素有 x,y,z
                self.process_hand_data(hand_landmarks.landmark, frame)
        else:
            self._handle_no_hand()

        return frame

    def process_hand_data(self, landmarks_list, frame=None):
        """
        处理手部数据 (核心逻辑)
        landmarks_list: 包含21个关键点的列表，每个点需有 .x, .y, .z 属性
        frame: (可选) 用于绘制UI和状态文本
        """
        if frame is not None:
            h, w, _ = frame.shape
        else:
            # 如果没有frame，使用默认分辨率计算（可能会影响像素级操作如平滑）
            h, w = SCREEN_HEIGHT, SCREEN_WIDTH

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # --- 🟢 模块一: 视觉感知 (识别手势) ---
        # 对关键点进行OneEuroFilter滤波，消除抖动
        filtered_landmarks = self.filter_landmarks(landmarks_list, current_time)
        detected_gesture = self.recognize_gesture(filtered_landmarks, h, w)

        # --- 🔵 模块二: 状态机 (带时间缓冲的模式切换) ---
        # 滑动窗口多数投票确认：更抗抖动
        self.gesture_history.append(detected_gesture)
        voted = self._majority_vote_gesture()
        if voted != MODE_NONE:
            self.confirmed_gesture = voted
        
        # 注意: 如果正在捏合(写字中)，则锁定模式切换
        is_pinching = self.check_pinch(filtered_landmarks, h, w)

        # 捏合沿检测（用于自动切笔/调试）
        pinch_start = is_pinching and (not self.prev_is_pinching)
        pinch_end = (not is_pinching) and self.prev_is_pinching
        self.prev_is_pinching = is_pinching

        # 关键修复：如果当前在 NAV（翻页）但用户开始捏合，自动切到 PEN，避免“写不上去”
        if pinch_start and self.auto_pen_on_pinch and self.current_mode == MODE_NAV:
            self.current_mode = MODE_PEN
            self.trigger_mode_switch_shortcut()

        if not is_pinching:
            # 使用确认的手势进行模式切换
            self.update_mode(self.confirmed_gesture, dt)

        # --- 🟠 模块三: 执行层 (平滑 & 动作) ---
        # 注意：执行层仍使用原始landmarks (或者滤波后的，这里保持逻辑一致性使用原始)
        # 如果 landmarks_list 是对象列表，可以直接用
        self.execute_action(landmarks_list, is_pinching, w, h)

        # --- UI 反馈 ---
        if frame is not None:
            self._draw_ui(frame, detected_gesture, is_pinching, landmarks_list, w, h)

    def _handle_no_hand(self):
        """处理无手状态"""
        if self.mouse_down:
            try:
                pyautogui.mouseUp()
                self.mouse_down = False
            except Exception:
                pass
            # 如果使用叠加层，结束笔画
            if self.use_overlay and self.overlay:
                self.overlay.end_stroke()
        
        # 【关键】手离开时隐藏光标，防止绿色十字残留
        if self.use_overlay and self.overlay:
            self.overlay.hide_cursor()
        
        # 重置手势历史
        self.gesture_history.clear()
        self.confirmed_gesture = MODE_NONE
        # 重置导航状态机
        if self.nav_state != STATE_IDLE:
            self.nav_state = STATE_IDLE
            self.neutral_stay_count = 0
        self.prev_is_pinching = False
        self._last_nav_eval_time = time.time()
        # 重置迟滞状态
        for k in self._finger_extended_state:
            self._finger_extended_state[k] = False
        self._thumb_open_state = False

    def _draw_ui(self, frame, detected_gesture, is_pinching, landmarks, w, h):
        """绘制UI状态"""
        mode_name = self.get_mode_name()
        cv2.putText(
            frame, f"Mode: {mode_name}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
        )
        cv2.putText(
            frame, f"Timer: {self.gesture_timer:.1f}s", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        # 显示检测到的手势和确认的手势
        detected_name = self._get_mode_name_from_gesture(detected_gesture)
        confirmed_name = self._get_mode_name_from_gesture(self.confirmed_gesture)
        cv2.putText(
            frame, f"Detected: {detected_name}", (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        cv2.putText(
            frame, f"Confirmed: {confirmed_name}", (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        # 显示实时捏合比率 (调试用)
        pinch_color = (0, 255, 0) if is_pinching else (0, 0, 255)
        cv2.putText(
            frame, f"Pinch Ratio: {self.last_pinch_ratio:.3f}", (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2
        )

        if self.debug_overlay:
            # NAV 调试信息：看一眼就知道“为什么不翻页”
            nav_state_name = {
                STATE_IDLE: "IDLE",
                STATE_SWIPE: "SWIPE",
                STATE_COOLDOWN: "COOLDOWN",
                STATE_WAIT_NEUTRAL: "WAIT_NEUTRAL",
            }.get(self.nav_state, "UNKNOWN")
            cv2.putText(
                frame,
                f"NAV: {nav_state_name} dx={self.last_nav_delta_x_norm:.3f} v={self.last_nav_velocity_norm_s:.2f} neutral={int(self.last_in_neutral_zone)}",
                (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                frame,
                "Keys: i=Pen  e=Eraser  n=Nav",
                (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                2,
            )
            com_ok = self.ppt_controller.active_view is not None
            overlay_on = self.use_overlay
            draw_method = "OVERLAY" if overlay_on else ("COM" if com_ok else "pyautogui")
            cv2.putText(
                frame,
                f"Draw: {draw_method}  MouseDown: {int(self.mouse_down)}  pts: {len(self.point_history)}",
                (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if (overlay_on or com_ok) else (0, 165, 255),
                2,
            )
            # 提示：按 O 开启叠加层
            if not com_ok and not overlay_on:
                cv2.putText(
                    frame,
                    "Press 'O' to enable overlay drawing on PPT",
                    (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
            elif overlay_on:
                cv2.putText(
                    frame,
                    "OVERLAY mode: drawing above PPT (press X to clear, O to disable)",
                    (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 100),
                    2,
                )
        
        # 显示三指捏合状态和距离信息
        if is_pinching:
            cv2.putText(
                frame, "3-FINGER PINCH", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            # 在画面上绘制三指连接线，可视化捏合状态
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]
            
            thumb_pt = (int(thumb.x * w), int(thumb.y * h))
            index_pt = (int(index.x * w), int(index.y * h))
            middle_pt = (int(middle.x * w), int(middle.y * h))
            
            # 绘制连接线
            cv2.line(frame, thumb_pt, index_pt, (0, 255, 0), 2)
            cv2.line(frame, thumb_pt, middle_pt, (0, 255, 0), 2)
            cv2.line(frame, index_pt, middle_pt, (0, 255, 0), 2)
            
            # 绘制三指中心点
            center_x = int((thumb.x + index.x + middle.x) / 3.0 * w)
            center_y = int((thumb.y + index.y + middle.y) / 3.0 * h)
            cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)

    def filter_landmarks(self, landmarks_list, t):
        """
        对21个关键点进行OneEuroFilter滤波，消除抖动
        返回一个包含滤波后坐标的列表
        """
        filtered = []
        for i, landmark in enumerate(landmarks_list):
            x = self.landmark_filters[(i, 'x')](landmark.x, t)
            y = self.landmark_filters[(i, 'y')](landmark.y, t)
            z = self.landmark_filters[(i, 'z')](landmark.z, t)
            # 创建一个简单的命名元组来存储滤波后的坐标
            filtered.append(type('Landmark', (), {'x': x, 'y': y, 'z': z})())
        return filtered

    def calculate_finger_angle(self, mcp, pip, tip):
        """
        计算手指弯曲角度（使用向量点积法）
        返回角度（度），0度表示完全伸直，180度表示完全弯曲
        具有旋转不变性
        """
        # 向量1: 从MCP指向PIP
        v_proximal = np.array([pip.x - mcp.x, pip.y - mcp.y])
        # 向量2: 从PIP指向TIP
        v_distal = np.array([tip.x - pip.x, tip.y - pip.y])
        
        # 计算点积和模长
        dot_product = np.dot(v_proximal, v_distal)
        norm_proximal = np.linalg.norm(v_proximal)
        norm_distal = np.linalg.norm(v_distal)
        
        # 避免除零
        if norm_proximal < 1e-6 or norm_distal < 1e-6:
            return 180.0  # 默认弯曲
        
        # 计算夹角（度）
        cos_angle = np.clip(dot_product / (norm_proximal * norm_distal), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle

    def check_thumb_state(self, landmarks):
        """
        拇指状态检测：使用距离比对法（Ratio-based heuristic）
        返回 True 表示拇指张开，False 表示闭合
        能够适应手掌离摄像头远近的变化
        """
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]  # 食指MCP
        pinky_mcp = landmarks[17]  # 小指MCP
        
        # 计算手掌宽度
        palm_width = math.hypot(
            index_mcp.x - pinky_mcp.x,
            index_mcp.y - pinky_mcp.y
        )
        
        # 计算拇指指尖到小指MCP的距离
        thumb_to_pinky = math.hypot(
            thumb_tip.x - pinky_mcp.x,
            thumb_tip.y - pinky_mcp.y
        )
        
        # 归一化比对：用比率适配远近变化
        if palm_width < 1e-6:
            return False
        
        ratio = thumb_to_pinky / palm_width
        return self._thumb_open_hysteresis(ratio)

    def recognize_gesture(self, landmarks, h, w):
        """
        识别手势：使用向量点积法计算手指角度，具有旋转不变性
        实现排他性逻辑，确保手势1、2、3不会相互混淆
        返回: MODE_PEN, MODE_ERASER, 或 MODE_NAV
        """
        # MediaPipe 手部关键点索引
        # 拇指: 4(指尖), 3(IP关节), 2(MCP关节)
        # 食指: 8(指尖), 6(PIP关节), 5(MCP关节)
        # 中指: 12(指尖), 10(PIP关节), 9(MCP关节)
        # 无名指: 16(指尖), 14(PIP关节), 13(MCP关节)
        # 小指: 20(指尖), 18(PIP关节), 17(MCP关节)

        # 手指关节配置: (TIP, PIP, MCP)
        finger_configs = [
            (8, 6, 5),   # 食指
            (12, 10, 9), # 中指
            (16, 14, 13), # 无名指
            (20, 18, 17)  # 小指
        ]
        
        finger_states = []

        # 检测拇指（迟滞）
        thumb_open = self.check_thumb_state(landmarks)
        finger_states.append(1 if thumb_open else 0)

        # 检测其他四指（角度 + 距离 + 迟滞）
        for idx_in_list, (tip_idx, pip_idx, mcp_idx) in enumerate(finger_configs, start=1):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            angle = self.calculate_finger_angle(mcp, pip, tip)
            tip_pip_dist = math.hypot(tip.x - pip.x, tip.y - pip.y)

            is_extended = self._finger_extended_hysteresis(idx_in_list, angle, tip_pip_dist)
            finger_states.append(1 if is_extended else 0)

        # fingers = [拇指, 食指, 中指, 无名指, 小指]
        # 恢复清晰的手势识别逻辑
        
        # 手势3 (MODE_NAV): 食指+中指+无名指都伸直
        if (finger_states[1] == 1 and  # 食指伸直
            finger_states[2] == 1 and  # 中指伸直
            finger_states[3] == 1):    # 无名指伸直
            return MODE_NAV
        
        # 手势2 (MODE_ERASER): 食指+中指伸直，无名指弯曲
        elif (finger_states[1] == 1 and  # 食指伸直
              finger_states[2] == 1 and  # 中指伸直
              finger_states[3] == 0):    # 无名指弯曲
            return MODE_ERASER
        
        # 手势1 (MODE_PEN): 食指伸直，中指弯曲
        elif (finger_states[1] == 1 and  # 食指伸直
              finger_states[2] == 0):    # 中指弯曲
            return MODE_PEN

        return MODE_NONE  # 默认

    def update_mode(self, detected_gesture, dt):
        """状态机逻辑: 防误触的时间积累"""
        
        # 冷却锁检查：如果刚刚结束写字(捏合)，禁止切换模式
        # 防止手指松开瞬间的形变被误判为其他手势
        if (time.time() - self.last_pinch_release_time) < PINCH_MODE_LOCK_COOLDOWN:
            self.gesture_timer = 0
            return

        if detected_gesture != MODE_NONE and detected_gesture == self.last_gesture:
            self.gesture_timer += dt
            if self.gesture_timer >= self.confirm_delay:
                if self.current_mode != detected_gesture:
                    self.current_mode = detected_gesture
                    self.trigger_mode_switch_shortcut()
                    self.gesture_timer = 0  # 重置
        else:
            self.gesture_timer = 0  # 手势变了，重置计时器

        self.last_gesture = detected_gesture

    def trigger_mode_switch_shortcut(self):
        """
        根据新模式设置PPT工具（使用COM接口，确定性状态管理）
        如果COM不可用，降级使用模拟按键
        """
        # 减少日志刷屏：只在非叠加层模式下打印
        if not self.use_overlay:
            print(f"切换模式到: {self.get_mode_name()}")
        
        # 如果使用透明叠加层，不需要切换 PPT 指针类型
        if self.use_overlay:
            return
        
        # 优先使用COM接口（确定性）
        if COM_AVAILABLE:
            # 使用 active_view 属性动态获取视图
            view = self.ppt_controller.active_view
            if view:
                if self.current_mode == MODE_PEN:
                    if self.ppt_controller.set_pointer_type(2):  # 画笔
                        return
                elif self.current_mode == MODE_ERASER:
                    if self.ppt_controller.set_pointer_type(5):  # 橡皮
                        return
                elif self.current_mode == MODE_NAV:
                    if self.ppt_controller.set_pointer_type(3):  # 激光笔
                        return
        
        # 降级方案：使用模拟按键（静默执行）
        try:
            if self.current_mode == MODE_PEN:
                pyautogui.hotkey('ctrl', 'p')
            elif self.current_mode == MODE_ERASER:
                pyautogui.hotkey('ctrl', 'e')
            elif self.current_mode == MODE_NAV:
                pyautogui.hotkey('ctrl', 'l')
        except Exception:
            pass

    def toggle_overlay_mode(self):
        """
        切换透明叠加层模式（按 'o' 键触发）
        当 COM 画线不可用时，用这个方案在 PPT 上方画画
        """
        if not OVERLAY_AVAILABLE:
            print("透明叠加层不可用，请检查 modules/transparent_overlay.py")
            return False
        
        self.use_overlay = not self.use_overlay
        
        if self.use_overlay:
            # 启动叠加层
            if not self._overlay_initialized:
                self.overlay = get_overlay()
                self.overlay.start()
                self._overlay_initialized = True
                print("透明叠加层已启动 - 现在可以在 PPT 上方画画了")
            else:
                self.overlay.set_visible(True)
                print("透明叠加层已显示")
        else:
            # 隐藏叠加层（但不销毁）
            if self.overlay:
                self.overlay.set_visible(False)
                print("透明叠加层已隐藏")
        
        return self.use_overlay

    def clear_overlay(self):
        """清除透明叠加层上的所有笔迹（按 'x' 键触发）"""
        if self.overlay:
            self.overlay.clear()
            print("透明叠加层已清空")

    def check_pinch(self, landmarks, h, w):
        """
        增强版捏合检测：检测拇指与食指 OR 中指的距离
        支持三指书写习惯
        """
        # 获取指尖坐标
        # landmarks 可能是原始MediaPipe对象或滤波后的列表
        try:
            thumb = landmarks[4]   # 拇指尖
            index = landmarks[8]    # 食指尖
            middle = landmarks[12]  # 中指尖
            # 辅助点用于计算手掌尺度
            index_mcp = landmarks[5]   # 食指MCP
            pinky_mcp = landmarks[17]  # 小指MCP
        except (IndexError, AttributeError):
            return False

        # 计算手掌参考宽度 (食指MCP到小指MCP)
        palm_width = math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)
        
        if palm_width < 1e-6:
            return False

        # 计算 拇指-食指 距离 (归一化坐标)
        dist_thumb_index = math.hypot(thumb.x - index.x, thumb.y - index.y)
        
        # 计算 拇指-中指 距离 (新增)
        dist_thumb_middle = math.hypot(thumb.x - middle.x, thumb.y - middle.y)
        
        # 取两者中较小的距离作为判定依据
        # 只要食指或中指任意一个靠近拇指，都算捏合
        min_dist = min(dist_thumb_index, dist_thumb_middle)
        
        # 计算捏合比例
        pinch_ratio = min_dist / palm_width
        
        # 调试信息：将捏合比率存入实例变量供UI显示
        self.last_pinch_ratio = pinch_ratio
        
        # 迟滞阈值设置（可注入，默认更抗抖，减少“点一下不成线/断断续续”）
        PINCH_TRIGGER_THRESHOLD = self.pinch_trigger_threshold
        PINCH_RELEASE_THRESHOLD = self.pinch_release_threshold

        # 状态机逻辑
        if self.mouse_down:
            # 如果已经是按下状态，使用较宽松的释放阈值
            if pinch_ratio > PINCH_RELEASE_THRESHOLD:
                return False  # 松手
            else:
                return True   # 保持捏合
        else:
            # 如果是松开状态，使用严格的触发阈值
            if pinch_ratio < PINCH_TRIGGER_THRESHOLD:
                return True   # 触发捏合
            else:
                return False  # 保持松开

    def execute_action(self, landmarks, is_pinching, w, h):
        """根据当前模式执行具体操作"""

        # 1. 获取三指中心点坐标（用于更稳定的控制）
        thumb = landmarks[4]
        index = landmarks[8]
        middle = landmarks[12]
        
        # 计算三指中心点（如果捏合）或仅使用食指尖（如果未捏合）
        if is_pinching:
            # 三指捏合时使用三指中心点
            center_x = (thumb.x + index.x + middle.x) / 3.0
            center_y = (thumb.y + index.y + middle.y) / 3.0
        else:
            # 未捏合时使用食指尖
            center_x = index.x
            center_y = index.y
        
        # 使用高级 CoordinateMapper 进行映射和平滑 (与绘图模式一致)
        # 传入归一化坐标 (0-1)，返回屏幕坐标 (0-W, 0-H)
        curr_x, curr_y = self.cursor_mapper.map((center_x, center_y))

        # 关键兜底：只要检测到捏合，就强制进入可书写模式并确保PPT指针也切到笔
        # 避免模式短时不稳定时仍停留在 NAV/激光，导致“只能出红点/写不上去线”
        if is_pinching and self.auto_pen_on_pinch and self.current_mode == MODE_NAV:
            self.current_mode = MODE_PEN
            self.trigger_mode_switch_shortcut()

        # ★ 如果使用透明叠加层，始终更新光标位置（让用户知道笔在哪）
        if self.use_overlay and self.overlay:
            is_eraser_mode = (self.current_mode == MODE_ERASER)
            self.overlay.update_cursor(
                int(curr_x), int(curr_y),
                is_drawing=is_pinching and not is_eraser_mode,
                is_erasing=is_pinching and is_eraser_mode
            )

        # 3. 分模式执行
        if self.current_mode == MODE_PEN or self.current_mode == MODE_ERASER:
            # 只有捏合时才画线/擦除
            if is_pinching:
                if not self.mouse_down:
                    # 开始捏合
                    self.mouse_down = True
                    self.point_history.clear()
                    # 记录起点
                    self.point_history.append((curr_x, curr_y))
                    
                    # 如果使用透明叠加层，开始笔画
                    if self.use_overlay and self.overlay:
                        if self.current_mode == MODE_PEN:
                            self.overlay.set_pen_color("#FF0000")  # 红色
                            self.overlay.start_stroke(int(curr_x), int(curr_y))
                        # 橡皮擦模式不需要 start_stroke，直接擦
                else:
                    # 更新历史点
                    self.point_history.append((curr_x, curr_y))
                
                    # 画线/擦除
                    if len(self.point_history) >= 2:
                        p1 = self.point_history[-2]
                        p2 = self.point_history[-1]
                        
                        # 方案1：透明叠加层（用户主动开启，最可靠）
                        if self.use_overlay and self.overlay:
                            if self.current_mode == MODE_PEN:
                                self.overlay.draw_to(int(p2[0]), int(p2[1]))
                            else:  # MODE_ERASER - 橡皮擦
                                self.overlay.erase_at(int(p2[0]), int(p2[1]), radius=35)
                        else:
                            # 方案2：COM 直接画线（PPT 放映模式）
                            com_ok = self.ppt_controller.draw_line(p1[0], p1[1], p2[0], p2[1])
                            
                            if not com_ok:
                                # 方案3：pyautogui 模拟拖拽（兜底）
                                try:
                                    if len(self.point_history) == 2:
                                        pyautogui.moveTo(p1[0], p1[1], duration=0)
                                        pyautogui.mouseDown(button='left')
                                    pyautogui.moveTo(p2[0], p2[1], duration=0)
                                except Exception:
                                    pass
            else:
                if self.mouse_down:
                    # 结束捏合
                    self.mouse_down = False
                    self.point_history.clear()
                    # 记录释放时间，启动模式切换冷却锁
                    self.last_pinch_release_time = time.time()
                    
                    # 如果使用透明叠加层，结束笔画
                    if self.use_overlay and self.overlay:
                        self.overlay.end_stroke()
                    else:
                        # pyautogui 模式需要 mouseUp
                        try:
                            pyautogui.mouseUp(button='left')
                        except Exception:
                            pass
                
                # 未捏合时仅移动光标 (不需要高级平滑，线性跟随即可)
                if not self.use_overlay:
                    try:
                        pyautogui.moveTo(curr_x, curr_y, duration=0)
                    except Exception:
                        pass

        elif self.current_mode == MODE_NAV:
            # 挥手翻页逻辑（NAV模式下不控制光标）
            # 使用归一化坐标，适配不同分辨率
            # 实现空间复位逻辑（Neutral Zone），彻底解决回位误触问题
            
            # 检查是否翻页（COM接口）- 实现"翻页后自动切笔"
            if self.ppt_controller.check_slide_changed():
                print("检测到翻页，已自动切换为画笔模式")
                # 关键修复：PPT 指针切到笔还不够，内部模式也要切到 PEN，否则“写不上去”
                if self.auto_pen_on_slide_change:
                    self.current_mode = MODE_PEN
                    self.trigger_mode_switch_shortcut()
                    return
            
            current_time = time.time()
            dt_nav = max(1e-3, current_time - self._last_nav_eval_time)
            self._last_nav_eval_time = current_time
            delta_x_norm = center_x - self.prev_hand_x_norm
            velocity_norm_s = delta_x_norm / dt_nav
            self.last_nav_delta_x_norm = float(delta_x_norm)
            self.last_nav_velocity_norm_s = float(velocity_norm_s)
            
            # 检查手部是否在安全区（归一化坐标）
            in_neutral_zone = (NEUTRAL_ZONE_X_MIN <= center_x <= NEUTRAL_ZONE_X_MAX and
                              NEUTRAL_ZONE_Y_MIN <= center_y <= NEUTRAL_ZONE_Y_MAX)
            self.last_in_neutral_zone = bool(in_neutral_zone)
            
            # 状态机逻辑：实现空间复位机制
            if self.nav_state == STATE_IDLE:
                # 空闲状态：检测挥手
                if (current_time - self.last_swipe_time) > self.swipe_cooldown:
                    if abs(delta_x_norm) > self.swipe_threshold and abs(velocity_norm_s) > self.swipe_velocity_threshold:
                        if delta_x_norm > self.swipe_threshold:
                            # 向右挥手 -> 上一页
                            try:
                                pyautogui.press('left')
                                print(f"上一页 (距离: {delta_x_norm:.3f})")
                                self.last_swipe_time = current_time
                                self.nav_state = STATE_WAIT_NEUTRAL  # 进入等待归位状态
                                self.neutral_stay_count = 0
                                self.prev_hand_x_norm = center_x
                            except Exception:
                                pass
                        elif delta_x_norm < -self.swipe_threshold:
                            # 向左挥手 -> 下一页
                            try:
                                pyautogui.press('right')
                                print(f"下一页 (距离: {delta_x_norm:.3f})")
                                self.last_swipe_time = current_time
                                self.nav_state = STATE_WAIT_NEUTRAL  # 进入等待归位状态
                                self.neutral_stay_count = 0
                                self.prev_hand_x_norm = center_x
                            except Exception:
                                pass
                    else:
                        # 移动距离不够，正常更新参考位置
                        self.prev_hand_x_norm = center_x
                else:
                    # 冷却期间，持续更新参考位置
                    self.prev_hand_x_norm = center_x
            
            elif self.nav_state == STATE_WAIT_NEUTRAL:
                # 等待归位状态：必须回到安全区才能重新检测挥手
                # 这是解决"回位反向操作"的核心机制
                if in_neutral_zone:
                    self.neutral_stay_count += 1
                    if self.neutral_stay_count >= self.neutral_stay_frames:
                        # 已归位，重置状态
                        self.nav_state = STATE_IDLE
                        self.neutral_stay_count = 0
                        print("手部已归位，可以继续挥手")
                else:
                    # 不在安全区，重置计数
                    self.neutral_stay_count = 0
                
                # 持续更新参考位置（但不检测挥手）
                self.prev_hand_x_norm = center_x

            # NAV模式下不移动光标，只检测挥手

    def get_mode_name(self):
        if self.current_mode == MODE_PEN:
            return "PEN"
        if self.current_mode == MODE_ERASER:
            return "ERASER"
        if self.current_mode == MODE_NAV:
            return "NAV"
        return "UNKNOWN"
    
    def _get_mode_name_from_gesture(self, gesture):
        """辅助方法：根据手势常量返回名称"""
        if gesture == MODE_PEN:
            return "PEN"
        if gesture == MODE_ERASER:
            return "ERASER"
        if gesture == MODE_NAV:
            return "NAV"
        return "NONE"

    def close(self):
        """清理资源"""
        if self.hands:
            self.hands.close()


# --- 🚀 主入口 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    # 独立运行时，使用内部 MediaPipe
    controller = PPTGestureController(external_mp=False)

    # 创建可调整大小的窗口
    window_name = "PPT Gesture Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, CAM_WIDTH, CAM_HEIGHT)

    print("=" * 60)
    print("PPT手势控制系统启动")
    print("=" * 60)
    print("手势说明:")
    print("  只有食指: 画笔模式 (PEN) -> 捏合书写 (带防抖)")
    print("  食指+中指: 橡皮模式 (ERASER) -> 捏合擦除 (带防抖)")
    print("  食指+中指+无名指: 导航/激光笔模式 (NAV/LASER)")
    print("  导航模式下左右挥手: 翻页")
    print("=" * 60)
    print("按 'q' 退出")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("警告: 无法读取摄像头帧")
                break

            frame = controller.process_frame(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")
