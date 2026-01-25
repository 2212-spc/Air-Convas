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
from collections import deque

# COM接口支持
try:
    import win32com.client
    COM_AVAILABLE = True
except ImportError:
    COM_AVAILABLE = False
    print("警告: win32com不可用，将使用模拟按键作为降级方案")

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
CONFIRM_DELAY = 1.0

# 手指伸直检测容差 (归一化坐标，0.0~1.0)
# 允许指尖稍微在关节下方，仍然判定为伸直
FINGER_EXTEND_TOLERANCE = 0.1  # 约10%的容差

# 手指伸直距离阈值 (归一化坐标，0.0~1.0)
# 指尖到关节的距离必须大于此值才认为手指真正伸直
FINGER_EXTEND_DISTANCE_THRESHOLD = 0.03  # 至少3%的距离

# 三指捏合判定阈值 (像素距离) - 较小的值使检测更精确
PINCH_THRESHOLD = 45  # 降低阈值，使三指捏合检测更精确

# 挥手判定阈值 (归一化坐标，0.0~1.0)
# 使用归一化坐标适配不同分辨率
SWIPE_THRESHOLD = 0.3  # 约10%的归一化距离


# 挥手冷却时间 (秒)
SWIPE_COOLDOWN = 0.5

# 空间复位逻辑参数（Neutral Zone）
NEUTRAL_ZONE_X_MIN = 0.2  # 屏幕中央安全区（归一化坐标）
NEUTRAL_ZONE_X_MAX = 0.8
NEUTRAL_ZONE_Y_MIN = 0
NEUTRAL_ZONE_Y_MAX = 1
NEUTRAL_STAY_FRAMES = 10  # 在安全区停留的帧数才认为归位

# OneEuroFilter 参数
ONEEURO_MIN_CUTOFF = 1.0  # 最小截止频率 (Hz)
ONEEURO_BETA = 0.007       # 速度系数
ONEEURO_DCUTOFF = 1.0      # 速度平滑截止频率

# 状态机迟滞参数
GESTURE_CONFIRM_FRAMES = 5  # 连续N帧确认才更新状态

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
        self.slide_show = None
        self.slide_show_view = None
        self.last_slide_index = -1
        
        if COM_AVAILABLE:
            try:
                self.app = win32com.client.GetActiveObject("PowerPoint.Application")
                # 获取当前演示文稿的幻灯片放映
                if self.app.Presentations.Count > 0:
                    pres = self.app.ActivePresentation
                    if pres.SlideShowWindow:
                        self.slide_show = pres.SlideShowWindow
                        self.slide_show_view = self.slide_show.View
                        self.last_slide_index = self.slide_show_view.CurrentSlide.SlideIndex
                        print("COM接口初始化成功")
            except Exception as e:
                print(f"COM初始化失败: {e}，将使用模拟按键")
                self.app = None
    
    def set_pointer_type(self, pointer_type):
        """
        设置PPT指针类型（确定性）
        pointer_type: 1=箭头, 2=画笔, 5=橡皮
        """
        if self.slide_show_view:
            try:
                self.slide_show_view.PointerType = pointer_type
                return True
            except Exception as e:
                print(f"设置指针类型失败: {e}")
                return False
        return False
    
    def check_slide_changed(self):
        """
        检测是否翻页，如果翻页则返回True并自动切笔
        这是实现"翻页后自动切笔"的核心功能
        """
        if self.slide_show_view:
            try:
                current_index = self.slide_show_view.CurrentSlide.SlideIndex
                if current_index != self.last_slide_index:
                    self.last_slide_index = current_index
                    # 翻页后自动切笔（用户核心需求）
                    self.set_pointer_type(2)  # 画笔
                    return True
            except Exception:
                pass
        return False
    
    def reconnect(self):
        """尝试重新连接PPT"""
        if COM_AVAILABLE:
            try:
                self.app = win32com.client.GetActiveObject("PowerPoint.Application")
                if self.app.Presentations.Count > 0:
                    pres = self.app.ActivePresentation
                    if pres.SlideShowWindow:
                        self.slide_show = pres.SlideShowWindow
                        self.slide_show_view = self.slide_show.View
                        self.last_slide_index = self.slide_show_view.CurrentSlide.SlideIndex
                        return True
            except Exception:
                pass
        return False


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


class HandController:
    def __init__(self):
        # 1. 初始化 MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7  # 提升到0.7，确保高质量跟踪
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 2. 状态变量
        self.current_mode = MODE_NAV  # 默认模式
        self.last_gesture = MODE_NONE
        self.gesture_timer = 0
        self.last_time = time.time()

        # 3. 平滑算法变量
        self.prev_x, self.prev_y = 0, 0

        # 4. 导航模式变量 (用于计算挥手速度)
        self.prev_hand_x_norm = 0  # 归一化坐标（0.0-1.0）
        self.last_swipe_time = 0
        
        # 5. 状态机变量（空间复位逻辑）
        self.nav_state = STATE_IDLE
        self.neutral_stay_count = 0
        
        # 6. PPT控制器（COM接口）
        self.ppt_controller = PPTController()

        # 5. 鼠标状态追踪
        self.mouse_down = False
        
        # 6. OneEuroFilter: 为21个关键点的x, y, z坐标创建滤波器
        self.landmark_filters = {}
        for i in range(21):
            for coord in ['x', 'y', 'z']:
                self.landmark_filters[(i, coord)] = OneEuroFilter(
                    min_cutoff=ONEEURO_MIN_CUTOFF,
                    beta=ONEEURO_BETA,
                    dcutoff=ONEEURO_DCUTOFF
                )
        
        # 7. 状态机迟滞：连续帧确认机制
        self.gesture_history = deque(maxlen=GESTURE_CONFIRM_FRAMES)
        self.confirmed_gesture = MODE_NONE

    def get_distance(self, p1, p2):
        """计算两点欧几里得距离"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def process_frame(self, frame):
        """核心处理循环"""
        # 镜像翻转，符合直觉
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制骨架 (调试用)
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # --- 🟢 模块一: 视觉感知 (识别手势) ---
                # 对关键点进行OneEuroFilter滤波，消除抖动
                filtered_landmarks = self.filter_landmarks(hand_landmarks, current_time)
                detected_gesture = self.recognize_gesture(filtered_landmarks, h, w)

                # --- 🔵 模块二: 状态机 (带时间缓冲的模式切换) ---
                # 状态机迟滞：连续N帧确认才更新状态
                self.gesture_history.append(detected_gesture)
                if len(self.gesture_history) == GESTURE_CONFIRM_FRAMES:
                    # 检查是否所有帧都是同一手势
                    if len(set(self.gesture_history)) == 1:
                        self.confirmed_gesture = detected_gesture
                    # 如果历史记录满了但手势不一致，清空重新开始
                    elif len(set(self.gesture_history)) > 1:
                        self.gesture_history.clear()
                
                # 注意: 如果正在捏合(写字中)，则锁定模式切换
                is_pinching = self.check_pinch(filtered_landmarks, h, w)

                if not is_pinching:
                    # 使用确认的手势进行模式切换
                    self.update_mode(self.confirmed_gesture, dt)

                # --- 🟠 模块三: 执行层 (平滑 & 动作) ---
                # 注意：执行层仍使用原始landmarks，因为滤波后的landmarks是列表格式
                self.execute_action(hand_landmarks, is_pinching, w, h)

                # --- UI 反馈 ---
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
                
                # 显示三指捏合状态和距离信息
                if is_pinching:
                    cv2.putText(
                        frame, "3-FINGER PINCH", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                    # 在画面上绘制三指连接线，可视化捏合状态
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]
                    middle = hand_landmarks.landmark[12]
                    
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
        else:
            # 无手时释放鼠标并重置状态
            if self.mouse_down:
                try:
                    pyautogui.mouseUp()
                    self.mouse_down = False
                except Exception:
                    pass
            # 重置手势历史
            self.gesture_history.clear()
            self.confirmed_gesture = MODE_NONE
            # 重置导航状态机
            if self.nav_state != STATE_IDLE:
                self.nav_state = STATE_IDLE
                self.neutral_stay_count = 0

        return frame

    def filter_landmarks(self, landmarks, t):
        """
        对21个关键点进行OneEuroFilter滤波，消除抖动
        返回一个包含滤波后坐标的列表
        """
        filtered = []
        for i, landmark in enumerate(landmarks.landmark):
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
        
        # 归一化比对：如果距离小于手掌宽度的60%，认为拇指闭合
        if palm_width < 1e-6:
            return False
        
        ratio = thumb_to_pinky / palm_width
        return ratio > 0.6  # 张开阈值

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

        # 检测拇指（使用距离比对法）
        thumb_open = self.check_thumb_state(landmarks)
        finger_states.append(1 if thumb_open else 0)

        # 检测其他四指（使用向量角度法）
        for tip_idx, pip_idx, mcp_idx in finger_configs:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            
            angle = self.calculate_finger_angle(mcp, pip, tip)
            
            # 角度 < 30度 认为伸直，> 90度 认为弯曲
            # 考虑到测量噪声，放宽至30度
            is_extended = angle < 30.0
            finger_states.append(1 if is_extended else 0)

        # fingers = [拇指, 食指, 中指, 无名指, 小指]
        # 实现排他性逻辑，确保手势不会相互混淆
        
        # 手势1 (MODE_PEN): 只有食指伸直，其他必须弯曲
        # 负向条件：中指、无名指、小指必须处于弯曲状态
        if (finger_states[1] == 1 and  # 食指伸直
            finger_states[2] == 0 and  # 中指弯曲
            finger_states[3] == 0 and  # 无名指弯曲
            finger_states[4] == 0):    # 小指弯曲
            return MODE_PEN
        
        # 手势2 (MODE_ERASER): 食指+中指伸直，无名指和小指必须弯曲
        # 负向条件：无名指、小指必须弯曲
        elif (finger_states[1] == 1 and  # 食指伸直
              finger_states[2] == 1 and  # 中指伸直
              finger_states[3] == 0 and  # 无名指弯曲
              finger_states[4] == 0):    # 小指弯曲
            return MODE_ERASER
        
        # 手势3 (MODE_NAV): 食指+中指+无名指伸直（美式）
        # 允许小指稍微弯曲（因为解剖学限制）
        elif (finger_states[1] == 1 and  # 食指伸直
              finger_states[2] == 1 and  # 中指伸直
              finger_states[3] == 1):   # 无名指伸直
            # 小指可以稍微弯曲，不严格要求
            return MODE_NAV

        return MODE_NONE  # 默认

    def update_mode(self, detected_gesture, dt):
        """状态机逻辑: 防误触的时间积累"""
        if detected_gesture != MODE_NONE and detected_gesture == self.last_gesture:
            self.gesture_timer += dt
            if self.gesture_timer >= CONFIRM_DELAY:
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
        print(f"切换模式到: {self.get_mode_name()}")
        
        # 优先使用COM接口（确定性）
        if COM_AVAILABLE:
            # 如果COM连接丢失，尝试重连
            if not self.ppt_controller.slide_show_view:
                self.ppt_controller.reconnect()
            
            if self.ppt_controller.slide_show_view:
                if self.current_mode == MODE_PEN:
                    if self.ppt_controller.set_pointer_type(2):  # 画笔
                        print("已通过COM接口切换到画笔模式")
                        return
                elif self.current_mode == MODE_ERASER:
                    if self.ppt_controller.set_pointer_type(5):  # 橡皮
                        print("已通过COM接口切换到橡皮模式")
                        return
                elif self.current_mode == MODE_NAV:
                    if self.ppt_controller.set_pointer_type(1):  # 箭头
                        print("已通过COM接口切换到箭头模式")
                        return
        
        # 降级方案：使用模拟按键
        try:
            if self.current_mode == MODE_PEN:
                pyautogui.hotkey('ctrl', 'p')
            elif self.current_mode == MODE_ERASER:
                pyautogui.hotkey('ctrl', 'e')
            elif self.current_mode == MODE_NAV:
                pyautogui.hotkey('ctrl', 'a')
        except Exception as e:
            print(f"快捷键执行失败: {e}")

    def check_pinch(self, landmarks, h, w):
        """检测拇指、食指、中指三指捏合"""
        # 获取三指指尖坐标
        # landmarks 可能是原始MediaPipe对象或滤波后的列表
        if hasattr(landmarks, 'landmark'):
            thumb = landmarks.landmark[4]   # 拇指尖
            index = landmarks.landmark[8]    # 食指尖
            middle = landmarks.landmark[12]  # 中指尖
        else:
            # 滤波后的列表格式
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]

        # 转换为像素坐标
        thumb_px = (thumb.x * w, thumb.y * h)
        index_px = (index.x * w, index.y * h)
        middle_px = (middle.x * w, middle.y * h)

        # 计算三个指尖之间的距离
        dist_thumb_index = math.hypot(thumb_px[0] - index_px[0], thumb_px[1] - index_px[1])
        dist_thumb_middle = math.hypot(thumb_px[0] - middle_px[0], thumb_px[1] - middle_px[1])
        dist_index_middle = math.hypot(index_px[0] - middle_px[0], index_px[1] - middle_px[1])

        # 三指捏合：三个指尖之间的距离都要小于阈值
        is_pinching = (dist_thumb_index < PINCH_THRESHOLD and
                      dist_thumb_middle < PINCH_THRESHOLD and
                      dist_index_middle < PINCH_THRESHOLD)

        return is_pinching

    def execute_action(self, landmarks, is_pinching, w, h):
        """根据当前模式执行具体操作"""

        # 1. 获取三指中心点坐标（用于更稳定的控制）
        thumb = landmarks.landmark[4]
        index = landmarks.landmark[8]
        middle = landmarks.landmark[12]
        
        # 计算三指中心点（如果捏合）或仅使用食指尖（如果未捏合）
        if is_pinching:
            # 三指捏合时使用三指中心点
            center_x = (thumb.x + index.x + middle.x) / 3.0
            center_y = (thumb.y + index.y + middle.y) / 3.0
        else:
            # 未捏合时使用食指尖
            center_x = index.x
            center_y = index.y
        
        raw_x = np.interp(center_x, [0, 1], [0, SCREEN_WIDTH])
        raw_y = np.interp(center_y, [0, 1], [0, SCREEN_HEIGHT])

        # 2. 指数平滑算法 (Exponential Smoothing)
        curr_x = self.prev_x * SMOOTHING_FACTOR + raw_x * (1 - SMOOTHING_FACTOR)
        curr_y = self.prev_y * SMOOTHING_FACTOR + raw_y * (1 - SMOOTHING_FACTOR)

        # 更新历史值
        self.prev_x, self.prev_y = curr_x, curr_y

        # 3. 分模式执行
        if self.current_mode == MODE_PEN or self.current_mode == MODE_ERASER:
            # 只有捏合时才按下鼠标写字/擦除
            if is_pinching:
                if not self.mouse_down:
                    # 开始捏合，按下鼠标
                    try:
                        pyautogui.mouseDown()
                        self.mouse_down = True
                    except Exception:
                        pass
                # 持续捏合，移动鼠标（保持按下状态）
                try:
                    pyautogui.moveTo(curr_x, curr_y, duration=0)
                except Exception:
                    pass
            else:
                if self.mouse_down:
                    # 结束捏合，释放鼠标
                    try:
                        pyautogui.mouseUp()
                        self.mouse_down = False
                    except Exception:
                        pass
                # 未捏合时仅移动光标
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
            
            current_time = time.time()
            delta_x_norm = center_x - self.prev_hand_x_norm
            
            # 检查手部是否在安全区（归一化坐标）
            in_neutral_zone = (NEUTRAL_ZONE_X_MIN <= center_x <= NEUTRAL_ZONE_X_MAX and
                              NEUTRAL_ZONE_Y_MIN <= center_y <= NEUTRAL_ZONE_Y_MAX)
            
            # 状态机逻辑：实现空间复位机制
            if self.nav_state == STATE_IDLE:
                # 空闲状态：检测挥手
                if (current_time - self.last_swipe_time) > SWIPE_COOLDOWN:
                    if abs(delta_x_norm) > SWIPE_THRESHOLD:
                        if delta_x_norm > SWIPE_THRESHOLD:
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
                        elif delta_x_norm < -SWIPE_THRESHOLD:
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
                    if self.neutral_stay_count >= NEUTRAL_STAY_FRAMES:
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
        self.hands.close()


# --- 🚀 主入口 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    controller = HandController()

    # 创建可调整大小的窗口
    window_name = "PPT Gesture Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, CAM_WIDTH, CAM_HEIGHT)

    print("=" * 60)
    print("PPT手势控制系统启动")
    print("=" * 60)
    print("手势说明:")
    print("  只有食指: 画笔模式 (PEN)")
    print("  食指+中指: 橡皮模式 (ERASER)")
    print("  食指+中指+无名指: 导航模式 (NAV)")
    print("  拇指+食指+中指三指捏合: 开始绘制/擦除")
    print("  导航模式下左右挥手: 翻页（不控制光标）")
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

