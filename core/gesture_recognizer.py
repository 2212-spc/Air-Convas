# -*- coding: utf-8 -*-
"""手势识别模块 - 识别捏合、挥手等手势，支持模式锁定防误触"""

import math
from collections import deque
from typing import Deque, Dict, List, Optional

from core.hand_detector import (
    Hand,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)
from core.hand_detector import distance as point_distance


class GestureRecognizer:
    """
    手势识别器 - 工具模式版
    
    核心改进：
    - 移除自动模式切换（Open Palm, Index Only）
    - 统一使用捏合（Pinch）作为主要交互手势
    - 保留辅助手势：
        - 双指（Index+Middle）：开关UI
        - 三指滑动：撤销/重做
    """
    
    def __init__(
        self,
        pinch_threshold: float,
        pinch_release_threshold: float,
        swipe_threshold: float,
        swipe_velocity_threshold: float = 0.015,
        swipe_cooldown_frames: int = 20,
        history_len: int = 15,
        pinch_confirm_frames: int = 3,
        pinch_release_confirm_frames: int = 1,
        pinch_velocity_boost: float = 0.02,
        pinch_history_len: int = 5,
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = pinch_release_threshold
        self.swipe_threshold = swipe_threshold
        self.swipe_velocity_threshold = swipe_velocity_threshold
        self.swipe_cooldown_frames = swipe_cooldown_frames
        self.history: Deque[tuple] = deque(maxlen=history_len)
        self.pinch_active = False
        self.swipe_cooldown = 0
        self.pinch_confirm_frames = pinch_confirm_frames
        self.pinch_release_confirm_frames = pinch_release_confirm_frames
        self._pinch_on_count = 0
        self._pinch_off_count = 0
        
        # 速度感知相关
        self.pinch_velocity_boost = pinch_velocity_boost
        self._pinch_dist_history: Deque[float] = deque(maxlen=pinch_history_len)
        self._pinch_velocity: float = 0.0

    def _thumb_up(self, hand: Hand) -> bool:
        """判断拇指是否伸展"""
        tip_x, _ = hand.landmarks_norm[THUMB_TIP]
        base_x, _ = hand.landmarks_norm[THUMB_TIP - 1]
        if hand.handedness == "LEFT":
            return tip_x < base_x
        return tip_x > base_x

    def _finger_up(self, hand: Hand, tip_idx: int) -> bool:
        """判断手指是否伸展"""
        tip_y = hand.landmarks_norm[tip_idx][1]
        base_y = hand.landmarks_norm[tip_idx - 2][1]
        return tip_y < base_y

    def fingers_up(self, hand: Hand) -> List[bool]:
        """返回五指伸展状态"""
        return [
            self._thumb_up(hand),
            self._finger_up(hand, INDEX_TIP),
            self._finger_up(hand, MIDDLE_TIP),
            self._finger_up(hand, RING_TIP),
            self._finger_up(hand, PINKY_TIP),
        ]

    def _update_pinch_velocity(self, pinch_dist: float) -> float:
        """更新捏合速度"""
        self._pinch_dist_history.append(pinch_dist)
        if len(self._pinch_dist_history) < 2:
            return 0.0
        
        dists = list(self._pinch_dist_history)
        if len(dists) >= 3:
            recent_change = (dists[-1] - dists[-3]) / 2
        else:
            recent_change = dists[-1] - dists[-2]
        
        self._pinch_velocity = recent_change
        return recent_change

    def _get_adaptive_threshold(self, base_threshold: float) -> float:
        """获取自适应捏合阈值"""
        if self._pinch_velocity < -0.005:
            speed_factor = min(1.0, abs(self._pinch_velocity) / 0.02)
            boost = self.pinch_velocity_boost * speed_factor
            return base_threshold + boost
        return base_threshold

    def _update_swipe(self, wrist_xy: tuple) -> Optional[str]:
        """更新挥手检测"""
        self.history.append(wrist_xy)
        if len(self.history) < self.history.maxlen:
            return None
        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1
            return None

        n = len(self.history)
        k = max(5, n // 3)
        xs = [p[0] for p in self.history]
        ys = [p[1] for p in self.history]
        x0 = sum(xs[:k]) / k
        y0 = sum(ys[:k]) / k
        x1 = sum(xs[-k:]) / k
        y1 = sum(ys[-k:]) / k
        dx, dy = x1 - x0, y1 - y0

        if abs(dx) < self.swipe_threshold and abs(dy) < self.swipe_threshold:
            return None
        
        speed = math.hypot(dx, dy) / max(n - k, 1)
        if speed < self.swipe_velocity_threshold:
            return None

        if abs(dx) > abs(dy):
            gesture = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        else:
            gesture = "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"

        self.swipe_cooldown = self.swipe_cooldown_frames
        return gesture

    def classify(self, hand: Hand) -> Dict:
        """
        分类手势 - 工具模式
        
        统一逻辑：
        - 捏合 (Pinch) = 执行当前工具动作 (画/擦/激光)
        - 双指 (Index+Middle) = 切换UI
        - 三指 (Index+Middle+Ring) = 撤销/重做/粒子
        """
        fingers = self.fingers_up(hand)

        # 捏合检测（加入手掌尺度自适应，解决“写字小/远近变化”导致的灵敏度不一致）
        thumb_tip = hand.landmarks_norm[THUMB_TIP]
        index_tip = hand.landmarks_norm[INDEX_TIP]
        pinch_dist = point_distance(thumb_tip, index_tip)

        self._update_pinch_velocity(pinch_dist)

        pinch_start = False
        pinch_end = False

        # 用掌宽做尺度（食指MCP到小指MCP），将绝对阈值按手远近缩放
        index_mcp = hand.landmarks_norm[5]
        pinky_mcp = hand.landmarks_norm[17]
        palm_width = point_distance(index_mcp, pinky_mcp)
        ref_palm_width = 0.22  # 经验参考（归一化）；越大越“灵敏”
        scale = max(0.6, min(1.6, palm_width / max(ref_palm_width, 1e-6)))

        adaptive_threshold = self._get_adaptive_threshold(self.pinch_threshold * scale)
        release_threshold = self.pinch_release_threshold * scale

        if pinch_dist < adaptive_threshold:
            # 防误触：在阈值附近“轻微张开/悬停不动”时不计入捏合确认
            # 典型场景：两指靠得很近但没有真正捏合，pinch_dist 会在阈值附近小幅抖动
            if not self.pinch_active:
                hover_band = 0.012  # 阈值附近带宽（归一化）
                near_threshold = (adaptive_threshold - hover_band) < pinch_dist < adaptive_threshold
                low_velocity = abs(self._pinch_velocity) < 0.002
                if near_threshold and low_velocity:
                    # 不累积确认帧，但也不回退（避免“慢捏合永远触发不了”）
                    pass
                else:
                    self._pinch_on_count += 1
            else:
                self._pinch_on_count += 1
            self._pinch_off_count = 0
        elif pinch_dist > release_threshold:
            self._pinch_off_count += 1
            self._pinch_on_count = 0

        # 进入捏合确认：慢动作更严格（2帧），快速捏合仍保持灵敏（1帧）
        required_on = self.pinch_confirm_frames
        if not self.pinch_active:
            # 快速合拢：1帧触发
            if self._pinch_velocity < -0.01:
                required_on = min(required_on, 1)
            else:
                # 慢动作：只有在“非常接近”阈值时才需要2帧，避免牺牲灵敏度
                deep_pinch = pinch_dist < (adaptive_threshold - 0.018)
                required_on = 1 if deep_pinch else max(required_on, 2)

        if not self.pinch_active and self._pinch_on_count >= required_on:
            self.pinch_active = True
            pinch_start = True
        if self.pinch_active and self._pinch_off_count >= self.pinch_release_confirm_frames:
            self.pinch_active = False
            pinch_end = True

        # ========== 辅助手势检测 ==========
        fist = not any(fingers)
        
        # 食指+中指（其他弯曲） -> UI切换
        index_middle = fingers[1] and fingers[2] and not any(fingers[i] for i in (0, 3, 4))
        
        # 三指（食指+中指+无名指，拇指和小指弯曲） -> 撤销/重做
        three_fingers = fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]
        
        # 只有食指（其他都弯曲） -> 工具快捷切换模式
        index_only = fingers[1] and not any(fingers[i] for i in (0, 2, 3, 4))

        # 挥手检测（只在非捏合状态）
        wrist = hand.landmarks_norm[0]
        index_mcp = hand.landmarks_norm[5]
        pinky_mcp = hand.landmarks_norm[17]
        palm_center = (
            (wrist[0] + index_mcp[0] + pinky_mcp[0]) / 3,
            (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3
        )
        swipe = self._update_swipe(palm_center)

        # 确定模式
        mode = "idle"
        
        # 捏合 = 激活 (Active)
        if self.pinch_active:
            mode = "active"
        elif pinch_end:
            mode = "active_end"  # 刚结束捏合
        elif three_fingers:
            mode = "particle"
        elif fist:
            mode = "pause"
        elif index_middle:
            mode = "click"

        return {
            "fingers": fingers,
            "pinching": self.pinch_active,
            "pinch_start": pinch_start,
            "pinch_end": pinch_end,
            "fist": fist,
            "index_middle": index_middle,
            "three_fingers": three_fingers,
            "index_only": index_only,
            "mode": mode,
            "swipe": swipe,
            "pinch_distance": pinch_dist,
            "pinch_velocity": self._pinch_velocity,
        }
    
    def reset_pinch_history(self) -> None:
        """重置捏合历史"""
        self._pinch_dist_history.clear()
        self._pinch_velocity = 0.0
        self._pinch_on_count = 0
        self._pinch_off_count = 0
