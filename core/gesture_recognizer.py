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
    def __init__(
        self,
        pinch_threshold: float,#三指捏合阈值
        pinch_release_threshold: float,#三指捏合释放阈值
        swipe_threshold: float,#挥手阈值
        swipe_velocity_threshold: float = 0.015,#挥手速度阈值
        swipe_cooldown_frames: int = 20,#挥手冷却帧数
        history_len: int = 15,#挥手历史帧数
        pinch_confirm_frames: int = 3,#捏合确认帧数
        pinch_release_confirm_frames: int = 1,  # 释放确认帧数，单独设置让断笔更快
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = pinch_release_threshold
        self.swipe_threshold = swipe_threshold
        self.swipe_velocity_threshold = swipe_velocity_threshold
        self.swipe_cooldown_frames = swipe_cooldown_frames
        self.history: Deque[tuple] = deque(maxlen=history_len)
        self.pinch_active = False#捏合状态
        self.swipe_cooldown = 0
        self.pinch_confirm_frames = pinch_confirm_frames#捏合确认帧数
        self.pinch_release_confirm_frames = pinch_release_confirm_frames#捏合释放确认帧数
        self._pinch_on_count = 0#捏合开启计数
        self._pinch_off_count = 0#捏合关闭计数

    def _thumb_up(self, hand: Hand) -> bool:
        tip_x, _ = hand.landmarks_norm[THUMB_TIP]#拇指尖x坐标
        base_x, _ = hand.landmarks_norm[THUMB_TIP - 1]#拇指基部x坐标
        #判断拇指是否向上
        if hand.handedness == "LEFT":
            return tip_x < base_x
        return tip_x > base_x

    def _finger_up(self, hand: Hand, tip_idx: int) -> bool:
        tip_y = hand.landmarks_norm[tip_idx][1]#指尖y坐标
        base_y = hand.landmarks_norm[tip_idx - 2][1]#指基部y坐标
        #判断手指是否向上
        return tip_y < base_y

    def fingers_up(self, hand: Hand) -> List[bool]:
        return [
            self._thumb_up(hand),
            self._finger_up(hand, INDEX_TIP),
            self._finger_up(hand, MIDDLE_TIP),
            self._finger_up(hand, RING_TIP),
            self._finger_up(hand, PINKY_TIP),
        ]

    def _update_swipe(self, wrist_xy: tuple) -> Optional[str]:#更新挥手
        self.history.append(wrist_xy)#将手腕坐标添加到历史帧数中
        if len(self.history) < self.history.maxlen:
            return None#如果历史帧数小于最大历史帧数，返回None
        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1#如果挥手冷却帧数大于0，挥手冷却帧数减1
            return None#如果挥手冷却帧数大于0，返回None

        # 使用“最近5帧均值 - 最早5帧均值”计算位移
        n = len(self.history)
        k = max(5, n // 3)
        xs = [p[0] for p in self.history]#x坐标列表
        ys = [p[1] for p in self.history]#y坐标列表
        x0 = sum(xs[:k]) / k#x0坐标
        y0 = sum(ys[:k]) / k#y0坐标
        x1 = sum(xs[-k:]) / k#x1坐标
        y1 = sum(ys[-k:]) / k#y1坐标
        dx, dy = x1 - x0, y1 - y0#x轴位移和y轴位移

        # 位移阈值
        if abs(dx) < self.swipe_threshold and abs(dy) < self.swipe_threshold:#如果x轴位移和y轴位移小于阈值，返回None，判定没有挥手
            return None
        # 速度阈值（单位：每帧归一化位移）；使用欧氏距离并按有效间隔归一化，减少抖动误触
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
        fingers = self.fingers_up(hand)

        # 三指捏合检测（拇指+食指+中指）
        thumb_tip = hand.landmarks_norm[THUMB_TIP]
        index_tip = hand.landmarks_norm[INDEX_TIP]
        middle_tip = hand.landmarks_norm[MIDDLE_TIP]

        # 计算三对距离
        dist_thumb_index = point_distance(thumb_tip, index_tip)
        dist_thumb_middle = point_distance(thumb_tip, middle_tip)
        dist_index_middle = point_distance(index_tip, middle_tip)

        # 取最大距离作为判断依据（三指都要靠近）
        max_dist = max(dist_thumb_index, dist_thumb_middle, dist_index_middle)
        # 兼容性：保留原有的 pinch_dist（拇指-食指距离）
        pinch_dist = dist_thumb_index

        pinch_start = False
        pinch_end = False

        # 三指捏合去抖逻辑：
        # - 开始：三指都靠近（最大距离 < 阈值）
        # - 结束：任意一对分开（最大距离 > 释放阈值）
        if max_dist < self.pinch_threshold:
            self._pinch_on_count += 1
            self._pinch_off_count = 0
        elif max_dist > self.pinch_release_threshold:
            # 任意一指离开即可触发释放
            self._pinch_off_count += 1
            self._pinch_on_count = 0
        else:
            # 中间区域，计数不增加
            self._pinch_on_count = 0
            self._pinch_off_count = 0

        if not self.pinch_active and self._pinch_on_count >= self.pinch_confirm_frames:
            self.pinch_active = True
            pinch_start = True
        # 使用专用的释放确认帧数，让断笔更灵敏
        if self.pinch_active and self._pinch_off_count >= self.pinch_release_confirm_frames:
            self.pinch_active = False
            pinch_end = True

        open_palm = sum(fingers) >= 4
        fist = not any(fingers)
        index_only = fingers[1] and not any(fingers[i] for i in (0, 2, 3, 4))
        index_middle = fingers[1] and fingers[2] and not any(fingers[i] for i in (0, 3, 4))
        # 三指竖起（食指+中指+无名指）= 粒子特效模式
        three_fingers = fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]

        # 使用手掌中心而非手腕进行挥手检测（更准确）
        # 手掌中心 = (手腕 + 食指根 + 小指根) / 3
        wrist = hand.landmarks_norm[0]  # WRIST
        index_mcp = hand.landmarks_norm[5]  # INDEX_MCP
        pinky_mcp = hand.landmarks_norm[17]  # PINKY_MCP
        palm_center = (
            (wrist[0] + index_mcp[0] + pinky_mcp[0]) / 3,
            (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3
        )
        swipe = self._update_swipe(palm_center)

        mode = "idle"
        if self.pinch_active:
            mode = "draw"
        elif three_fingers:
            mode = "particle"  # 粒子特效模式
        elif open_palm:
            mode = "erase"
        elif index_only:
            mode = "move"
        elif fist:
            mode = "pause"
        elif index_middle:
            mode = "click"

        return {
            "fingers": fingers,
            "pinching": self.pinch_active,
            "pinch_start": pinch_start,
            "pinch_end": pinch_end,
            "open_palm": open_palm,
            "fist": fist,
            "index_only": index_only,
            "index_middle": index_middle,
            "three_fingers": three_fingers,
            "mode": mode,
            "swipe": swipe,
            "pinch_distance": pinch_dist,
        }
