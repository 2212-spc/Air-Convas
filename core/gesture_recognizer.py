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
        pinch_threshold: float,
        pinch_release_threshold: float,
        swipe_threshold: float,
        swipe_velocity_threshold: float = 0.015,
        swipe_cooldown_frames: int = 20,
        history_len: int = 15,
        pinch_confirm_frames: int = 3,
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
        self._pinch_on_count = 0
        self._pinch_off_count = 0

    def _thumb_up(self, hand: Hand) -> bool:
        tip_x, _ = hand.landmarks_norm[THUMB_TIP]
        base_x, _ = hand.landmarks_norm[THUMB_TIP - 1]
        if hand.handedness == "LEFT":
            return tip_x < base_x
        return tip_x > base_x

    def _finger_up(self, hand: Hand, tip_idx: int) -> bool:
        tip_y = hand.landmarks_norm[tip_idx][1]
        base_y = hand.landmarks_norm[tip_idx - 2][1]
        return tip_y < base_y

    def fingers_up(self, hand: Hand) -> List[bool]:
        return [
            self._thumb_up(hand),
            self._finger_up(hand, INDEX_TIP),
            self._finger_up(hand, MIDDLE_TIP),
            self._finger_up(hand, RING_TIP),
            self._finger_up(hand, PINKY_TIP),
        ]

    def _update_swipe(self, wrist_xy: tuple) -> Optional[str]:
        self.history.append(wrist_xy)
        if len(self.history) < self.history.maxlen:
            return None
        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1
            return None

        # 使用“最近5帧均值 - 最早5帧均值”
        n = len(self.history)
        k = max(5, n // 3)
        xs = [p[0] for p in self.history]
        ys = [p[1] for p in self.history]
        x0 = sum(xs[:k]) / k
        y0 = sum(ys[:k]) / k
        x1 = sum(xs[-k:]) / k
        y1 = sum(ys[-k:]) / k
        dx, dy = x1 - x0, y1 - y0

        # 位移阈值
        if abs(dx) < self.swipe_threshold and abs(dy) < self.swipe_threshold:
            return None
        # 速度阈值（单位：每帧归一化位移）
        speed = (abs(dx) + abs(dy)) / max(n - 1, 1)
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
        pinch_dist = point_distance(
            hand.landmarks_norm[THUMB_TIP], hand.landmarks_norm[INDEX_TIP]
        )

        pinch_start = False
        pinch_end = False
        # 去抖逻辑
        if pinch_dist < self.pinch_threshold:
            self._pinch_on_count += 1
            self._pinch_off_count = 0
        elif pinch_dist > self.pinch_release_threshold:
            self._pinch_off_count += 1
            self._pinch_on_count = 0
        else:
            # 中间区域，计数不增加
            self._pinch_on_count = 0
            self._pinch_off_count = 0

        if not self.pinch_active and self._pinch_on_count >= self.pinch_confirm_frames:
            self.pinch_active = True
            pinch_start = True
        if self.pinch_active and self._pinch_off_count >= self.pinch_confirm_frames:
            self.pinch_active = False
            pinch_end = True

        open_palm = sum(fingers) >= 4
        fist = not any(fingers)
        index_only = fingers[1] and not any(fingers[i] for i in (0, 2, 3, 4))
        index_middle = fingers[1] and fingers[2] and not any(fingers[i] for i in (0, 3, 4))

        swipe = self._update_swipe(hand.landmarks_norm[0])

        mode = "idle"
        if self.pinch_active:
            mode = "draw"
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
            "mode": mode,
            "swipe": swipe,
            "pinch_distance": pinch_dist,
        }
