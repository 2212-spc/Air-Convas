from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - dependency provided by requirements
    raise ImportError("mediapipe is required for hand detection") from exc

# Landmark indices for readability
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


@dataclass
class Hand:
    landmarks: List[Tuple[int, int]]
    landmarks_norm: List[Tuple[float, float]]
    bbox: Tuple[int, int, int, int]
    handedness: str
    confidence: float


class HandDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,  # 降低到0.5，提高灵敏度
        tracking_confidence: float = 0.5,
    ) -> None:
        mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray) -> List[Hand]:
        """Detect hands in a BGR frame."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        hands: List[Hand] = []
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            pts_norm = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            label = handedness.classification[0].label.upper()
            score = handedness.classification[0].score

            hands.append(
                Hand(
                    landmarks=pts_px,
                    landmarks_norm=pts_norm,
                    bbox=bbox,
                    handedness=label,
                    confidence=score,
                )
            )
        return hands

    def draw_hand(self, frame_bgr: np.ndarray, hand: Hand, simple: bool = True) -> None:
        """Draw landmarks and connections for a detected hand.

        Args:
            frame_bgr: BGR frame to draw on
            hand: Hand object with landmarks
            simple: If True, only draw key landmarks (faster). If False, draw all.
        """
        mp_hands = mp.solutions.hands

        if simple:
            # 简化模式：只绘制关键的5个指尖，性能更好
            key_landmarks = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            for idx in key_landmarks:
                x, y = hand.landmarks[idx]
                cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        else:
            # 完整模式：绘制所有关键点和连接线
            for x, y in hand.landmarks:
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = hand.landmarks[start_idx]
                end_point = hand.landmarks[end_idx]
                cv2.line(frame_bgr, start_point, end_point, (255, 0, 0), 2)


def distance(pt1: Sequence[float], pt2: Sequence[float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))
