# -*- coding: utf-8 -*-
"""手部检测模块 - 使用 MediaPipe Hands 进行手部关键点检测"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Any, NewType

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("mediapipe is required for hand detection") from exc

# 关键点索引常量
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# [Type Hints] 定义语义化类型别名，提升代码可读性
PixelPoint = Tuple[int, int]        # 屏幕像素坐标 (x, y)
NormPoint = Tuple[float, float]     # 归一化坐标 (0.0~1.0)
BoundingBox = Tuple[int, int, int, int]  # 边界框 (min_x, min_y, max_x, max_y)


@dataclass
class Hand:
    """
    手部数据模型 (Data Transfer Object)
    
    用于在检测器和上层应用之间传递检测结果。
    """
    landmarks: List[PixelPoint]      # 21个关键点的像素坐标
    landmarks_norm: List[NormPoint]  # 21个关键点的归一化坐标
    bbox: BoundingBox                # 手部外接矩形框
    handedness: str                  # "LEFT" 或 "RIGHT"
    confidence: float                # 检测置信度 (0.0~1.0)


class HandDetector:
    """
    手部检测器封装类
    """
    
    # [Type Hints] 显式声明 MediaPipe 对象类型
    _mp_draw: Any
    _hands: Any

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
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
        """释放 MediaPipe 资源"""
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray) -> List[Hand]:
        """
        在 BGR 图像中检测手部。
        
        Args:
            frame_bgr (np.ndarray): OpenCV 读取的原始帧 (BGR 格式)。

        Returns:
            List[Hand]: 检测到的手部对象列表。如果没有检测到，返回空列表。
        """
        # MediaPipe 需要 RGB 输入
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 执行推理
        results = self._hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        hands: List[Hand] = []
        
        # 遍历检测到的每只手
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # 转换坐标
            pts_norm: List[NormPoint] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            pts_px: List[PixelPoint] = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # 计算边界框
            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            bbox: BoundingBox = (min(xs), min(ys), max(xs), max(ys))
            
            # 解析左右手信息
            label = str(handedness.classification[0].label.upper())
            score = float(handedness.classification[0].score)

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
        """
        在画面上绘制手部骨架。

        Args:
            frame_bgr (np.ndarray): 目标绘制图像。
            hand (Hand): 手部数据对象。
            simple (bool): 简化绘制模式。True 只画指尖（高性能），False 画完整骨架。
        """
        mp_hands = mp.solutions.hands

        if simple:
            # 简化模式：只绘制关键的5个指尖
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
    """计算两点间的欧几里得距离"""
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))