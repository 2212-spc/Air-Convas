# -*- coding: utf-8 -*-
"""
手部检测核心模块 (Hand Detector)

基于 Google MediaPipe Hands 框架封装，提供高性能的实时手部追踪能力。
负责从原始视频帧中提取 21 个 3D 手部关键点，并进行坐标归一化处理。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Any, NewType

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("mediapipe is required for hand detection") from exc

# MediaPipe 关键点索引映射
# 0: 手腕 (WRIST)
# 1-4: 拇指 (THUMB) -> [1:CMC, 2:MCP, 3:IP, 4:TIP]
# 5-8: 食指 (INDEX) -> [5:MCP, 6:PIP, 7:DIP, 8:TIP]
# ... 以此类推
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# 语义化类型定义
PixelPoint = Tuple[int, int]        # 屏幕像素坐标 (x, y)
NormPoint = Tuple[float, float]     # 归一化坐标 (0.0~1.0)
BoundingBox = Tuple[int, int, int, int]  # 边界框 (min_x, min_y, max_x, max_y)


@dataclass
class Hand:
    """
    手部实体类 (Hand Entity)
    
    封装单只手的所有检测信息，作为系统内部传输的标准数据结构。
    
    Attributes:
        landmarks (List[PixelPoint]): 映射到屏幕分辨率的像素坐标列表 [(x,y), ...]。
        landmarks_norm (List[NormPoint]): 归一化的相对坐标列表 [(0.5, 0.5), ...]。
        bbox (BoundingBox): 手部区域的轴对齐包围盒 (AABB)。
        handedness (str): 手性分类结果 ('LEFT' 或 'RIGHT')。
        confidence (float): 模型输出的置信度分数。
    """
    landmarks: List[PixelPoint]
    landmarks_norm: List[NormPoint]
    bbox: BoundingBox
    handedness: str
    confidence: float


class HandDetector:
    """
    手部检测器 (HandDetector)
    
    封装 MediaPipe 解决方案，管理推理引擎的生命周期。
    支持 BGR 到 RGB 的色彩空间转换，以及多手检测结果的解析。

    Attributes:
        _mp_draw: MediaPipe 绘图工具（用于调试绘制）。
        _hands: MediaPipe Hands 图计算对象。
    """
    
    _mp_draw: Any
    _hands: Any

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        """
        初始化检测器。

        Args:
            max_num_hands (int): 最大检测手数。默认为 1（单手模式性能最佳）。
            detection_confidence (float): 首次检测的最小置信度阈值 (0.0~1.0)。
            tracking_confidence (float): 关键点追踪的最小置信度阈值。
                                       若追踪失败，下一帧将自动重新触发全图检测。
        """
        mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self) -> None:
        """释放 MediaPipe 图计算资源，避免内存泄漏。"""
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray) -> List[Hand]:
        """
        执行手部检测推理。

        流程：
        1. BGR -> RGB 色彩空间转换 (MediaPipe 要求 RGB 输入)。
        2. 图模型推理。
        3. 解析 landmarks 并进行坐标反归一化 (Mapped to pixels)。
        4. 计算 BBox 和手性。

        Args:
            frame_bgr (np.ndarray): 输入的 BGR 图像帧。

        Returns:
            List[Hand]: 检测到的 Hand 对象列表。
        """
        # OpenCV 默认使用 BGR，MediaPipe 需要 RGB
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 这是一个耗时操作 (CPU Inference)
        results = self._hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        hands: List[Hand] = []
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # 提取关键点
            pts_norm: List[NormPoint] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            # 反归一化映射到像素坐标
            pts_px: List[PixelPoint] = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # 计算外接矩形 (ROI)
            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            bbox: BoundingBox = (min(xs), min(ys), max(xs), max(ys))
            
            # MediaPipe 的左右手标签通常是基于“镜像前”的视角
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
        可视化绘制。

        Args:
            frame_bgr (np.ndarray): 绘制的目标图像（原地修改）。
            hand (Hand): 检测到的手部数据。
            simple (bool): 
                - True: 极简模式。只在 5 个指尖绘制绿点，减少视觉干扰，提升 FPS。
                - False: 调试模式。绘制完整的 21 点骨架结构。
        """
        mp_hands = mp.solutions.hands

        if simple:
            key_landmarks = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            for idx in key_landmarks:
                x, y = hand.landmarks[idx]
                cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        else:
            # 绘制节点
            for x, y in hand.landmarks:
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

            # 绘制骨骼连接线
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = hand.landmarks[start_idx]
                end_point = hand.landmarks[end_idx]
                cv2.line(frame_bgr, start_point, end_point, (255, 0, 0), 2)


def distance(pt1: Sequence[float], pt2: Sequence[float]) -> float:
    """
    计算两点间的欧几里得距离 (L2 Norm)。
    
    Args:
        pt1, pt2: 两个点的坐标 (x, y)。

    Returns:
        float: 距离值。
    """
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))