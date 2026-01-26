# -*- coding: utf-8 -*-
"""异步手部检测模块 - 将推理放到独立线程以减少主循环阻塞"""

import threading
from typing import List, Optional
from collections import deque
import time

import numpy as np
import cv2

from core.hand_detector import HandDetector, Hand


class AsyncHandDetector:
    """
    异步手部检测器 - 在独立线程中运行 MediaPipe 推理
    
    特点：
    - 主线程不阻塞，始终使用最新可用的检测结果
    - 自动跳过积压的帧，只处理最新帧
    - 线程安全的结果共享
    
    使用方式：
        detector = AsyncHandDetector()
        detector.start()
        
        while True:
            frame = cap.read()
            detector.submit_frame(frame)  # 非阻塞
            hands = detector.get_result()  # 返回最新结果或 None
            ...
        
        detector.stop()
    """
    
    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        infer_width: int = 640,
        infer_height: int = 360,
    ):
        self._detector = HandDetector(
            max_num_hands=max_num_hands,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
        )
        self._infer_width = infer_width
        self._infer_height = infer_height
        
        # 线程控制
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # 帧队列（只保留最新帧）
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()
        
        # 结果队列
        self._result_lock = threading.Lock()
        self._latest_result: List[Hand] = []
        self._result_timestamp: float = 0.0
    
    def start(self) -> None:
        """启动推理线程"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止推理线程"""
        self._running = False
        self._frame_ready.set()  # 唤醒线程以便退出
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._detector.close()
    
    def submit_frame(self, frame: np.ndarray) -> None:
        """
        提交新帧进行检测（非阻塞）
        
        只保留最新帧，旧帧会被丢弃
        """
        # 缩放到推理分辨率
        frame_small = cv2.resize(
            frame, 
            (self._infer_width, self._infer_height), 
            interpolation=cv2.INTER_LINEAR
        )
        
        with self._frame_lock:
            self._latest_frame = frame_small
        self._frame_ready.set()
    
    def get_result(self) -> List[Hand]:
        """
        获取最新的检测结果（非阻塞）
        
        返回最新可用的结果，如果没有则返回空列表
        """
        with self._result_lock:
            return self._latest_result
    
    def _inference_loop(self) -> None:
        """推理线程主循环"""
        while self._running:
            # 等待新帧
            self._frame_ready.wait(timeout=0.1)
            if not self._running:
                break
            
            # 获取最新帧
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            self._frame_ready.clear()
            
            if frame is None:
                continue
            
            # 执行检测
            try:
                hands = self._detector.detect(frame)
            except Exception:
                hands = []
            
            # 更新结果
            with self._result_lock:
                self._latest_result = hands
                self._result_timestamp = time.time()


class SyncAsyncHandDetector:
    """
    同步/异步混合检测器 - 提供统一接口，可切换模式
    
    用于平滑过渡到异步模式，同时保持向后兼容
    """
    
    def __init__(
        self,
        async_mode: bool = True,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        infer_width: int = 640,
        infer_height: int = 360,
    ):
        self.async_mode = async_mode
        self._infer_width = infer_width
        self._infer_height = infer_height
        
        if async_mode:
            self._async_detector = AsyncHandDetector(
                max_num_hands=max_num_hands,
                detection_confidence=detection_confidence,
                tracking_confidence=tracking_confidence,
                infer_width=infer_width,
                infer_height=infer_height,
            )
            self._sync_detector = None
        else:
            self._sync_detector = HandDetector(
                max_num_hands=max_num_hands,
                detection_confidence=detection_confidence,
                tracking_confidence=tracking_confidence,
            )
            self._async_detector = None
    
    def start(self) -> None:
        """启动（异步模式下启动线程）"""
        if self._async_detector:
            self._async_detector.start()
    
    def stop(self) -> None:
        """停止"""
        if self._async_detector:
            self._async_detector.stop()
        if self._sync_detector:
            self._sync_detector.close()
    
    def _scale_hands(self, hands: List[Hand], target_w: int, target_h: int) -> List[Hand]:
        """将检测结果缩放到目标分辨率"""
        if not hands:
            return []
        
        scaled_hands = []
        for hand in hands:
            # 使用归一化坐标重新计算像素坐标
            new_landmarks = [
                (int(lm[0] * target_w), int(lm[1] * target_h)) 
                for lm in hand.landmarks_norm
            ]
            
            # 重新计算包围盒
            xs = [p[0] for p in new_landmarks]
            ys = [p[1] for p in new_landmarks]
            if xs and ys:
                new_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                new_bbox = (0, 0, 0, 0)
            
            scaled_hands.append(Hand(
                landmarks=new_landmarks,
                landmarks_norm=hand.landmarks_norm,
                bbox=new_bbox,
                handedness=hand.handedness,
                confidence=hand.confidence
            ))
        return scaled_hands

    def detect(self, frame: np.ndarray) -> List[Hand]:
        """
        检测手部
        
        异步模式：提交帧并返回最新可用结果
        同步模式：阻塞式检测
        """
        h, w = frame.shape[:2]
        hands = []

        if self._async_detector:
            self._async_detector.submit_frame(frame)
            hands = self._async_detector.get_result()
        else:
            # 同步模式：缩放后检测
            frame_small = cv2.resize(
                frame, 
                (self._infer_width, self._infer_height), 
                interpolation=cv2.INTER_LINEAR
            )
            hands = self._sync_detector.detect(frame_small)
        
        # 统一进行坐标缩放，确保返回的结果与输入 frame 的尺寸匹配
        return self._scale_hands(hands, w, h)
    
    def draw_hand(self, frame: np.ndarray, hand: Hand, simple: bool = True) -> None:
        """
        在帧上绘制手部关键点
        
        代理到底层的 HandDetector
        """
        if self._async_detector:
            self._async_detector._detector.draw_hand(frame, hand, simple)
        elif self._sync_detector:
            self._sync_detector.draw_hand(frame, hand, simple)

