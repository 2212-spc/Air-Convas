# -*- coding: utf-8 -*-
"""
异步手部检测模块 (Async Detector)

核心目标：解决 MediaPipe 推理耗时导致的 UI 卡顿问题。
实现原理：
1. 生产者-消费者模型：主线程提交帧，子线程处理帧。
2. 丢帧策略 (Frame Dropping)：推理线程只处理最新帧，自动丢弃积压的旧帧，
   确保 UI 渲染永远不会因为等待推理而阻塞。
"""

import threading
import time
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np

from core.hand_detector import HandDetector, Hand, PixelPoint, NormPoint, BoundingBox


class AsyncHandDetector:
    """
    异步手部检测器 (AsyncHandDetector)
    
    在独立守护线程 (Daemon Thread) 中运行 MediaPipe 推理。
    使用线程锁 (Lock) 和事件 (Event) 实现线程安全的数据交换。
    
    Attributes:
        _detector (HandDetector): 底层同步检测器实例。
        _infer_width (int): 推理时的缩放宽度。
        _infer_height (int): 推理时的缩放高度。
        _running (bool): 线程运行标志。
        _frame_lock (Lock): 保护 _latest_frame 的互斥锁。
        _result_lock (Lock): 保护 _latest_result 的互斥锁。
    """
    
    # [Type Hints] 显式声明属性类型
    _detector: HandDetector
    _infer_width: int
    _infer_height: int
    _running: bool
    _thread: Optional[threading.Thread]
    _frame_lock: threading.Lock
    _latest_frame: Optional[np.ndarray]
    _frame_ready: threading.Event
    _result_lock: threading.Lock
    _latest_result: List[Hand]
    _result_timestamp: float
    
    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        infer_width: int = 640,
        infer_height: int = 360,
    ) -> None:
        """
        初始化异步检测器。

        Args:
            max_num_hands (int): 最大检测手数。
            detection_confidence (float): 检测置信度阈值。
            tracking_confidence (float): 追踪置信度阈值。
            infer_width (int): 推理图像宽度（降低分辨率可显著提速）。
            infer_height (int): 推理图像高度。
        """
        self._detector = HandDetector(
            max_num_hands=max_num_hands,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
        )
        self._infer_width = infer_width
        self._infer_height = infer_height
        
        # 线程控制
        self._running = False
        self._thread = None
        
        # 帧缓冲区（容量为1，只保留最新）
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_ready = threading.Event()
        
        # 结果缓冲区
        self._result_lock = threading.Lock()
        self._latest_result = []
        self._result_timestamp = 0.0
    
    def start(self) -> None:
        """启动后台推理线程。"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止后台线程并释放资源。"""
        self._running = False
        self._frame_ready.set()  # 唤醒线程以便让它有机会检查 _running 标志并退出
        
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        self._detector.close()
    
    def submit_frame(self, frame: np.ndarray) -> None:
        """
        提交新帧进行检测（非阻塞）。
        
        如果推理线程正在忙，旧的待处理帧会被直接覆盖（丢帧策略），
        保证系统总是处理最新数据，而不是处理几秒前的延迟画面。

        Args:
            frame (np.ndarray): 原始 BGR 图像帧。
        """
        # 预先缩放到推理分辨率，减少在锁内的时间开销
        frame_small = cv2.resize(
            frame, 
            (self._infer_width, self._infer_height), 
            interpolation=cv2.INTER_LINEAR
        )
        
        with self._frame_lock:
            self._latest_frame = frame_small
        
        # 通知线程有新数据
        self._frame_ready.set()
    
    def get_result(self) -> List[Hand]:
        """
        获取最新的检测结果（非阻塞）。
        
        Returns:
            List[Hand]: 最新可用的手部检测结果。如果尚未有结果，返回空列表。
        """
        with self._result_lock:
            # 返回副本或引用皆可，列表本身是可变的，但 Hand 对象最好视为不可变
            return list(self._latest_result)
    
    def _inference_loop(self) -> None:
        """
        后台推理主循环。
        
        逻辑：等待事件 -> 获取帧 -> 推理 -> 更新结果 -> 循环。
        """
        while self._running:
            # 阻塞等待新帧信号，避免空转占用 CPU
            # timeout=0.1 确保即使没有新帧，线程也能定期醒来检查 _running 状态
            self._frame_ready.wait(timeout=0.1)
            
            if not self._running:
                break
            
            # 取出最新帧，并清空缓冲区
            frame: Optional[np.ndarray] = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None
                    self._frame_ready.clear() # 重置信号
            
            if frame is None:
                continue
            
            # 执行耗时推理（释放 GIL，不阻塞主线程）
            try:
                hands = self._detector.detect(frame)
            except Exception as e:
                print(f"[AsyncDetector] Inference error: {e}")
                hands = []
            
            # 更新结果
            with self._result_lock:
                self._latest_result = hands
                self._result_timestamp = time.time()


class SyncAsyncHandDetector:
    """
    通用手部检测器接口 (Facade Pattern)
    
    对外提供统一的 API，内部可配置为同步或异步模式。
    
    设计目的：
    1. 开发调试时使用同步模式 (Async=False)，便于定位错误。
    2. 生产运行时使用异步模式 (Async=True)，获得最佳性能。
    """
    
    # [Type Hints]
    async_mode: bool
    _infer_width: int
    _infer_height: int
    _async_detector: Optional[AsyncHandDetector]
    _sync_detector: Optional[HandDetector]
    
    def __init__(
        self,
        async_mode: bool = True,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        infer_width: int = 640,
        infer_height: int = 360,
    ) -> None:
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
        """启动检测器（仅异步模式有效）。"""
        if self._async_detector:
            self._async_detector.start()
    
    def stop(self) -> None:
        """停止检测器并释放资源。"""
        if self._async_detector:
            self._async_detector.stop()
        if self._sync_detector:
            self._sync_detector.close()
    
    def _scale_hands(self, hands: List[Hand], target_w: int, target_h: int) -> List[Hand]:
        """
        坐标重映射。
        
        因为推理是在低分辨率 (infer_width, infer_height) 下进行的，
        结果需要映射回原始画面分辨率 (target_w, target_h)。
        
        Args:
            hands (List[Hand]): 推理结果。
            target_w (int): 目标宽度。
            target_h (int): 目标高度。

        Returns:
            List[Hand]: 坐标已缩放的手部列表。
        """
        if not hands:
            return []
        
        scaled_hands: List[Hand] = []
        for hand in hands:
            # 使用归一化坐标重新计算像素坐标
            new_landmarks: List[PixelPoint] = [
                (int(lm[0] * target_w), int(lm[1] * target_h)) 
                for lm in hand.landmarks_norm
            ]
            
            # 重新计算包围盒
            xs = [p[0] for p in new_landmarks]
            ys = [p[1] for p in new_landmarks]
            if xs and ys:
                new_bbox: BoundingBox = (min(xs), min(ys), max(xs), max(ys))
            else:
                new_bbox = (0, 0, 0, 0)
            
            scaled_hands.append(Hand(
                landmarks=new_landmarks,
                landmarks_norm=hand.landmarks_norm, # 归一化坐标保持不变
                bbox=new_bbox,
                handedness=hand.handedness,
                confidence=hand.confidence
            ))
        return scaled_hands

    def detect(self, frame: np.ndarray) -> List[Hand]:
        """
        执行手部检测。
        
        Async Mode: 提交当前帧，并立即返回上一次推理的最新结果（非阻塞）。
        Sync Mode: 阻塞直到当前帧推理完成。

        Args:
            frame (np.ndarray): 输入图像。

        Returns:
            List[Hand]: 手部检测结果列表。
        """
        h, w = frame.shape[:2]
        hands: List[Hand] = []

        if self._async_detector:
            self._async_detector.submit_frame(frame)
            hands = self._async_detector.get_result()
        elif self._sync_detector:
            # 同步模式：手动缩放后检测
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
        绘制手部关键点（代理方法）。
        """
        if self._async_detector:
            self._async_detector._detector.draw_hand(frame, hand, simple)
        elif self._sync_detector:
            self._sync_detector.draw_hand(frame, hand, simple)