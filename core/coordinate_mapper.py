# -*- coding: utf-8 -*-
"""坐标映射模块 - 处理坐标转换与平滑算法"""

from typing import Optional, Tuple, Literal, List

import numpy as np

from utils.smoothing import OneEuroFilter, AdaptiveEmaFilter


class CoordinateMapper:
    """
    坐标映射器 (CoordinateMapper)
    
    负责将摄像头捕捉到的归一化坐标 (0.0~1.0) 映射到屏幕像素坐标，
    并应用多种平滑算法以消除抖动，实现平滑跟手效果。

    Attributes:
        screen_w (int): 目标屏幕宽度（像素）。
        screen_h (int): 目标屏幕高度（像素）。
        x1, y1, x2, y2 (float): 活动区域的归一化边界坐标。
        smoothing_factor (float): EMA 平滑因子。
        smoothing_mode (str): 当前使用的平滑模式 ('ema', 'one_euro', 'adaptive')。
    """
    
    # [Type Hints] 显式声明类属性类型
    screen_w: int
    screen_h: int
    x1: float
    y1: float
    x2: float
    y2: float
    smoothing_factor: float
    smoothing_mode: str
    
    # 滤波器状态类型定义
    _smoothed: Optional[np.ndarray]
    _one_euro: OneEuroFilter
    _adaptive: AdaptiveEmaFilter

    def __init__(
        self,
        screen_size: Tuple[int, int],
        active_region: Tuple[float, float, float, float],
        smoothing_factor: float = 0.4,
        smoothing_mode: Literal['ema', 'one_euro', 'adaptive'] = 'one_euro',
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.007,
    ) -> None:
        """
        初始化坐标映射器。

        Args:
            screen_size (Tuple[int, int]): 屏幕分辨率 (width, height)。
            active_region (Tuple[float, float, float, float]): 手部活动区域 (x1, y1, x2, y2)。
            smoothing_factor (float, optional): EMA 平滑系数，仅在 mode='ema' 时有效。默认为 0.4。
            smoothing_mode (Literal, optional): 平滑算法模式。默认为 'one_euro'。
            one_euro_min_cutoff (float, optional): 1€ Filter 的最小截止频率。默认为 1.0。
            one_euro_beta (float, optional): 1€ Filter 的速度系数。默认为 0.007。
        """
        self.screen_w, self.screen_h = screen_size
        self.x1, self.y1, self.x2, self.y2 = active_region
        self.smoothing_factor = smoothing_factor
        self.smoothing_mode = smoothing_mode
        
        # EMA 状态初始化
        self._smoothed = None
        
        # 1€ Filter 初始化
        self._one_euro = OneEuroFilter(
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
        )
        
        # 自适应 EMA 初始化
        self._adaptive = AdaptiveEmaFilter(
            min_alpha=0.3,
            max_alpha=0.9,
            speed_threshold=50.0,
        )

    def reset(self) -> None:
        """
        重置所有滤波器的内部状态。
        
        当手部丢失或重新检测到手部时应当调用此方法，防止坐标跳变。
        """
        self._smoothed = None
        self._one_euro.reset()
        self._adaptive.reset()

    def map(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """
        将归一化坐标映射到屏幕坐标，并应用平滑滤波。

        Args:
            norm_point (Tuple[float, float]): 归一化的 (x, y) 坐标，范围通常在 0.0 到 1.0 之间。

        Returns:
            Tuple[int, int]: 映射并平滑后的屏幕像素坐标 (x, y)。
        """
        x: float
        y: float
        x, y = norm_point
        
        # 限制坐标范围
        x = min(max(x, self.x1), self.x2)
        y = min(max(y, self.y1), self.y2)

        # 在活动区域内归一化
        xr: float = (x - self.x1) / max(self.x2 - self.x1, 1e-6)
        yr: float = (y - self.y1) / max(self.y2 - self.y1, 1e-6)

        target: Tuple[float, float] = (xr * self.screen_w, yr * self.screen_h)

        smoothed_x: float = 0.0
        smoothed_y: float = 0.0

        # 根据平滑模式选择滤波器
        if self.smoothing_mode == 'one_euro':
            smoothed_x, smoothed_y = self._one_euro.push(target)
        elif self.smoothing_mode == 'adaptive':
            smoothed_x, smoothed_y = self._adaptive.push(target)
        else:  # 'ema' 或其他
            target_arr = np.array(target, dtype=np.float32)
            if self._smoothed is None:
                self._smoothed = target_arr
            else:
                alpha = self.smoothing_factor
                self._smoothed = alpha * self._smoothed + (1 - alpha) * target_arr
            smoothed_x = float(self._smoothed[0])
            smoothed_y = float(self._smoothed[1])

        return int(smoothed_x), int(smoothed_y)