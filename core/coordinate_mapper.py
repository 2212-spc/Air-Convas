# -*- coding: utf-8 -*-
"""坐标映射模块 - 将归一化坐标映射到屏幕坐标并进行平滑处理"""

from typing import Optional, Tuple, Literal, List

import numpy as np

from utils.smoothing import OneEuroFilter, AdaptiveEmaFilter


class CoordinateMapper:
    """
    坐标映射器 - 将归一化坐标映射到屏幕坐标并进行平滑处理
    """
    
    # [Type Hints] 显式声明类属性类型，提升代码健壮性
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
        """重置平滑状态"""
        self._smoothed = None
        self._one_euro.reset()
        self._adaptive.reset()

    def map(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """Map normalized camera coords (0-1) to screen pixels with smoothing."""
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