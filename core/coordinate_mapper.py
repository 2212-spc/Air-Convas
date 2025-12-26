from typing import Optional, Tuple
import time

import numpy as np

from utils.one_euro_filter import OneEuroFilter2D, OneEuroPresets


class CoordinateMapper:
    def __init__(
        self,
        screen_size: Tuple[int, int],
        active_region: Tuple[float, float, float, float],
        smoothing_factor: float = 0.4,
        use_one_euro: bool = True,  # 默认使用 One Euro Filter
        one_euro_preset: str = "DRAWING",  # 预设：DRAWING, CURSOR, LASER, PRECISION
    ) -> None:
        self.screen_w, self.screen_h = screen_size
        self.x1, self.y1, self.x2, self.y2 = active_region
        self.smoothing_factor = smoothing_factor
        self.use_one_euro = use_one_euro
        self._preset_name = one_euro_preset  # 保存预设名称用于显示
        
        # EMA 平滑状态（备用方案）
        self._smoothed: Optional[np.ndarray] = None
        
        # One Euro Filter（推荐方案）
        if use_one_euro:
            preset = getattr(OneEuroPresets, one_euro_preset, OneEuroPresets.DRAWING)
            self.one_euro = OneEuroFilter2D(**preset)
        else:
            self.one_euro = None
    
    def set_one_euro_preset(self, preset_name: str) -> None:
        """切换 One Euro Filter 预设"""
        if self.use_one_euro:
            preset = getattr(OneEuroPresets, preset_name, OneEuroPresets.DRAWING)
            self.one_euro = OneEuroFilter2D(**preset)
            self._preset_name = preset_name
            print(f"Switched to One Euro preset: {preset_name}")

    def reset(self) -> None:
        """重置平滑状态"""
        self._smoothed = None
        if self.one_euro:
            self.one_euro.reset()

    def map(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """Map normalized camera coords (0-1) to screen pixels with smoothing."""
        x, y = norm_point
        x = min(max(x, self.x1), self.x2)
        y = min(max(y, self.y1), self.y2)

        # Normalize within the active region
        xr = (x - self.x1) / max(self.x2 - self.x1, 1e-6)
        yr = (y - self.y1) / max(self.y2 - self.y1, 1e-6)

        # 映射到屏幕坐标
        target_x = xr * self.screen_w
        target_y = yr * self.screen_h

        # 应用平滑
        if self.use_one_euro and self.one_euro:
            # 使用 One Euro Filter（自适应平滑）
            timestamp = time.time()
            smoothed_x, smoothed_y = self.one_euro.filter(target_x, target_y, timestamp)
        else:
            # 使用传统 EMA（指数移动平均）
            target = np.array([target_x, target_y], dtype=np.float32)
            if self._smoothed is None:
                self._smoothed = target
            else:
                alpha = self.smoothing_factor
                self._smoothed = alpha * self._smoothed + (1 - alpha) * target
            smoothed_x, smoothed_y = self._smoothed[0], self._smoothed[1]

        return int(smoothed_x), int(smoothed_y)
