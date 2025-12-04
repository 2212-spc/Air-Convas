from typing import Optional, Tuple

import numpy as np


class CoordinateMapper:
    def __init__(
        self,
        screen_size: Tuple[int, int],
        active_region: Tuple[float, float, float, float],
        smoothing_factor: float = 0.4,
    ) -> None:
        self.screen_w, self.screen_h = screen_size
        self.x1, self.y1, self.x2, self.y2 = active_region
        self.smoothing_factor = smoothing_factor
        self._smoothed: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._smoothed = None

    def map(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """Map normalized camera coords (0-1) to screen pixels with smoothing."""
        x, y = norm_point
        x = min(max(x, self.x1), self.x2)
        y = min(max(y, self.y1), self.y2)

        # Normalize within the active region
        xr = (x - self.x1) / max(self.x2 - self.x1, 1e-6)
        yr = (y - self.y1) / max(self.y2 - self.y1, 1e-6)

        target = np.array(
            [xr * self.screen_w, yr * self.screen_h],
            dtype=np.float32,
        )

        if self._smoothed is None:
            self._smoothed = target
        else:
            alpha = self.smoothing_factor
            self._smoothed = alpha * self._smoothed + (1 - alpha) * target

        return int(self._smoothed[0]), int(self._smoothed[1])
