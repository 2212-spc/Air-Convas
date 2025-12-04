from typing import Optional, Sequence, Tuple

import numpy as np


class EmaSmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def push(self, value: Sequence[float]) -> Tuple[float, float]:
        arr = np.array(value, dtype=np.float32)
        if self._state is None:
            self._state = arr
        else:
            self._state = self.alpha * self._state + (1 - self.alpha) * arr
        return float(self._state[0]), float(self._state[1])
