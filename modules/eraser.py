# -*- coding: utf-8 -*-
"""橡皮擦模块"""

from typing import Tuple

from modules.canvas import Canvas


class Eraser:
    def __init__(self, canvas: Canvas, size: int) -> None:
        self.canvas = canvas
        self.size = size

    def erase(self, point: Tuple[int, int]) -> None:
        self.canvas.erase(point, self.size)
