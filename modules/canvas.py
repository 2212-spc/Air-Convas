import cv2
import numpy as np


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def draw_line(self, pt1, pt2, color, thickness: int) -> None:
        cv2.line(self._canvas, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def erase(self, center, radius: int) -> None:
        cv2.circle(self._canvas, center, radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    def clear(self) -> None:
        self._canvas[:] = 0

    def get_canvas(self) -> np.ndarray:
        return self._canvas

    def save(self, filename: str) -> None:
        cv2.imwrite(filename, self._canvas)
