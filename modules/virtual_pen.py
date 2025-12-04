from typing import Optional, Tuple

from modules.canvas import Canvas
from utils.smoothing import EmaSmoother


class VirtualPen:
    def __init__(
        self,
        canvas: Canvas,
        color: Tuple[int, int, int],
        thickness: int,
        smoothing: Optional[EmaSmoother] = None,
    ) -> None:
        self.canvas = canvas
        self.color = color
        self.thickness = thickness
        self.prev_point: Optional[Tuple[int, int]] = None
        self.smoothing = smoothing
        self.points: list[Tuple[int, int]] = []

    def start_stroke(self) -> None:
        self.prev_point = None
        if self.smoothing:
            self.smoothing.reset()
        self.points = []

    def draw(self, point: Tuple[int, int]) -> Tuple[int, int]:
        if self.smoothing:
            point = tuple(map(int, self.smoothing.push(point)))
        if self.prev_point is not None:
            self.canvas.draw_line(self.prev_point, point, self.color, self.thickness)
        self.prev_point = point
        self.points.append(point)
        return point

    def end_stroke(self) -> list[Tuple[int, int]]:
        finished_points = self.points
        self.start_stroke()
        return finished_points
