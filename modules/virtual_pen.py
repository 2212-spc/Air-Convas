from typing import Optional, Tuple
import math

from modules.canvas import Canvas
from modules.brush_manager import BrushManager
from utils.smoothing import EmaSmoother


class VirtualPen:
    def __init__(
        self,
        canvas: Canvas,
        brush_manager: BrushManager,
        smoothing: Optional[EmaSmoother] = None,
        jump_threshold: int = 80,  # 位置跳变阈值（像素），超过此值自动断笔
    ) -> None:
        self.canvas = canvas
        self.brush_manager = brush_manager
        self.smoothing = smoothing
        self.jump_threshold = jump_threshold
        self.prev_point: Optional[Tuple[int, int]] = None
        self.points: list[Tuple[int, int]] = []
        self._stroke_broken = False  # 标记笔画是否被跳变中断

    def start_stroke(self) -> None:
        self.prev_point = None
        if self.smoothing:
            self.smoothing.reset()
        self.points = []
        self._stroke_broken = False

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """计算两点距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def draw(self, point: Tuple[int, int]) -> Tuple[int, int]:
        if self.smoothing:
            point = tuple(map(int, self.smoothing.push(point)))

        # 位置跳变检测：如果移动距离超过阈值，自动断笔
        if self.prev_point is not None:
            dist = self._distance(self.prev_point, point)
            if dist > self.jump_threshold:
                # 距离太大，认为是新笔画开始，不连接
                self._stroke_broken = True
                self.prev_point = point
                self.points.append(point)
                return point

        if self.prev_point is not None:
            # 使用BrushManager绘制
            self.brush_manager.draw_line(self.canvas.get_canvas(), self.prev_point, point)
        self.prev_point = point
        self.points.append(point)
        return point

    def end_stroke(self) -> list[Tuple[int, int]]:
        finished_points = self.points
        self.start_stroke()
        return finished_points

    @property
    def was_stroke_broken(self) -> bool:
        """返回上一次绘制是否因跳变而中断"""
        return self._stroke_broken
