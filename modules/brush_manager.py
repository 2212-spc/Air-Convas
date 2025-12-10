from typing import Tuple, List
import cv2
import numpy as np


class BrushManager:
    """笔刷管理器 - 管理颜色、粗细、笔刷类型"""

    # 预设颜色（BGR格式）
    COLORS = {
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "orange": (0, 165, 255),
    }

    COLOR_NAMES = list(COLORS.keys())

    # 预设粗细
    THICKNESSES = [2, 3, 5, 8, 12]

    # 笔刷类型
    BRUSH_TYPES = ["solid", "dashed", "glow", "marker"]

    def __init__(self):
        self.current_color_index = 0  # yellow
        self.current_thickness_index = 1  # 3
        self.current_brush_type_index = 0  # solid

    @property
    def color(self) -> Tuple[int, int, int]:
        """当前颜色"""
        color_name = self.COLOR_NAMES[self.current_color_index]
        return self.COLORS[color_name]

    @property
    def color_name(self) -> str:
        """当前颜色名称"""
        return self.COLOR_NAMES[self.current_color_index]

    @property
    def thickness(self) -> int:
        """当前粗细"""
        return self.THICKNESSES[self.current_thickness_index]

    @property
    def brush_type(self) -> str:
        """当前笔刷类型"""
        return self.BRUSH_TYPES[self.current_brush_type_index]

    def next_color(self):
        """切换到下一个颜色"""
        self.current_color_index = (self.current_color_index + 1) % len(self.COLOR_NAMES)

    def prev_color(self):
        """切换到上一个颜色"""
        self.current_color_index = (self.current_color_index - 1) % len(self.COLOR_NAMES)

    def next_thickness(self):
        """增加粗细"""
        self.current_thickness_index = (self.current_thickness_index + 1) % len(self.THICKNESSES)

    def prev_thickness(self):
        """减少粗细"""
        self.current_thickness_index = (self.current_thickness_index - 1) % len(self.THICKNESSES)

    def next_brush_type(self):
        """切换笔刷类型"""
        self.current_brush_type_index = (self.current_brush_type_index + 1) % len(self.BRUSH_TYPES)

    def draw_line(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int]
    ):
        """根据当前笔刷类型绘制线条"""
        if self.brush_type == "solid":
            # 实线
            cv2.line(canvas, pt1, pt2, self.color, self.thickness, lineType=cv2.LINE_AA)

        elif self.brush_type == "dashed":
            # 虚线效果
            self._draw_dashed_line(canvas, pt1, pt2)

        elif self.brush_type == "glow":
            # 发光效果
            self._draw_glow_line(canvas, pt1, pt2)

        elif self.brush_type == "marker":
            # 马克笔效果（半透明）
            self._draw_marker_line(canvas, pt1, pt2)

    def _draw_dashed_line(self, canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """绘制虚线"""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 1:
            return

        # 虚线参数
        dash_length = 10
        gap_length = 5
        total_length = dash_length + gap_length

        num_dashes = int(distance / total_length)

        for i in range(num_dashes):
            t1 = i * total_length / distance
            t2 = (i * total_length + dash_length) / distance
            p1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
            p2 = (int(x1 + t2 * dx), int(y1 + t2 * dy))
            cv2.line(canvas, p1, p2, self.color, self.thickness, lineType=cv2.LINE_AA)

    def _draw_glow_line(self, canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """绘制发光线条"""
        # 外层光晕
        cv2.line(canvas, pt1, pt2, self.color, self.thickness + 4, lineType=cv2.LINE_AA)
        # 中层
        cv2.line(canvas, pt1, pt2, self.color, self.thickness + 2, lineType=cv2.LINE_AA)
        # 内层亮核
        lighter_color = tuple(min(255, int(c * 1.3)) for c in self.color)
        cv2.line(canvas, pt1, pt2, lighter_color, self.thickness, lineType=cv2.LINE_AA)

    def _draw_marker_line(self, canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """绘制马克笔效果（使用叠加模拟半透明）"""
        # 马克笔效果：粗一点，颜色稍淡
        marker_thickness = self.thickness + 2
        cv2.line(canvas, pt1, pt2, self.color, marker_thickness, lineType=cv2.LINE_AA)

    def get_status_text(self) -> str:
        """获取状态文本"""
        return f"Brush: {self.color_name} | Size: {self.thickness} | Type: {self.brush_type}"
