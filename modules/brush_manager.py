"""
笔刷管理器 - 优化颜色板和笔刷效果
[优化说明]
1. 替换为更现代、更舒适的Air Palette配色方案
2. 颜色不再是纯饱和色，增加了视觉舒适度
3. 优化了特殊笔刷的绘制逻辑
"""
from typing import Tuple, List
import cv2
import numpy as np


class BrushManager:
    """笔刷管理器 - 管理颜色、粗细、笔刷类型"""

    # === 优化的配色方案 (BGR格式) ===
    # 相比原版纯色，这套颜色亮度更高，且避免了刺眼的高饱和度，显示效果更高级
    COLORS = {
        "cyan": (255, 255, 0),       # 青色 - 极佳的深色背景高亮色
        "magenta": (255, 0, 255),    # 品红 - 鲜艳醒目
        "lime": (50, 255, 50),       # 荧光绿 - 替代普通绿
        "sun-yellow": (0, 215, 255), # 太阳黄 - 替代刺眼的纯黄
        "orange": (0, 140, 255),     # 鲜橙色
        "hot-pink": (180, 105, 255), # 热粉色
        "white": (240, 240, 240),    # 柔和白
        "sky-blue": (255, 200, 100), # 天蓝色
    }

    COLOR_NAMES = list(COLORS.keys())

    # 预设粗细
    THICKNESSES = [2, 4, 6, 8, 12, 16]

    # 笔刷类型
    BRUSH_TYPES = ["solid", "dashed", "glow", "marker"]

    def __init__(self):
        self.current_color_index = 0
        # 默认从第3个粗细开始 (即6)，更适合演示
        self.current_thickness_index = 2
        self.current_brush_type_index = 0

    @property
    def color(self) -> Tuple[int, int, int]:
        color_name = self.COLOR_NAMES[self.current_color_index]
        return self.COLORS[color_name]

    @property
    def color_name(self) -> str:
        return self.COLOR_NAMES[self.current_color_index]

    @property
    def thickness(self) -> int:
        return self.THICKNESSES[self.current_thickness_index]

    @property
    def brush_type(self) -> str:
        return self.BRUSH_TYPES[self.current_brush_type_index]

    def next_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.COLOR_NAMES)

    def prev_color(self):
        self.current_color_index = (self.current_color_index - 1) % len(self.COLOR_NAMES)

    def next_thickness(self):
        self.current_thickness_index = (self.current_thickness_index + 1) % len(self.THICKNESSES)

    def prev_thickness(self):
        self.current_thickness_index = (self.current_thickness_index - 1) % len(self.THICKNESSES)

    def next_brush_type(self):
        self.current_brush_type_index = (self.current_brush_type_index + 1) % len(self.BRUSH_TYPES)

    def draw_line(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int]
    ):
        """根据当前笔刷类型在画布上绘制线条"""
        # 确保坐标为整数
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        
        if self.brush_type == "solid":
            cv2.line(canvas, pt1, pt2, self.color, self.thickness, lineType=cv2.LINE_AA)

        elif self.brush_type == "dashed":
            self._draw_dashed_line(canvas, pt1, pt2)

        elif self.brush_type == "glow":
            self._draw_glow_line(canvas, pt1, pt2)

        elif self.brush_type == "marker":
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

        # 动态调整虚线的线段长度和间隙
        dash_length = max(5, self.thickness * 1.5)
        gap_length = max(5, self.thickness * 1.5)
        total_length = dash_length + gap_length
        
        # 计算需要画多少段
        num_dashes = int(distance / total_length) + 1

        for i in range(num_dashes):
            start_dist = i * total_length
            end_dist = start_dist + dash_length
            
            if start_dist >= distance: break

            start_ratio = start_dist / distance
            end_ratio = min(end_dist / distance, 1.0)
            
            p1 = (int(x1 + start_ratio * dx), int(y1 + start_ratio * dy))
            p2 = (int(x1 + end_ratio * dx), int(y1 + end_ratio * dy))
            cv2.line(canvas, p1, p2, self.color, self.thickness, lineType=cv2.LINE_AA)

    def _draw_glow_line(self, canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """绘制发光效果线条"""
        # 1. 绘制宽而淡的外部光晕
        glow_color = tuple(int(c * 0.3) for c in self.color)
        cv2.line(canvas, pt1, pt2, glow_color, self.thickness * 4, lineType=cv2.LINE_AA)
        
        # 2. 绘制主色线条
        cv2.line(canvas, pt1, pt2, self.color, self.thickness, lineType=cv2.LINE_AA)
        
        # 3. 绘制中心高亮白线
        white_core = (255, 255, 255)
        core_thickness = max(1, self.thickness // 4)
        cv2.line(canvas, pt1, pt2, white_core, core_thickness, lineType=cv2.LINE_AA)

    def _draw_marker_line(self, canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """绘制马克笔效果（半透明叠加）"""
        # 创建一个临时图层来绘制半透明线
        overlay = canvas.copy()
        
        # 绘制一条很宽的线
        marker_thickness = self.thickness + 8
        cv2.line(overlay, pt1, pt2, self.color, marker_thickness, lineType=cv2.LINE_AA)
        
        # 将临时图层以较低的不透明度叠加到画布上 (alpha=0.3)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    def get_status_text(self) -> str:
        """获取状态栏显示的文本"""
        # 将颜色名称的首字母大写
        color_display = self.color_name.replace('-', ' ').title()
        return f"Brush: {color_display} | Size: {self.thickness} | Type: {self.brush_type.title()}"