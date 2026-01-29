# -*- coding: utf-8 -*-
"""笔刷管理器 - 负责绘图状态管理与笔触渲染实现"""

import time
import colorsys
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np


class BrushManager:
    """
    笔刷管理器 (BrushManager)
    
    采用单例模式思想，集中管理绘图系统的全局状态，包括：
    1. 当前颜色 (Color)
    2. 当前粗细 (Thickness)
    3. 笔刷特效 (Solid, Dashed, Glow, etc.)
    4. 当前工具 (Pen, Eraser, Laser)
    
    并提供了不同笔刷效果的底层渲染实现。

    Attributes:
        COLORS (Dict): 预设颜色调色板。
        THICKNESSES (List): 预设粗细等级。
        BRUSH_TYPES (List): 支持的笔刷类型列表。
        TOOLS (List): 支持的工具模式列表。
    """

    # [Type Hints] 类常量定义
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "orange": (0, 165, 255),
    }

    COLOR_NAMES: List[str] = list(COLORS.keys())
    THICKNESSES: List[int] = [2, 4, 6, 10, 15]
    BRUSH_TYPES: List[str] = ["solid", "dashed", "glow", "marker", "rainbow"]
    TOOLS: List[str] = ["pen", "eraser", "laser"]

    # 实例属性声明
    current_color_index: int
    current_thickness_index: int
    current_brush_type_index: int
    current_tool_index: int
    dash_phase: float
    dash_length: int
    gap_length: int
    _dashed_debug_printed: bool

    def __init__(self) -> None:
        """初始化笔刷管理器，设置默认状态。"""
        self.current_color_index = 0  # 默认黄色
        self.current_thickness_index = 2  # 默认中等粗细 (index 2)
        self.current_brush_type_index = 0  # 默认实线
        self.current_tool_index = 0  # 默认画笔
        
        # 虚线相位参数
        self.dash_phase = 0.0
        self.dash_length = 35
        self.gap_length = 15

    @property
    def color(self) -> Tuple[int, int, int]:
        """
        获取当前绘画颜色 (BGR 格式)。
        
        如果是 'rainbow' 笔刷，则根据时间戳生成动态彩虹色。
        """
        if self.brush_type == "rainbow":
            t = time.time() * 2.0
            rgb = colorsys.hsv_to_rgb(t % 1.0, 1.0, 1.0)
            return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
        color_name = self.COLOR_NAMES[self.current_color_index]
        return self.COLORS[color_name]

    @property
    def color_name(self) -> str:
        """获取当前颜色名称（用于 UI 显示）。"""
        if self.brush_type == "rainbow":
            return "Rainbow"
        return self.COLOR_NAMES[self.current_color_index]

    @property
    def thickness(self) -> int:
        """获取当前画笔粗细（像素）。"""
        return self.THICKNESSES[self.current_thickness_index]

    @property
    def brush_type(self) -> str:
        """获取当前笔刷特效类型。"""
        return self.BRUSH_TYPES[self.current_brush_type_index]
    
    @property
    def tool(self) -> str:
        """获取当前活动工具名称。"""
        return self.TOOLS[self.current_tool_index]

    def next_color(self) -> None:
        """循环切换到下一个颜色。"""
        self.current_color_index = (self.current_color_index + 1) % len(self.COLOR_NAMES)

    def prev_color(self) -> None:
        """循环切换到上一个颜色。"""
        self.current_color_index = (self.current_color_index - 1) % len(self.COLOR_NAMES)

    def next_thickness(self) -> None:
        """增加画笔粗细。"""
        self.current_thickness_index = (self.current_thickness_index + 1) % len(self.THICKNESSES)

    def prev_thickness(self) -> None:
        """减少画笔粗细。"""
        self.current_thickness_index = (self.current_thickness_index - 1) % len(self.THICKNESSES)

    def next_brush_type(self) -> None:
        """循环切换笔刷特效。"""
        self.current_brush_type_index = (self.current_brush_type_index + 1) % len(self.BRUSH_TYPES)
        
    def next_tool(self) -> None:
        """循环切换工具 (Pen -> Eraser -> Laser)。"""
        self.current_tool_index = (self.current_tool_index + 1) % len(self.TOOLS)

    def draw_line(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        thickness: int = None
    ) -> None:
        """
        在画布上绘制线条，自动应用当前的颜色和笔刷特效。

        Args:
            canvas (np.ndarray): OpenCV 图像画布。
            pt1 (Tuple[int, int]): 起点坐标。
            pt2 (Tuple[int, int]): 终点坐标。
            thickness (int, optional): 指定粗细。如果为 None，则使用管理器当前的 thickness。
        """
        current_color = self.color
        use_thickness = thickness if thickness is not None else self.thickness

        if self.brush_type == "solid":
            cv2.line(canvas, pt1, pt2, current_color, use_thickness, lineType=cv2.LINE_AA)

        elif self.brush_type == "dashed":
            self._draw_dashed_line(canvas, pt1, pt2, current_color, use_thickness)

        elif self.brush_type == "glow":
            self._draw_glow_line(canvas, pt1, pt2, current_color, use_thickness)

        elif self.brush_type == "marker":
            self._draw_marker_line(canvas, pt1, pt2, current_color, use_thickness)
            
        elif self.brush_type == "rainbow":
            cv2.line(canvas, pt1, pt2, current_color, use_thickness, lineType=cv2.LINE_AA)

    def reset_dash_phase(self) -> None:
        """
        重置虚线相位。
        
        应当在 `start_stroke` (开始新的一笔) 时调用，确保虚线从实线段开始绘制。
        """
        self.dash_phase = 0.0
    
    def _draw_dashed_line(
        self, 
        canvas: np.ndarray, 
        pt1: Tuple[int, int], 
        pt2: Tuple[int, int], 
        color: Tuple[int, int, int], 
        thickness: int
    ) -> None:
        """
        绘制连续虚线算法。
        
        通过维护 `self.dash_phase` 状态，确保在多帧连续绘制线段时，
        虚线的样式（实线-空隙）能够连贯，不会因为线段切分而闪烁。
        """
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.5:
            return

        dashed_thickness = thickness + max(2, int(thickness * 0.3))
        period = self.dash_length + self.gap_length
        
        if not hasattr(self, '_dashed_debug_printed'):
            print(f"[虚线] 实线={self.dash_length}px, 空隙={self.gap_length}px")
            self._dashed_debug_printed = True

        current_distance = 0.0
        
        while current_distance < distance:
            phase_in_period = self.dash_phase % period
            
            if phase_in_period < self.dash_length:
                remaining_dash = self.dash_length - phase_in_period
                segment_length = min(remaining_dash, distance - current_distance)
                
                t1 = current_distance / distance
                t2 = (current_distance + segment_length) / distance
                p1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
                p2 = (int(x1 + t2 * dx), int(y1 + t2 * dy))
                
                if segment_length > 0.5:
                    cv2.line(canvas, p1, p2, color, dashed_thickness, lineType=cv2.LINE_AA)
                
                current_distance += segment_length
                self.dash_phase += segment_length
            else:
                remaining_gap = period - phase_in_period
                segment_length = min(remaining_gap, distance - current_distance)
                
                current_distance += segment_length
                self.dash_phase += segment_length

    def _draw_glow_line(
        self, 
        canvas: np.ndarray, 
        pt1: Tuple[int, int], 
        pt2: Tuple[int, int], 
        color: Tuple[int, int, int], 
        thickness: int
    ) -> None:
        """绘制霓虹发光效果（多层叠加）。"""
        outer_color = tuple(int(c * 0.3) for c in color)
        cv2.line(canvas, pt1, pt2, outer_color, thickness + 8, lineType=cv2.LINE_AA)
        
        mid_color = tuple(int(c * 0.6) for c in color)
        cv2.line(canvas, pt1, pt2, mid_color, thickness + 4, lineType=cv2.LINE_AA)
        
        cv2.line(canvas, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        
        highlight = tuple(min(255, int(c * 1.5)) for c in color)
        cv2.line(canvas, pt1, pt2, highlight, max(1, thickness // 3), lineType=cv2.LINE_AA)

    def _draw_marker_line(
        self, 
        canvas: np.ndarray, 
        pt1: Tuple[int, int], 
        pt2: Tuple[int, int], 
        color: Tuple[int, int, int], 
        thickness: int
    ) -> None:
        """绘制马克笔效果（模拟半透明叠加）。"""
        marker_color = tuple(int(c * 0.7) for c in color)
        marker_thickness = thickness + 4
        cv2.line(canvas, pt1, pt2, marker_color, marker_thickness, lineType=cv2.LINE_AA)

    def get_status_text(self) -> str:
        """返回状态栏调试文本。"""
        return f"Tool: {self.tool.upper()} | Brush: {self.color_name} | Size: {self.thickness} | Type: {self.brush_type}"