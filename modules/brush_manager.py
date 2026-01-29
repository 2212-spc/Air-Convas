# -*- coding: utf-8 -*-
"""笔刷管理器 - 管理颜色、粗细、笔刷类型（含彩虹笔）及工具选择"""

import time
import colorsys
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np


class BrushManager:
    """笔刷管理器 - 管理颜色、粗细、笔刷类型"""

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

    # 预设粗细 (加粗一点，保证可见性)
    THICKNESSES: List[int] = [2, 4, 6, 10, 15]

    # 笔刷类型
    BRUSH_TYPES: List[str] = ["solid", "dashed", "glow", "marker", "rainbow"]
    
    # 核心工具类型（互斥选择）
    TOOLS: List[str] = ["pen", "eraser", "laser"]

    # [Type Hints] 实例属性类型声明
    current_color_index: int
    current_thickness_index: int
    current_brush_type_index: int
    current_tool_index: int
    dash_phase: float
    dash_length: int
    gap_length: int
    _dashed_debug_printed: bool

    def __init__(self) -> None:
        self.current_color_index = 0  # yellow
        self.current_thickness_index = 2  # 默认 6px (index 2)
        self.current_brush_type_index = 0  # solid
        self.current_tool_index = 0  # pen (默认画笔)
        
        # 虚线相位追踪（用于连续虚线效果）
        self.dash_phase = 0.0  # 当前虚线相位
        self.dash_length = 35  # 实线段长度（0.7cm，70%）
        self.gap_length = 15   # 空隙长度（0.3cm，30%）

    @property
    def color(self) -> Tuple[int, int, int]:
        """当前颜色"""
        if self.brush_type == "rainbow":
            # 彩虹笔颜色随时间变化
            t = time.time() * 2.0  # 速度系数
            # HSV转BGR
            rgb = colorsys.hsv_to_rgb(t % 1.0, 1.0, 1.0)
            return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
        color_name = self.COLOR_NAMES[self.current_color_index]
        return self.COLORS[color_name]

    @property
    def color_name(self) -> str:
        """当前颜色名称"""
        if self.brush_type == "rainbow":
            return "Rainbow"
        return self.COLOR_NAMES[self.current_color_index]

    @property
    def thickness(self) -> int:
        """当前粗细"""
        return self.THICKNESSES[self.current_thickness_index]

    @property
    def brush_type(self) -> str:
        """当前笔刷类型"""
        return self.BRUSH_TYPES[self.current_brush_type_index]
    
    @property
    def tool(self) -> str:
        """当前选中工具"""
        return self.TOOLS[self.current_tool_index]

    def next_color(self) -> None:
        """切换到下一个颜色"""
        self.current_color_index = (self.current_color_index + 1) % len(self.COLOR_NAMES)

    def prev_color(self) -> None:
        """切换到上一个颜色"""
        self.current_color_index = (self.current_color_index - 1) % len(self.COLOR_NAMES)

    def next_thickness(self) -> None:
        """增加粗细"""
        self.current_thickness_index = (self.current_thickness_index + 1) % len(self.THICKNESSES)

    def prev_thickness(self) -> None:
        """减少粗细"""
        self.current_thickness_index = (self.current_thickness_index - 1) % len(self.THICKNESSES)

    def next_brush_type(self) -> None:
        """切换笔刷类型"""
        self.current_brush_type_index = (self.current_brush_type_index + 1) % len(self.BRUSH_TYPES)
        
    def next_tool(self) -> None:
        """切换工具"""
        self.current_tool_index = (self.current_tool_index + 1) % len(self.TOOLS)

    def draw_line(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        thickness: int = None
    ) -> None:
        """根据当前笔刷类型绘制线条"""
        current_color = self.color  # 获取当前动态颜色
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
        """重置虚线相位（开始新笔画时调用）"""
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
        绘制虚线 - 沿着路径连续绘制，使用相位累积
        像普通线段一样，但从中间一些地方断开
        以1组为单位，其中80%是实线，20%是空白
        """
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.5:
            return

        # 虚线使用更粗的线条
        dashed_thickness = thickness + max(2, int(thickness * 0.3))
        
        # 虚线周期参数
        period = self.dash_length + self.gap_length  # 完整周期长度
        
        # 调试信息（只在第一次绘制时打印）
        if not hasattr(self, '_dashed_debug_printed'):
            print(f"[虚线] 实线={self.dash_length}px(0.7cm, 70%), 空隙={self.gap_length}px(0.3cm, 30%), 粗细={thickness}→{dashed_thickness}")
            self._dashed_debug_printed = True

        # 沿着 pt1 -> pt2 绘制虚线，考虑当前相位
        current_distance = 0.0
        
        while current_distance < distance:
            # 当前相位在周期中的位置
            phase_in_period = self.dash_phase % period
            
            # 判断当前位置是实线还是空隙
            if phase_in_period < self.dash_length:
                # 在实线段内
                remaining_dash = self.dash_length - phase_in_period
                segment_length = min(remaining_dash, distance - current_distance)
                
                # 计算线段的起始和结束点
                t1 = current_distance / distance
                t2 = (current_distance + segment_length) / distance
                p1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
                p2 = (int(x1 + t2 * dx), int(y1 + t2 * dy))
                
                # 绘制实线段
                if segment_length > 0.5:
                    cv2.line(canvas, p1, p2, color, dashed_thickness, lineType=cv2.LINE_AA)
                
                current_distance += segment_length
                self.dash_phase += segment_length
            else:
                # 在空隙段内，跳过
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
        """绘制发光线条 - 改进光晕效果"""
        # 宽大的外层光晕（暗淡）
        outer_color = tuple(int(c * 0.3) for c in color)
        cv2.line(canvas, pt1, pt2, outer_color, thickness + 8, lineType=cv2.LINE_AA)
        
        # 中层光晕
        mid_color = tuple(int(c * 0.6) for c in color)
        cv2.line(canvas, pt1, pt2, mid_color, thickness + 4, lineType=cv2.LINE_AA)
        
        # 内层亮核
        cv2.line(canvas, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        
        # 中心高光（接近白色）
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
        """绘制马克笔效果 - 降低不透明度模拟"""
        # 马克笔效果：更粗，但只绘制一层，通过主循环的 addWeighted 实现半透明叠加
        # 这里我们降低颜色的亮度来模拟半透明
        marker_color = tuple(int(c * 0.7) for c in color)
        marker_thickness = thickness + 4
        cv2.line(canvas, pt1, pt2, marker_color, marker_thickness, lineType=cv2.LINE_AA)

    def get_status_text(self) -> str:
        """获取状态文本"""
        return f"Tool: {self.tool.upper()} | Brush: {self.color_name} | Size: {self.thickness} | Type: {self.brush_type}"