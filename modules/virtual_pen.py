# -*- coding: utf-8 -*-
"""虚拟画笔模块 - 实现基于速度感知的钢笔笔触与贝塞尔曲线平滑"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import math

from modules.canvas import Canvas
from modules.brush_manager import BrushManager
from utils.smoothing import EmaSmoother


def catmull_rom_spline(
    p0: Tuple[int, int], 
    p1: Tuple[int, int], 
    p2: Tuple[int, int], 
    p3: Tuple[int, int], 
    num_points: int = 8
) -> List[Tuple[int, int]]:
    """
    计算 Catmull-Rom 样条曲线上的插值点序列。
    
    Catmull-Rom 样条是一种穿过所有控制点的插值曲线。
    绘制的是 p1 到 p2 之间的曲线段，p0 和 p3 作为切向控制点。

    Args:
        p0 (Tuple[int, int]): 前一个控制点 (控制切线方向)。
        p1 (Tuple[int, int]): 起始点。
        p2 (Tuple[int, int]): 结束点。
        p3 (Tuple[int, int]): 后一个控制点 (控制切线方向)。
        num_points (int): 插值点数量，越高曲线越平滑。默认为 8。

    Returns:
        List[Tuple[int, int]]: 插值生成的点坐标列表。
    """
    result: List[Tuple[int, int]] = []
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        t2 = t * t
        t3 = t2 * t
        
        # Catmull-Rom 矩阵公式
        x = 0.5 * ((2 * p1[0]) +
                   (-p0[0] + p2[0]) * t +
                   (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                   (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        
        y = 0.5 * ((2 * p1[1]) +
                   (-p0[1] + p2[1]) * t +
                   (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                   (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        
        result.append((int(x), int(y)))
    
    return result


class VirtualPen:
    """
    虚拟钢笔 (VirtualPen)
    
    实现真实书写体验的核心类，具备两大特性：
    1. 速度感知 (Velocity Sensitivity): 移动速度越快，笔画越细（模拟钢笔轻压）；速度越慢，笔画越粗（模拟重压）。
    2. 曲线平滑 (Curve Smoothing): 使用 4 点滑动窗口和 Catmull-Rom 样条算法，将折线点平滑为连续曲线。

    Attributes:
        canvas (Canvas): 目标画布对象。
        brush_manager (BrushManager): 笔刷状态管理器。
        _window (List): 存储最近 4 个坐标点的滑动窗口，用于计算样条曲线。
    """
    
    # [Type Hints] 显式声明属性类型
    canvas: Canvas
    brush_manager: BrushManager
    smoothing: Optional[EmaSmoother]
    jump_threshold: int
    enable_bezier: bool
    bezier_segments: int
    enable_pen_effect: bool
    min_thickness_ratio: float
    max_thickness_ratio: float
    speed_threshold: float
    thickness_smoothing: float
    prev_point: Optional[Tuple[int, int]]
    points: List[Tuple[int, int]]
    _stroke_broken: bool
    _window: List[Tuple[int, int]]
    _last_curve_end: Optional[Tuple[int, int]]
    _current_thickness: float
    _last_speed: float

    def __init__(
        self,
        canvas: Canvas,
        brush_manager: BrushManager,
        smoothing: Optional[EmaSmoother] = None,
        jump_threshold: int = 80,
        enable_bezier: bool = True,
        bezier_segments: int = 8,
        enable_pen_effect: bool = True,
        min_thickness_ratio: float = 0.4,
        max_thickness_ratio: float = 1.2,
        speed_threshold: float = 30.0,
        thickness_smoothing: float = 0.3,
    ) -> None:
        """
        初始化虚拟画笔。

        Args:
            canvas (Canvas): 绘图画布。
            brush_manager (BrushManager): 笔刷管理器。
            smoothing (Optional[EmaSmoother]): 输入坐标的 EMA 平滑器。
            jump_threshold (int): 防跳变阈值。
            enable_bezier (bool): 是否启用贝塞尔/Catmull-Rom 平滑。
            enable_pen_effect (bool): 是否启用速度感知粗细效果。
            min_thickness_ratio (float): 快速移动时的最小粗细比例。
            max_thickness_ratio (float): 慢速移动时的最大粗细比例。
            speed_threshold (float): 达到最小粗细的参考速度。
            thickness_smoothing (float): 粗细变化的平滑系数 (0.0~1.0)。
        """
        self.canvas = canvas
        self.brush_manager = brush_manager
        self.smoothing = smoothing
        self.jump_threshold = jump_threshold
        self.enable_bezier = enable_bezier
        self.bezier_segments = bezier_segments
        
        # 钢笔效果
        self.enable_pen_effect = enable_pen_effect
        self.min_thickness_ratio = min_thickness_ratio
        self.max_thickness_ratio = max_thickness_ratio
        self.speed_threshold = speed_threshold
        self.thickness_smoothing = thickness_smoothing
        
        # 绘图状态
        self.prev_point = None
        self.points = []
        self._stroke_broken = False
        
        # 滑动窗口缓冲（最多4个点）
        self._window = []
        self._last_curve_end = None
        
        # 钢笔效果状态
        self._current_thickness = 1.0  # 当前粗细比例
        self._last_speed = 0.0

    def start_stroke(self) -> None:
        """开始新笔画。"""
        self.prev_point = None
        if self.smoothing:
            self.smoothing.reset()
        self.points = []
        self._stroke_broken = False
        self._window = []
        self._last_curve_end = None
        self._current_thickness = 1.0
        self._last_speed = 0.0
        # 重置虚线相位（如果是虚线笔刷）
        if self.brush_manager.brush_type == "dashed":
            self.brush_manager.reset_dash_phase()

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """计算两点欧几里得距离。"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _calculate_thickness(self, speed: float) -> int:
        """
        根据当前移动速度计算动态笔画粗细。
        
        算法逻辑：
        1. 将速度归一化到 (0, speed_threshold) 区间。
        2. 线性插值计算目标比例 (max_ratio -> min_ratio)。
        3. 使用指数平滑 (EMA) 平滑粗细变化，避免线条粗细突变。
        
        Args:
            speed (float): 当前两帧之间的移动距离（像素）。

        Returns:
            int: 计算后的像素粗细值。
        """
        base_thickness = self.brush_manager.thickness
        
        if not self.enable_pen_effect:
            return base_thickness
        
        # 速度归一化（0-1）
        speed_normalized = min(1.0, speed / self.speed_threshold)
        
        # 速度快时细，速度慢时粗
        # ratio 从 max_thickness_ratio 到 min_thickness_ratio
        ratio = self.max_thickness_ratio - speed_normalized * (self.max_thickness_ratio - self.min_thickness_ratio)
        
        # 平滑过渡
        self._current_thickness = (
            self.thickness_smoothing * ratio + 
            (1 - self.thickness_smoothing) * self._current_thickness
        )
        
        return max(1, int(base_thickness * self._current_thickness))

    def _draw_line_segment(self, pt1: Tuple[int, int], pt2: Tuple[int, int], thickness: int = None) -> None:
        """绘制原子线段，并自动处理笔刷特效和圆形端点连接。"""
        canvas_img = self.canvas.get_canvas()
        
        if thickness is None:
            thickness = self.brush_manager.thickness
        
        # 使用笔刷管理器绘制（支持虚线/发光/马克笔/彩虹）
        self.brush_manager.draw_line(canvas_img, pt1, pt2, thickness)
        
        # 绘制圆形端点，使线条更圆润（虚线不需要端点圆圈）
        if self.brush_manager.brush_type in ("solid", "rainbow"):
            color = self.brush_manager.color
            radius = max(1, thickness // 2)
            cv2.circle(canvas_img, pt2, radius, color, -1, lineType=cv2.LINE_AA)

    def _draw_smooth_segment(self, thickness: int) -> None:
        """
        基于滑动窗口绘制 Catmull-Rom 平滑曲线段。
        
        包含“直线保护”逻辑：当检测到 4 点近似共线时，
        直接绘制直线而非曲线，避免手写小字时笔画被过度平滑扭曲。
        """
        if len(self._window) < 4:
            return
        
        p0, p1, p2, p3 = self._window[-4], self._window[-3], self._window[-2], self._window[-1]

        # 直线保护逻辑：计算 p1-p2 偏离 p0-p3 连线的程度
        vx = p3[0] - p0[0]
        vy = p3[1] - p0[1]
        denom = (vx * vx + vy * vy) ** 0.5 + 1e-6
        ux = p2[0] - p1[0]
        uy = p2[1] - p1[1]
        cross = abs(vx * uy - vy * ux)
        
        if (cross / denom) < 2.0:
            self._draw_line_segment(p1, p2, thickness)
            self._last_curve_end = p2
            return
        
        # 计算曲线点
        curve_points = catmull_rom_spline(p0, p1, p2, p3, self.bezier_segments)
        
        # 从上次曲线结束点开始绘制
        start_pt = self._last_curve_end if self._last_curve_end else curve_points[0]
        
        for pt in curve_points:
            self._draw_line_segment(start_pt, pt, thickness)
            start_pt = pt
        
        self._last_curve_end = curve_points[-1]

    def draw(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        绘制入口函数。
        
        处理平滑、速度计算、粗细调整、跳变检测，并将点推入滑动窗口以生成曲线。

        Args:
            point (Tuple[int, int]): 当前帧捕捉到的原始坐标点。

        Returns:
            Tuple[int, int]: 实际使用的平滑后坐标点。
        """
        if self.smoothing:
            point = tuple(map(int, self.smoothing.push(point)))

        # 计算速度
        speed = 0.0
        if self.prev_point is not None:
            speed = self._distance(self.prev_point, point)
        
        # 计算当前粗细
        thickness = self._calculate_thickness(speed)

        # 位置跳变检测
        if self.prev_point is not None and speed > self.jump_threshold:
            hard_break = self.jump_threshold * 2
            if speed <= hard_break:
                self._draw_line_segment(self.prev_point, point, thickness)
                self._window = [point]
                self._last_curve_end = point
            else:
                self._stroke_broken = True
                self._window = [point]
                self._last_curve_end = None
            self.prev_point = point
            self.points.append(point)
            return point

        # 添加到窗口和点列表
        self._window.append(point)
        self.points.append(point)

        # 绘制策略
        if self.enable_bezier:
            if len(self._window) == 1:
                # 第1个点：画起始圆点
                canvas_img = self.canvas.get_canvas()
                if self.brush_manager.brush_type == "dashed":
                    dashed_thickness = thickness + max(2, int(thickness * 0.3))
                    radius = max(1, dashed_thickness // 2)
                    cv2.circle(canvas_img, point, radius, 
                              self.brush_manager.color, -1, lineType=cv2.LINE_AA)
                else:
                    radius = max(1, thickness // 2)
                    cv2.circle(canvas_img, point, radius, 
                              self.brush_manager.color, -1, lineType=cv2.LINE_AA)
            elif len(self._window) == 2:
                self._draw_line_segment(self._window[-2], self._window[-1], thickness)
                self._last_curve_end = point
            elif len(self._window) == 3:
                self._draw_line_segment(self._window[-2], self._window[-1], thickness)
                self._last_curve_end = point
            else:
                # 曲线平滑
                self._draw_smooth_segment(thickness)
                if len(self._window) > 4:
                    self._window = self._window[-4:]
        else:
            if self.prev_point is not None:
                self._draw_line_segment(self.prev_point, point, thickness)

        self.prev_point = point
        self._last_speed = speed
        return point

    def end_stroke(self) -> List[Tuple[int, int]]:
        """结束当前笔画，并绘制剩余的曲线缓冲段。"""
        # 绘制最后几个点
        if self.enable_bezier and len(self._window) >= 2:
            thickness = self._calculate_thickness(self._last_speed)
            if self._last_curve_end and len(self._window) >= 1:
                for i in range(len(self._window)):
                    pt = self._window[i]
                    if self._last_curve_end != pt:
                        self._draw_line_segment(self._last_curve_end, pt, thickness)
                        self._last_curve_end = pt
        
        finished_points = self.points.copy()
        self.start_stroke()
        return finished_points

    @property
    def was_stroke_broken(self) -> bool:
        """bool: 上一帧是否发生了笔画中断（大跳变）。"""
        return self._stroke_broken
    
    @property
    def current_stroke_length(self) -> int:
        """int: 当前笔画包含的点数。"""
        return len(self.points)
    
    @property
    def current_thickness(self) -> float:
        """float: 当前粗细比例（用于UI显示）。"""
        return self._current_thickness