# -*- coding: utf-8 -*-
"""虚拟画笔模块 - 钢笔效果：速度感知笔画粗细 + 贝塞尔平滑 + 形状识别"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, TYPE_CHECKING
import math
import time

from modules.canvas import Canvas
from modules.brush_manager import BrushManager
from utils.smoothing import EmaSmoother, catmull_rom_spline

# 类型检查时导入（避免循环依赖）
if TYPE_CHECKING:
    from modules.shape_recognizer import RecognizedShape


class VirtualPen:
    """
    虚拟钢笔 - 速度感知笔画粗细 + 贝塞尔平滑
    
    钢笔效果原理：
    - 移动速度快 → 笔画细（模拟轻压）
    - 移动速度慢 → 笔画粗（模拟重压）
    - 粗细平滑过渡，避免突变
    """
    
    def __init__(
        self,
        canvas: Canvas,
        brush_manager: BrushManager,
        smoothing: Optional[EmaSmoother] = None,
        jump_threshold: int = 80,
        enable_bezier: bool = True,
        bezier_segments: int = 8,
        # 钢笔效果参数
        enable_pen_effect: bool = True,
        min_thickness_ratio: float = 0.4,   # 最细时的粗细比例
        max_thickness_ratio: float = 1.2,   # 最粗时的粗细比例
        speed_threshold: float = 30.0,      # 速度阈值（像素/帧）
        thickness_smoothing: float = 0.3,   # 粗细平滑系数
        # 形状识别参数
        enable_shape_recognition: bool = True,  # 启用形状识别
    ) -> None:
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
        
        # 形状识别
        self.enable_shape_recognition = enable_shape_recognition
        self.last_recognized_shape = None
        
        # 悬停检测（用于确认是否需要形状识别）
        self._hover_start_time = None  # 悬停开始时间
        self._hover_threshold = 0.5  # 悬停阈值（秒）
        self._hover_region_radius = 15  # 悬停区域半径（像素）- 在此范围内移动算悬停
        self._hover_center = None  # 悬停中心点
        self._is_hovering = False  # 是否正在悬停
        
        # 延迟导入，避免循环依赖
        if self.enable_shape_recognition:
            try:
                from modules.shape_recognizer import GeometricShapeRecognizer
                self._shape_recognizer = GeometricShapeRecognizer()
            except ImportError as e:
                print(f"警告：无法加载形状识别器，功能已禁用 ({e})")
                self.enable_shape_recognition = False
                self._shape_recognizer = None
        else:
            self._shape_recognizer = None
        
        # 绘图状态
        self.prev_point: Optional[Tuple[int, int]] = None
        self.points: List[Tuple[int, int]] = []
        self._stroke_broken = False
        
        # 滑动窗口缓冲（最多4个点）
        self._window: List[Tuple[int, int]] = []
        self._last_curve_end: Optional[Tuple[int, int]] = None
        
        # 钢笔效果状态
        self._current_thickness: float = 1.0  # 当前粗细比例
        self._last_speed: float = 0.0

    def start_stroke(self) -> None:
        """开始新笔画"""
        self.prev_point = None
        if self.smoothing:
            self.smoothing.reset()
        self.points = []
        self._stroke_broken = False
        self._window = []
        self._last_curve_end = None
        self._current_thickness = 1.0
        self._last_speed = 0.0
        
        # 重置悬停状态
        self._hover_start_time = None
        self._hover_center = None
        self._is_hovering = False
        
        # 重置虚线相位（如果是虚线笔刷）
        if self.brush_manager.brush_type == "dashed":
            self.brush_manager.reset_dash_phase()

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """计算两点距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _calculate_thickness(self, speed: float) -> int:
        """
        根据速度计算笔画粗细（钢笔效果）
        
        速度快 → 细笔画（轻盈）
        速度慢 → 粗笔画（厚重）
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
        """绘制一段线条（带圆角连接）"""
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
        """绘制平滑曲线段"""
        if len(self._window) < 4:
            return
        
        p0, p1, p2, p3 = self._window[-4], self._window[-3], self._window[-2], self._window[-1]

        # 小字/慢写时常见问题：Catmull-Rom 会把近似直线拉成不自然弧线
        # 这里加入“直线保护”：四点近似共线时直接画线段，避免弧线化
        vx = p3[0] - p0[0]
        vy = p3[1] - p0[1]
        denom = (vx * vx + vy * vy) ** 0.5 + 1e-6
        ux = p2[0] - p1[0]
        uy = p2[1] - p1[1]
        cross = abs(vx * uy - vy * ux)
        # cross/denom 近似表示“偏离直线的像素量”
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
        绘制一个点
        
        实现钢笔效果：根据移动速度调整笔画粗细
        同时检测悬停状态（用于形状识别确认）
        """
        if self.smoothing:
            point = tuple(map(int, self.smoothing.push(point)))

        # 计算速度
        speed = 0.0
        if self.prev_point is not None:
            speed = self._distance(self.prev_point, point)
        
        # 悬停检测（在小范围内移动算悬停）
        if self._hover_center is None:
            # 首次进入，设置悬停中心
            self._hover_center = point
            self._hover_start_time = time.time()
        else:
            # 检查是否在悬停区域内
            distance_from_center = self._distance(point, self._hover_center)
            
            if distance_from_center <= self._hover_region_radius:
                # 仍在悬停区域内，计算悬停时间
                hover_duration = time.time() - self._hover_start_time
                
                if hover_duration >= self._hover_threshold:
                    if not self._is_hovering:  # 首次达到悬停阈值
                        self._is_hovering = True
                        print(f"[悬停检测] ✓ 悬停确认（{hover_duration:.2f}s ≥ {self._hover_threshold}s）")
            else:
                # 移动超出悬停区域，重置
                if self._is_hovering:
                    print(f"[悬停检测] ✗ 移动超出悬停区域，重置")
                self._hover_center = point
                self._hover_start_time = time.time()
                self._is_hovering = False
        
        # 计算当前粗细
        thickness = self._calculate_thickness(speed)

        # 位置跳变检测
        if self.prev_point is not None and speed > self.jump_threshold:
            # raw 点抖动时可能出现瞬时跳变，直接断笔会造成“断断续续”
            # 分两档处理：
            # - 中等跳变：画一段直线桥接，避免断笔
            # - 巨大跳变：真正断笔（避免跨屏长线）
            hard_break = self.jump_threshold * 2
            if speed <= hard_break:
                self._draw_line_segment(self.prev_point, point, thickness)
                # 继续笔画，但重置曲线窗口，避免曲线形变
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
                # 第1个点：画起始圆点（所有笔刷类型都需要起始点）
                canvas_img = self.canvas.get_canvas()
                # 虚线使用加粗的起始点（与虚线粗细一致）
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
        """
        结束笔画
        
        如果启用形状识别，会尝试识别并优化手绘形状：
        - 圆形 → 标准圆
        - 矩形 → 标准矩形
        - 正方形 → 标准正方形
        - 直线 → 标准直线
        """
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
        
        # ========== 形状识别（只在画笔工具 + 悬停确认时启用） ==========
        if (self.enable_shape_recognition and 
            self._shape_recognizer and 
            len(finished_points) > 0 and
            self.brush_manager.tool == "pen" and
            self._is_hovering):  # ✅ 只在悬停确认后才识别
            
            try:
                print(f"[悬停确认] 开始形状识别...")
                recognized = self._shape_recognizer.recognize(finished_points)
                self.last_recognized_shape = recognized
                
                # 如果识别到标准形状，用标准形状替换手绘
                if recognized.shape_type != "none":
                    self._replace_with_standard_shape(recognized)
                    print(f"✓ 识别到 {recognized.shape_type}，置信度: {recognized.confidence:.2f}")
            except Exception as e:
                print(f"形状识别错误: {e}")
        elif self.enable_shape_recognition and not self._is_hovering and len(finished_points) > 0:
            print(f"[提示] 未检测到悬停确认，跳过形状识别")
        
        self.start_stroke()
        return finished_points
    
    def _replace_with_standard_shape(self, shape: "RecognizedShape") -> None:
        """
        用标准形状替换手绘笔画
        
        关键策略：
        1. 从历史栈获取上一个画布状态（没有当前手绘的状态）
        2. 用上一个状态覆盖当前画布（完全清除当前手绘）
        3. 在干净的画布上绘制标准形状
        
        注意：不在这里保存笔画，由外部统一调用 canvas.save_stroke()
        """
        if not shape.points or len(self.points) == 0:
            return
        
        canvas_img = self.canvas.get_canvas()
        thickness = self.brush_manager.thickness
        color = self.brush_manager.color
        
        # ===== 关键步骤1：获取历史栈中的上一个状态（没有当前手绘） =====
        try:
            # 访问 Canvas 的私有历史栈
            history_stack = self.canvas._history._history
            
            if len(history_stack) > 0:
                # 获取上一个画布状态（在开始当前笔画之前的状态）
                previous_canvas = history_stack[-1].copy()
                
                # ===== 关键步骤2：用上一个状态覆盖当前画布（清除当前手绘） =====
                canvas_img[:] = previous_canvas
                
                print(f"[形状替换] 已清除原手绘，恢复到历史状态")
            else:
                # 如果是第一笔（历史栈为空），清空整个画布
                canvas_img[:] = 0
                print(f"[形状替换] 第一笔，清空画布")
        except Exception as e:
            print(f"[形状替换] 警告：无法访问历史栈 ({e})，尝试直接清除")
            # 降级方案：直接清除区域
            points_array = np.array(self.points)
            x_min, y_min = points_array.min(axis=0)
            x_max, y_max = points_array.max(axis=0)
            margin = thickness + 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(canvas_img.shape[1], x_max + margin)
            y_max = min(canvas_img.shape[0], y_max + margin)
            canvas_img[y_min:y_max, x_min:x_max] = 0
        
        # ===== 关键步骤3：绘制标准形状（在干净的画布上） =====
        if shape.shape_type == "circle" and shape.center and shape.radius:
            # 绘制标准圆
            cv2.circle(canvas_img, shape.center, int(shape.radius), color, thickness, lineType=cv2.LINE_AA)
            print(f"[形状替换] 已绘制标准圆")
        
        elif shape.shape_type in ["rectangle", "square"]:
            # 绘制标准矩形/正方形（闭合路径）
            pts = np.array(shape.points[:4], dtype=np.int32)  # 只取前4个角点
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas_img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            print(f"[形状替换] 已绘制标准{shape.shape_type}")
        
        elif shape.shape_type == "line":
            # 绘制标准直线
            if len(shape.points) >= 2:
                pt1 = shape.points[0]
                pt2 = shape.points[-1]
                cv2.line(canvas_img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
                print(f"[形状替换] 已绘制标准直线")
        
        # 注意：不在这里保存，由 main.py 统一保存

    @property
    def was_stroke_broken(self) -> bool:
        return self._stroke_broken
    
    @property
    def current_stroke_length(self) -> int:
        return len(self.points)
    
    @property
    def current_thickness(self) -> float:
        """返回当前粗细比例（用于UI显示）"""
        return self._current_thickness