"""
虚拟画笔模块 - 实现平滑曲线绘制
[优化说明]
1. 彻底重构了绘图逻辑，引入了基于中点的二次贝塞尔曲线(Quadratic Bezier)算法。
2. 不再是简单的点对点直线连接，而是生成平滑的曲线路径，大幅提升书写流畅度和美观度。
3. 优化了防粘连(跳变检测)逻辑。
"""
from typing import Optional, Tuple, List
import math
import numpy as np

from modules.canvas import Canvas
from modules.brush_manager import BrushManager
from utils.smoothing import EmaSmoother


class VirtualPen:
    def __init__(
        self,
        canvas: Canvas,
        brush_manager: BrushManager,
        smoothing: Optional[EmaSmoother] = None,
        jump_threshold: int = 100, # 增加了一点阈值，避免误断
    ) -> None:
        self.canvas = canvas
        self.brush_manager = brush_manager
        self.smoothing = smoothing
        self.jump_threshold = jump_threshold
        
        # 绘图缓冲区，用于存储最近的点以生成曲线
        # 只需要保存最近的3个点即可计算贝塞尔曲线
        self.points_buffer: List[Tuple[float, float]] = []
        # 保存整笔的所有原始点
        self.stroke_points: List[Tuple[int, int]] = []
        self._stroke_broken = False

    def start_stroke(self) -> None:
        """开始新的一笔，重置缓冲区"""
        self.points_buffer = []
        self.stroke_points = []
        self._stroke_broken = False
        if self.smoothing:
            self.smoothing.reset()

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_midpoint(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """计算中点"""
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    def _draw_quadratic_bezier(self, p0, p1, p2):
        """
        绘制二次贝塞尔曲线并画在画布上
        起点: p0和p1的中点
        终点: p1和p2的中点
        控制点: p1
        """
        mid1 = self._get_midpoint(p0, p1)
        mid2 = self._get_midpoint(p1, p2)
        
        # 根据距离动态计算插值步数，保证平滑度一致
        dist = self._distance(mid1, mid2)
        # 至少2步，大约每3个像素一个插值点，步数越多越平滑但计算量越大
        steps = max(2, int(dist / 3)) 
        
        last_pt = mid1
        # 生成 t 从 0 到 1 的序列（不包含0，因为起点已经是last_pt）
        for t in np.linspace(0, 1, steps + 1)[1:]:
            # 二次贝塞尔公式: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            # 这里起点是mid1, 控制点是p1, 终点是mid2
            u = 1 - t
            tt = t * t
            uu = u * u
            
            x = uu * mid1[0] + 2 * u * t * p1[0] + tt * mid2[0]
            y = uu * mid1[1] + 2 * u * t * p1[1] + tt * mid2[1]
            
            curr_pt = (x, y)
            
            # 调用 BrushManager 绘制这一小段线
            # BrushManager 内部会将浮点坐标转为整数
            self.brush_manager.draw_line(self.canvas.get_canvas(), last_pt, curr_pt)
            last_pt = curr_pt

    def draw(self, point_int: Tuple[int, int]) -> Tuple[int, int]:
        """
        处理新的输入点并绘图
        :param point_int: 原始整数坐标点
        :return: 平滑处理后的点（如果启用平滑）或原始点
        """
        # 1. 坐标平滑处理 (低通滤波)
        # 将整数点转为浮点数进行高精度计算
        point_float = (float(point_int[0]), float(point_int[1]))
        
        if self.smoothing:
            # 平滑器返回的是浮点数坐标
            processed_point = self.smoothing.push(point_float)
        else:
            processed_point = point_float

        # 记录用于最终输出的整数点
        final_output_point = (int(processed_point[0]), int(processed_point[1]))

        # 2. 跳变检测 (防粘连)
        if self.points_buffer:
            last_pt = self.points_buffer[-1]
            # 如果两帧之间距离过大，认为是误识别或跳变，断开笔画
            if self._distance(last_pt, processed_point) > self.jump_threshold:
                self._stroke_broken = True
                # 重置缓冲区为当前新点，开始新的一段
                self.points_buffer = [processed_point] 
                self.stroke_points.append(final_output_point)
                return final_output_point

        self.points_buffer.append(processed_point)
        self.stroke_points.append(final_output_point)

        # 3. 贝塞尔曲线绘图逻辑
        buffer_len = len(self.points_buffer)

        if buffer_len >= 3:
            # 取最后三个点用于计算曲线
            # 我们绘制的是倒数第3点和倒数第2点中点 -> 倒数第2点和倒数第1点中点 之间的曲线
            p0 = self.points_buffer[-3]
            p1 = self.points_buffer[-2]
            p2 = self.points_buffer[-1]
            self._draw_quadratic_bezier(p0, p1, p2)
            # 为了保持缓冲区较小，移除最旧的点
            self.points_buffer.pop(0)
        
        elif buffer_len == 2:
            # 只有两个点时，无法画曲线，先用直线连接起点到中点，避免起始延迟感
            p0 = self.points_buffer[-2]
            p1 = self.points_buffer[-1]
            mid = self._get_midpoint(p0, p1)
            self.brush_manager.draw_line(self.canvas.get_canvas(), p0, mid)

        return final_output_point

    def end_stroke(self) -> List[Tuple[int, int]]:
        """结束笔画，补齐最后一段"""
        if len(self.points_buffer) >= 2:
            # 补齐从最后一段曲线终点(即倒数两点的中点)到最后一个实际点的直线
            p_last_2 = self.points_buffer[-2]
            p_last = self.points_buffer[-1]
            mid = self._get_midpoint(p_last_2, p_last)
            self.brush_manager.draw_line(self.canvas.get_canvas(), mid, p_last)
            
        finished_points = self.stroke_points
        # 准备下一次书写
        self.start_stroke()
        return finished_points

    @property
    def was_stroke_broken(self) -> bool:
        return self._stroke_broken