# -*- coding: utf-8 -*-
"""图形识别模块 - 识别并美化手绘的圆形、矩形、三角形、直线"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ShapeRecognizer:
    """
    图形识别器 - 支持识别和美化几何图形
    
    支持的图形：
    - 圆形 (circle)
    - 矩形 (rectangle)
    - 三角形 (triangle)
    - 直线 (line) - 新增
    """
    
    def __init__(
        self,
        closedness_thresh: float = 0.2,
        circle_score_thresh: float = 0.75,
        line_variance_thresh: float = 0.015,  # 直线方差阈值（归一化）
        min_line_length: int = 50,            # 最小直线长度（像素）
        enable_line_assist: bool = True,      # 是否启用直线辅助
    ) -> None:
        """
        初始化图形识别器
        
        Args:
            closedness_thresh: 闭合度阈值（越小越严格）
            circle_score_thresh: 圆形得分阈值
            line_variance_thresh: 直线方差阈值（点到拟合直线的归一化距离方差）
            min_line_length: 最小直线长度
            enable_line_assist: 是否启用直线辅助功能
        """
        self.closedness_thresh = closedness_thresh
        self.circle_score_thresh = circle_score_thresh
        self.line_variance_thresh = line_variance_thresh
        self.min_line_length = min_line_length
        self.enable_line_assist = enable_line_assist

    def _closedness(self, pts: np.ndarray) -> float:
        """计算轮廓闭合度"""
        perim = float(cv2.arcLength(pts, False))
        if perim < 1e-6:
            return 1.0
        start = pts[0, 0]
        end = pts[-1, 0]
        dist = np.linalg.norm(start - end)
        return dist / perim

    def _circle_score(self, pts: np.ndarray) -> float:
        """计算圆形得分"""
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        area_circle = np.pi * (radius**2)
        area_contour = float(cv2.contourArea(pts))
        if area_circle < 1e-6:
            return 0.0
        return area_contour / area_circle

    def _is_line(self, points: List[Tuple[int, int]]) -> Tuple[bool, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        检测笔画是否接近直线
        
        使用最小二乘法拟合直线，计算点到直线的方差。
        如果方差小于阈值，则认为是直线。
        
        Args:
            points: 笔画点列表
        
        Returns:
            (is_line, (start_point, end_point)) - 是否为直线，以及直线的起点和终点
        """
        if len(points) < 3:
            return False, None
        
        pts = np.array(points, dtype=np.float32)
        
        # 计算起点和终点的距离
        start = pts[0]
        end = pts[-1]
        line_length = np.linalg.norm(end - start)
        
        if line_length < self.min_line_length:
            return False, None
        
        # 使用 PCA 或简单的线性回归来拟合直线
        # 这里使用点到起点-终点连线的距离方差
        
        # 直线方向向量
        direction = end - start
        direction_norm = direction / (line_length + 1e-6)
        
        # 计算每个点到直线的距离
        distances = []
        for pt in pts:
            # 点到直线的垂直距离
            v = pt - start
            # 投影长度
            proj_len = np.dot(v, direction_norm)
            # 投影点
            proj_pt = start + proj_len * direction_norm
            # 垂直距离
            dist = np.linalg.norm(pt - proj_pt)
            distances.append(dist)
        
        # 计算归一化方差（相对于线长）
        variance = np.var(distances) / (line_length + 1e-6)
        
        # 判断是否为直线
        if variance < self.line_variance_thresh:
            return True, (tuple(map(int, start)), tuple(map(int, end)))
        
        return False, None

    def recognize(self, points: List[Tuple[int, int]]) -> Optional[str]:
        """
        识别图形类型
        
        Args:
            points: 笔画点列表
        
        Returns:
            图形类型字符串，或 None
        """
        if len(points) < 5:
            return None
        
        # 首先检测直线（优先级最高，因为直线检测更快）
        if self.enable_line_assist:
            is_line, _ = self._is_line(points)
            if is_line:
                return "line"
        
        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        closedness = self._closedness(contour)
        closed = closedness < self.closedness_thresh

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        circle_score = self._circle_score(contour)

        if closed and vertices == 3:
            return "triangle"
        if closed and vertices == 4:
            return "rectangle"
        if closed and circle_score > self.circle_score_thresh:
            return "circle"
        return None

    def beautify(
        self,
        points: List[Tuple[int, int]],
        canvas_img: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> Optional[str]:
        """
        美化图形
        
        Args:
            points: 笔画点列表
            canvas_img: 画布图像
            color: 绘制颜色
            thickness: 线条粗细
        
        Returns:
            识别到的图形类型，或 None
        """
        shape = self.recognize(points)
        if not shape:
            return None

        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        padding = thickness + 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(canvas_img.shape[1], x + w + padding)
        y2 = min(canvas_img.shape[0], y + h + padding)

        # 清除原始笔迹区域
        cv2.rectangle(canvas_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # 绘制标准几何图形
        if shape == "line":
            # 直线美化
            _, line_points = self._is_line(points)
            if line_points:
                start_pt, end_pt = line_points
                cv2.line(canvas_img, start_pt, end_pt, color, thickness, lineType=cv2.LINE_AA)
        elif shape == "triangle":
            hull = cv2.convexHull(contour)
            cv2.polylines(canvas_img, [hull], True, color, thickness, lineType=cv2.LINE_AA)
        elif shape == "rectangle":
            cv2.rectangle(
                canvas_img,
                (x, y),
                (x + w, y + h),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        elif shape == "circle":
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(
                canvas_img,
                (int(cx), int(cy)),
                int(radius),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        return shape
    
    def set_line_assist(self, enabled: bool) -> None:
        """
        设置直线辅助功能开关
        
        Args:
            enabled: 是否启用
        """
        self.enable_line_assist = enabled
    
    def is_line_assist_enabled(self) -> bool:
        """返回直线辅助是否启用"""
        return self.enable_line_assist