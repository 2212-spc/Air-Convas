# -*- coding: utf-8 -*-
"""
几何形状识别器 - 基于几何算法的智能识别
支持圆形、矩形、正方形的自动识别和优化
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RecognizedShape:
    """识别到的形状"""
    shape_type: str  # "circle", "rectangle", "square", "line", "none"
    confidence: float  # 0.0-1.0
    points: List[Tuple[int, int]]  # 标准化后的关键点
    center: Optional[Tuple[int, int]] = None
    radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class GeometricShapeRecognizer:
    """
    形状识别器
    
    算法原理：
    1. 圆形识别：最小二乘法拟合圆 + 圆度评分
    2. 矩形识别：Douglas-Peucker 简化 + 角点检测
    3. 正方形识别：矩形 + 边长比例检测
    """
    
    def __init__(
        self,
        min_points: int = 10,  # 最少点数（降低要求）
        closure_threshold: float = 60.0,  # 闭合阈值（像素，放宽标准）
        circle_confidence_threshold: float = 0.50,  # 圆形置信度阈值（放宽标准）
        rectangle_confidence_threshold: float = 0.62,  # 矩形置信度阈值
    ):
        self.min_points = min_points
        self.closure_threshold = closure_threshold
        self.circle_confidence_threshold = circle_confidence_threshold
        self.rectangle_confidence_threshold = rectangle_confidence_threshold
        
        self.enabled = True
    
    def recognize(self, stroke_points: List[Tuple[int, int]]) -> RecognizedShape:
        """
        识别笔画形状
        
        Args:
            stroke_points: 笔画的点列表 [(x, y), ...]
        
        Returns:
            RecognizedShape: 识别结果
        
        识别优先级（从高到低）：
        1. 圆形（最高优先级，即使不完全闭合也尝试）
        2. 矩形/正方形（闭合形状）
        3. 直线（开放形状）
        """
        if not self.enabled or len(stroke_points) < self.min_points:
            return RecognizedShape("none", 0.0, stroke_points)
        
        # 转换为 numpy 数组
        points = np.array(stroke_points, dtype=np.float32)
        
        # 1. 检查是否闭合
        is_closed = self._is_closed(points)
        
        # 2. 【最高优先级】始终尝试识别圆形（即使不完全闭合）
        # 原因：用户画圆时可能没有完美闭合，但形状明显是圆形
        circle_result = self._recognize_circle(points)
        print(f"[形状识别] 圆形置信度: {circle_result.confidence:.3f} (阈值: {self.circle_confidence_threshold:.3f}, 闭合: {is_closed})")
        if circle_result.confidence >= self.circle_confidence_threshold:
            return circle_result
        
        # 3. 如果闭合，尝试识别矩形/正方形
        if is_closed:
            rect_result = self._recognize_rectangle(points)
            print(f"[形状识别] 矩形置信度: {rect_result.confidence:.3f} (阈值: {self.rectangle_confidence_threshold:.3f})")
            if rect_result.confidence >= self.rectangle_confidence_threshold:
                return rect_result
        
        # 4. 如果不闭合，尝试识别直线（最低优先级）
        if not is_closed:
            line_result = self._recognize_line(points)
            print(f"[形状识别] 直线置信度: {line_result.confidence:.3f} (阈值: 0.85)")
            # 提高直线识别阈值，避免误判
            if line_result.confidence >= 0.85:  # 提高到 0.85
                return line_result
        
        # 5. 无法识别
        print(f"[形状识别] 未识别到标准形状")
        return RecognizedShape("none", 0.0, stroke_points)
    
    def _is_closed(self, points: np.ndarray) -> bool:
        """检查路径是否闭合"""
        if len(points) < 3:
            return False
        
        start = points[0]
        end = points[-1]
        distance = np.linalg.norm(start - end)
        
        return distance <= self.closure_threshold
    
    # ==================== 圆形识别 ====================
    
    def _recognize_circle(self, points: np.ndarray) -> RecognizedShape:
        """
        圆形识别算法
        
        步骤：
        1. 最小二乘法拟合圆（找圆心和半径）
        2. 计算拟合误差
        3. 计算圆度评分
        """
        # 1. 拟合圆
        center, radius = self._fit_circle_least_squares(points)
        
        if radius <= 0:
            return RecognizedShape("none", 0.0, points.tolist())
        
        # 2. 计算拟合误差
        distances = np.linalg.norm(points - center, axis=1)
        errors = np.abs(distances - radius)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # 3. 计算置信度（放宽误差容忍度）
        # 误差越小，置信度越高（使用宽松的阈值）
        error_score = max(0, 1.0 - (mean_error / (radius * 0.28)))  # 允许 28% 的误差
        error_score = min(1.0, error_score)  # 限制在 0-1
        
        consistency_score = max(0, 1.0 - (std_error / (radius * 0.25)))  # 允许 25% 的标准差
        consistency_score = min(1.0, consistency_score)  # 限制在 0-1
        
        # 4. 计算圆度（周长和面积的关系）
        circularity = self._calculate_circularity(points)
        
        # 综合评分（圆度权重较高）
        confidence = (error_score * 0.30 + consistency_score * 0.20 + circularity * 0.50)
        
        # 5. 生成标准圆的点
        if confidence >= self.circle_confidence_threshold:
            circle_points = self._generate_circle_points(center, radius)
            return RecognizedShape(
                "circle",
                confidence,
                circle_points,
                center=tuple(center.astype(int)),
                radius=float(radius)
            )
        
        return RecognizedShape("none", confidence, points.tolist())
    
    def _fit_circle_least_squares(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        最小二乘法拟合圆
        
        原理：
        圆的方程：(x - cx)² + (y - cy)² = r²
        展开：x² + y² - 2*cx*x - 2*cy*y + cx² + cy² - r² = 0
        令：A = -2*cx, B = -2*cy, C = cx² + cy² - r²
        得：x² + y² + A*x + B*y + C = 0
        
        Returns:
            center: (cx, cy)
            radius: r
        """
        n = len(points)
        
        # 构建矩阵
        x = points[:, 0]
        y = points[:, 1]
        
        # 求解线性方程组
        # [sum(x²)  sum(xy)  sum(x)]   [A]   [sum(x³ + xy²)]
        # [sum(xy)  sum(y²)  sum(y)] * [B] = [sum(x²y + y³)]
        # [sum(x)   sum(y)   n     ]   [C]   [sum(x² + y²)]
        
        xx = x * x
        yy = y * y
        xy = x * y
        
        A_matrix = np.array([
            [np.sum(xx), np.sum(xy), np.sum(x)],
            [np.sum(xy), np.sum(yy), np.sum(y)],
            [np.sum(x),  np.sum(y),  n]
        ])
        
        b_vector = np.array([
            -np.sum(xx * x + x * yy),
            -np.sum(xx * y + yy * y),
            -np.sum(xx + yy)
        ])
        
        try:
            # 求解 A * [A, B, C]^T = b
            solution = np.linalg.solve(A_matrix, b_vector)
            A, B, C = solution
            
            # 计算圆心和半径
            cx = -A / 2
            cy = -B / 2
            radius = np.sqrt(cx**2 + cy**2 - C)
            
            return np.array([cx, cy]), radius
        except np.linalg.LinAlgError:
            # 奇异矩阵，返回质心和平均半径
            center = np.mean(points, axis=0)
            radius = np.mean(np.linalg.norm(points - center, axis=1))
            return center, radius
    
    def _calculate_circularity(self, points: np.ndarray) -> float:
        """
        计算圆度
        
        完美圆形：周长² / 面积 = 4π ≈ 12.566
        """
        # 计算周长
        perimeter = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            perimeter += np.linalg.norm(p2 - p1)
        
        # 计算面积（Green's theorem）
        area = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            area += p1[0] * p2[1] - p2[0] * p1[1]
        area = abs(area) / 2
        
        if area < 1e-6:
            return 0.0
        
        # 计算圆度
        circularity_value = (perimeter ** 2) / area
        ideal_circularity = 4 * np.pi  # ≈ 12.566
        
        # 归一化到 0-1（宽松的评分标准）
        # 越接近 4π，圆度越高
        diff = abs(circularity_value - ideal_circularity)
        # 使用宽松的容忍度：允许偏差达到 ideal_circularity 的 1.4 倍
        score = max(0, 1.0 - diff / (ideal_circularity * 1.4))
        
        return score
    
    def _generate_circle_points(self, center: np.ndarray, radius: float, num_points: int = 100) -> List[Tuple[int, int]]:
        """生成标准圆的点"""
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        points = list(zip(x.astype(int), y.astype(int)))
        return points
    
    # ==================== 矩形/正方形识别 ====================
    
    def _recognize_rectangle(self, points: np.ndarray) -> RecognizedShape:
        """
        矩形/正方形识别算法
        
        步骤：
        1. Douglas-Peucker 算法简化路径
        2. 检测角点（应该有 4 个）
        3. 检测角度（应该接近 90 度）
        4. 检测边长关系（正方形：四边相等，矩形：对边相等）
        """
        # 1. 简化路径
        epsilon = self._calculate_epsilon(points)
        simplified = self._douglas_peucker(points, epsilon)
        
        # 2. 检查是否有 4 个角点
        if len(simplified) < 4 or len(simplified) > 6:
            return RecognizedShape("none", 0.0, points.tolist())
        
        # 3. 如果有 5 个点（包含起点重复），移除最后一个
        if len(simplified) == 5 and np.allclose(simplified[0], simplified[-1], atol=10):
            simplified = simplified[:-1]
        
        if len(simplified) != 4:
            return RecognizedShape("none", 0.0, points.tolist())
        
        # 4. 检测角度
        angles = self._calculate_angles(simplified)
        angle_score = self._score_right_angles(angles)
        
        if angle_score < 0.7:
            return RecognizedShape("none", 0.0, points.tolist())
        
        # 5. 检测边长
        side_lengths = self._calculate_side_lengths(simplified)
        
        # 6. 判断是正方形还是矩形
        is_square, shape_score = self._is_square(side_lengths)
        
        if is_square:
            shape_type = "square"
        else:
            shape_type = "rectangle"
        
        # 7. 综合置信度
        confidence = (angle_score * 0.6 + shape_score * 0.4)
        
        # 8. 生成标准矩形/正方形
        if confidence >= self.rectangle_confidence_threshold:
            rect_points = self._generate_rectangle_points(simplified)
            
            # 计算中心和尺寸
            center = np.mean(simplified, axis=0)
            width = max(side_lengths[0], side_lengths[2])
            height = max(side_lengths[1], side_lengths[3])
            
            return RecognizedShape(
                shape_type,
                confidence,
                rect_points,
                center=tuple(center.astype(int)),
                width=float(width),
                height=float(height)
            )
        
        return RecognizedShape("none", confidence, points.tolist())
    
    def _douglas_peucker(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Douglas-Peucker 算法简化路径
        
        原理：递归地移除距离直线较近的点
        """
        if len(points) < 3:
            return points
        
        # 找到距离起点-终点连线最远的点
        start = points[0]
        end = points[-1]
        
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(points) - 1):
            distance = self._point_to_line_distance(points[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # 如果最大距离大于阈值，递归简化
        if max_distance > epsilon:
            left = self._douglas_peucker(points[:max_index + 1], epsilon)
            right = self._douglas_peucker(points[max_index:], epsilon)
            
            # 合并，去除重复点
            result = np.vstack([left[:-1], right])
            return result
        else:
            # 否则，只保留起点和终点
            return np.array([start, end])
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """计算点到直线的距离"""
        # 向量叉积法
        numerator = abs((line_end[0] - line_start[0]) * (line_start[1] - point[1]) -
                       (line_start[0] - point[0]) * (line_end[1] - line_start[1]))
        denominator = np.linalg.norm(line_end - line_start)
        
        if denominator < 1e-6:
            return np.linalg.norm(point - line_start)
        
        return numerator / denominator
    
    def _calculate_epsilon(self, points: np.ndarray) -> float:
        """计算 Douglas-Peucker 的 epsilon 参数"""
        # 基于点的总长度
        total_length = 0
        for i in range(len(points) - 1):
            total_length += np.linalg.norm(points[i + 1] - points[i])
        
        # epsilon = 总长度的 2%
        return total_length * 0.02
    
    def _calculate_angles(self, corners: np.ndarray) -> List[float]:
        """计算角点的角度"""
        angles = []
        n = len(corners)
        
        for i in range(n):
            p1 = corners[(i - 1) % n]
            p2 = corners[i]
            p3 = corners[(i + 1) % n]
            
            # 计算两个向量
            v1 = p1 - p2
            v2 = p3 - p2
            
            # 计算夹角
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_deg = np.degrees(angle)
            
            angles.append(angle_deg)
        
        return angles
    
    def _score_right_angles(self, angles: List[float]) -> float:
        """评分角度接近 90 度的程度"""
        scores = []
        for angle in angles:
            # 计算与 90 度的差异
            diff = abs(angle - 90)
            # 允许 ±20 度的误差
            score = max(0, 1.0 - diff / 20.0)
            scores.append(score)
        
        return np.mean(scores)
    
    def _calculate_side_lengths(self, corners: np.ndarray) -> List[float]:
        """计算边长"""
        side_lengths = []
        n = len(corners)
        
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)
        
        return side_lengths
    
    def _is_square(self, side_lengths: List[float]) -> Tuple[bool, float]:
        """
        判断是否为正方形
        
        Returns:
            is_square: 是否为正方形
            score: 形状评分（边长一致性）
        """
        if len(side_lengths) != 4:
            return False, 0.0
        
        mean_length = np.mean(side_lengths)
        std_length = np.std(side_lengths)
        
        # 计算变异系数
        cv = std_length / (mean_length + 1e-6)
        
        # 正方形：四边长度相等（允许 10% 误差）
        if cv < 0.1:
            score = 1.0 - cv / 0.1
            return True, score
        
        # 矩形：对边相等
        # 检查对边差异
        diff1 = abs(side_lengths[0] - side_lengths[2]) / mean_length
        diff2 = abs(side_lengths[1] - side_lengths[3]) / mean_length
        
        if diff1 < 0.15 and diff2 < 0.15:
            score = 1.0 - (diff1 + diff2) / 2 / 0.15
            return False, score
        
        return False, 0.5
    
    def _generate_rectangle_points(self, corners: np.ndarray) -> List[Tuple[int, int]]:
        """生成标准矩形的点"""
        points = []
        n = len(corners)
        
        # 沿着每条边生成点
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            
            # 在两点之间插值
            num_segment_points = 25
            for j in range(num_segment_points):
                t = j / num_segment_points
                point = p1 * (1 - t) + p2 * t
                points.append(tuple(point.astype(int)))
        
        return points
    
    # ==================== 直线识别 ====================
    
    def _recognize_line(self, points: np.ndarray) -> RecognizedShape:
        """
        直线识别算法
        
        原理：最小二乘法拟合直线，计算拟合误差
        """
        if len(points) < 2:
            return RecognizedShape("none", 0.0, points.tolist())
        
        # 1. 拟合直线
        start, end, error = self._fit_line(points)
        
        # 2. 计算置信度
        length = np.linalg.norm(end - start)
        if length < 1e-6:
            return RecognizedShape("none", 0.0, points.tolist())
        
        # 误差越小，置信度越高
        confidence = max(0, 1.0 - error / length)
        
        if confidence >= 0.70:
            line_points = [tuple(start.astype(int)), tuple(end.astype(int))]
            return RecognizedShape("line", confidence, line_points)
        
        return RecognizedShape("none", confidence, points.tolist())
    
    def _fit_line(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        拟合直线
        
        Returns:
            start: 起点
            end: 终点
            error: 平均误差
        """
        # 使用 PCA 拟合直线
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        # SVD 分解
        U, S, Vt = np.linalg.svd(centered)
        
        # 主方向
        direction = Vt[0]
        
        # 投影到主方向
        projections = np.dot(centered, direction)
        
        # 找到最远的两个点
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        start = mean + min_proj * direction
        end = mean + max_proj * direction
        
        # 计算拟合误差
        errors = []
        for point in points:
            error = self._point_to_line_distance(point, start, end)
            errors.append(error)
        
        mean_error = np.mean(errors)
        
        return start, end, mean_error


# ==================== 全局实例 ====================

# 几何形状识别器实例（用于圆形、矩形等识别）
geometric_shape_recognizer = GeometricShapeRecognizer()


# ==================== 原有的直线辅助识别器（保持兼容） ====================

class ShapeRecognizer:
    """
    原有的形状识别器（直线辅助功能）
    保持向后兼容
    """
    def __init__(
        self,
        enable_line_assist: bool = True,
        line_variance_thresh: float = 0.015,
        min_line_length: int = 50,
    ):
        self.enable_line_assist = enable_line_assist
        self.line_variance_thresh = line_variance_thresh
        self.min_line_length = min_line_length
    
    def set_line_assist(self, enabled: bool):
        """设置直线辅助开关"""
        self.enable_line_assist = enabled
    
    def beautify(self, points: List[Tuple[int, int]], canvas, color, thickness) -> Optional[str]:
        """
        尝试美化图形（直线辅助）
        
        Args:
            points: 笔画点列表
            canvas: 画布
            color: 颜色
            thickness: 粗细
        
        Returns:
            识别到的形状类型，如果没有识别到返回 None
        """
        if not self.enable_line_assist or len(points) < 10:
            return None
        
        # 检测是否为直线
        if self._is_line(points):
            # 绘制直线（从第一个点到最后一个点）
            start = points[0]
            end = points[-1]
            
            # 计算长度
            length = np.linalg.norm(np.array(end) - np.array(start))
            
            if length >= self.min_line_length:
                # 擦除原始笔画区域
                # （简化处理：不擦除，直接覆盖绘制）
                
                # 绘制标准直线
                cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)
                
                return "line"
        
        return None
    
    def _is_line(self, points: List[Tuple[int, int]]) -> bool:
        """
        检测笔画是否为直线
        
        使用 PCA (主成分分析) 判断点是否在一条直线上
        """
        if len(points) < 10:
            return False
        
        points_array = np.array(points, dtype=np.float32)
        
        # 计算中心点
        mean = np.mean(points_array, axis=0)
        
        # 中心化
        centered = points_array - mean
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered.T)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # 计算方差比（第二主成分 / 第一主成分）
        if eigenvalues[1] < 1e-6:
            return False
        
        variance_ratio = eigenvalues[0] / eigenvalues[1]
        
        # 如果方差比很小（点集中在一条线上），则认为是直线
        return variance_ratio < self.line_variance_thresh
