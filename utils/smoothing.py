# -*- coding: utf-8 -*-
"""平滑算法模块 - EMA 指数移动平均、1€ Filter 等平滑算法"""

import math
from typing import Optional, Sequence, Tuple, List

import numpy as np


def catmull_rom_spline(p0: Tuple[int, int], p1: Tuple[int, int], 
                       p2: Tuple[int, int], p3: Tuple[int, int], 
                       num_points: int = 8) -> List[Tuple[int, int]]:
    """
    计算 Catmull-Rom 样条曲线上的点序列
    绘制的是 p1 到 p2 之间的曲线段
    """
    result = []
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        t2 = t * t
        t3 = t2 * t
        
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


class EmaSmoother:
    """简单的指数移动平均平滑器"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def push(self, value: Sequence[float]) -> Tuple[float, float]:
        arr = np.array(value, dtype=np.float32)
        if self._state is None:
            self._state = arr
        else:
            self._state = self.alpha * self._state + (1 - self.alpha) * arr
        return float(self._state[0]), float(self._state[1])


class LowPassFilter:
    """低通滤波器 - 用于 1€ Filter 内部"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._prev: Optional[float] = None
    
    def reset(self) -> None:
        self._prev = None
    
    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self._prev is None:
            self._prev = value
        else:
            self._prev = self.alpha * value + (1 - self.alpha) * self._prev
        return self._prev


class OneEuroFilter:
    """
    1€ Filter - 自适应平滑滤波器
    
    特点：
    - 低速移动时高平滑（减少抖动）
    - 高速移动时低平滑（更跟手）
    
    参考论文：
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
    https://cristal.univ-lille.fr/~casiez/1euro/
    
    参数说明：
    - min_cutoff: 最小截止频率（越小越平滑，但延迟越大）
    - beta: 速度系数（越大则高速时越跟手）
    - d_cutoff: 微分信号的截止频率
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float = 30.0  # 预估帧率
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        
        # 为 x, y 分别创建滤波器
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._y_filter = LowPassFilter()
        self._dy_filter = LowPassFilter()
        
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._prev_time: Optional[float] = None
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._y_filter.reset()
        self._dy_filter.reset()
        self._prev_x = None
        self._prev_y = None
        self._prev_time = None
    
    def _alpha(self, cutoff: float) -> float:
        """计算低通滤波器的 alpha 值"""
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def _filter_1d(
        self,
        value: float,
        prev_value: Optional[float],
        x_filter: LowPassFilter,
        dx_filter: LowPassFilter
    ) -> float:
        """对单个维度进行 1€ 滤波"""
        # 计算速度（微分）
        if prev_value is None:
            dx = 0.0
        else:
            dx = (value - prev_value) * self.freq
        
        # 对速度进行低通滤波
        edx = dx_filter.filter(dx, self._alpha(self.d_cutoff))
        
        # 根据速度计算自适应截止频率
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # 对位置进行低通滤波
        return x_filter.filter(value, self._alpha(cutoff))
    
    def push(self, value: Sequence[float], timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        输入新的坐标点，返回平滑后的坐标
        
        Args:
            value: (x, y) 坐标
            timestamp: 可选的时间戳（秒），用于计算精确的频率
        
        Returns:
            平滑后的 (x, y) 坐标
        """
        x, y = float(value[0]), float(value[1])
        
        # 如果提供了时间戳，动态计算频率
        if timestamp is not None and self._prev_time is not None:
            dt = timestamp - self._prev_time
            if dt > 0:
                self.freq = 1.0 / dt
            self._prev_time = timestamp
        elif timestamp is not None:
            self._prev_time = timestamp
        
        # 分别对 x, y 进行滤波
        filtered_x = self._filter_1d(x, self._prev_x, self._x_filter, self._dx_filter)
        filtered_y = self._filter_1d(y, self._prev_y, self._y_filter, self._dy_filter)
        
        self._prev_x = x
        self._prev_y = y
        
        return filtered_x, filtered_y


class AdaptiveEmaFilter:
    """
    自适应 EMA 滤波器 - 简化版的速度自适应平滑
    
    特点：
    - 根据移动速度动态调整 alpha
    - 速度快时 alpha 接近 1（更跟手）
    - 速度慢时 alpha 较小（更平滑）
    """
    
    def __init__(
        self,
        min_alpha: float = 0.3,  # 最小 alpha（静止时）
        max_alpha: float = 0.9,  # 最大 alpha（快速移动时）
        speed_threshold: float = 50.0  # 速度阈值（像素/帧）
    ):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.speed_threshold = speed_threshold
        self._state: Optional[np.ndarray] = None
    
    def reset(self) -> None:
        self._state = None
    
    def push(self, value: Sequence[float]) -> Tuple[float, float]:
        arr = np.array(value, dtype=np.float32)
        
        if self._state is None:
            self._state = arr
            return float(arr[0]), float(arr[1])
        
        # 计算移动距离（速度近似）
        delta = np.linalg.norm(arr - self._state)
        
        # 根据速度计算自适应 alpha
        # 速度越快，alpha 越接近 max_alpha（更跟手）
        speed_ratio = min(1.0, delta / self.speed_threshold)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * speed_ratio
        
        # 应用 EMA
        self._state = alpha * arr + (1 - alpha) * self._state
        
        return float(self._state[0]), float(self._state[1])


class KalmanFilter2D:
    """
    2D卡尔曼滤波器 - 用于手部坐标跟踪
    
    状态向量: [x, y, vx, vy] (位置 + 速度)
    
    优势：
    - 预测性：可以预测下一帧位置，减少延迟
    - 平滑性：自动过滤测量噪声
    - 速度感知：利用速度信息提高准确性
    
    参考：阿里云美效SDK的卡尔曼滤波 + 匈牙利算法目标跟踪
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0 / 30.0  # 帧间隔（秒）
    ):
        self.dt = dt
        
        # 状态向量 [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float64)
        
        # 状态转移矩阵 F
        # x' = x + vx * dt
        # y' = y + vy * dt
        # vx' = vx
        # vy' = vy
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # 观测矩阵 H（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # 过程噪声协方差 Q
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        
        # 测量噪声协方差 R
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        
        # 估计误差协方差 P
        self.P = np.eye(4, dtype=np.float64)
        
        # 是否已初始化
        self._initialized = False
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self.state = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self._initialized = False
    
    def predict(self) -> Tuple[float, float]:
        """
        预测下一状态
        
        Returns:
            预测的 (x, y) 位置
        """
        # 状态预测: x' = F * x
        self.state = self.F @ self.state
        
        # 协方差预测: P' = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return float(self.state[0]), float(self.state[1])
    
    def update(self, measurement: Sequence[float]) -> Tuple[float, float]:
        """
        用测量值更新状态
        
        Args:
            measurement: (x, y) 测量坐标
        
        Returns:
            更新后的 (x, y) 位置
        """
        z = np.array(measurement, dtype=np.float64)
        
        if not self._initialized:
            # 第一次测量：直接初始化状态
            self.state[0] = z[0]
            self.state[1] = z[1]
            self.state[2] = 0.0  # 初始速度为0
            self.state[3] = 0.0
            self._initialized = True
            return float(z[0]), float(z[1])
        
        # 预测步骤
        self.predict()
        
        # 计算卡尔曼增益: K = P * H^T * (H * P * H^T + R)^-1
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新: x = x + K * (z - H * x)
        y = z - self.H @ self.state  # 测量残差
        self.state = self.state + K @ y
        
        # 协方差更新: P = (I - K * H) * P
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        
        return float(self.state[0]), float(self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取当前估计的速度"""
        return float(self.state[2]), float(self.state[3])
    
    def push(self, value: Sequence[float]) -> Tuple[float, float]:
        """兼容接口：输入测量值，返回滤波后的位置"""
        return self.update(value)


class TemporalFilter:
    """
    时域滤波器 - 基于历史帧平滑当前坐标
    
    使用加权移动平均，近期帧权重更高
    
    参考：阿里云美效SDK的时域滤波模块
    """
    
    def __init__(self, window_size: int = 5):
        """
        初始化时域滤波器
        
        Args:
            window_size: 历史帧窗口大小
        """
        from collections import deque
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        
        # 生成权重（指数增长，近期帧权重更高）
        raw_weights = [math.exp(i * 0.5) for i in range(window_size)]
        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights]
    
    def reset(self) -> None:
        """重置滤波器"""
        self.history.clear()
    
    def push(self, value: Sequence[float]) -> Tuple[float, float]:
        """
        输入新的坐标点，返回平滑后的坐标
        
        Args:
            value: (x, y) 坐标
        
        Returns:
            平滑后的 (x, y) 坐标
        """
        point = (float(value[0]), float(value[1]))
        self.history.append(point)
        
        if len(self.history) == 1:
            return point
        
        # 加权平均
        n = len(self.history)
        weights = self.weights[-n:]  # 取最后n个权重
        weight_sum = sum(weights)
        
        x = sum(p[0] * w for p, w in zip(self.history, weights)) / weight_sum
        y = sum(p[1] * w for p, w in zip(self.history, weights)) / weight_sum
        
        return x, y


class CombinedFilter:
    """
    组合滤波器 - 卡尔曼滤波 + 时域滤波
    
    先用卡尔曼滤波进行预测和状态估计，
    再用时域滤波进一步平滑输出
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        temporal_window: int = 3
    ):
        self.kalman = KalmanFilter2D(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        self.temporal = TemporalFilter(window_size=temporal_window)
    
    def reset(self) -> None:
        """重置所有滤波器"""
        self.kalman.reset()
        self.temporal.reset()
    
    def push(self, value: Sequence[float]) -> Tuple[float, float]:
        """
        输入测量值，返回滤波后的位置
        
        Args:
            value: (x, y) 测量坐标
        
        Returns:
            滤波后的 (x, y) 坐标
        """
        # 先用卡尔曼滤波
        kalman_result = self.kalman.update(value)
        # 再用时域滤波平滑
        return self.temporal.push(kalman_result)
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取卡尔曼滤波估计的速度"""
        return self.kalman.get_velocity()
