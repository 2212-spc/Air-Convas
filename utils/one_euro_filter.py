"""
One Euro Filter - 自适应低延迟滤波器
专为人机交互设计，被广泛应用于 VR/AR 设备（Meta Quest、Apple Vision Pro）

原理：
- 快速移动时：降低平滑强度，保持低延迟
- 慢速/静止时：增强平滑强度，消除抖动

论文：https://cristal.univ-lille.fr/~casiez/1euro/
"""
import math
from typing import Optional


class LowPassFilter:
    """低通滤波器（指数移动平均的变体）"""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev: Optional[float] = None
    
    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is None:
            alpha = self.alpha
        
        if self.prev is None:
            self.prev = value
        else:
            self.prev = alpha * value + (1 - alpha) * self.prev
        
        return self.prev
    
    def reset(self) -> None:
        self.prev = None


class OneEuroFilter:
    """
    One Euro Filter 实现
    
    参数说明：
    - freq: 采样频率（Hz），如 30fps = 30
    - min_cutoff: 最小截止频率，控制最小平滑强度（越小越平滑，默认1.0）
    - beta: 速度系数，控制对速度的响应（越大对速度越敏感，默认0.007）
    - d_cutoff: 速度滤波的截止频率（默认1.0）
    """
    
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(self.d_cutoff))
        
        self.prev_value: Optional[float] = None
        self.prev_time: Optional[float] = None
    
    def _alpha(self, cutoff: float) -> float:
        """计算平滑系数"""
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        滤波一个值
        
        Args:
            value: 当前值
            timestamp: 时间戳（秒），如果为None则使用固定频率
        
        Returns:
            平滑后的值
        """
        # 首次调用，直接返回
        if self.prev_value is None:
            self.prev_value = value
            self.prev_time = timestamp
            return value
        
        # 计算时间间隔（用于动态频率）
        if timestamp is not None and self.prev_time is not None:
            te = timestamp - self.prev_time
            if te > 0:
                freq = 1.0 / te
            else:
                freq = self.freq
        else:
            freq = self.freq
        
        # 估计导数（速度）
        dx = (value - self.prev_value) * freq
        
        # 平滑导数
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff))
        
        # 根据速度动态调整截止频率
        # 速度越大，截止频率越高，平滑越弱，延迟越低
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # 应用自适应平滑
        filtered = self.x_filter.filter(value, self._alpha(cutoff))
        
        # 更新状态
        self.prev_value = filtered
        self.prev_time = timestamp
        
        return filtered
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.prev_value = None
        self.prev_time = None


class OneEuroFilter2D:
    """2D 点的 One Euro Filter（用于坐标平滑）"""
    
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self.filter_x = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
    
    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> tuple[float, float]:
        """
        滤波一个2D点
        
        Args:
            x, y: 当前坐标
            timestamp: 时间戳（秒）
        
        Returns:
            (filtered_x, filtered_y)
        """
        filtered_x = self.filter_x.filter(x, timestamp)
        filtered_y = self.filter_y.filter(y, timestamp)
        return filtered_x, filtered_y
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self.filter_x.reset()
        self.filter_y.reset()


# 预设参数配置
class OneEuroPresets:
    """常用的参数预设"""
    
    # 画图模式：平滑优先，但保持一定响应性
    DRAWING = {
        "freq": 30.0,
        "min_cutoff": 1.0,      # 基础平滑
        "beta": 0.007,          # 中等速度响应
        "d_cutoff": 1.0
    }
    
    # 光标模式：响应优先，轻度平滑
    CURSOR = {
        "freq": 30.0,
        "min_cutoff": 1.5,      # 较弱平滑
        "beta": 0.01,           # 较强速度响应
        "d_cutoff": 1.0
    }
    
    # 激光笔模式：极致跟手
    LASER = {
        "freq": 30.0,
        "min_cutoff": 2.0,      # 最弱平滑
        "beta": 0.015,          # 最强速度响应
        "d_cutoff": 1.0
    }
    
    # 高精度模式：强平滑，适合精细绘制
    PRECISION = {
        "freq": 30.0,
        "min_cutoff": 0.5,      # 强平滑
        "beta": 0.005,          # 弱速度响应
        "d_cutoff": 1.0
    }

