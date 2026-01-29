# -*- coding: utf-8 -*-
"""
粒子系统模块 (Particle System)

实现基于物理的 2D 粒子效果，用于增强手指移动时的视觉反馈（拖尾）。
包含简单的牛顿物理模拟（速度、重力）和生命周期管理。
"""

from typing import List, Tuple, Optional, NewType

import numpy as np
import cv2

# [Type Hints] 定义物理计算相关的类型别名
Vector2D = np.ndarray           # 二维向量 [x, y] (float32)
ColorRGB = Tuple[int, int, int] # 颜色元组 (B, G, R)
Position = Tuple[int, int]      # 屏幕坐标 (x, y)


class Particle:
    """
    单个粒子实体 (Particle Entity)
    
    使用 __slots__ 优化内存布局，避免为每个粒子创建 __dict__。
    这对于每帧需要更新和渲染数百个对象的系统至关重要。

    Attributes:
        position (Vector2D): 当前位置 [x, y]。
        velocity (Vector2D): 当前速度向量 [vx, vy]。
        lifetime (float): 剩余生命周期 (秒)。当 < 0 时粒子消亡。
        color (ColorRGB): 粒子基色。
        size (float): 当前渲染半径。
    """
    __slots__ = ['position', 'velocity', 'lifetime', 'color', 'size']
    
    # [Type Hints] 显式声明 __slots__ 属性类型
    position: Vector2D
    velocity: Vector2D
    lifetime: float
    color: ColorRGB
    size: float

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        lifetime: float,
        color: ColorRGB,
        size: float
    ) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.lifetime = lifetime
        self.color = color
        self.size = size


class ParticleSystem:
    """
    粒子系统管理器 (ParticleSystem)
    
    核心特性：
    1. 对象池管理：自动回收死亡粒子，限制最大数量。
    2. 物理模拟：应用重力加速度 (Gravity) 和阻力/衰减 (Fade Rate)。
    3. 视觉渲染：使用多层同心圆模拟发光 (Glow) 效果。

    Attributes:
        max_particles (int): 场景中允许存在的最大粒子数。
        emit_count (int): 每次触发发射时产生的粒子数量。
        gravity (float): 垂直方向的重力加速度分量。
        fade_rate (float): 粒子大小的每帧衰减系数 (0.0~1.0)。
    """

    # [Type Hints] 系统参数类型定义
    max_particles: int
    emit_count: int
    gravity: float
    fade_rate: float
    particles: List[Particle]

    def __init__(
        self,
        max_particles: int = 300,
        emit_count: int = 5,
        gravity: float = 0.1,
        fade_rate: float = 0.98
    ) -> None:
        """
        初始化粒子系统。

        Args:
            max_particles (int): 最大粒子容量。
            emit_count (int): 单次发射量。
            gravity (float): 重力参数 (正值向下)。
            fade_rate (float): 衰减系数 (例如 0.98 代表每帧缩小 2%)。
        """
        self.max_particles = max_particles
        self.emit_count = emit_count
        self.gravity = gravity
        self.fade_rate = fade_rate
        self.particles = []

    def emit(self, position: Position, color: ColorRGB) -> None:
        """
        在指定位置发射一组新粒子。
        
        粒子会被赋予随机的初速度方向、大小和生命周期，
        并对基础颜色进行微小的随机扰动，以产生自然的色彩丰富度。

        Args:
            position (Position): 发射源坐标 (x, y)。
            color (ColorRGB): 粒子的基础颜色 (B, G, R)。
        """
        # 防止粒子数量过多
        if len(self.particles) >= self.max_particles:
            # 移除最老的粒子腾出空间 (FIFO 策略)
            num_to_remove = self.emit_count
            if len(self.particles) > self.max_particles - num_to_remove:
                self.particles = self.particles[num_to_remove:]

        for _ in range(self.emit_count):
            # 随机速度和方向
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 2.0)
            velocity = (
                speed * np.cos(angle),
                speed * np.sin(angle)
            )

            # 随机生命周期
            lifetime = np.random.uniform(0.5, 1.5)

            # 随机大小
            size = np.random.uniform(2.0, 5.0)

            # 颜色带随机变化 (Color Jitter)
            r, g, b = color
            color_var = (
                max(0, min(255, r + np.random.randint(-30, 30))),
                max(0, min(255, g + np.random.randint(-30, 30))),
                max(0, min(255, b + np.random.randint(-30, 30)))
            )

            particle = Particle(
                position=(float(position[0]), float(position[1])),
                velocity=velocity,
                lifetime=lifetime,
                color=color_var,
                size=size
            )
            self.particles.append(particle)

    def update(self, dt: float = 0.016) -> None:
        """
        更新物理状态。

        包含：
        1. 位置更新 (p = p + v)
        2. 速度更新 (v.y = v.y + g)
        3. 生命周期与大小衰减
        4. 清理死亡粒子

        Args:
            dt (float): 时间步长（当前版本未使用，预留接口）。
        """
        alive_particles: List[Particle] = []

        for particle in self.particles:
            # 更新位置
            particle.position += particle.velocity

            # 应用重力
            particle.velocity[1] += self.gravity

            # 更新生命周期
            particle.lifetime -= dt

            # 大小衰减
            particle.size *= self.fade_rate

            # 只保留存活的粒子
            if particle.lifetime > 0 and particle.size > 0.5:
                alive_particles.append(particle)

        self.particles = alive_particles

    def render(self, frame: np.ndarray) -> None:
        """
        将粒子渲染到 OpenCV 帧上。

        使用 Alpha Blending 模拟发光效果：
        根据剩余生命周期计算透明度，绘制多层不同大小和透明度的圆，
        形成中心亮、边缘柔和的光晕效果。

        Args:
            frame (np.ndarray): 目标图像帧。
        """
        h, w = frame.shape[:2]

        for particle in self.particles:
            # 计算透明度（基于剩余生命）
            alpha = min(1.0, particle.lifetime)

            # 粒子位置
            x, y = int(particle.position[0]), int(particle.position[1])

            # 严格边界检查（包括粒子半径）
            radius = int(particle.size)
            if x < -radius or y < -radius or x >= w + radius or y >= h + radius:
                continue

            # 绘制发光效果（多层圆）
            try:
                # 外层（半透明大圆）
                outer_color = tuple(int(c * alpha * 0.3) for c in particle.color)
                cv2.circle(frame, (x, y), radius + 3, outer_color, -1, lineType=cv2.LINE_AA)

                # 中层
                mid_color = tuple(int(c * alpha * 0.6) for c in particle.color)
                cv2.circle(frame, (x, y), radius + 1, mid_color, -1, lineType=cv2.LINE_AA)

                # 内层（亮核心）
                inner_color = tuple(int(c * alpha) for c in particle.color)
                cv2.circle(frame, (x, y), radius, inner_color, -1, lineType=cv2.LINE_AA)
            except Exception:
                continue

    def clear(self) -> None:
        """清除所有粒子"""
        self.particles.clear()

    def get_count(self) -> int:
        """获取当前粒子数量"""
        return len(self.particles)