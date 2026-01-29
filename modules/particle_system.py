# -*- coding: utf-8 -*-
"""粒子系统模块 - 为手指移动添加拖尾粒子效果"""

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
    
    使用 __slots__ 限制属性动态绑定，大幅减少内存占用，
    适合在每一帧都需要处理数百个实例的场景。
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
    
    负责粒子的发射、物理更新（重力/衰减）和渲染循环。
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
        self.max_particles = max_particles
        self.emit_count = emit_count
        self.gravity = gravity
        self.fade_rate = fade_rate
        self.particles = []

    def emit(self, position: Position, color: ColorRGB) -> None:
        """在指定位置发射粒子"""
        # 防止粒子数量过多
        if len(self.particles) >= self.max_particles:
            # 移除最老的粒子腾出空间
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

            # 颜色带随机变化
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
        """更新所有粒子状态"""
        # 向量化更新（更高效）
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
        """渲染所有粒子到画面"""
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
            # 使用 try-except 防止绘制时的边界问题
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
                # 忽略绘制错误，继续处理其他粒子
                continue

    def clear(self) -> None:
        """清除所有粒子"""
        self.particles.clear()

    def get_count(self) -> int:
        """获取当前粒子数量"""
        return len(self.particles)