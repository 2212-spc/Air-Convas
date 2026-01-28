# -*- coding: utf-8 -*-
"""
粒子模型库
定义各种精美的粒子模型轮廓
"""
import numpy as np
from typing import List, Tuple


def generate_heart(num_points: int = 200) -> List[Tuple[float, float]]:
    """生成爱心形状点集"""
    points = []
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        # 心形参数方程
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        # 归一化到 [-1, 1]
        x = x / 20
        y = -y / 20  # 翻转Y轴
        points.append((x, y))
    return points


def generate_flower(num_points: int = 200, petals: int = 6) -> List[Tuple[float, float]]:
    """生成花朵形状点集"""
    points = []
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        # 玫瑰线（花瓣）
        r = np.cos(petals * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        points.append((x, y))
    return points


def generate_star(num_points: int = 200, points_num: int = 5) -> List[Tuple[float, float]]:
    """生成五角星/多角星形状点集"""
    points = []
    outer_radius = 1.0
    inner_radius = 0.4

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        # 判断是外点还是内点
        point_index = (i * points_num * 2) // num_points
        if point_index % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius

        actual_angle = 2 * np.pi * point_index / (points_num * 2)
        x = r * np.sin(actual_angle)
        y = -r * np.cos(actual_angle)
        points.append((x, y))

    return points


def generate_saturn(num_points: int = 200) -> List[Tuple[float, float]]:
    """生成土星形状点集（行星+光环）"""
    points = []
    # 70% 是光环
    ring_points = int(num_points * 0.7)
    planet_points = num_points - ring_points

    # 行星主体（圆形）
    for i in range(planet_points):
        t = 2 * np.pi * i / planet_points
        r = 0.4
        x = r * np.cos(t)
        y = r * np.sin(t)
        points.append((x, y))

    # 光环（椭圆）
    for i in range(ring_points):
        t = 2 * np.pi * i / ring_points
        a = 0.95  # 长轴
        b = 0.2   # 短轴
        x = a * np.cos(t)
        y = b * np.sin(t)
        points.append((x, y))

    return points


def generate_buddha(num_points: int = 200) -> List[Tuple[float, float]]:
    """生成佛像轮廓形状点集（简化版：坐姿轮廓）"""
    points = []

    # 分段构建佛像轮廓
    # 头部（圆形，上半部分）
    head_points = int(num_points * 0.25)
    head_radius = 0.25
    head_center_y = 0.5
    for i in range(head_points):
        t = np.pi + np.pi * i / head_points  # 从180度到360度
        x = head_radius * np.cos(t)
        y = head_center_y + head_radius * np.sin(t)
        points.append((x, y))

    # 身体（梯形轮廓）
    body_points = int(num_points * 0.5)
    for i in range(body_points):
        progress = i / body_points
        if progress < 0.5:
            # 左侧
            x = -0.25 - progress * 1.0
            y = 0.25 - progress * 1.5
        else:
            # 右侧
            x = -0.75 + (progress - 0.5) * 2.0
            y = -0.5 + (progress - 0.5) * 1.5
        points.append((x, y))

    # 底座（椭圆）
    base_points = num_points - head_points - body_points
    for i in range(base_points):
        t = 2 * np.pi * i / base_points
        a = 0.6
        b = 0.15
        x = a * np.cos(t)
        y = -0.7 + b * np.sin(t)
        points.append((x, y))

    return points


def generate_firework(num_points: int = 200, rays: int = 12) -> List[Tuple[float, float]]:
    """生成烟花爆炸形状点集"""
    points = []

    for i in range(num_points):
        # 随机分布在各个射线上
        ray_index = (i * rays) // num_points
        angle = 2 * np.pi * ray_index / rays

        # 沿射线的位置（有一定随机性）
        t = (i % (num_points // rays)) / (num_points // rays)
        r = 0.3 + 0.7 * t

        # 添加一些抖动使其更自然
        angle_jitter = (np.sin(i * 13.7) * 0.1)
        r_jitter = 1.0 + np.sin(i * 7.3) * 0.15

        x = r * r_jitter * np.cos(angle + angle_jitter)
        y = r * r_jitter * np.sin(angle + angle_jitter)
        points.append((x, y))

    return points


def generate_circle(num_points: int = 200) -> List[Tuple[float, float]]:
    """生成圆形点集"""
    points = []
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        x = np.cos(t)
        y = np.sin(t)
        points.append((x, y))
    return points


def generate_spiral(num_points: int = 200, turns: int = 3) -> List[Tuple[float, float]]:
    """生成螺旋形状点集"""
    points = []
    for i in range(num_points):
        t = turns * 2 * np.pi * i / num_points
        r = t / (turns * 2 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        points.append((x, y))
    return points


class ParticleModelLibrary:
    """粒子模型库"""

    MODELS = {
        "heart": generate_heart,
        "flower": generate_flower,
        "star": generate_star,
        "saturn": generate_saturn,
        "buddha": generate_buddha,
        "firework": generate_firework,
        "circle": generate_circle,
        "spiral": generate_spiral,
    }

    @classmethod
    def get_model(cls, name: str, num_points: int = 200) -> List[Tuple[float, float]]:
        """获取模型点集"""
        if name not in cls.MODELS:
            name = "circle"
        return cls.MODELS[name](num_points)

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有可用模型"""
        return list(cls.MODELS.keys())

