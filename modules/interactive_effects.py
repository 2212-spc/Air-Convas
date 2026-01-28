# -*- coding: utf-8 -*-
"""
互动特效模块 - 手势控制的视觉表演效果（增强版）

包含:
1. 互动粒子场 - 华丽的粒子系统，带拖尾和发光
2. 海浪效果 - 波涛汹涌的大海，有浪花和泡沫
3. 星空漩涡 - 壮观的星系漩涡效果
"""

import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Optional


class InteractiveParticleField:
    """
    增强版互动粒子场
    
    特点:
    - 500+粒子，带拖尾轨迹
    - 发光效果和颜色渐变
    - 粒子之间有连线效果
    - 更强的力场交互
    """
    
    def __init__(self, width: int = 1280, height: int = 720, num_particles: int = 400):
        self.width = width
        self.height = height
        self.num_particles = num_particles
        
        # 粒子属性: x, y, vx, vy, size, hue, life, trail_len
        self.particles = np.zeros((num_particles, 8), dtype=np.float32)
        # 粒子轨迹 (每个粒子最多10个历史点)
        self.trails = np.zeros((num_particles, 10, 2), dtype=np.float32)
        self.trail_idx = np.zeros(num_particles, dtype=np.int32)
        
        self._init_particles()
        
        # 力场参数
        self.repel_radius = 200
        self.attract_radius = 250
        self.repel_strength = 15.0
        self.attract_strength = 8.0
        self.friction = 0.97
        self.max_speed = 20.0
        
        # 连线距离阈值
        self.connection_dist = 80
        
    def _init_particles(self):
        """初始化粒子"""
        self.particles[:, 0] = np.random.uniform(0, self.width, self.num_particles)   # x
        self.particles[:, 1] = np.random.uniform(0, self.height, self.num_particles)  # y
        self.particles[:, 2] = np.random.uniform(-2, 2, self.num_particles)           # vx
        self.particles[:, 3] = np.random.uniform(-2, 2, self.num_particles)           # vy
        self.particles[:, 4] = np.random.uniform(2, 5, self.num_particles)            # size
        self.particles[:, 5] = np.random.uniform(160, 200, self.num_particles)        # hue (青蓝色系)
        self.particles[:, 6] = np.random.uniform(0.5, 1.0, self.num_particles)        # life/brightness
        self.particles[:, 7] = np.random.randint(5, 10, self.num_particles)           # trail_len
        
        # 初始化轨迹
        for i in range(self.num_particles):
            self.trails[i, :, 0] = self.particles[i, 0]
            self.trails[i, :, 1] = self.particles[i, 1]
        
    def update(self, hand_pos: Optional[Tuple[int, int]] = None, 
               is_open_palm: bool = False, is_pinching: bool = False):
        """更新粒子"""
        
        # 更新轨迹
        for i in range(self.num_particles):
            idx = self.trail_idx[i]
            self.trails[i, idx, 0] = self.particles[i, 0]
            self.trails[i, idx, 1] = self.particles[i, 1]
            self.trail_idx[i] = (idx + 1) % 10
        
        # 手势力场
        if hand_pos is not None:
            hx, hy = hand_pos
            dx = self.particles[:, 0] - hx
            dy = self.particles[:, 1] - hy
            dist = np.sqrt(dx * dx + dy * dy) + 0.1
            nx = dx / dist
            ny = dy / dist
            
            if is_open_palm:
                # 爆炸式推开
                mask = dist < self.repel_radius
                force = self.repel_strength * np.power(1 - dist[mask] / self.repel_radius, 2)
                self.particles[mask, 2] += nx[mask] * force
                self.particles[mask, 3] += ny[mask] * force
                # 推开时粒子变亮
                self.particles[mask, 6] = np.minimum(1.0, self.particles[mask, 6] + 0.1)
                
            elif is_pinching:
                # 漩涡式吸引
                mask = dist < self.attract_radius
                force = self.attract_strength * (1 - dist[mask] / self.attract_radius)
                # 添加切向力形成漩涡
                tangent_x = -ny[mask]
                tangent_y = nx[mask]
                self.particles[mask, 2] += (-nx[mask] * force * 0.7 + tangent_x * force * 0.5)
                self.particles[mask, 3] += (-ny[mask] * force * 0.7 + tangent_y * force * 0.5)
            else:
                # 轻微吸引
                mask = dist < self.attract_radius * 0.4
                force = 2.0 * (1 - dist[mask] / (self.attract_radius * 0.4))
                self.particles[mask, 2] -= nx[mask] * force
                self.particles[mask, 3] -= ny[mask] * force
        
        # 摩擦力
        self.particles[:, 2] *= self.friction
        self.particles[:, 3] *= self.friction
        
        # 限速
        speed = np.sqrt(self.particles[:, 2]**2 + self.particles[:, 3]**2)
        mask = speed > self.max_speed
        self.particles[mask, 2] *= self.max_speed / speed[mask]
        self.particles[mask, 3] *= self.max_speed / speed[mask]
        
        # 更新位置
        self.particles[:, 0] += self.particles[:, 2]
        self.particles[:, 1] += self.particles[:, 3]
        
        # 边界环绕
        self.particles[:, 0] = np.mod(self.particles[:, 0], self.width)
        self.particles[:, 1] = np.mod(self.particles[:, 1], self.height)
        
        # 色相缓慢变化
        self.particles[:, 5] = np.mod(self.particles[:, 5] + 0.3, 180)
        
        # 亮度衰减
        self.particles[:, 6] = np.maximum(0.3, self.particles[:, 6] - 0.005)
        
    def render(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]] = None,
               is_open_palm: bool = False, is_pinching: bool = False):
        """渲染粒子"""
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        
        # 绘制力场
        if hand_pos is not None:
            hx, hy = int(hand_pos[0]), int(hand_pos[1])
            if 0 <= hx < w and 0 <= hy < h:
                if is_open_palm:
                    # 爆炸波纹
                    for r in range(4):
                        radius = int(self.repel_radius * (0.3 + r * 0.25))
                        alpha = 0.4 - r * 0.1
                        cv2.circle(overlay, (hx, hy), radius, 
                                  (0, int(100 * alpha), int(255 * alpha)), 2, cv2.LINE_AA)
                elif is_pinching:
                    # 漩涡效果
                    for r in range(5):
                        radius = int(self.attract_radius * (0.2 + r * 0.15))
                        alpha = 0.5 - r * 0.1
                        cv2.circle(overlay, (hx, hy), radius,
                                  (int(255 * alpha), int(200 * alpha), 0), 1, cv2.LINE_AA)
        
        # 绘制粒子连线（性能优化：只检查部分粒子）
        sample_size = min(100, self.num_particles)
        sample_idx = np.random.choice(self.num_particles, sample_size, replace=False)
        for i in sample_idx:
            x1, y1 = int(self.particles[i, 0]), int(self.particles[i, 1])
            if not (0 <= x1 < w and 0 <= y1 < h):
                continue
            for j in range(i + 1, min(i + 20, self.num_particles)):
                x2, y2 = int(self.particles[j, 0]), int(self.particles[j, 1])
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < self.connection_dist:
                    alpha = 1 - dist / self.connection_dist
                    color = (int(100 * alpha), int(150 * alpha), int(200 * alpha))
                    cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        # 绘制粒子轨迹
        for i in range(self.num_particles):
            trail_len = int(self.particles[i, 7])
            hue = int(self.particles[i, 5])
            brightness = self.particles[i, 6]
            
            # 轨迹点
            for t in range(trail_len - 1):
                idx1 = (self.trail_idx[i] - t - 1) % 10
                idx2 = (self.trail_idx[i] - t - 2) % 10
                x1, y1 = int(self.trails[i, idx1, 0]), int(self.trails[i, idx1, 1])
                x2, y2 = int(self.trails[i, idx2, 0]), int(self.trails[i, idx2, 1])
                
                if (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                    alpha = (trail_len - t) / trail_len * brightness * 0.5
                    hsv = np.uint8([[[hue, 255, int(255 * alpha)]]])
                    color = tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))
                    cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        # 绘制粒子本体
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            size = int(self.particles[i, 4])
            hue = int(self.particles[i, 5])
            brightness = self.particles[i, 6]
            
            if 0 <= x < w and 0 <= y < h:
                # HSV转BGR
                hsv = np.uint8([[[hue, 255, int(255 * brightness)]]])
                color = tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))
                
                # 外发光
                cv2.circle(overlay, (x, y), size + 3, tuple(c // 3 for c in color), -1, cv2.LINE_AA)
                # 核心
                cv2.circle(overlay, (x, y), size, color, -1, cv2.LINE_AA)
                # 高光
                cv2.circle(overlay, (x, y), max(1, size // 2), (255, 255, 255), -1, cv2.LINE_AA)
        
        cv2.addWeighted(frame, 0.7, overlay, 0.8, 0, frame)


class WaveEffect:
    """
    增强版海浪效果 - 波涛汹涌的大海
    
    特点:
    - 多层叠加波浪
    - 浪花和泡沫效果
    - 手势影响波浪高度和强度
    - 渐变天空背景
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.time = 0.0
        
        # 波浪参数 (更多层次)
        self.waves = [
            # 远景波浪 (慢，小)
            {"amp": 15, "freq": 0.008, "speed": 0.8, "phase": 0, "y": 0.45, "color": (80, 60, 40)},
            {"amp": 20, "freq": 0.012, "speed": 1.0, "phase": 1.5, "y": 0.50, "color": (100, 80, 50)},
            # 中景波浪
            {"amp": 30, "freq": 0.015, "speed": 1.5, "phase": 0.5, "y": 0.58, "color": (120, 100, 60)},
            {"amp": 35, "freq": 0.02, "speed": 1.8, "phase": 2.0, "y": 0.65, "color": (150, 120, 70)},
            # 近景波浪 (快，大)
            {"amp": 45, "freq": 0.025, "speed": 2.2, "phase": 1.0, "y": 0.75, "color": (180, 150, 90)},
            {"amp": 50, "freq": 0.03, "speed": 2.5, "phase": 2.5, "y": 0.85, "color": (200, 170, 100)},
        ]
        
        # 浪花粒子
        self.foam_particles = []
        self.max_foam = 100
        
        # 手势影响
        self.hand_x = width // 2
        self.hand_y = height // 2
        self.hand_influence = 0.0
        self.storm_level = 0.0  # 风暴程度
        
    def update(self, hand_pos: Optional[Tuple[int, int]] = None, dt: float = 0.033):
        """更新波浪"""
        self.time += dt
        
        if hand_pos is not None:
            self.hand_x, self.hand_y = hand_pos
            self.hand_influence = min(1.0, self.hand_influence + 0.08)
            # 手越低，风暴越强
            self.storm_level = min(1.0, (self.hand_y / self.height) * self.hand_influence)
        else:
            self.hand_influence = max(0.0, self.hand_influence - 0.03)
            self.storm_level = max(0.0, self.storm_level - 0.02)
        
        # 生成浪花
        if len(self.foam_particles) < self.max_foam and np.random.random() < 0.3 + self.storm_level * 0.5:
            self.foam_particles.append({
                "x": np.random.uniform(0, self.width),
                "y": np.random.uniform(self.height * 0.5, self.height * 0.9),
                "vx": np.random.uniform(-2, 2),
                "vy": np.random.uniform(-3, -1),
                "size": np.random.uniform(2, 6),
                "life": np.random.uniform(0.5, 1.5),
                "max_life": 1.5,
            })
        
        # 更新浪花
        alive_foam = []
        for f in self.foam_particles:
            f["life"] -= dt
            f["x"] += f["vx"]
            f["y"] += f["vy"]
            f["vy"] += 0.5  # 重力
            if f["life"] > 0 and 0 <= f["x"] < self.width and 0 <= f["y"] < self.height:
                alive_foam.append(f)
        self.foam_particles = alive_foam
            
    def render(self, frame: np.ndarray):
        """渲染波浪"""
        h, w = frame.shape[:2]
        
        # 绘制渐变天空
        for y in range(int(h * 0.45)):
            progress = y / (h * 0.45)
            # 日落渐变
            r = int(30 + progress * 50)
            g = int(20 + progress * 40)
            b = int(60 + progress * 30)
            frame[y, :] = (b, g, r)
        
        # 绘制波浪
        for wave in self.waves:
            points = []
            base_y = int(h * wave["y"])
            amp = wave["amp"] * (1 + self.storm_level * 0.8)
            freq = wave["freq"]
            speed = wave["speed"] * (1 + self.storm_level * 0.5)
            
            for x in range(0, w + 5, 4):
                # 多重正弦叠加
                y = base_y
                y += amp * math.sin(freq * x + self.time * speed + wave["phase"])
                y += amp * 0.5 * math.sin(freq * 2 * x + self.time * speed * 1.3)
                y += amp * 0.25 * math.sin(freq * 3 * x + self.time * speed * 0.7)
                
                # 手势影响
                if self.hand_influence > 0:
                    dist = abs(x - self.hand_x)
                    if dist < 200:
                        influence = (1 - dist / 200) * self.hand_influence
                        y -= influence * 60 * (1 + self.storm_level)
                
                points.append((x, int(y)))
            
            # 闭合
            points.append((w, h))
            points.append((0, h))
            
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            
            # 绘制波浪主体
            color = wave["color"]
            cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
            
            # 波浪顶部高光
            highlight = tuple(min(255, c + 40) for c in color)
            for i in range(len(points) - 3):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if y1 < base_y:  # 只在波峰画高光
                    cv2.line(frame, (x1, y1), (x2, y2), highlight, 2, cv2.LINE_AA)
        
        # 绘制浪花/泡沫
        for f in self.foam_particles:
            x, y = int(f["x"]), int(f["y"])
            size = int(f["size"] * (f["life"] / f["max_life"]))
            alpha = f["life"] / f["max_life"]
            if size > 0 and 0 <= x < w and 0 <= y < h:
                color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)


class StarVortex:
    """
    增强版星空漩涡 - 壮观的星系效果
    
    特点:
    - 更多星星，多层次
    - 流星效果
    - 漩涡更加壮观
    - 星云背景
    """
    
    def __init__(self, width: int = 1280, height: int = 720, num_stars: int = 500):
        self.width = width
        self.height = height
        self.num_stars = num_stars
        
        # 星星: x, y, base_x, base_y, size, brightness, angle, layer, color_hue
        self.stars = np.zeros((num_stars, 9), dtype=np.float32)
        self._init_stars()
        
        self.vortex_center = None
        self.vortex_strength = 0.0
        self.time = 0.0
        
        # 流星
        self.meteors = []
        self.max_meteors = 5
        
    def _init_stars(self):
        """初始化星星"""
        self.stars[:, 0] = np.random.uniform(0, self.width, self.num_stars)   # x
        self.stars[:, 1] = np.random.uniform(0, self.height, self.num_stars)  # y
        self.stars[:, 2] = self.stars[:, 0].copy()  # base_x
        self.stars[:, 3] = self.stars[:, 1].copy()  # base_y
        self.stars[:, 4] = np.random.uniform(1, 4, self.num_stars)            # size
        self.stars[:, 5] = np.random.uniform(0.2, 1.0, self.num_stars)        # brightness
        self.stars[:, 6] = np.random.uniform(0, 2 * math.pi, self.num_stars)  # angle
        self.stars[:, 7] = np.random.randint(0, 3, self.num_stars)            # layer (0=远, 2=近)
        self.stars[:, 8] = np.random.choice([0, 30, 60, 120], self.num_stars) # hue (不同颜色的星)
        
    def update(self, hand_pos: Optional[Tuple[int, int]] = None, 
               is_pinching: bool = False, dt: float = 0.033):
        """更新星空"""
        self.time += dt
        
        if hand_pos is not None and is_pinching:
            self.vortex_center = hand_pos
            self.vortex_strength = min(1.0, self.vortex_strength + 0.08)
        else:
            self.vortex_strength = max(0.0, self.vortex_strength - 0.02)
        
        if self.vortex_center is not None and self.vortex_strength > 0.1:
            cx, cy = self.vortex_center
            dx = self.stars[:, 0] - cx
            dy = self.stars[:, 1] - cy
            dist = np.sqrt(dx * dx + dy * dy) + 1
            
            max_dist = 400
            mask = dist < max_dist
            
            angles = np.arctan2(dy, dx)
            
            # 根据层次不同速度旋转
            layer_speed = 1 + self.stars[mask, 7] * 0.5
            rotation_speed = 5.0 * self.vortex_strength * (1 - dist[mask] / max_dist) * layer_speed
            angles[mask] += rotation_speed * dt
            
            # 向心吸引
            attract = 80 * self.vortex_strength * (1 - dist[mask] / max_dist)
            new_dist = np.maximum(30, dist[mask] - attract * dt)
            
            self.stars[mask, 0] = cx + np.cos(angles[mask]) * new_dist
            self.stars[mask, 1] = cy + np.sin(angles[mask]) * new_dist
        else:
            # 缓慢回位
            self.stars[:, 0] += (self.stars[:, 2] - self.stars[:, 0]) * 0.01
            self.stars[:, 1] += (self.stars[:, 3] - self.stars[:, 1]) * 0.01
        
        # 闪烁
        self.stars[:, 5] = 0.4 + 0.6 * np.abs(np.sin(self.time * 2 + self.stars[:, 6]))
        
        # 流星
        if len(self.meteors) < self.max_meteors and np.random.random() < 0.02:
            self.meteors.append({
                "x": np.random.uniform(0, self.width),
                "y": 0,
                "vx": np.random.uniform(-5, -2),
                "vy": np.random.uniform(8, 15),
                "length": np.random.uniform(50, 150),
                "life": 1.0,
            })
        
        # 更新流星
        alive = []
        for m in self.meteors:
            m["x"] += m["vx"]
            m["y"] += m["vy"]
            m["life"] -= dt * 0.5
            if m["life"] > 0 and m["y"] < self.height:
                alive.append(m)
        self.meteors = alive
            
    def render(self, frame: np.ndarray):
        """渲染星空"""
        h, w = frame.shape[:2]
        
        # 暗化背景
        frame[:] = (frame * 0.3).astype(np.uint8)
        
        # 星云背景
        if self.vortex_center and self.vortex_strength > 0.2:
            cx, cy = int(self.vortex_center[0]), int(self.vortex_center[1])
            for r in range(8):
                radius = int((8 - r) * 40 * self.vortex_strength)
                alpha = 0.15 * self.vortex_strength * (1 - r / 8)
                color = (int(50 * alpha), int(30 * alpha), int(80 * alpha))
                cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
        
        # 绘制星星（按层次）
        for layer in range(3):
            layer_mask = self.stars[:, 7] == layer
            layer_stars = self.stars[layer_mask]
            
            for star in layer_stars:
                x, y = int(star[0]), int(star[1])
                size = int(star[4] * (1 + layer * 0.3))
                brightness = star[5]
                hue = int(star[8])
                
                if 0 <= x < w and 0 <= y < h:
                    # 颜色
                    if hue == 0:
                        color = (int(255 * brightness), int(255 * brightness), int(255 * brightness))
                    else:
                        hsv = np.uint8([[[hue, 100, int(255 * brightness)]]])
                        color = tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))
                    
                    # 外发光
                    cv2.circle(frame, (x, y), size + 2, tuple(c // 4 for c in color), -1, cv2.LINE_AA)
                    # 核心
                    cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)
                    
                    # 大星星十字光芒
                    if star[4] > 2.5:
                        length = int(size * 3 * brightness)
                        faint = tuple(c // 2 for c in color)
                        cv2.line(frame, (x - length, y), (x + length, y), faint, 1, cv2.LINE_AA)
                        cv2.line(frame, (x, y - length), (x, y + length), faint, 1, cv2.LINE_AA)
        
        # 绘制流星
        for m in self.meteors:
            x, y = int(m["x"]), int(m["y"])
            length = int(m["length"] * m["life"])
            alpha = m["life"]
            
            # 流星尾巴
            end_x = int(x + m["vx"] * length / 10)
            end_y = int(y - m["vy"] * length / 10)
            
            for i in range(5):
                t = i / 5
                px = int(x + (end_x - x) * t)
                py = int(y + (end_y - y) * t)
                if 0 <= px < w and 0 <= py < h:
                    size = int(3 * (1 - t) * alpha)
                    if size > 0:
                        intensity = int(255 * alpha * (1 - t))
                        cv2.circle(frame, (px, py), size, (intensity, intensity, intensity), -1, cv2.LINE_AA)


class InteractiveEffectsManager:
    """互动特效管理器"""
    
    EFFECT_NAMES = ["particles", "waves", "stars"]
    EFFECT_LABELS = {
        "particles": "Particles",
        "waves": "Ocean",
        "stars": "Galaxy"
    }
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        
        self.particle_field = InteractiveParticleField(width, height, num_particles=400)
        self.wave_effect = WaveEffect(width, height)
        self.star_vortex = StarVortex(width, height, num_stars=500)
        
        self.current_effect = "particles"
        self.is_active = False
        self._last_time = time.time()
        
    def toggle(self):
        self.is_active = not self.is_active
        return self.is_active
        
    def next_effect(self):
        idx = self.EFFECT_NAMES.index(self.current_effect)
        idx = (idx + 1) % len(self.EFFECT_NAMES)
        self.current_effect = self.EFFECT_NAMES[idx]
        return self.current_effect
        
    def set_effect(self, effect_name: str):
        if effect_name in self.EFFECT_NAMES:
            self.current_effect = effect_name
            
    def get_effect_label(self) -> str:
        return self.EFFECT_LABELS.get(self.current_effect, self.current_effect)
        
    def update(self, hand_pos: Optional[Tuple[int, int]] = None,
               is_open_palm: bool = False, is_pinching: bool = False):
        if not self.is_active:
            return
            
        current_time = time.time()
        dt = current_time - self._last_time
        self._last_time = current_time
        
        if self.current_effect == "particles":
            self.particle_field.update(hand_pos, is_open_palm, is_pinching)
        elif self.current_effect == "waves":
            self.wave_effect.update(hand_pos, dt)
        elif self.current_effect == "stars":
            self.star_vortex.update(hand_pos, is_pinching, dt)
            
    def render(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]] = None,
               is_open_palm: bool = False, is_pinching: bool = False):
        if not self.is_active:
            return
            
        if self.current_effect == "particles":
            self.particle_field.render(frame, hand_pos, is_open_palm, is_pinching)
        elif self.current_effect == "waves":
            self.wave_effect.render(frame)
        elif self.current_effect == "stars":
            self.star_vortex.render(frame)
            
        # 显示特效名称
        label = f"Effect: {self.get_effect_label()}"
        cv2.putText(frame, label, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
