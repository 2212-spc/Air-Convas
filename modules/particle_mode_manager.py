"""
粒子模式管理器 - 完整版
支持多种模型、平滑过渡、自呼吸效果、手势控制
"""
import cv2
import numpy as np
import time
from typing import Tuple, List, Optional
from modules.particle_models import ParticleModelLibrary


class Particle:
    """单个粒子"""
    def __init__(self, target_x: float, target_y: float):
        self.target_x = target_x  # 目标位置（归一化坐标 -1 到 1）
        self.target_y = target_y
        self.current_x = target_x
        self.current_y = target_y
        self.vx = 0.0
        self.vy = 0.0
        self.trail: List[Tuple[float, float]] = []
        self.max_trail = 5
        
    def update(self, dt: float = 0.016, smoothness: float = 0.15):
        """更新粒子位置（弹簧效果）"""
        # 计算到目标的距离
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        
        # 弹簧力
        self.vx += dx * smoothness
        self.vy += dy * smoothness
        
        # 阻尼
        damping = 0.85
        self.vx *= damping
        self.vy *= damping
        
        # 更新位置
        self.current_x += self.vx * dt * 60
        self.current_y += self.vy * dt * 60
        
        # 更新拖尾
        self.trail.append((self.current_x, self.current_y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
    
    def set_target(self, x: float, y: float):
        """设置新的目标位置"""
        self.target_x = x
        self.target_y = y


class ParticleModeManager:
    """粒子模式管理器"""
    
    def __init__(self):
        self.active = False
        self.particles: List[Particle] = []
        self.current_model = "heart"
        self.current_color = (255, 100, 150)  # BGR格式
        
        # 中心位置（屏幕比例）
        self.center_x = 0.5
        self.center_y = 0.5
        
        # 缩放和呼吸
        self.base_scale = 200.0  # 基础大小（像素）
        self.current_scale = 200.0
        self.target_scale = 200.0
        self.hand_scale_factor = 1.0  # 手势控制的缩放
        
        # 自呼吸效果
        self.breathing_enabled = True
        self.breathing_time = 0.0
        self.breathing_speed = 1.0  # 呼吸速度
        self.breathing_amplitude = 0.15  # 呼吸幅度（15%）
        
        # 旋转效果
        self.rotation = 0.0
        self.rotation_speed = 0.2  # 度/帧
        
        # 过渡动画
        self.transitioning = False
        self.transition_progress = 0.0
        self.transition_speed = 0.05
        
        # 性能优化
        self.last_update_time = time.time()
        self.update_interval = 1.0 / 60.0  # 60 FPS
        
    def initialize_particles(self, num_particles: int = 300):
        """初始化粒子系统"""
        self.particles.clear()
        model_points = ParticleModelLibrary.get_model(self.current_model, num_particles)
        
        for x, y in model_points:
            particle = Particle(x, y)
            self.particles.append(particle)
        
        print(f"初始化 {len(self.particles)} 个粒子，模型: {self.current_model}")
    
    def set_model(self, model_name: str):
        """切换模型（平滑过渡）"""
        if model_name == self.current_model:
            return
        
        self.current_model = model_name
        
        # 获取新模型的点
        new_points = ParticleModelLibrary.get_model(model_name, len(self.particles))
        
        # 为每个粒子设置新的目标位置
        for i, particle in enumerate(self.particles):
            if i < len(new_points):
                x, y = new_points[i]
                particle.set_target(x, y)
        
        # 开始过渡动画
        self.transitioning = True
        self.transition_progress = 0.0
        
        print(f"切换到模型: {model_name}")
    
    def set_color(self, color: Tuple[int, int, int]):
        """设置粒子颜色（BGR）"""
        self.current_color = color
        print(f"粒子颜色: {color}")
    
    def update_hand_control(self, hand_open: float):
        """
        根据手掌开合程度更新缩放
        hand_open: 0.0 (完全闭合) 到 1.0 (完全张开)
        """
        # 映射到缩放因子：0.5x (闭合) 到 2.0x (张开)
        min_scale = 0.5
        max_scale = 2.0
        self.hand_scale_factor = min_scale + (max_scale - min_scale) * hand_open
        self.target_scale = self.base_scale * self.hand_scale_factor
        self.breathing_enabled = False  # 手势控制时禁用自呼吸
    
    def enable_breathing(self):
        """启用自呼吸效果"""
        self.breathing_enabled = True
    
    def update(self, screen_width: int, screen_height: int, has_hand: bool = False):
        """更新粒子系统"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 限制更新频率
        if dt < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 如果没有手势，启用自呼吸
        if not has_hand:
            self.breathing_enabled = True
        
        # 更新呼吸效果
        if self.breathing_enabled:
            self.breathing_time += dt
            breathing_scale = 1.0 + self.breathing_amplitude * np.sin(
                self.breathing_time * self.breathing_speed * 2 * np.pi
            )
            self.target_scale = self.base_scale * breathing_scale
        
        # 平滑缩放过渡
        scale_diff = self.target_scale - self.current_scale
        self.current_scale += scale_diff * 0.1
        
        # 更新旋转（可选）
        # self.rotation += self.rotation_speed
        # if self.rotation >= 360:
        #     self.rotation -= 360
        
        # 更新过渡进度
        if self.transitioning:
            self.transition_progress += self.transition_speed
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.transitioning = False
        
        # 更新每个粒子
        for particle in self.particles:
            particle.update(dt, smoothness=0.12)
    
    def render(self, frame: np.ndarray):
        """在屏幕中心渲染粒子"""
        height, width = frame.shape[:2]
        
        # 计算中心位置（像素）
        center_x_px = int(width * self.center_x)
        center_y_px = int(height * self.center_y)
        
        # 渲染每个粒子
        for particle in self.particles:
            # 转换归一化坐标到屏幕坐标
            x = center_x_px + int(particle.current_x * self.current_scale)
            y = center_y_px + int(particle.current_y * self.current_scale)
            
            # 边界检查
            if 0 <= x < width and 0 <= y < height:
                # 绘制粒子（圆点）
                particle_size = max(2, int(4 * (self.current_scale / self.base_scale)))
                cv2.circle(frame, (x, y), particle_size, self.current_color, -1, lineType=cv2.LINE_AA)
                
                # 可选：绘制发光效果
                glow_size = particle_size + 2
                glow_color = tuple(int(c * 0.5) for c in self.current_color)
                cv2.circle(frame, (x, y), glow_size, glow_color, 1, lineType=cv2.LINE_AA)
                
                # 可选：绘制拖尾
                if len(particle.trail) > 1:
                    trail_points = []
                    for tx, ty in particle.trail[-3:]:
                        trail_x = center_x_px + int(tx * self.current_scale)
                        trail_y = center_y_px + int(ty * self.current_scale)
                        if 0 <= trail_x < width and 0 <= trail_y < height:
                            trail_points.append((trail_x, trail_y))
                    
                    if len(trail_points) > 1:
                        for i in range(len(trail_points) - 1):
                            alpha = (i + 1) / len(trail_points)
                            color = tuple(int(c * alpha * 0.3) for c in self.current_color)
                            cv2.line(frame, trail_points[i], trail_points[i+1], 
                                   color, 1, lineType=cv2.LINE_AA)
        
        # 绘制中心指示（调试用，可选）
        # cv2.circle(frame, (center_x_px, center_y_px), 5, (0, 255, 0), 2)
    
    def get_particle_count(self) -> int:
        """获取粒子数量"""
        return len(self.particles)
    
    def reset(self):
        """重置粒子系统"""
        self.particles.clear()
        self.current_scale = self.base_scale
        self.target_scale = self.base_scale
        self.breathing_time = 0.0
        self.rotation = 0.0
