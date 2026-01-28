# -*- coding: utf-8 -*-
"""
3D粒子系统 - 完整版
支持透视投影、景深、发光效果、自动旋转
"""
import cv2
import numpy as np
import time
from typing import Tuple, List
from modules.particle_models_3d import Particle3D, ParticleModel3DLibrary


def perspective_projection(x: float, y: float, z: float, 
                          fov: float = 500, 
                          distance: float = 5.0) -> Tuple[float, float, float]:
    """
    3D透视投影到2D屏幕
    返回 (screen_x, screen_y, depth)
    """
    # 相机距离
    z_cam = z + distance

    # 防止除零
    if z_cam <= 0.1:
        z_cam = 0.1

    # 透视投影
    scale = fov / z_cam
    screen_x = x * scale
    screen_y = y * scale

    return screen_x, screen_y, z_cam


def rotate_3d(x: float, y: float, z: float, 
              angle_x: float, angle_y: float, angle_z: float) -> Tuple[float, float, float]:
    """
    3D旋转变换
    角度单位：弧度
    """
    # 绕X轴旋转
    if angle_x != 0:
        cos_x = np.cos(angle_x)
        sin_x = np.sin(angle_x)
        y_new = y * cos_x - z * sin_x
        z_new = y * sin_x + z * cos_x
        y, z = y_new, z_new

    # 绕Y轴旋转
    if angle_y != 0:
        cos_y = np.cos(angle_y)
        sin_y = np.sin(angle_y)
        x_new = x * cos_y + z * sin_y
        z_new = -x * sin_y + z * cos_y
        x, z = x_new, z_new

    # 绕Z轴旋转
    if angle_z != 0:
        cos_z = np.cos(angle_z)
        sin_z = np.sin(angle_z)
        x_new = x * cos_z - y * sin_z
        y_new = x * sin_z + y * cos_z
        x, y = x_new, y_new

    return x, y, z


class ParticleSystem3D:
    """3D粒子系统"""

    def __init__(self):
        self.active = False
        self.particles: List[Particle3D] = []
        self.current_model = "heart"
        self.current_color = (50, 50, 255)  # BGR 红色

        # 3D相机参数（调整视野以容纳更多粒子）
        self.camera_distance = 3.0
        self.target_camera_distance = 3.0
        self.fov = 700  # 增大FOV，让更多粒子可见

        # 旋转参数（缓慢）
        self.rotation_x = -90.0  # 绕X轴旋转-90度，让爱心竖起来
        self.rotation_y = 180.0  # 绕Y轴180度
        self.rotation_z = 0.0
        self.auto_rotate = True
        self.rotation_speed_x = 0.05  # X轴旋转速度（斜着转）
        self.rotation_speed_y = 0.10  # Y轴旋转速度
        self.rotation_speed_z = 0.03  # Z轴旋转速度

        # 呼吸效果（细微）
        self.breathing_enabled = True
        self.breathing_time = 0.0
        self.breathing_speed = 0.4  # 缓慢呼吸
        self.breathing_amplitude = 0.06  # 细微幅度（6%）
        self.base_scale = 1.0
        self.current_scale = 1.0

        # 手势控制
        self.hand_scale_factor = 1.0

        # 粒子大小（有粗有细，更小更细腻）
        self.glow_intensity = 1.0
        self.particle_size_base = 1  # 基础大小1px（更小）
        self.particle_size_variation = 0.5  # 大小变化范围

        # 性能优化（更快响应）
        self.last_update_time = time.time()
        self.update_interval = 1.0 / 120.0  # 提高到120Hz，更快响应

        # 过渡动画
        self.transitioning = False
        self.transition_progress = 0.0

    def initialize_particles(self, num_particles: int = 5000):
        """初始化3D粒子（优化性能和内存）"""
        self.particles.clear()
        print(f"开始生成 {num_particles} 个粒子的 {self.current_model} 模型...")
        model_points = ParticleModel3DLibrary.get_model(self.current_model, num_particles)

        for i, (x, y, z) in enumerate(model_points):
            particle = Particle3D(x, y, z)
            self.particles.append(particle)

        print(f"✓ 成功初始化 {len(self.particles)} 个3D粒子！")

    def set_model(self, model_name: str):
        """切换3D模型"""
        if model_name == self.current_model:
            return

        old_model = self.current_model
        self.current_model = model_name

        # 获取新模型
        new_points = ParticleModel3DLibrary.get_model(model_name, len(self.particles))

        # 平滑过渡
        for i, particle in enumerate(self.particles):
            if i < len(new_points):
                x, y, z = new_points[i]
                particle.set_target(x, y, z)

        self.transitioning = True
        self.transition_progress = 0.0

        print(f"3D模型切换: {old_model} -> {model_name}")

    def set_color(self, color: Tuple[int, int, int]):
        """设置粒子颜色"""
        self.current_color = color

    def update_hand_control(self, hand_open: float):
        """
        手势控制缩放（极端效果）
        - hand_open = 0.0（完全合拢）→ scale = 1.0（正常大小）
        - hand_open = 1.0（完全张开）→ scale = 8.0（最大放大）
        - 使用二次方映射，让张开更有爆发感
        """
        # 二次方映射：让张开时有更强的爆发感
        # 0.0 -> 1.0, 1.0 -> 8.0
        hand_open_enhanced = hand_open ** 1.5  # 使用1.5次方，让小幅度张开有明显效果
        
        self.hand_scale_factor = 1.0 + 7.0 * hand_open_enhanced
        self.target_camera_distance = 3.0 / self.hand_scale_factor
        self.breathing_enabled = False

    def enable_breathing(self):
        """启用呼吸"""
        self.breathing_enabled = True
    
    def reset_to_normal_size(self):
        """
        重置为正常大小（不呼吸，不缩放）
        用于没有检测到手时的默认状态
        """
        self.breathing_enabled = False
        self.hand_scale_factor = 1.0
        self.target_camera_distance = 3.0

    def update(self, width: int, height: int, has_hand: bool = False):
        """更新3D粒子系统"""
        current_time = time.time()
        dt = current_time - self.last_update_time

        if dt < self.update_interval:
            return

        self.last_update_time = current_time

        # 根据呼吸状态和手势控制更新缩放
        if self.breathing_enabled:
            # 呼吸效果（目前不使用，保留代码）
            self.breathing_time += dt
            breathing_scale = 1.0 + self.breathing_amplitude * np.sin(
                self.breathing_time * self.breathing_speed * 2 * np.pi
            )
            self.current_scale = breathing_scale
            self.target_camera_distance = 3.0
        else:
            # 手势控制或正常状态
            self.current_scale = self.hand_scale_factor

        # 平滑相机距离（合拢时超快，张开时流畅）
        cam_diff = self.target_camera_distance - self.camera_distance
        # 使用非线性插值
        if abs(cam_diff) > 0.01:
            # 收缩时（相机距离增加）使用极快的响应
            if cam_diff > 0:  # 收缩（缩小）
                self.camera_distance += cam_diff * 0.50  # 收缩时极快响应（原0.30）
            else:  # 张开（放大）
                self.camera_distance += cam_diff * 0.25  # 张开时流畅响应（原0.22）
        else:
            self.camera_distance = self.target_camera_distance

        # 自动旋转（多轴，土星斜着转）
        if self.auto_rotate:
            # 只有土星才多轴旋转，爱心和星空只Y轴
            if self.current_model == "saturn":
                self.rotation_x += self.rotation_speed_x * dt * 60
                self.rotation_y += self.rotation_speed_y * dt * 60
                self.rotation_z += self.rotation_speed_z * dt * 60
                if self.rotation_x >= 360:
                    self.rotation_x -= 360
                if self.rotation_z >= 360:
                    self.rotation_z -= 360
            else:
                self.rotation_y += self.rotation_speed_y * dt * 60

            if self.rotation_y >= 360:
                self.rotation_y -= 360

        # 更新粒子（提高smoothness加快响应）
        for particle in self.particles:
            particle.update(dt, smoothness=0.18)  # 从0.12提升到0.18

        # 过渡动画
        if self.transitioning:
            self.transition_progress += 0.03
            if self.transition_progress >= 1.0:
                self.transitioning = False

    def render(self, frame: np.ndarray):
        """渲染3D粒子系统"""
        if len(self.particles) == 0:
            # 没有粒子时，显示提示
            cv2.putText(frame, "Initializing particles...", 
                       (frame.shape[1]//2 - 150, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            return

        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # 创建粒子数据列表（用于Z排序）
        particle_data = []

        for particle in self.particles:
            # 应用缩放
            x = particle.x * self.current_scale
            y = particle.y * self.current_scale
            z = particle.z * self.current_scale

            # 3D旋转（X、Y、Z三轴）
            angle_x = np.radians(self.rotation_x)
            angle_y = np.radians(self.rotation_y)
            angle_z = np.radians(self.rotation_z)
            x, y, z = rotate_3d(x, y, z, angle_x, angle_y, angle_z)

            # 透视投影
            screen_x, screen_y, depth = perspective_projection(
                x, y, z, self.fov, self.camera_distance
            )

            # 转换到屏幕坐标
            px = int(center_x + screen_x)
            py = int(center_y - screen_y)  # Y轴翻转

            # 边界检查
            if 0 <= px < width and 0 <= py < height:
                particle_data.append((px, py, depth, x, y, z))

        # 性能优化：不进行Z排序（节省大量时间）
        # particle_data.sort(key=lambda p: p[2], reverse=True)

        # 渲染粒子
        for i, (px, py, depth, x, y, z) in enumerate(particle_data):
            # 获取粒子的大小因子（每个粒子自带）
            size_factor = 1.0
            if i < len(self.particles):
                size_factor = self.particles[i].size_factor

            # 景深效果（远的粒子更小更暗）
            depth_factor = 1.0 / (1.0 + depth * 0.1)

            # 有粗有细，整体小一圈
            size = max(1, int(self.particle_size_base * depth_factor * size_factor))

            # 颜色随深度变化（近的更亮）
            color_intensity = min(1.0, 0.4 + depth_factor * 0.6)
            color = tuple(int(c * color_intensity) for c in self.current_color)

            # 绘制主粒子（性能优化：去掉抗锯齿）
            cv2.circle(frame, (px, py), size, color, -1)

            # 发光效果（减少发光粒子数量，提升性能）
            if self.glow_intensity > 0 and size >= 2 and i % 3 == 0:  # 只给1/3的粒子加发光
                glow_size = size + 1
                glow_color = tuple(int(c * 0.4) for c in self.current_color)
                cv2.circle(frame, (px, py), glow_size, glow_color, 1)

    def get_particle_count(self) -> int:
        """获取粒子数量"""
        return len(self.particles)

    def reset(self):
        """重置系统"""
        self.particles.clear()
        self.camera_distance = 3.0
        self.rotation_y = 0.0
        self.breathing_time = 0.0
        self.current_scale = 1.0

