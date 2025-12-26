"""
3D体积粒子模型库
使用隐式方程和体积拒绝采样生成真正的3D粒子
"""
import numpy as np
from typing import List, Tuple, Optional


class Particle3D:
    """3D粒子"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.size_factor = 0.4 + np.random.random() * 0.8  # 有粗有细，但整体更小
        
    def update(self, dt: float = 0.016, smoothness: float = 0.12):
        """更新粒子位置（3D弹簧物理）"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z
        
        self.vx += dx * smoothness
        self.vy += dy * smoothness
        self.vz += dz * smoothness
        
        damping = 0.85
        self.vx *= damping
        self.vy *= damping
        self.vz *= damping
        
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60
        self.z += self.vz * dt * 60
    
    def set_target(self, x: float, y: float, z: float):
        """设置新的目标位置"""
        self.target_x = x
        self.target_y = y
        self.target_z = z


def volumetric_heart(num_particles: int = 1000) -> List[Tuple[float, float, float]]:
    """
    3D圆润爱心 - 优化版，外圈密集显示轮廓，内圈稀疏
    """
    particles = []
    attempts = 0
    max_attempts = num_particles * 100
    
    while len(particles) < num_particles and attempts < max_attempts:
        # 在边界框内随机采样
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, 1.5)
        z = np.random.uniform(-1.5, 1.5)
        
        # 更圆润的爱心方程（调整参数让形状更圆）
        term1 = (x**2 + 2.0 * y**2 + z**2 - 1) ** 3  # 减小y系数让更圆
        term2 = x**2 * z**3 + 0.08 * y**2 * z**3  # 减小系数让更平滑
        implicit_value = term1 - term2
        
        # 计算到表面的距离
        dist_to_surface = abs(implicit_value) ** 0.3
        
        # 外圈密集（轮廓清晰）
        if dist_to_surface < 0.15:  # 表面附近
            # 外圈接受概率高（80%）
            if np.random.random() < 0.8:
                particles.append((x, y, z))
        # 内圈稀疏
        elif implicit_value < 0:  # 内部
            # 内圈接受概率低（10%）
            if np.random.random() < 0.1:
                particles.append((x, y, z))
        
        attempts += 1
    
    # 归一化
    if particles:
        particles = np.array(particles)
        max_val = np.max(np.abs(particles))
        if max_val > 0:
            particles = particles / max_val
        return [(x, y, z) for x, y, z in particles]
    
    return [(0, 0, 0)] * num_particles


def volumetric_lotus(num_particles: int = 1000) -> List[Tuple[float, float, float]]:
    """
    3D立体莲花 - 分层数学构建
    底部莲台 + 中层荷叶 + 上层花蕊
    """
    particles = []
    
    # 分配粒子数量
    base_count = int(num_particles * 0.3)    # 30% 莲台
    petals_count = int(num_particles * 0.5)  # 50% 花瓣
    center_count = num_particles - base_count - petals_count  # 20% 花蕊
    
    # 1. 底部莲台（圆盘 + 波浪边缘）
    for i in range(base_count):
        angle = 2 * np.pi * i / base_count
        r = 0.3 + 0.1 * np.random.random()
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = -0.6 + 0.1 * np.sin(8 * angle)  # 波浪边缘
        particles.append((x, y, z))
    
    # 2. 中层花瓣（6瓣，每瓣是椭球的一部分）
    petal_count = 6
    particles_per_petal = petals_count // petal_count
    for petal_idx in range(petal_count):
        base_angle = 2 * np.pi * petal_idx / petal_count
        
        for i in range(particles_per_petal):
            # 花瓣内的位置
            u = np.random.random()  # 径向
            v = np.random.random()  # 高度
            
            # 花瓣形状（椭球切片）
            angle = base_angle + (v - 0.5) * 0.8
            r = 0.4 + 0.5 * u
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = -0.3 + 0.6 * u - 0.3 * u * u  # 抛物线高度
            
            particles.append((x, y, z))
    
    # 3. 上层花蕊（球形簇）
    for i in range(center_count):
        # 球形均匀采样
        phi = np.arccos(2 * np.random.random() - 1)
        theta = 2 * np.pi * np.random.random()
        r = 0.15 * np.random.random() ** (1/3)  # 立方根保证体积均匀
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = 0.3 + r * np.cos(phi)
        
        particles.append((x, y, z))
    
    return particles


def volumetric_star_field(num_particles: int = 1000) -> List[Tuple[float, float, float]]:
    """
    满天星海 - 全屏立方体空间随机分布
    优化版：更多聚类，营造沉浸宇宙感
    """
    particles = []
    
    # 60%核心星海（密集）
    core_count = int(num_particles * 0.6)
    for i in range(core_count):
        # 立方体空间密集分布
        x = np.random.uniform(-1.2, 1.2)
        y = np.random.uniform(-1.2, 1.2)
        z = np.random.uniform(-1.2, 1.2)
        
        # 添加星系团聚类
        if np.random.random() < 0.4:  # 40% 粒子聚集
            cluster_angle = 2 * np.pi * np.random.random()
            cluster_dist = 0.3 * np.random.random()
            x += cluster_dist * np.cos(cluster_angle)
            y += cluster_dist * np.sin(cluster_angle)
        
        particles.append((x, y, z))
    
    # 40%外围星海（稀疏）
    outer_count = num_particles - core_count
    for i in range(outer_count):
        # 更大范围稀疏分布
        x = np.random.uniform(-2.0, 2.0)
        y = np.random.uniform(-2.0, 2.0)
        z = np.random.uniform(-2.0, 2.0)
        
        particles.append((x, y, z))
    
    return particles


def volumetric_saturn(num_particles: int = 1000) -> List[Tuple[float, float, float]]:
    """
    3D土星 - 球体 + 环形带（优化版）
    """
    particles = []
    
    # 40% 行星主体（密集球体）
    planet_count = int(num_particles * 0.4)
    for i in range(planet_count):
        # 球形均匀采样
        phi = np.arccos(2 * np.random.random() - 1)
        theta = 2 * np.pi * np.random.random()
        r = 0.45 * np.random.random() ** (1/3)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        particles.append((x, y, z))
    
    # 50% 主光环（密集）
    main_ring_count = int(num_particles * 0.5)
    for i in range(main_ring_count):
        angle = 2 * np.pi * np.random.random()
        r = 0.65 + 0.35 * np.random.random()  # 环的半径
        thickness = 0.03 * (np.random.random() - 0.5)  # 环的厚度
        
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = thickness
        
        particles.append((x, y, z))
    
    # 10% 外光环（稀疏）
    outer_ring_count = num_particles - planet_count - main_ring_count
    for i in range(outer_ring_count):
        angle = 2 * np.pi * np.random.random()
        r = 1.1 + 0.3 * np.random.random()  # 外环半径
        thickness = 0.02 * (np.random.random() - 0.5)
        
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = thickness
        
        particles.append((x, y, z))
    
    return particles


def volumetric_buddha(num_particles: int = 1000) -> List[Tuple[float, float, float]]:
    """
    3D佛像 - 简化体积模型
    头部球体 + 身体圆锥 + 底座
    """
    particles = []
    
    # 头部（球体，上部）
    head_count = int(num_particles * 0.25)
    for i in range(head_count):
        phi = np.arccos(2 * np.random.random() - 1)
        theta = 2 * np.pi * np.random.random()
        r = 0.25 * np.random.random() ** (1/3)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = 0.5 + r * np.cos(phi)
        
        particles.append((x, y, z))
    
    # 身体（圆锥体）
    body_count = int(num_particles * 0.5)
    for i in range(body_count):
        angle = 2 * np.pi * np.random.random()
        height = np.random.random()  # 0到1
        r = 0.3 + 0.4 * height  # 越往下越宽
        
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = 0.5 - 0.8 * height
        
        particles.append((x, y, z))
    
    # 底座（扁平圆盘）
    base_count = num_particles - head_count - body_count
    for i in range(base_count):
        angle = 2 * np.pi * np.random.random()
        r = 0.7 * np.random.random()
        
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = -0.7 + 0.1 * (np.random.random() - 0.5)
        
        particles.append((x, y, z))
    
    return particles


def volumetric_firework(num_particles: int = 1000) -> List[Tuple[float, float, float]]:
    """
    3D烟花爆炸 - 径向爆炸 + 重力下落
    """
    particles = []
    
    rays = 20  # 爆炸射线数量
    
    for i in range(num_particles):
        # 球形爆炸方向
        phi = np.arccos(2 * np.random.random() - 1)
        theta = 2 * np.pi * np.random.random()
        
        # 距离（越远粒子越少）
        r = 0.8 * (np.random.random() ** 0.5)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # 添加一些下落效果（重力）
        z -= 0.2 * r
        
        particles.append((x, y, z))
    
    return particles


class ParticleModel3DLibrary:
    """3D粒子模型库"""
    
    MODELS = {
        "heart": volumetric_heart,
        "lotus": volumetric_lotus,
        "star_field": volumetric_star_field,
        "saturn": volumetric_saturn,
        "buddha": volumetric_buddha,
        "firework": volumetric_firework,
    }
    
    @classmethod
    def get_model(cls, name: str, num_particles: int = 1000) -> List[Tuple[float, float, float]]:
        """获取3D模型点集"""
        if name not in cls.MODELS:
            name = "heart"
        return cls.MODELS[name](num_particles)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有可用模型"""
        return list(cls.MODELS.keys())

