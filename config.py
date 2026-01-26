# -*- coding: utf-8 -*-
"""AirCanvas 配置文件"""

CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 推理分辨率（降低以提升性能，归一化坐标不受影响）
INFER_WIDTH = 640
INFER_HEIGHT = 360

# 异步推理模式（将 MediaPipe 推理放到独立线程）
# True: 主循环不阻塞，更流畅但可能有1帧延迟
# False: 同步推理，延迟更稳定
ASYNC_INFERENCE = True

# Screen (adjust to actual display resolution)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Active regions (normalized x1, y1, x2, y2)
# 全屏光标映射区域，避免只能在窗口一角移动
ACTIVE_REGION_CURSOR = (0.0, 0.0, 1.0, 1.0)
# 绘图区域（先全屏，若需可再收窄）
ACTIVE_REGION_DRAW = (0.0, 0.0, 1.0, 1.0)

# Gesture thresholds
# 两指捏合阈值（拇指+食指）
PINCH_THRESHOLD = 0.08              # 基础捏合阈值
PINCH_RELEASE_THRESHOLD = 0.12      # 释放阈值（迟滞防抖）
SWIPE_THRESHOLD = 0.15
SWIPE_COOLDOWN_FRAMES = 20
SWIPE_VELOCITY_THRESHOLD = 0.008
PINCH_CONFIRM_FRAMES = 1            # 捏合确认帧数
PINCH_RELEASE_CONFIRM_FRAMES = 1    # 释放确认帧数

# 速度感知捏合参数（新增）
PINCH_VELOCITY_BOOST = 0.02         # 快速捏合时的阈值增益

# Pen / eraser
PEN_COLOR = (0, 255, 255)  # BGR yellow
PEN_THICKNESS = 6
ERASER_SIZE = 40
STROKE_JUMP_THRESHOLD = 150
DRAW_LOCK_FRAMES = 3                # 笔画结束后的冷却帧数

# Smoothing（分别控制鼠标和绘图）
CURSOR_SMOOTHING_FACTOR = 0.15
DRAW_SMOOTHING_FACTOR = 0.20

# 1€ Filter 参数（自适应平滑：低速平滑，高速跟手）
# 原理：cutoff = min_cutoff + beta × 速度
#   低速书写: cutoff ≈ min_cutoff (高平滑，消除抖动)
#   高速书写: cutoff 增大 (低延迟，跟手)
# 
# 参数调优指南：
#   - MIN_CUTOFF 越小：平滑度越高，延迟稍高
#   - BETA 越大：高速时越跟手，但可能抖动
#
# 当前配置（稳定优先，减少抖动）：
ONE_EURO_MIN_CUTOFF = 0.8           # 较低，增强平滑
ONE_EURO_BETA = 0.007               # 较低，减少抖动

# 贝塞尔曲线参数
BEZIER_ENABLED = True               # 启用贝塞尔曲线平滑
BEZIER_SEGMENTS = 8                 # 每段曲线的插值点数

# 钢笔效果参数（新增）
PEN_EFFECT_ENABLED = True           # 启用钢笔效果（速度感知粗细）
PEN_MIN_THICKNESS_RATIO = 0.4       # 最细时的粗细比例（快速移动）
PEN_MAX_THICKNESS_RATIO = 1.2       # 最粗时的粗细比例（慢速移动）
PEN_SPEED_THRESHOLD = 25.0          # 速度阈值（像素/帧）
PEN_THICKNESS_SMOOTHING = 0.25      # 粗细平滑系数（越小越平滑）

# 直线辅助参数（新增）
LINE_ASSIST_ENABLED = True          # 是否启用直线辅助
LINE_VARIANCE_THRESH = 0.015        # 直线方差阈值（归一化）
MIN_LINE_LENGTH = 50                # 最小直线长度（像素）

# 撤销/重做参数（新增）
MAX_HISTORY = 50                    # 最大历史记录数

# Particles
MAX_PARTICLES = 100                 # 最大粒子数
PARTICLE_EMIT_COUNT = 3             # 每次发射数量

# AR悬浮窗口书写系统
# 白板尺寸（阿里云风格：缩放后完整显示，ROI在白板上移动）
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# ROI悬浮窗口
ROI_SCALE = 0.4            # ROI窗口占视口的比例（更小=更精细书写）
ROI_PADDING = 0.1          # 摄像头边缘裁剪比例
ROI_OPACITY = 0.7          # ROI窗口透明度

# 边缘自动滚动
EDGE_MARGIN = 60           # 边缘检测区域宽度（像素）
SCROLL_SPEED = 12.0        # 最大滚动速度

# 网格背景
GRID_SIZE = 25             # 网格大小（像素）

# UI
WINDOW_NAME = "AirCanvas"
