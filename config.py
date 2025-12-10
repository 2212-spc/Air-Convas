CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Screen (adjust to actual display resolution)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Active regions (normalized x1, y1, x2, y2)
# 全屏光标映射区域，避免只能在窗口一角移动
ACTIVE_REGION_CURSOR = (0.0, 0.0, 1.0, 1.0)
# 绘图区域（先全屏，若需可再收窄）
ACTIVE_REGION_DRAW = (0.0, 0.0, 1.0, 1.0)

# Gesture thresholds
# 三指捏合阈值（拇指+食指+中指）：更明确的手势，断开更容易
PINCH_THRESHOLD = 0.16              # 三指贴合触发阈值（增大=更容易触发）
PINCH_RELEASE_THRESHOLD = 0.18      # 任意一指分开即释放
SWIPE_THRESHOLD = 0.15              # 提高到15%，更符合自然挥手幅度
SWIPE_COOLDOWN_FRAMES = 20
SWIPE_VELOCITY_THRESHOLD = 0.008    # 降低速度要求，更容易触发
PINCH_CONFIRM_FRAMES = 1            # 降到1帧，更快响应
PINCH_RELEASE_CONFIRM_FRAMES = 1    # 1帧即释放，断笔灵敏

# Pen / eraser
PEN_COLOR = (0, 255, 255)  # BGR yellow
PEN_THICKNESS = 3
ERASER_SIZE = 40
STROKE_JUMP_THRESHOLD = 80  # 笔画跳变阈值（像素），超过此距离自动断笔防止粘连
DRAW_LOCK_FRAMES = 5  # 捏合结束后冷却帧数，防止误连笔

# Smoothing（分别控制鼠标和绘图）
# 降低平滑系数以减少延迟，提高响应速度
CURSOR_SMOOTHING_FACTOR = 0.08   # 鼠标更跟手（降低以减少延迟）
DRAW_SMOOTHING_FACTOR = 0.15     # 绘图更流畅但仍保持响应（降低以减少延迟）

# Particles (placeholder for later phases)
MAX_PARTICLES = 300
PARTICLE_EMIT_COUNT = 5

# UI
WINDOW_NAME = "AirCanvas"
