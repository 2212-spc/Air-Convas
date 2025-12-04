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
PINCH_THRESHOLD = 0.035
PINCH_RELEASE_THRESHOLD = 0.11
SWIPE_THRESHOLD = 0.10
SWIPE_COOLDOWN_FRAMES = 20
SWIPE_VELOCITY_THRESHOLD = 0.015  # 最小单位帧速度
PINCH_CONFIRM_FRAMES = 2          # 捏合/释放去抖确认帧数

# Pen / eraser
PEN_COLOR = (0, 255, 255)  # BGR yellow
PEN_THICKNESS = 3
ERASER_SIZE = 40

# Smoothing（分别控制鼠标和绘图）
CURSOR_SMOOTHING_FACTOR = 0.10  # 鼠标更跟手
DRAW_SMOOTHING_FACTOR = 0.28    # 绘图更顺滑且低延迟

# Particles (placeholder for later phases)
MAX_PARTICLES = 300
PARTICLE_EMIT_COUNT = 5

# UI
WINDOW_NAME = "AirCanvas"
