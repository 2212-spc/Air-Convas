# AirCanvas - 隔空绘手

基于手势识别的虚拟演示系统，通过摄像头捕捉手势实现空中绘图、PPT控制和AR增强效果。

## 功能特性

### 核心功能
- 手部检测（MediaPipe Hands）
- 三指捏合绘图（拇指+食指+中指）
- 手势UI界面（无需键盘选择颜色/粗细/笔刷）
- 多种笔刷类型（实线、虚线、发光、马克笔）
- 橡皮擦（五指张开）
- 图形美化（自动识别圆形、矩形、三角形）

### PPT演示控制
- 挥手翻页（左右上下）
- PPT墨迹模式（直接在演示文稿中绘图）
- 多种PPT工具模式（鼠标/画笔/翻页）

### AR增强效果
- 粒子特效（三指竖起触发）
- 激光笔指示器
- 掌心HUD信息显示

## 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n aircanvas python=3.10
conda activate aircanvas

# 安装依赖
pip install -r requirements.txt
```

依赖包：
- opencv-python：图像处理
- mediapipe：手部检测
- numpy：数值计算
- pyautogui：鼠标/键盘控制

## 使用方法

### 启动程序

```bash
python main.py
```

### 手势控制

| 手势 | 功能 |
|------|------|
| 三指捏合（拇指+食指+中指） | 开始绘画 |
| 松开任意一指 | 停止绘画（断笔） |
| 食指竖起 | 移动光标 / 激光笔 / UI悬停 |
| 食指+中指竖起 | 显示/隐藏手势UI |
| 三指竖起（食指+中指+无名指） | 粒子特效模式 |
| 五指张开 | 橡皮擦模式 |
| 握拳 | 暂停所有操作 |
| 向右挥手 | PPT下一页 |
| 向左挥手 | PPT上一页 |
| 向上挥手 | PPT第一页 |
| 向下挥手 | PPT最后一页 |
| 手掌静止1秒 | 显示掌心HUD |

### 手势UI操作

1. 伸出食指+中指 → 显示/隐藏UI面板
2. 伸出食指悬停在按钮上 → 按钮高亮
3. 三指捏合 → 选择当前悬停的按钮

UI面板包含：
- 左侧：颜色选择（红/绿/蓝/黄/白/青/品红/橙）
- 底部：线条粗细（2/4/6/8/10/12像素）
- 右侧：笔刷类型（实线/虚线/发光/马克笔）

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| q | 退出程序 |
| c | 清空画布 |
| s | 保存画布到 `captures/canvas_N.png` |
| w | 切换全屏/窗口模式 |
| h | 显示/隐藏帮助信息 |
| f | 切换PPT墨迹模式 |
| t | 切换PPT工具（鼠标/画笔/翻页） |
| [ / ] | 切换颜色 |
| - / + | 调整线条粗细 |
| b | 切换笔刷类型 |
| 1 | 开关粒子特效 |
| 2 | 开关激光笔 |
| 3 | 开关掌心HUD |
| r | 重置演讲计时器 |

## 配置参数

编辑 `config.py` 可调整：

```python
# 摄像头设置
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 三指捏合阈值
PINCH_THRESHOLD = 0.16           # 三指贴合触发距离（增大=更灵敏）
PINCH_RELEASE_THRESHOLD = 0.18   # 释放距离（任意一指分开即断笔）
PINCH_CONFIRM_FRAMES = 1         # 确认帧数（1=即时响应）

# 挥手阈值
SWIPE_THRESHOLD = 0.15           # 挥手位移阈值
SWIPE_VELOCITY_THRESHOLD = 0.008 # 挥手速度阈值

# 画笔设置
PEN_COLOR = (0, 255, 255)  # BGR黄色
PEN_THICKNESS = 3
ERASER_SIZE = 40
STROKE_JUMP_THRESHOLD = 80 # 笔画跳变断笔阈值（像素）
DRAW_LOCK_FRAMES = 5       # 断笔后冷却帧数

# 平滑系数（数值越小越跟手，越大越平滑）
CURSOR_SMOOTHING_FACTOR = 0.08
DRAW_SMOOTHING_FACTOR = 0.15

# AR效果
MAX_PARTICLES = 300
PARTICLE_EMIT_COUNT = 5
```

## 项目结构

```
AirCanvas/
├── main.py                  # 主程序入口
├── config.py                # 配置参数
├── requirements.txt         # 依赖列表
├── CLAUDE.md               # 开发指南
├── core/                    # 核心模块
│   ├── hand_detector.py    # 手部检测
│   ├── gesture_recognizer.py # 手势识别（三指捏合）
│   └── coordinate_mapper.py  # 坐标映射与平滑
├── modules/                 # 功能模块
│   ├── canvas.py           # 画布
│   ├── virtual_pen.py      # 画笔（含跳变断笔）
│   ├── eraser.py           # 橡皮擦
│   ├── brush_manager.py    # 笔刷管理器
│   ├── gesture_ui.py       # 手势UI界面
│   ├── ppt_controller.py   # PPT控制
│   ├── shape_recognizer.py # 图形识别
│   ├── particle_system.py  # 粒子系统
│   ├── laser_pointer.py    # 激光笔
│   └── palm_hud.py         # 掌心HUD
└── utils/                   # 工具函数
    └── smoothing.py        # 平滑算法
```

## 技术亮点

### 三指捏合手势
- 使用拇指+食指+中指三点距离检测
- 三指都靠近才触发绘图，任意一指离开即断笔
- 比传统两指捏合更明确，误触率更低

### 防粘连机制
1. **非对称确认帧数**：开始需确认，释放即时响应
2. **位置跳变检测**：移动距离超过阈值自动断笔
3. **冷却期保护**：断笔后短暂禁止重连

### 手势UI
- 无需键盘即可选择颜色、粗细、笔刷
- 悬停锁定机制减少手部抖动影响
- 增大碰撞检测区域提高选择成功率

## 已知问题

- 需要良好的光照条件
- 手部需在画面中完整可见
- 复杂背景可能影响检测准确度

## 许可证

MIT License
