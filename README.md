# AirCanvas - 隔空绘手 ✨

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.12+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-orange.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

基于手势识别的虚拟演示系统，通过摄像头捕捉手势实现**空中绘图**、**PPT控制**和**AR增强效果**。

## ✨ 功能特性

### 🖌️ 核心绘图
- **捏合绘图**：拇指+食指捏合即可书写
- **多种笔刷**：实线、虚线、发光、马克笔、彩虹
- **多种颜色**：8种预设颜色，支持手势/鼠标切换
- **线条粗细**：6档可调
- **橡皮擦**：捏合时擦除
- **激光笔**：GoodNotes风格消失墨迹

### 🎯 智能图形识别
- **停留触发**：画完后停留0.5秒自动识别
- **支持图形**：圆形、矩形、三角形、直线
- **一键美化**：手绘图形自动变规整

### 🎮 手势UI
- **悬停选择**：手指悬停按钮0.4秒自动选中
- **捏合确认**：捏合手势确认选择
- **鼠标支持**：所有按钮支持鼠标点击
- **智能死区**：UI区域不会误画

### 📊 PPT演示控制
- **挥手翻页**：左右挥手切换幻灯片
- **墨迹模式**：直接在PPT中绘图批注
- **手势切换**：笔/橡皮/导航三种模式

### 🎨 AR增强效果
- **粒子特效**：绘画时的华丽粒子效果
- **互动特效**：粒子场、海浪、星空漩涡
- **激光指示**：移动模式下的激光光标

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/AirCanvas.git
cd AirCanvas

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
python main.py
```

## 🎮 操作指南

### 手势控制

| 手势 | 功能 |
|------|------|
| 👌 捏合（拇指+食指） | 绘画/擦除/激光 |
| ☝️ 食指单指 | 移动光标/UI悬停 |
| ✌️ 食指+中指 | - |
| 🖐️ 五指张开 | - |
| ✋ 向左/右挥手 | PPT翻页 |
| 三指滑动 | 撤销(左)/重做(右) |

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `c` | 清空画布 |
| `s` | 保存画布 |
| `u` | 显示/隐藏UI |
| `z` | 撤销 |
| `y` | 重做 |
| `t` | 切换工具 |
| `l` | 直线辅助开关 |
| `1` | 粒子效果开关 |
| `2` | 激光笔开关 |
| `3` | 掌心HUD开关 |
| `4` | 互动特效开关 |
| `5` | 切换互动特效 |
| `Tab` | 切换AirCanvas/PPT模式 |

### 鼠标操作

所有UI按钮均支持鼠标点击：
- 左侧：工具切换（笔/橡皮/激光）
- 顶部：颜色选择
- 右侧：笔刷类型
- 底部：线条粗细
- 右下角：Help按钮

## 📁 项目结构

```
AirCanvas/
├── main.py                    # 主程序入口
├── config.py                  # 配置参数
├── requirements.txt           # 依赖列表
├── CLAUDE.md                  # 开发指南
├── CHANGELOG.md               # 更新日志
├── README.md                  # 项目说明
├── README_CN.md               # 中文用户指南
│
├── core/                      # 核心模块
│   ├── hand_detector.py       # 手部检测
│   ├── gesture_recognizer.py  # 手势识别
│   ├── coordinate_mapper.py   # 坐标映射
│   └── async_detector.py      # 异步检测器
│
├── modules/                   # 功能模块
│   ├── canvas.py              # 画布（含撤销/重做）
│   ├── virtual_pen.py         # 虚拟画笔
│   ├── eraser.py              # 橡皮擦
│   ├── brush_manager.py       # 笔刷管理
│   ├── gesture_ui.py          # 手势UI界面
│   ├── shape_recognizer.py    # 图形识别
│   ├── temporary_ink.py       # 消失墨迹
│   ├── visual_effects.py      # 视觉特效
│   ├── particle_system.py     # 粒子系统
│   ├── interactive_effects.py # 互动特效
│   ├── laser_pointer.py       # 激光笔
│   ├── palm_hud.py            # 掌心HUD
│   ├── ppt_gesture_controller.py # PPT手势控制
│   ├── tutorial_manager.py    # 教程管理
│   └── viewport.py            # 视口管理
│
└── utils/                     # 工具函数
    └── smoothing.py           # 平滑算法（1€滤波器等）
```

## ⚙️ 配置参数

编辑 `config.py` 可调整：

```python
# 摄像头设置
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 捏合手势阈值
PINCH_THRESHOLD = 0.065        # 触发距离
PINCH_RELEASE_THRESHOLD = 0.15 # 释放距离

# 平滑参数（1€滤波器）
ONE_EURO_MIN_CUTOFF = 1.5      # 最小截止频率
ONE_EURO_BETA = 0.008          # 速度系数

# 画笔设置
PEN_THICKNESS = 3
ERASER_SIZE = 40
```

## 🔧 依赖

- **opencv-python** >= 4.12.0 - 图像处理
- **mediapipe** >= 0.10.14 - 手部检测
- **numpy** >= 2.2.6 - 数值计算
- **pyautogui** >= 0.9.54 - 鼠标/键盘控制
- **pywin32** >= 311 - Windows COM接口（PPT控制）

## 📝 技术亮点

### 自适应捏合检测
- 基于手掌宽度动态调整阈值
- 远近距离均保持稳定灵敏度
- 迟滞机制防止状态抖动

### 智能图形识别
- **停留触发**：只有停留时才识别，避免干扰书写
- **闭合度检测**：判断图形是否闭合
- **顶点分析**：区分三角形、矩形
- **圆度评分**：识别圆形

### 高精度平滑
- **1€滤波器**：自适应低延迟平滑
- **Catmull-Rom样条**：平滑曲线插值
- **直线保护**：小笔画不被曲线化

### 双坐标空间
- **画布空间**：用于绘图（带平滑）
- **UI空间**：用于界面交互（无延迟）

## 🐛 已知问题

- 需要良好的光照条件
- 手部需在画面中完整可见
- 复杂背景可能影响检测准确度
- 某些摄像头可能需要手动调整对焦

## 📄 许可证

MIT License

## 👥 贡献者

- **ZephyrHan** - 项目维护
- **MortalSnow** - 核心开发

---

**享受 AirCanvas 带来的空中绘画体验！** ✨🎨
