# 更新日志 (Changelog)

## [v2.0.0] - 2026-01-27

### 🎯 重大更新

本次更新为 Air-Convas 添加了完整的英文教程系统、鼠标交互支持、智能按钮死区、以及虚线笔刷功能，大幅提升了用户体验和绘画准确性。

---

## 📚 新增功能

### 1. 英文教程系统 (Tutorial System)

**文件：** `modules/tutorial_manager.py`

#### 功能特性
- ✅ **启动教程**：首次运行时自动显示，引导用户了解基本操作
- ✅ **3 页简洁教程**：
  - 第 1 页：手势操作说明（双指切换模式、捏合绘画、张开橡皮擦、三指撤销/重做）
  - 第 2 页：UI 布局说明（左侧工具、右侧颜色和笔刷、底部粗细）
  - 第 3 页：准备开始提示
- ✅ **透明背景 + 浅蓝边框**：占据屏幕 80%，视觉效果清晰
- ✅ **多种导航方式**：
  - 键盘任意键翻页
  - 鼠标点击翻页
- ✅ **Help 按钮**：位于右下角，随时可点击重新查看教程
- ✅ **居中标题 + 左对齐内容**：排版清晰易读
- ✅ **黄色加粗小标题 + 白色详细说明**：层次分明

#### 实现细节
- 使用 OpenCV 绘制半透明遮罩层
- 多行文本自动换行和对齐
- 支持键盘和鼠标事件捕获
- 教程结束后显示 Help 按钮

#### 使用方法
```python
# 在 main.py 中集成
from modules.tutorial_manager import TutorialManager

tutorial_manager = TutorialManager()
tutorial_manager.start_tutorial()  # 启动教程

# 渲染教程
if tutorial_manager.visible:
    frame = tutorial_manager.render(frame)
```

---

### 2. 鼠标交互支持 (Mouse Interaction)

**文件：** `main.py`, `modules/gesture_ui.py`, `modules/tutorial_manager.py`

#### 功能特性
- ✅ **鼠标点击 UI 按钮**：所有工具、颜色、粗细、笔刷、动作按钮均可鼠标点击
- ✅ **鼠标点击教程翻页**：点击屏幕任意位置翻页
- ✅ **鼠标点击 Help 按钮**：重新进入教程
- ✅ **视觉反馈**：点击按钮时产生涟漪效果

#### 实现细节
```python
# 在 main.py 中添加鼠标回调
def mouse_callback(event, x, y, flags, param):
    global mouse_clicked, mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
        mouse_click_pos = (x, y)

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
```

#### 支持的鼠标操作
- 点击工具按钮（Pen/Eraser/Laser）
- 点击颜色按钮（8 种颜色）
- 点击粗细按钮（6 种粗细）
- 点击笔刷按钮（Solid/Dashed/Glow/Marker/Rainbow）
- 点击动作按钮（Clear/FX）
- 点击 Help 按钮

---

### 3. 智能按钮死区 (Smart Dead Zone)

**文件：** `modules/gesture_ui.py`, `main.py`

#### 功能特性
- ✅ **防止误触**：在 UI 按钮区域无法开始绘画
- ✅ **智能锁定**：手指接近按钮时自动锁定，无需精确命中
- ✅ **绘画中穿越**：绘画过程中可穿越死区，不会中断笔画
- ✅ **死区范围**：按钮周围 25px 范围内
- ✅ **智能锁定范围**：按钮周围 50px 范围内

#### 实现细节

**1. 死区检测**
```python
def is_in_dead_zone(self, point: Tuple[int, int], brush_manager) -> bool:
    """检查点是否在任何 UI 按钮的死区范围内"""
    x, y = point
    dead_zone_margin = 25  # 死区边距
    
    # 检查所有按钮（工具、颜色、粗细、笔刷、动作）
    for button in all_buttons:
        if point_in_extended_rect(x, y, button, dead_zone_margin):
            return True
    return False
```

**2. 智能锁定**
```python
def get_locked_ui_item(self, point: Tuple[int, int], brush_manager) -> Optional[Tuple[str, int]]:
    """智能锁定最近的 UI 按钮"""
    lock_distance = 50  # 锁定距离
    closest_item = None
    min_dist_sq = lock_distance ** 2
    
    # 找到距离最近的按钮
    for button in all_buttons:
        dist_sq = distance_to_button_center(point, button)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_item = button
    
    return closest_item
```

**3. 绘画状态管理**
```python
# 在 main.py 中
is_drawing = False

# 开始绘画时检查死区
if g["pinch_start"]:
    if gesture_ui.is_in_dead_zone(draw_pt, brush_manager):
        print("在按钮区域，无法开始画画")
    else:
        pen.start_stroke()
        is_drawing = True

# 绘画中可穿越死区
if g["pinching"]:
    if is_drawing:  # 已经在绘画，继续
        pen.draw(draw_pt)
    elif not gesture_ui.is_in_dead_zone(draw_pt, brush_manager):
        # 不在死区，可以开始
        pen.start_stroke()
        is_drawing = True
```

#### 使用效果
- ✅ 点击按钮时不会留下笔迹
- ✅ 手指接近按钮时自动锁定，提高选择成功率
- ✅ 绘画过程中经过按钮区域不会中断

---

### 4. 虚线笔刷 (Dashed Brush)

**文件：** `modules/brush_manager.py`, `modules/virtual_pen.py`

#### 功能特性
- ✅ **沿路径连续绘制**：像普通线段一样，但从中间一些地方断开
- ✅ **相位累积算法**：保证虚线沿整个笔画路径连续，不会每段重新开始
- ✅ **自定义比例**：实线 70%，空白 30%（0.7cm 实线，0.3cm 空白）
- ✅ **自动加粗**：虚线比实线粗 30%，视觉效果更清晰
- ✅ **使用当前颜色**：虚线颜色与选择的颜色一致

#### 实现细节

**1. 相位追踪**
```python
# 在 BrushManager.__init__() 中
self.dash_phase = 0.0      # 当前虚线相位
self.dash_length = 35      # 实线段长度（0.7cm，70%）
self.gap_length = 15       # 空隙长度（0.3cm，30%）
```

**2. 相位累积算法**
```python
def _draw_dashed_line(self, canvas, pt1, pt2, color, thickness):
    """沿着路径连续绘制，使用相位累积"""
    distance = calculate_distance(pt1, pt2)
    period = self.dash_length + self.gap_length  # 50px
    
    current_distance = 0.0
    while current_distance < distance:
        phase_in_period = self.dash_phase % period
        
        if phase_in_period < self.dash_length:
            # 在实线段内，绘制
            segment_length = min(remaining_dash, distance - current_distance)
            draw_segment(...)
            self.dash_phase += segment_length
        else:
            # 在空隙段内，跳过
            segment_length = min(remaining_gap, distance - current_distance)
            self.dash_phase += segment_length
        
        current_distance += segment_length
```

**3. 笔画开始时重置相位**
```python
# 在 VirtualPen.start_stroke() 中
def start_stroke(self):
    ...
    if self.brush_manager.brush_type == "dashed":
        self.brush_manager.reset_dash_phase()
```

#### 虚线参数
- **实线长度**：35 像素（0.7cm，70%）
- **空隙长度**：15 像素（0.3cm，30%）
- **周期长度**：50 像素（1.0cm）
- **粗细增强**：原始粗细 + 30%（最少 +2 像素）

#### 使用方法
1. 选择 **Pen** 工具
2. 选择 **Dashed** 笔刷（右侧第 2 个按钮）
3. 选择颜色和粗细
4. 双指捏合开始画画
5. 看到沿着路径的连续虚线效果

---

## 🔧 UI 优化

### UI 布局调整

**文件：** `modules/gesture_ui.py`

#### 改进内容
- ✅ **按钮间距加大**：所有按钮间距增加，避免误触
- ✅ **按钮位置调整**：工具按钮靠左，颜色和笔刷按钮靠右，粗细按钮在底部
- ✅ **所有按钮在屏幕内**：确保所有按钮都在可见范围内
- ✅ **颜色按钮上移**：所有颜色按钮都在屏幕内可见

#### 具体调整
```python
# 工具面板 - 左侧
tool_panel_x = 30

# 动作面板 - 左侧下方
action_panel_x = 30

# 颜色面板 - 右侧
color_panel_x = self.screen_width - 80
color_panel_y_start = 120  # 上移，确保所有颜色可见

# 笔刷面板 - 右侧下方
brush_panel_x = self.screen_width - 140

# 粗细面板 - 底部居中
thickness_panel_y = self.screen_height - 80
```

---

## 🐛 Bug 修复

### 1. 连续画画问题

**问题：** 之前只能画一笔就停止

**原因：** 死区逻辑错误，导致 `start_stroke()` 被重复调用

**修复：**
```python
# 修复前
if not is_drawing and in_dead_zone:
    # 逻辑混乱

# 修复后
if in_dead_zone:
    # 在死区内完全不允许开始
else:
    # 只在非死区时允许开始
    pen.start_stroke()
    is_drawing = True
```

**效果：** 现在可以连续画多笔，不受限制

### 2. 橡皮工具状态管理

**问题：** 橡皮工具没有设置 `is_drawing` 状态

**修复：**
```python
# 在 pinch_start 时
elif brush_manager.tool == "eraser":
    is_drawing = True  # 橡皮也需要标记

# 在 pinch_end 时
is_drawing = False  # 所有工具都清除状态
```

**效果：** 橡皮工具状态管理一致

### 3. 虚线端点问题

**问题：** 虚线的起始点和端点圆圈影响虚线效果

**修复：**
```python
# 虚线不绘制端点圆圈
if self.brush_manager.brush_type in ("solid", "rainbow"):
    cv2.circle(canvas_img, pt2, radius, color, -1)
# 虚线不需要端点
```

**效果：** 虚线效果更清晰

---

## 📝 代码结构改进

### 新增文件
1. **`modules/tutorial_manager.py`** (新增)
   - `TutorialPage` 类：教程页面数据结构
   - `TutorialManager` 类：教程管理器
   - 教程渲染、导航、Help 按钮管理

### 修改文件
1. **`main.py`**
   - 集成 `TutorialManager`
   - 添加鼠标回调函数 `mouse_callback`
   - 优化绘画状态管理（`is_drawing`）
   - 集成死区检测逻辑
   - 添加智能锁定逻辑

2. **`modules/gesture_ui.py`**
   - 添加 `handle_mouse_click()` 方法
   - 添加 `is_in_dead_zone()` 方法
   - 添加 `get_locked_ui_item()` 方法
   - 添加 `get_item_color()` 辅助方法
   - 调整 UI 布局参数

3. **`modules/brush_manager.py`**
   - 添加虚线相位追踪（`dash_phase`, `dash_length`, `gap_length`）
   - 添加 `reset_dash_phase()` 方法
   - 重写 `_draw_dashed_line()` 方法（相位累积算法）
   - 调整虚线参数（35px 实线，15px 空隙）

4. **`modules/virtual_pen.py`**
   - 在 `start_stroke()` 中调用 `reset_dash_phase()`
   - 优化虚线起始点绘制
   - 移除虚线端点圆圈

---

## 🎨 用户体验提升

### 教程系统
- ✅ 新用户首次运行时自动显示教程
- ✅ 3 页简洁教程，快速上手
- ✅ Help 按钮随时可查看

### 鼠标交互
- ✅ 不依赖手势也能操作所有 UI
- ✅ 鼠标点击比手势更精确
- ✅ 适合演示和测试

### 智能死区
- ✅ 点击按钮不会留下笔迹
- ✅ 智能锁定提高选择成功率
- ✅ 绘画中穿越死区不中断

### 虚线笔刷
- ✅ 真正的虚线效果（沿路径连续）
- ✅ 可调整比例和粗细
- ✅ 视觉效果清晰

---

## 📊 技术亮点

### 1. 相位累积算法
- **创新点**：虚线不是"每段单独画"，而是"沿路径连续累积相位"
- **优势**：虚线看起来自然、连续，像真正的虚线
- **实现**：全局相位追踪 + 每笔画重置

### 2. 智能锁定机制
- **创新点**：按钮不需要精确命中，接近即可锁定
- **优势**：大幅提高手势选择成功率
- **实现**：计算到所有按钮中心的距离，锁定最近的

### 3. 死区 + 绘画状态管理
- **创新点**：区分"未开始"和"绘画中"两种状态
- **优势**：防止误触，但不影响连续绘画
- **实现**：`is_drawing` 标志 + 死区检测

### 4. 教程系统架构
- **创新点**：数据驱动的教程页面
- **优势**：易于扩展和修改
- **实现**：`TutorialPage` 数据结构 + `TutorialManager` 渲染引擎

---

## 🔬 测试验证

### 教程系统
- ✅ 首次启动自动显示
- ✅ 键盘翻页正常
- ✅ 鼠标点击翻页正常
- ✅ Help 按钮点击重新进入
- ✅ 教程结束后显示 Help 按钮

### 鼠标交互
- ✅ 所有按钮都可鼠标点击
- ✅ 点击产生涟漪效果
- ✅ 状态正确更新

### 智能死区
- ✅ 死区内无法开始绘画
- ✅ 绘画中可穿越死区
- ✅ 智能锁定提高选择成功率

### 虚线笔刷
- ✅ 虚线沿路径连续
- ✅ 比例正确（70% 实线，30% 空白）
- ✅ 粗细正确（自动加粗 30%）
- ✅ 颜色正确（使用选择的颜色）

---

## 📦 兼容性

- **Python 版本**：3.10+
- **主要依赖**：
  - opencv-python >= 4.12.0
  - mediapipe >= 0.10.14
  - numpy >= 2.2.6
- **操作系统**：Windows / macOS / Linux

---

## 🚀 未来计划

### 功能扩展
- [ ] 更多笔刷类型（水彩、喷枪等）
- [ ] 图层支持
- [ ] 录屏功能
- [ ] 手势自定义

### 性能优化
- [ ] GPU 加速
- [ ] 多线程渲染
- [ ] 内存优化

---

## 👥 贡献者

- **ZephyrHan** (@2212-spc) - 项目维护
- **MortalSnow** (@Seren666) - 核心开发
- **Claude** - AI 辅助开发

---

## 📄 许可证

MIT License

---

## 🙏 致谢

感谢所有为 Air-Convas 项目做出贡献的开发者和用户！

特别感谢：
- MediaPipe 团队提供优秀的手部检测库
- OpenCV 团队提供强大的图像处理工具
- 所有提供反馈和建议的用户

---

**享受 Air-Convas 带来的空中绘画体验！** ✨🎨

