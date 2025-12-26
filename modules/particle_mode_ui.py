"""
粒子模式UI面板
简洁现代的UI，支持模型选择、颜色选择、鼠标和手势控制
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Callable


class Button:
    """按钮类"""
    def __init__(self, x: int, y: int, w: int, h: int, text: str, action: Callable):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.action = action
        self.hovered = False
        self.active = False
    
    def contains(self, px: int, py: int) -> bool:
        """检查点是否在按钮内"""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    
    def render(self, frame: np.ndarray):
        """渲染按钮"""
        # 背景色
        if self.active:
            color = (100, 200, 100)  # 绿色（激活）
        elif self.hovered:
            color = (200, 200, 200)  # 亮灰（悬停）
        else:
            color = (150, 150, 150)  # 灰色（正常）
        
        # 绘制圆角矩形
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # 边框
        border_color = (255, 255, 255) if self.hovered else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     border_color, 2)
        
        # 文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), font, font_scale, 
                   (255, 255, 255), thickness, cv2.LINE_AA)


class ColorButton:
    """颜色按钮"""
    def __init__(self, x: int, y: int, size: int, color: Tuple[int, int, int], name: str):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.name = name
        self.hovered = False
        self.active = False
    
    def contains(self, px: int, py: int) -> bool:
        """检查点是否在按钮内"""
        cx, cy = self.x + self.size // 2, self.y + self.size // 2
        return (px - cx) ** 2 + (py - cy) ** 2 <= (self.size // 2) ** 2
    
    def render(self, frame: np.ndarray):
        """渲染颜色按钮"""
        cx, cy = self.x + self.size // 2, self.y + self.size // 2
        radius = self.size // 2
        
        # 颜色圆
        cv2.circle(frame, (cx, cy), radius, self.color, -1, lineType=cv2.LINE_AA)
        
        # 边框
        if self.active:
            cv2.circle(frame, (cx, cy), radius + 3, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        elif self.hovered:
            cv2.circle(frame, (cx, cy), radius + 2, (200, 200, 200), 2, lineType=cv2.LINE_AA)


class ParticleModeUI:
    """粒子模式UI面板"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visible = False
        self.entering = False
        self.exiting = False
        self.animation_progress = 0.0
        
        # UI位置（右侧面板）
        self.panel_width = 200
        self.panel_x = width - self.panel_width - 20
        self.panel_y = 50
        
        # 按钮
        self.buttons = []
        self.color_buttons = []
        self.mouse_pos = (0, 0)
        
        # 回调函数
        self.on_model_change = None
        self.on_color_change = None
        self.on_confirm = None
        self.on_cancel = None
        
        self._create_ui()
    
    def _create_ui(self):
        """创建UI元素"""
        x = self.panel_x
        y = self.panel_y + 60
        btn_w = 180
        btn_h = 40
        spacing = 10
        
        # 模型按钮（只保留3个）
        models = [
            ("3D Heart", "heart"),
            ("Star Field", "star_field"),
            ("Saturn", "saturn"),
        ]
        
        for text, model in models:
            btn = Button(x, y, btn_w, btn_h, text, 
                        lambda m=model: self.on_model_change and self.on_model_change(m))
            self.buttons.append(btn)
            y += btn_h + spacing
        
        # 颜色按钮
        y += 20
        colors = [
            ((50, 50, 255), "Red"),       # 红色（默认）
            ((255, 100, 150), "Pink"),    # 粉红
            ((100, 100, 255), "Blue"),    # 蓝色
            ((100, 255, 100), "Green"),   # 绿色
            ((255, 255, 100), "Yellow"),  # 黄色
            ((200, 100, 255), "Purple"),  # 紫色
        ]
        
        color_size = 35
        colors_per_row = 3
        for i, (color, name) in enumerate(colors):
            col = i % colors_per_row
            row = i // colors_per_row
            cx = x + 30 + col * (color_size + 20)
            cy = y + row * (color_size + 20)
            btn = ColorButton(cx, cy, color_size, color, name)
            self.color_buttons.append(btn)
        
        y += (len(colors) // colors_per_row + 1) * (color_size + 20) + 20
        
        # 确定/取消按钮
        confirm_btn = Button(x, y, 85, 40, "OK(1)", 
                            lambda: self.on_confirm and self.on_confirm())
        cancel_btn = Button(x + 95, y, 85, 40, "Exit(2)", 
                           lambda: self.on_cancel and self.on_cancel())
        self.buttons.extend([confirm_btn, cancel_btn])
    
    def show(self):
        """显示UI（带动画）"""
        if not self.visible and not self.entering:
            self.entering = True
            self.animation_progress = 0.0
    
    def hide(self):
        """隐藏UI（带动画）"""
        if self.visible and not self.exiting:
            self.exiting = True
            self.animation_progress = 1.0
    
    def update_animation(self):
        """更新动画"""
        if self.entering:
            self.animation_progress += 0.1
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                self.entering = False
                self.visible = True
        
        if self.exiting:
            self.animation_progress -= 0.1
            if self.animation_progress <= 0.0:
                self.animation_progress = 0.0
                self.exiting = False
                self.visible = False
    
    def handle_mouse(self, event: int, x: int, y: int, flags, param):
        """处理鼠标事件"""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_MOUSEMOVE:
            # 更新悬停状态
            for btn in self.buttons:
                btn.hovered = btn.contains(x, y)
            for btn in self.color_buttons:
                btn.hovered = btn.contains(x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # 点击按钮
            for btn in self.buttons:
                if btn.contains(x, y):
                    btn.action()
                    return
            
            for btn in self.color_buttons:
                if btn.contains(x, y):
                    if self.on_color_change:
                        self.on_color_change(btn.color)
                    # 更新激活状态
                    for cb in self.color_buttons:
                        cb.active = False
                    btn.active = True
                    return
    
    def set_active_model(self, model_name: str):
        """设置激活的模型"""
        model_map = {
            "heart": "3D Heart",
            "star_field": "Star Field",
            "saturn": "Saturn",
        }
        active_text = model_map.get(model_name, "")
        for btn in self.buttons[:-2]:  # 排除确定/取消按钮
            btn.active = (btn.text == active_text)
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """渲染UI"""
        if not self.visible and not self.entering and not self.exiting:
            return frame
        
        # 更新动画
        self.update_animation()
        
        if self.animation_progress <= 0:
            return frame
        
        # 创建副本
        output = frame.copy()
        
        # 应用暗化效果（背景变暗）
        dark_overlay = output.copy()
        cv2.rectangle(dark_overlay, (0, 0), (self.width, self.height), 
                     (0, 0, 0), -1)
        alpha = 0.4 * self.animation_progress
        cv2.addWeighted(dark_overlay, alpha, output, 1 - alpha, 0, output)
        
        # 绘制面板背景（滑入动画）
        panel_offset = int((1 - self.animation_progress) * (self.panel_width + 50))
        panel_x = self.panel_x + panel_offset
        
        # 半透明背景
        panel_bg = output.copy()
        cv2.rectangle(panel_bg, 
                     (panel_x - 10, self.panel_y - 10),
                     (panel_x + self.panel_width + 10, self.height - 50),
                     (40, 40, 40), -1)
        cv2.addWeighted(panel_bg, 0.85, output, 0.15, 0, output)
        
        # 边框
        cv2.rectangle(output,
                     (panel_x - 10, self.panel_y - 10),
                     (panel_x + self.panel_width + 10, self.height - 50),
                     (100, 100, 100), 2)
        
        # 标题
        title = "Particle Mode"
        cv2.putText(output, title, (panel_x, self.panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制按钮（应用偏移）
        for btn in self.buttons:
            btn_copy = Button(btn.x + panel_offset, btn.y, btn.w, btn.h, btn.text, btn.action)
            btn_copy.hovered = btn.hovered
            btn_copy.active = btn.active
            btn_copy.render(output)
        
        for btn in self.color_buttons:
            btn_copy = ColorButton(btn.x + panel_offset, btn.y, btn.size, btn.color, btn.name)
            btn_copy.hovered = btn.hovered
            btn_copy.active = btn.active
            btn_copy.render(output)
        
        # 提示文字
        tip1 = "Hand: Open=EXPLODE"
        tip2 = "Close=Shrink (7x)"
        cv2.putText(output, tip1, (panel_x, self.height - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(output, tip2, (panel_x, self.height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        
        return output

