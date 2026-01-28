# -*- coding: utf-8 -*-
"""
Tutorial Manager - 开始引导界面
用英文展示使用说明，支持多页浏览
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class TutorialPage:
    """单个教程页面"""
    
    def __init__(self, title: str, content: List[str], icon: Optional[str] = None):
        self.title = title
        self.content = content  # 内容行列表
        self.icon = icon  # 可选图标标识


class TutorialManager:
    """教程管理器"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.is_active = True  # 是否显示教程
        self.current_page = 0
        self.show_help_button = False  # 完成教程后显示Help按钮
        
        # 界面尺寸：占80%屏幕
        self.box_width = int(width * 0.8)
        self.box_height = int(height * 0.8)
        self.box_x = (width - self.box_width) // 2
        self.box_y = (height - self.box_height) // 2
        
        # 样式
        self.bg_alpha = 0.85  # 背景透明度
        self.border_color = (255, 200, 100)  # 浅蓝色边框 (BGR)
        self.border_thickness = 4
        self.text_color = (255, 255, 255)
        self.title_color = (255, 200, 100)
        self.highlight_color = (100, 255, 255)
        
        # Help按钮 - 右下角小按钮
        self.help_button_width = 70
        self.help_button_height = 35
        self.help_button_x = width - self.help_button_width - 15
        self.help_button_y = height - self.help_button_height - 15
        self.help_button_hover = False
        
        # 创建教程页面
        self.pages = self._create_pages()
        
        # 点击检测
        self._click_triggered = False
        
    def _create_pages(self) -> List[TutorialPage]:
        """创建所有教程页面"""
        pages = []
        
        # Page 1: Basic Gestures and Functions
        pages.append(TutorialPage(
            "HAND GESTURE CONTROLS",
            [
                "1. PINCH - Draw & Select buttons",
                "   Put thumb + index finger close together",
                "   Hold = Draw  |  Release = Stop",
                "",
                "2. TWO FINGERS - Show/Hide menu",
                "   Raise index + middle finger together",
                "   Tap quickly to toggle UI panel",
                "",
                "3. THREE FINGERS - Undo/Redo",
                "   Swipe LEFT = Undo last action",
                "   Swipe RIGHT = Redo action",
                "",
                "",
                "TOOLS (Press T to switch):",
                "  Pen - Permanent drawing",
                "  Eraser - Remove strokes",
                "  Laser - Fades after 1.5 seconds",
                "",
            ]
        ))
        
        # Page 2: UI Panel Layout
        pages.append(TutorialPage(
            "INTERFACE & MODE SWITCHING",
            [
                "Show/Hide UI: Use 2-finger gesture",
                "",
                "1. LEFT SIDE - Tools & Actions",
                "   Tools: Pen / Eraser / Laser",
                "   Actions: Clear canvas / Effects",
                "",
                "2. MIDDLE - Color Selection",
                "   Color circles - Choose drawing color",
                "   (Only visible in Pen mode)",
                "",
                "3. BOTTOM CENTER - Line Thickness",
                "   Thickness bars - Adjust line width",
                "",
                "4. RIGHT SIDE - Brush Styles",
                "   Brush types: Solid / Dash / Glow / Marker / Rainbow",
                "",
                "",
                "HOW TO SELECT:",
                "   Hover finger + Pinch  OR  Mouse Click",
            ]
        ))
        
        # Page 3: Ready to Start (Final)
        pages.append(TutorialPage(
            "READY TO CREATE!",
            [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ))
        
        return pages
    
    def handle_click(self, point: Tuple[int, int]) -> bool:
        """
        处理鼠标点击
        返回True表示点击被处理
        """
        if not self.is_active and self.show_help_button:
            # 检查Help按钮点击
            x, y = point
            if (self.help_button_x <= x <= self.help_button_x + self.help_button_width and
                self.help_button_y <= y <= self.help_button_y + self.help_button_height):
                self.restart_tutorial()
                return True
        
        if self.is_active:
            # 教程激活时，任意点击都切换到下一页
            self.next_page()
            return True
            
        return False
    
    def handle_key(self, key: int) -> bool:
        """
        处理键盘按键
        返回True表示按键被处理（教程消耗了这个按键）
        """
        if self.is_active and key != 255:  # 255 = 无按键
            self.next_page()
            return True
        return False
    
    def next_page(self):
        """切换到下一页"""
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
        else:
            # 最后一页，关闭教程
            self.is_active = False
            self.show_help_button = True
    
    def restart_tutorial(self):
        """重新开始教程"""
        self.current_page = 0
        self.is_active = True
        self.show_help_button = False
    
    def update_help_button_hover(self, point: Tuple[int, int]):
        """更新Help按钮悬停状态"""
        if not self.show_help_button or self.is_active:
            self.help_button_hover = False
            return
            
        x, y = point
        self.help_button_hover = (
            self.help_button_x <= x <= self.help_button_x + self.help_button_width and
            self.help_button_y <= y <= self.help_button_y + self.help_button_height
        )
    
    def render(self, frame: np.ndarray, cursor_pos: Optional[Tuple[int, int]] = None):
        """
        渲染教程界面或Help按钮
        cursor_pos: 鼠标/手指位置，用于悬停检测
        """
        if self.is_active:
            self._render_tutorial(frame, cursor_pos)
        elif self.show_help_button:
            self._render_help_button(frame, cursor_pos)
    
    def _render_tutorial(self, frame: np.ndarray, cursor_pos: Optional[Tuple[int, int]]):
        """渲染教程页面"""
        page = self.pages[self.current_page]
        
        # 创建半透明背景
        overlay = np.zeros_like(frame)
        
        # 绘制主框背景（深色半透明）
        cv2.rectangle(overlay,
                     (self.box_x, self.box_y),
                     (self.box_x + self.box_width, self.box_y + self.box_height),
                     (30, 30, 30), -1)
        
        # 混合背景
        cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0, frame)
        
        # 绘制边框（浅蓝色）
        cv2.rectangle(frame,
                     (self.box_x, self.box_y),
                     (self.box_x + self.box_width, self.box_y + self.box_height),
                     self.border_color, self.border_thickness, cv2.LINE_AA)
        
        # 绘制标题
        title_y = self.box_y + 70
        title_scale = 1.2
        title_thickness = 3
        (title_w, title_h), _ = cv2.getTextSize(page.title, cv2.FONT_HERSHEY_DUPLEX, title_scale, title_thickness)
        title_x = self.box_x + (self.box_width - title_w) // 2
        
        # 标题阴影
        cv2.putText(frame, page.title, (title_x + 3, title_y + 3),
                   cv2.FONT_HERSHEY_DUPLEX, title_scale, (0, 0, 0), title_thickness + 1, cv2.LINE_AA)
        # 标题文字
        cv2.putText(frame, page.title, (title_x, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, title_scale, self.title_color, title_thickness, cv2.LINE_AA)
        
        # 绘制分隔线
        line_y = title_y + 30
        cv2.line(frame,
                (self.box_x + 80, line_y),
                (self.box_x + self.box_width - 80, line_y),
                self.border_color, 2, cv2.LINE_AA)
        
        # 绘制内容
        content_start_y = line_y + 50
        line_height = 32
        content_scale = 0.65
        content_thickness = 2
        
        for i, line in enumerate(page.content):
            y = content_start_y + i * line_height
            
            # 跳过超出边界的行
            if y > self.box_y + self.box_height - 80:
                break
            
            # 高亮显示（以数字开头的行）
            if line.strip() and line.strip()[0].isdigit():
                color = self.highlight_color
                thickness = 2
            else:
                color = self.text_color
                thickness = 1
            
            # 所有内容都左对齐
            x = self.box_x + 100
            
            cv2.putText(frame, line, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, content_scale, color, thickness, cv2.LINE_AA)
        
        # 绘制页码
        page_indicator = f"{self.current_page + 1} / {len(self.pages)}"
        indicator_y = self.box_y + self.box_height - 30
        (ind_w, ind_h), _ = cv2.getTextSize(page_indicator, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        indicator_x = self.box_x + (self.box_width - ind_w) // 2
        cv2.putText(frame, page_indicator, (indicator_x, indicator_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)
        
        # 最后一页显示额外提示
        if self.current_page == len(self.pages) - 1:
            final_text = "PRESS ANY KEY OR CLICK TO START"
            final_scale = 0.85
            final_thickness = 3
            (final_w, final_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_DUPLEX, final_scale, final_thickness)
            final_x = self.box_x + (self.box_width - final_w) // 2
            final_y = self.box_y + self.box_height - 80
            
            # 闪烁效果
            import time
            alpha = abs(np.sin(time.time() * 3))
            color = (
                int(255 * alpha),
                int(255 * alpha),
                int(255 * alpha)
            )
            
            cv2.putText(frame, final_text, (final_x, final_y),
                       cv2.FONT_HERSHEY_DUPLEX, final_scale, color, final_thickness, cv2.LINE_AA)
    
    def _render_help_button(self, frame: np.ndarray, cursor_pos: Optional[Tuple[int, int]]):
        """渲染Help按钮"""
        if cursor_pos:
            self.update_help_button_hover(cursor_pos)
        
        # 按钮背景
        if self.help_button_hover:
            bg_color = (100, 150, 200)
            border_color = (150, 200, 255)
        else:
            bg_color = (60, 100, 140)
            border_color = (100, 150, 200)
        
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (self.help_button_x, self.help_button_y),
                     (self.help_button_x + self.help_button_width,
                      self.help_button_y + self.help_button_height),
                     bg_color, -1, cv2.LINE_AA)
        
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # 按钮边框
        cv2.rectangle(frame,
                     (self.help_button_x, self.help_button_y),
                     (self.help_button_x + self.help_button_width,
                      self.help_button_y + self.help_button_height),
                     border_color, 2, cv2.LINE_AA)
        
        # 按钮文字
        text = "HELP"
        text_scale = 0.5
        text_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_scale, text_thickness)
        text_x = self.help_button_x + (self.help_button_width - text_w) // 2
        text_y = self.help_button_y + (self.help_button_height + text_h) // 2
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

