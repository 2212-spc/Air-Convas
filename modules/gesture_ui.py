# -*- coding: utf-8 -*-
"""
手势控制的可视化UI界面 - 综合控制台版 (标签优化与布局对齐版)
修改内容：
1. 左侧面板宽度统一增加至 120px，以容纳四字标签
2. 标签全面汉化并统一长度：直线辅助、橡皮擦、激光笔、清空画布、粒子特效
3. 文本自动居中对齐
"""

from typing import Tuple, Optional, Dict, List, Any
import cv2
import numpy as np
import math
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import config

@dataclass
class HUDStyle:
    """HUD 样式配置类"""
    bg_color: Tuple[int, int, int, int] = (20, 20, 20, 220)
    text_color_main: Tuple[int, int, int] = (255, 255, 255)
    text_color_sub: Tuple[int, int, int] = (180, 180, 180)
    accent_color_pen: Tuple[int, int, int] = (0, 255, 255)    # 青色
    accent_color_eraser: Tuple[int, int, int] = (255, 80, 80) # 红色
    accent_color_laser: Tuple[int, int, int] = (0, 100, 255)  # 橙色/深蓝
    font_size_large: int = 28
    font_size_main: int = 20
    font_size_small: int = 14
    capsule_radius: int = 18
    margin: int = 15

class GestureUI:
    """手势控制的UI管理器 - 集成 HUD 与交互控件"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visible = True
        self.style = HUDStyle()

        # ================== 字体初始化 ==================
        self._init_fonts()

        # ================== 1. 控件布局 (左侧扩容以容纳4字) ==================
        # 左侧工具栏
        self.tool_panel_x = 10
        self.tool_panel_y_start = 80
        self.tool_button_width = 120  # [修改] 增加宽度以容纳 "橡皮擦" 等
        self.tool_button_height = 55
        self.tool_button_spacing = 65

        # 左侧动作栏
        self.action_panel_x = self.tool_panel_x
        self.action_panel_y_start = self.tool_panel_y_start + self.tool_button_spacing * 3 + 20
        self.action_button_width = 120 # [修改] 统一宽度
        self.action_button_height = 50
        self.action_button_spacing = 60
        
        # [修改] 更新为更详细的中文标签
        self.action_items = [
            ("clear", "清空画布"), 
            ("particles", "粒子特效"), 
            ("effects", "互动特效")
        ]

        # 顶部颜色栏
        self.color_panel_y = 80 
        self.color_panel_x_start = 160 # [修改] 左侧变宽了，颜色栏稍微右移避免重叠
        self.color_button_size = 36
        self.color_button_spacing = 50

        # 底部粗细栏
        self.thickness_panel_y = height - 60
        self.thickness_button_width = 60
        self.thickness_button_height = 45
        self.thickness_button_spacing = 70

        # 右侧笔刷栏
        self.brush_panel_x = width - 85
        self.brush_panel_y_start = 80
        self.brush_button_width = 75
        self.brush_button_height = 55
        self.brush_button_spacing = 65

        # ================== 2. 交互状态 ==================
        self.hit_tolerance = 25
        self.hover_item = None
        self._hover_frames = 0
        self._last_hover = None
        self._selection_flash = 0
        self._last_selected = None
        self.dwell_frames = 18 
        self._pending_selection = None

    def _init_fonts(self):
        self.font_path = config.UI_FONT_PATH
        try:
            if self.font_path:
                self.font = ImageFont.truetype(self.font_path, self.style.font_size_main)
                self.font_small = ImageFont.truetype(self.font_path, self.style.font_size_small)
                self.font_large = ImageFont.truetype(self.font_path, self.style.font_size_large)
            else:
                raise IOError("Font path not set")
        except Exception:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def toggle_visibility(self):
        self.visible = not self.visible

    def render(self, frame: np.ndarray, brush_manager, action_state: Optional[dict] = None, hud_data: Optional[dict] = None):
        if not self.visible:
            return
        if action_state is None: action_state = {}
        if hud_data is None: hud_data = {}

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, 'RGBA')

        if self._selection_flash > 0:
            self._selection_flash -= 1

        self._render_hud(draw, brush_manager, hud_data, action_state)
        self._render_panels(draw, brush_manager, action_state)
        self._render_dwell_progress(draw)

        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ================== HUD 渲染子系统 ==================

    def _render_hud(self, draw: ImageDraw, brush_manager, hud_data: dict, action_state: dict):
        # 1. 左侧 FPS
        fps = hud_data.get('fps', 0.0)
        mode = hud_data.get('mode', 'DRAW')
        fps_color = (50, 200, 100) if fps > 20 else ((255, 200, 0) if fps > 10 else (255, 50, 50))
        
        self._draw_status_capsule(
            draw, x=20, y=15, w=140, h=40,
            text=f"FPS: {fps:.1f}", sub_text=mode,
            accent_color=fps_color
        )

        # 2. 中间工具栏 (大标题)
        tool_name = brush_manager.tool.upper()
        # [修改] 中文映射表
        tool_cn_map = {"PEN": "画笔", "ERASER": "橡皮擦", "LASER": "激光笔"}
        tool_cn = tool_cn_map.get(tool_name, tool_name)
        
        if tool_name == "PEN":
            theme_color = self._bgr_to_rgb(brush_manager.color)
            sub_info = f"{brush_manager.thickness}px | {brush_manager.brush_type}"
        elif tool_name == "LASER":
            theme_color = self.style.accent_color_laser
            sub_info = "演示模式"
        else:
            theme_color = self.style.accent_color_eraser
            sub_info = f"Size: {config.ERASER_SIZE}"

        center_x = self.width // 2
        self._draw_status_capsule(
            draw, x=center_x - 110, y=10, w=220, h=50,
            text=tool_cn, sub_text=sub_info,
            accent_color=theme_color, bg_color=(30, 30, 30, 240), is_active=True
        )

        # 3. 右侧历史记录 (逻辑优化：显示可撤销步数)
        right_start_x = self.width - 20
        history_info = hud_data.get('history', '0/0')
        
        import re
        nums = re.findall(r'\d+', history_info)
        
        # [优化] 解析数字并减 1，让初始状态显示为 0
        try:
            raw_total = int(nums[0]) if len(nums) > 0 else 0
            redo_val = int(nums[1]) if len(nums) > 1 else 0
            
            # 如果只有1条记录(空白页)，显示撤销为0；否则显示 总数-1
            undo_display = max(0, raw_total - 1)
        except:
            undo_display = 0
            redo_val = 0
        
        display_text = f"撤销:{undo_display} 重做:{redo_val}"
        
        # 动态计算宽度
        bbox = draw.textbbox((0, 0), display_text, font=self.font)
        text_w = bbox[2] - bbox[0]
        capsule_w = max(140, text_w + 50) 

        self._draw_status_capsule(
            draw, x=right_start_x - capsule_w, y=15, w=capsule_w, h=40,
            text=display_text, accent_color=(200, 200, 200)
        )
        
        # 4. 状态小图标 (适配4字标签宽度)
        # [修改] 标签全称
        status_icons = []
        if action_state.get('particles'): status_icons.append(("粒子特效", (255, 215, 0))) 
        if action_state.get('line_assist'): status_icons.append(("直线辅助", (100, 200, 255))) # [修改] 原"直线"改为"直线辅助"
        if action_state.get('pen_effect'): status_icons.append(("压感模拟", (255, 150, 200))) # [修改] 原"压感"改为"压感模拟"
        
        icon_start_x = right_start_x - capsule_w - 20 
        icon_width = 85 # [修改] 增加宽度以容纳4个汉字
        
        for i, (label, color) in enumerate(status_icons):
            icon_x = icon_start_x - (i + 1) * (icon_width + 10)
            
            # 背景
            self._draw_rounded_rect_pil(draw, icon_x, 15, icon_width, 32, (40, 40, 40, 200), radius=10)
            
            # 文字居中
            bbox = draw.textbbox((0, 0), label, font=self.font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((icon_x + (icon_width - tw)//2, 15 + (32 - th)//2 - 2), label, font=self.font_small, fill=color)

        # 5. Toast
        message = hud_data.get('message', '')
        if message:
            self._draw_toast_message(draw, message)

    def _draw_status_capsule(self, draw, x, y, w, h, text, sub_text=None, accent_color=(255,255,255), bg_color=None, is_active=False):
        if bg_color is None: bg_color = self.style.bg_color
        radius = h // 2
        self._draw_rounded_rect_pil(draw, x, y, w, h, bg_color, radius=radius)
        if is_active:
             self._draw_rounded_rect_pil(draw, x, y, w, h, outline=accent_color, width=2, radius=radius)
        
        bar_h = h - 16
        draw.line((x + 12, y + 8, x + 12, y + 8 + bar_h), fill=accent_color, width=3)
        
        text_x = x + 28
        title_y = y + 4 if sub_text else y + h//2 - 12
        draw.text((text_x, title_y), text, font=self.font, fill=self.style.text_color_main)
        if sub_text:
            draw.text((text_x, y + 24), sub_text, font=self.font_small, fill=self.style.text_color_sub)

    def _draw_toast_message(self, draw, message):
        bbox = draw.textbbox((0, 0), message, font=self.font_large)
        text_w = bbox[2] - bbox[0]
        toast_w = text_w + 80
        toast_h = 50
        toast_x = (self.width - toast_w) // 2
        self._draw_rounded_rect_pil(draw, toast_x, 80, toast_w, toast_h, (0, 0, 0, 180), radius=25)
        draw.text((toast_x + 40, 88), message, font=self.font_large, fill=(255, 255, 255))

    # ================== 面板渲染系统 ==================

    def _render_panels(self, draw, brush_manager, action_state):
        # 1. 左侧工具 (标签更新)
        # [修改] 使用新的标签列表
        labels = ["画笔", "橡皮擦", "激光笔"]
        for i, label in enumerate(labels):
            y = self.tool_panel_y_start + i * self.tool_button_spacing
            self._render_button_pil(draw, self.tool_panel_x, y, self.tool_button_width, self.tool_button_height,
                                    label, (i == brush_manager.current_tool_index), 
                                    (self.hover_item == ("tool", i)), "tool", i)

        # 2. 左侧动作 (标签在init中更新)
        for i, (key, label) in enumerate(self.action_items):
            y = self.action_panel_y_start + i * self.action_button_spacing
            is_active = action_state.get(key, False)
            is_sel = is_active
            if key == "clear" and self._selection_flash > 0 and self._last_selected == ("action", i): is_sel = True
            
            self._render_button_pil(draw, self.action_panel_x, y, self.action_button_width, self.action_button_height,
                                    label, is_sel, (self.hover_item == ("action", i)), "action", i)

        # 3. 颜色
        if brush_manager.tool == "pen":
            for i, c_name in enumerate(brush_manager.COLOR_NAMES):
                rgb = self._bgr_to_rgb(brush_manager.COLORS[c_name])
                x = self.color_panel_x_start + i * self.color_button_spacing
                y = self.color_panel_y
                r = self.color_button_size // 2
                if self.hover_item == ("color", i): r += 3
                if i == brush_manager.current_color_index:
                    draw.ellipse((x-r-3, y-r-3, x+r+3, y+r+3), outline=(255,255,255), width=2)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=rgb)

        # 4. 粗细
        t_start_x = (self.width - (len(brush_manager.THICKNESSES) * self.thickness_button_spacing)) // 2
        for i, thick in enumerate(brush_manager.THICKNESSES):
            x = t_start_x + i * self.thickness_button_spacing
            y = self.thickness_panel_y
            self._render_button_pil(draw, x, y, self.thickness_button_width, self.thickness_button_height,
                                    "", (i == brush_manager.current_thickness_index), 
                                    (self.hover_item == ("thickness", i)), "thickness", i)
            ly = y + self.thickness_button_height // 2
            draw.line((x+10, ly, x+self.thickness_button_width-10, ly), fill="white", width=thick)

        # 5. 右侧笔刷 (含预览)
        if brush_manager.tool == "pen":
            brush_labels = {"solid": "实线", "dashed": "虚线", "glow": "发光", "marker": "马克", "rainbow": "彩虹"}
            for i, b_type in enumerate(brush_manager.BRUSH_TYPES):
                y = self.brush_panel_y_start + i * self.brush_button_spacing
                label = brush_labels.get(b_type, b_type)
                
                self._render_button_pil(draw, self.brush_panel_x, y, self.brush_button_width, self.brush_button_height,
                                        label, (i == brush_manager.current_brush_type_index),
                                        (self.hover_item == ("brush", i)), "brush", i,
                                        hide_text=True)
                
                preview_y = y + 20
                start_x = self.brush_panel_x + 10
                end_x = self.brush_panel_x + self.brush_button_width - 10
                self._draw_brush_preview_pil(draw, b_type, start_x, end_x, preview_y)
                
                bbox = draw.textbbox((0, 0), label, font=self.font_small)
                tw = bbox[2] - bbox[0]
                text_x = self.brush_panel_x + (self.brush_button_width - tw) // 2
                draw.text((text_x, y + 35), label, font=self.font_small, fill=(200, 200, 200))

    def _render_button_pil(self, draw, x, y, w, h, text, is_selected, is_hover, item_type, index, hide_text=False):
        is_flash = (self._selection_flash > 0 and self._last_selected == (item_type, index))
        bg = (50, 200, 100, 255) if is_flash else ((0, 120, 215, 230) if is_selected else ((70, 70, 70, 220) if is_hover else (40, 40, 40, 200)))
        
        self._draw_rounded_rect_pil(draw, x, y, w, h, bg, radius=10)
        
        if is_selected:
            self._draw_rounded_rect_pil(draw, x, y, w, h, outline=(255, 255, 255, 200), width=2, radius=10)
        elif is_hover:
            self._draw_rounded_rect_pil(draw, x, y, w, h, outline=(100, 200, 255, 150), width=1, radius=10)
            
        if text and not hide_text:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            tx = x + (w - (bbox[2] - bbox[0])) // 2
            ty = y + (h - (bbox[3] - bbox[1])) // 2 - 2
            draw.text((tx, ty), text, font=self.font, fill=(255, 255, 255))

    def _draw_brush_preview_pil(self, draw, brush_type, x1, x2, y):
        width = x2 - x1
        if brush_type == "solid":
            draw.line((x1, y, x2, y), fill="white", width=3)
        elif brush_type in ["dashed", "dash"]:
            seg = width // 5
            draw.line((x1, y, x1+seg, y), fill="white", width=3)
            draw.line((x1+2*seg, y, x1+3*seg, y), fill="white", width=3)
            draw.line((x1+4*seg, y, x2, y), fill="white", width=3)
        elif brush_type == "glow":
            draw.line((x1, y, x2, y), fill=(100, 100, 255, 128), width=6)
            draw.line((x1, y, x2, y), fill="white", width=2)
        elif brush_type == "marker":
            draw.line((x1, y, x2, y), fill=(255, 255, 255, 180), width=5)
        elif brush_type == "rainbow":
            seg = width // 3
            draw.line((x1, y, x1+seg, y), fill=(255, 0, 0), width=3)
            draw.line((x1+seg, y, x1+2*seg, y), fill=(0, 255, 0), width=3)
            draw.line((x1+2*seg, y, x2, y), fill=(0, 0, 255), width=3)

    def _render_dwell_progress(self, draw):
        if self.hover_item is None or self._hover_frames < 3: return
        progress = self.get_dwell_progress()
        if progress <= 0: return
        cx, cy = self._get_item_center_coords(self.hover_item)
        if cx is None: return
        
        radius = 28
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.arc(bbox, 0, 360, fill=(100, 100, 100, 100), width=4)
        draw.arc(bbox, -90, -90 + 360 * progress, fill=(0, 255, 200, 255), width=4)

    def _draw_rounded_rect_pil(self, draw, x, y, w, h, fill=None, outline=None, width=1, radius=10):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)

    def _bgr_to_rgb(self, bgr):
        return (bgr[2], bgr[1], bgr[0])

    def _get_item_center_coords(self, item_tuple):
        t, i = item_tuple
        if t == "tool":
            return (self.tool_panel_x + self.tool_button_width//2, self.tool_panel_y_start + i * self.tool_button_spacing + self.tool_button_height//2)
        elif t == "action":
            return (self.action_panel_x + self.action_button_width//2, self.action_panel_y_start + i * self.action_button_spacing + self.action_button_height//2)
        elif t == "color":
            return (self.color_panel_x_start + i * self.color_button_spacing, self.color_panel_y)
        elif t == "thickness":
            t_start_x = (self.width - (5 * self.thickness_button_spacing)) // 2 
            return (t_start_x + i * self.thickness_button_spacing + self.thickness_button_width//2, self.thickness_panel_y + self.thickness_button_height//2)
        elif t == "brush":
            return (self.brush_panel_x + self.brush_button_width//2, self.brush_panel_y_start + i * self.brush_button_spacing + self.brush_button_height//2)
        return (None, None)

    # 交互逻辑保持不变
    def update_hover(self, point, brush_manager):
        if not self.visible: return None
        x, y = point; tol = self.hit_tolerance; new_hover = None
        
        for i in range(len(brush_manager.TOOLS)):
            bx, by = self.tool_panel_x, self.tool_panel_y_start + i * self.tool_button_spacing
            if self._hit_rect(x, y, bx, by, self.tool_button_width, self.tool_button_height, tol): new_hover = ("tool", i); break
        
        if not new_hover:
            for i in range(len(self.action_items)):
                bx, by = self.action_panel_x, self.action_panel_y_start + i * self.action_button_spacing
                if self._hit_rect(x, y, bx, by, self.action_button_width, self.action_button_height, tol): new_hover = ("action", i); break
        
        if not new_hover and brush_manager.tool == "pen":
            for i in range(len(brush_manager.COLOR_NAMES)):
                cx = self.color_panel_x_start + i * self.color_button_spacing; cy = self.color_panel_y
                if (x-cx)**2 + (y-cy)**2 <= (self.color_button_size//2 + tol)**2: new_hover = ("color", i); break

        if not new_hover:
            t_start_x = (self.width - (len(brush_manager.THICKNESSES) * self.thickness_button_spacing)) // 2
            for i in range(len(brush_manager.THICKNESSES)):
                bx = t_start_x + i * self.thickness_button_spacing; by = self.thickness_panel_y
                if self._hit_rect(x, y, bx, by, self.thickness_button_width, self.thickness_button_height, tol): new_hover = ("thickness", i); break
        
        if not new_hover and brush_manager.tool == "pen":
            for i in range(len(brush_manager.BRUSH_TYPES)):
                bx, by = self.brush_panel_x, self.brush_panel_y_start + i * self.brush_button_spacing
                if self._hit_rect(x, y, bx, by, self.brush_button_width, self.brush_button_height, tol): new_hover = ("brush", i); break
        
        if new_hover == self._last_hover and new_hover: self._hover_frames += 1
        else: self._hover_frames = 0
        self._last_hover = new_hover; self.hover_item = new_hover
        if self.hover_item and self._hover_frames >= self.dwell_frames: self._pending_selection = self.hover_item; self._hover_frames = 0
        return self.hover_item

    def _hit_rect(self, mx, my, x, y, w, h, tol):
        return (x - tol <= mx <= x + w + tol) and (y - tol <= my <= y + h + tol)
    
    def get_dwell_progress(self):
        if not self.hover_item: return 0.0
        return min(1.0, self._hover_frames / self.dwell_frames)

    def consume_pending_selection(self, brush_manager):
        result = {"selected": False, "item_type": None, "action": None}
        if self._pending_selection:
            t, i = self._pending_selection; self._pending_selection = None
            result = {"selected": True, "item_type": t, "action": None}
            self._selection_flash = 5; self._last_selected = (t, i)
            if t == "tool": brush_manager.current_tool_index = i
            elif t == "color": brush_manager.current_color_index = i
            elif t == "thickness": brush_manager.current_thickness_index = i
            elif t == "brush": brush_manager.current_brush_type_index = i
            elif t == "action": result["action"] = self.action_items[i][0]
        return result

    def select_hover_item(self, brush_manager):
        if self.hover_item: self._pending_selection = self.hover_item; return self.consume_pending_selection(brush_manager)
        return {"selected": False, "item_type": None, "action": None}
    
    def handle_mouse_click(self, point, brush_manager):
        self.update_hover(point, brush_manager)
        if self.hover_item: self.select_hover_item(brush_manager); return True
        return False
    
    def is_in_dead_zone(self, point, brush_manager):
        self.update_hover(point, brush_manager); return self.hover_item is not None