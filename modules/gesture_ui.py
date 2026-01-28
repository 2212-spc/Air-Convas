# -*- coding: utf-8 -*-
"""
手势控制的可视化UI界面 (布局优化版 - 支持长文本)
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config

class GestureUI:
    """手势控制的UI管理器"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visible = True

        # UI布局参数 - 所有按键都在屏幕内，间距合适
        
        # 1. 工具栏 (左侧)
        self.tool_panel_x = 30  # 左侧留30px边距
        self.tool_panel_y_start = 120  # 顶部留空间
        self.tool_button_width = 120  # 合适的宽度
        self.tool_button_height = 80  # 合适的高度
        self.tool_button_spacing = 100  # 合适的间距

        # 1.1 快捷动作栏 (Clear / FX)
        self.action_panel_x = self.tool_panel_x
        self.action_panel_y_start = self.tool_panel_y_start + self.tool_button_spacing * 3 + 20  # 工具下方
        self.action_button_width = 120
        self.action_button_height = 70
        self.action_button_spacing = 85
        self.action_items = [("clear", "Clear"), ("particles", "FX")]

        # 2. 颜色栏 (中左区域)
        self.color_panel_x = 200  # 工具栏右边
        self.color_panel_y_start = 80  # 上移，确保所有颜色在屏幕内
        self.color_button_size = 65  # 合适大小
        self.color_button_spacing = 80  # 减小间距，确保8个颜色都可见

        # 3. 粗细栏 (底部中央)
        self.thickness_panel_y = height - 110  # 底部留110px
        self.thickness_panel_x_start = width // 2 - 320  # 居中
        self.thickness_button_width = 125  # 合适宽度
        self.thickness_button_height = 75  # 合适高度
        self.thickness_button_spacing = 140  # 合适间距

        # 4. 笔刷类型栏 (右侧)
        self.brush_panel_x = width - 160  # 右侧留160px
        self.brush_panel_y_start = 120  # 和其他对齐
        self.brush_button_width = 120  # 合适宽度
        self.brush_button_height = 85  # 合适高度
        self.brush_button_spacing = 100  # 合适间距

        # 碰撞检测容差（方便点击）
        self.hit_tolerance = 45  # 合适的容差

        # 悬停和选择状态
        self.hover_item = None  # (type, index) 例如 ("color", 2)
        self.pinch_ready = False  # 是否准备捏合选择
        self._hover_lock_frames = 0  # 悬停锁定计数器，防止抖动
        self._hover_frames = 0
        self._last_hover = None
        self._dwell_item = None
        self.dwell_frames = 12

    def toggle_visibility(self):
        self.visible = not self.visible
    
    def is_in_dead_zone(self, point: Tuple[int, int], brush_manager) -> bool:
        """
        检查点是否在按钮死区内（按钮区域+周围边距）
        死区：防止在按钮区域误触开始画画
        """
        if not self.visible:
            return False
        
        x, y = point
        margin = 25  # 按钮周围的额外死区边距
        
        # 1. 检查工具按钮区域（包括周围空隙）
        for i in range(len(brush_manager.TOOLS)):
            btn_x = self.tool_panel_x
            btn_y = self.tool_panel_y_start + i * self.tool_button_spacing
            if self._point_in_rect(x, y, 
                                   btn_x - margin, 
                                   btn_y - margin,
                                   self.tool_button_width + 2 * margin,
                                   self.tool_button_height + 2 * margin):
                return True
        
        # 2. 检查动作按钮区域
        for i in range(len(self.action_items)):
            btn_x = self.action_panel_x
            btn_y = self.action_panel_y_start + i * self.action_button_spacing
            if self._point_in_rect(x, y,
                                   btn_x - margin,
                                   btn_y - margin,
                                   self.action_button_width + 2 * margin,
                                   self.action_button_height + 2 * margin):
                return True
        
        # 3. 检查颜色按钮区域
        if brush_manager.tool == "pen":
            for i in range(len(brush_manager.COLOR_NAMES)):
                btn_x = self.color_panel_x
                btn_y = self.color_panel_y_start + i * self.color_button_spacing
                if self._point_in_circle(x, y, btn_x, btn_y, 
                                        self.color_button_size // 2 + margin):
                    return True
        
        # 4. 检查粗细按钮区域
        for i in range(len(brush_manager.THICKNESSES)):
            btn_x = self.thickness_panel_x_start + i * self.thickness_button_spacing
            btn_y = self.thickness_panel_y
            if self._point_in_rect(x, y,
                                   btn_x - margin,
                                   btn_y - margin,
                                   self.thickness_button_width + 2 * margin,
                                   self.thickness_button_height + 2 * margin):
                return True
        
        # 5. 检查笔刷按钮区域
        if brush_manager.tool == "pen":
            for i in range(len(brush_manager.BRUSH_TYPES)):
                btn_x = self.brush_panel_x
                btn_y = self.brush_panel_y_start + i * self.brush_button_spacing
                if self._point_in_rect(x, y,
                                       btn_x - margin,
                                       btn_y - margin,
                                       self.brush_button_width + 2 * margin,
                                       self.brush_button_height + 2 * margin):
                    return True
        
        return False

    def handle_mouse_click(self, point: Tuple[int, int], brush_manager) -> bool:
        """
        处理鼠标点击
        返回: True表示点击被处理
        """
        if not self.visible:
            return False
        
        # 更新悬停状态以找到点击的项
        self.update_hover(point, brush_manager)
        
        if self.hover_item:
            # 选择点击的项
            self.select_hover_item(brush_manager)
            return True
        
        return False
    
    def update_hover(self, point: Tuple[int, int], brush_manager) -> Optional[Tuple[str, int]]:
        """
        更新悬停状态（带智能锁定）
        返回: (类型, 索引) 如果悬停在某个按钮上
        """
        if not self.visible:
            return None

        x, y = point
        new_hover = None
        tol = self.hit_tolerance  # 容差
        
        # 智能锁定：记录最近的按钮及距离
        nearest_button = None
        nearest_distance = float('inf')
        smart_lock_threshold = 60  # 智能锁定距离阈值（像素）

        # 检查工具按钮 (Tool)
        for i in range(len(brush_manager.TOOLS)):
            btn_x = self.tool_panel_x + self.tool_button_width // 2  # 按钮中心X
            btn_y = self.tool_panel_y_start + i * self.tool_button_spacing + self.tool_button_height // 2  # 按钮中心Y
            
            # 计算距离
            distance = ((x - btn_x) ** 2 + (y - btn_y) ** 2) ** 0.5
            
            # 直接点击检测
            if self._point_in_rect(x, y, 
                                   self.tool_panel_x - tol, 
                                   self.tool_panel_y_start + i * self.tool_button_spacing - tol,
                                   self.tool_button_width + 2 * tol,
                                   self.tool_button_height + 2 * tol):
                new_hover = ("tool", i)
                break
            
            # 智能锁定：记录最近的按钮
            if distance < nearest_distance and distance < smart_lock_threshold:
                nearest_distance = distance
                nearest_button = ("tool", i)

        # 检查动作按钮 (Action)
        if new_hover is None:
            for i in range(len(self.action_items)):
                btn_x = self.action_panel_x + self.action_button_width // 2
                btn_y = self.action_panel_y_start + i * self.action_button_spacing + self.action_button_height // 2
                
                distance = ((x - btn_x) ** 2 + (y - btn_y) ** 2) ** 0.5
                
                if self._point_in_rect(x, y, 
                                       self.action_panel_x - tol,
                                       self.action_panel_y_start + i * self.action_button_spacing - tol,
                                       self.action_button_width + 2 * tol,
                                       self.action_button_height + 2 * tol):
                    new_hover = ("action", i)
                    break
                
                if distance < nearest_distance and distance < smart_lock_threshold:
                    nearest_distance = distance
                    nearest_button = ("action", i)

        # 检查颜色按钮 (Color) - 只有非橡皮/激光模式下才有效
        if new_hover is None and brush_manager.tool == "pen":
            for i in range(len(brush_manager.COLOR_NAMES)):
                btn_x = self.color_panel_x
                btn_y = self.color_panel_y_start + i * self.color_button_spacing
                
                distance = ((x - btn_x) ** 2 + (y - btn_y) ** 2) ** 0.5
                
                if self._point_in_circle(x, y, btn_x, btn_y, self.color_button_size // 2 + tol):
                    new_hover = ("color", i)
                    break
                
                if distance < nearest_distance and distance < smart_lock_threshold:
                    nearest_distance = distance
                    nearest_button = ("color", i)

        # 检查粗细按钮 (Thickness)
        if new_hover is None:
            for i in range(len(brush_manager.THICKNESSES)):
                btn_x = self.thickness_panel_x_start + i * self.thickness_button_spacing + self.thickness_button_width // 2
                btn_y = self.thickness_panel_y + self.thickness_button_height // 2
                
                distance = ((x - btn_x) ** 2 + (y - btn_y) ** 2) ** 0.5
                
                if self._point_in_rect(x, y, 
                                       self.thickness_panel_x_start + i * self.thickness_button_spacing - tol,
                                       self.thickness_panel_y - tol,
                                       self.thickness_button_width + 2 * tol,
                                       self.thickness_button_height + 2 * tol):
                    new_hover = ("thickness", i)
                    break
                
                if distance < nearest_distance and distance < smart_lock_threshold:
                    nearest_distance = distance
                    nearest_button = ("thickness", i)

        # 检查笔刷类型按钮 (BrushType) - 只有画笔模式有效
        if new_hover is None and brush_manager.tool == "pen":
            for i in range(len(brush_manager.BRUSH_TYPES)):
                btn_x = self.brush_panel_x + self.brush_button_width // 2
                btn_y = self.brush_panel_y_start + i * self.brush_button_spacing + self.brush_button_height // 2
                
                distance = ((x - btn_x) ** 2 + (y - btn_y) ** 2) ** 0.5
                
                if self._point_in_rect(x, y, 
                                       self.brush_panel_x - tol,
                                       self.brush_panel_y_start + i * self.brush_button_spacing - tol,
                                       self.brush_button_width + 2 * tol,
                                       self.brush_button_height + 2 * tol):
                    new_hover = ("brush", i)
                    break
                
                if distance < nearest_distance and distance < smart_lock_threshold:
                    nearest_distance = distance
                    nearest_button = ("brush", i)
        
        # 智能锁定：如果没有直接点击到按钮，但接近某个按钮，则锁定到该按钮
        if new_hover is None and nearest_button is not None:
            new_hover = nearest_button
            print(f"智能锁定到按钮: {nearest_button[0]} #{nearest_button[1]} (距离: {nearest_distance:.1f}px)")

        # 混合
        overlay = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        alpha = config.UI_THEME["opacity"]
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _render_button(self, draw, x, y, w, h, text, is_selected, is_hover, active_color_key, has_preview=False, brush_type=None):
        # 颜色获取
        if is_selected:
            bg_color = self._get_color(active_color_key)
            text_color = self._get_color("text_active")
        elif is_hover:
            bg_color = self._get_color("bg_hover")
            text_color = self._get_color("text_normal")
        else:
            bg_color = self._get_color("bg_normal")
            text_color = self._get_color("text_normal")

        # 背景
        self._draw_rounded_rect(draw, (x, y, w, h), bg_color, radius=config.UI_RADIUS)
        if is_selected:
             self._draw_rounded_rect(draw, (x, y, w, h), (255,255,255,0), radius=config.UI_RADIUS, width=2)

        # 笔刷预览
        text_y_offset = 0
        if has_preview and brush_type:
            self._draw_brush_preview_pil(draw, brush_type, x + 15, x + w - 15, y + 20)
            text_y_offset = 12 # 稍微上移文字，因为有预览图

        # 文字居中绘制
        if text:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            # 计算精确居中位置
            draw.text((x + (w - text_w)/2, y + (h - text_h)/2 + text_y_offset), text, font=self.font, fill=text_color)

    # ================= 辅助函数 =================
    
    def _get_color(self, key):
        bgr = config.UI_THEME.get(key, (128, 128, 128))
        return (bgr[2], bgr[1], bgr[0], 255)

    def _draw_rounded_rect(self, draw, rect, color, radius=10, width=0):
        x, y, w, h = rect
        draw.rounded_rectangle((x, y, x+w, y+h), radius=radius, fill=color if width==0 else None, outline=color if width>0 else None, width=width)

    def _draw_brush_preview_pil(self, draw, brush_type, x1, x2, y):
        if brush_type == "solid":
            draw.line((x1, y, x2, y), fill="white", width=3)
        elif brush_type in ["dashed", "dash"]:
            step = 10
            for i in range(int(x1), int(x2), step * 2):
                draw.line((i, y, min(i+step, x2), y), fill="white", width=3)
        elif brush_type == "glow":
            draw.line((x1, y, x2, y), fill=(100, 100, 255, 150), width=6)
            draw.line((x1, y, x2, y), fill="white", width=2)
        elif brush_type == "rainbow":
            w = (x2 - x1) / 3
            draw.line((x1, y, x1+w, y), fill="red", width=3)
            draw.line((x1+w, y, x1+2*w, y), fill="green", width=3)
            draw.line((x1+2*w, y, x2, y), fill="blue", width=3)
        else:
            draw.line((x1, y, x2, y), fill="white", width=3)

    # ================= 交互判定 (更新为动态坐标) =================

    def update_hover(self, point: Tuple[int, int], brush_manager) -> Optional[Tuple[str, int]]:
        if not self.visible: return None
        x, y = point
        tol = self.hit_tolerance
        
        # 1. 检查左侧面板 (Tool + Action)
        # Tools
        for i in range(len(brush_manager.TOOLS)):
            btn_y = self.panel_start_y + i * (self.left_btn_height + self.left_spacing)
            if self._point_in_rect(x, y, self.left_panel_x - tol, btn_y - tol, 
                                   self.left_btn_width + 2*tol, self.left_btn_height + 2*tol):
                return self._set_hover("tool", i)
        
        # Actions
        action_items = [("clear", "Clear"), ("particles", "FX")]
        tool_count = len(brush_manager.TOOLS)
        for i in range(len(action_items)):
            global_index = tool_count + i
            btn_y = self.panel_start_y + global_index * (self.left_btn_height + self.left_spacing)
            if self._point_in_rect(x, y, self.left_panel_x - tol, btn_y - tol, 
                                   self.left_btn_width + 2*tol, self.left_btn_height + 2*tol):
                return self._set_hover("action", i)

        # 2. 检查中间颜色 (仅Pen)
        if brush_manager.tool == "pen":
            for i in range(len(brush_manager.COLOR_NAMES)):
                cy = self.panel_start_y + i * (self.color_btn_size + self.color_spacing) + self.color_btn_size / 2
                if self._point_in_circle(x, y, self.color_panel_center_x, cy, self.color_btn_size // 2 + tol):
                    return self._set_hover("color", i)

        # 3. 检查右侧笔触 (仅Pen)
        if brush_manager.tool == "pen":
            for i in range(len(brush_manager.BRUSH_TYPES)):
                btn_y = self.panel_start_y + i * (self.right_btn_height + self.right_spacing)
                if self._point_in_rect(x, y, self.right_panel_x - tol, btn_y - tol,
                                       self.right_btn_width + 2*tol, self.right_btn_height + 2*tol):
                    return self._set_hover("brush", i)

        # 4. 底部粗细
        thickness_total_w = len(brush_manager.THICKNESSES) * self.thickness_btn_width + (len(brush_manager.THICKNESSES) - 1) * 20
        t_start_x = (self.width - thickness_total_w) // 2
        for i in range(len(brush_manager.THICKNESSES)):
            btn_x = t_start_x + i * (self.thickness_btn_width + 20)
            if self._point_in_rect(x, y, btn_x - tol, self.thickness_panel_y - tol,
                                   self.thickness_btn_width + 2*tol, self.thickness_btn_height + 2*tol):
                return self._set_hover("thickness", i)

        self._clear_hover()
        return None

    def _set_hover(self, item_type, index):
        self.hover_item = (item_type, index)
        self._hover_lock_frames = 5
        
        if self.hover_item == self._last_hover:
            self._hover_frames += 1
        else:
            self._hover_frames = 0
        self._last_hover = self.hover_item
        
        if item_type in ("tool", "action") and self._hover_frames >= self.dwell_frames:
            self._dwell_item = self.hover_item
            self._hover_frames = 0
        return self.hover_item

    def _clear_hover(self):
        if self._hover_lock_frames > 0:
            self._hover_lock_frames -= 1
        else:
            self.hover_item = None
            self._hover_frames = 0
            self._last_hover = None

    def select_hover_item(self, brush_manager) -> dict:
        result = {"selected": False, "item_type": None, "action": None}
        if not self.hover_item: return result
        
        item_type, index = self.hover_item
        result["item_type"] = item_type
        result["selected"] = True

        if item_type == "tool":
            brush_manager.current_tool_index = index
        elif item_type == "color":
            brush_manager.current_color_index = index
        elif item_type == "thickness":
            brush_manager.current_thickness_index = index
        elif item_type == "brush":
            brush_manager.current_brush_type_index = index
        elif item_type == "action":
            action_items = ["clear", "particles"]
            if 0 <= index < len(action_items):
                result["action"] = action_items[index]
        
        return result

    def consume_dwell_item(self):
        if self._dwell_item:
            item = self._dwell_item
            self._dwell_item = None
            return item
        return None

    def _point_in_circle(self, px, py, cx, cy, radius):
        return (px - cx)**2 + (py - cy)**2 <= radius**2

    def _point_in_rect(self, px, py, rx, ry, rw, rh):
        return rx <= px <= rx + rw and ry <= py <= ry + rh