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

        # --- 字体设置 ---
        self.font_path = config.UI_FONT_PATH
        self.font_size = config.UI_FONT_SIZE_MAIN
        try:
            self.font = ImageFont.truetype(self.font_path, self.font_size) if self.font_path else ImageFont.load_default()
            self.font_small = ImageFont.truetype(self.font_path, config.UI_FONT_SIZE_SMALL) if self.font_path else ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

        # ================== 核心布局参数计算 ==================
        
        # 1. 垂直布局约束 (让左右两侧总高度一致)
        # 设定面板总高度为屏幕高度的 60% 左右，垂直居中
        self.panel_total_height = int(height * 0.65)
        self.panel_start_y = (height - self.panel_total_height) // 2
        
        # 2. 左侧面板配置 (工具 + 动作)
        self.left_item_count = 5  # 3个工具 + 2个动作
        self.left_btn_width = 140 # [修改] 加宽以容纳“清空画布”等4字词
        self.left_btn_height = 65
        
        # 自动计算垂直间距
        if self.left_item_count > 1:
            self.left_spacing = (self.panel_total_height - (self.left_item_count * self.left_btn_height)) / (self.left_item_count - 1)
        else:
            self.left_spacing = 0
            
        self.left_panel_x = 50  # 左边距
        
        # 3. 颜色面板配置 (紧跟左侧，不重叠)
        self.color_item_count = 7 
        self.color_btn_size = 50 # 圆形直径
        
        # 自动计算颜色垂直间距
        if self.color_item_count > 1:
            self.color_spacing = (self.panel_total_height - (self.color_item_count * self.color_btn_size)) / (self.color_item_count - 1)
        else:
            self.color_spacing = 0
            
        # [动态计算] 颜色面板 X 坐标 = 左侧面板X + 按钮宽 + 间距(25px) + 半径修正
        self.color_panel_gap = 25
        self.color_panel_center_x = self.left_panel_x + self.left_btn_width + self.color_panel_gap + (self.color_btn_size // 2)

        # 4. 右侧笔触面板配置 (与左侧对称)
        self.right_item_count = 5
        self.right_btn_width = 140 # [修改] 保持对称，也加宽
        self.right_btn_height = 65
        
        if self.right_item_count > 1:
            self.right_spacing = (self.panel_total_height - (self.right_item_count * self.right_btn_height)) / (self.right_item_count - 1)
        else:
            self.right_spacing = 0
            
        self.right_panel_x = width - self.right_btn_width - 50 # 右对齐，边距与左侧一致

        # 5. 底部粗细面板 (保持在底部中央)
        self.thickness_panel_y = height - 80
        self.thickness_btn_width = 100
        self.thickness_btn_height = 50
        self.thickness_start_x = (width - (5 * self.thickness_btn_width + 4 * 20)) // 2 
        
        # [修改] 更新为完整文本映射
        self.tool_labels = {
            "pen": "画笔",
            "eraser": "橡皮擦", 
            "laser": "激光笔"
        }
        # 动作标签在 render 中定义，但也先预设好
        self.action_labels_map = {
            "clear": "清空画布",
            "particles": "粒子特效"
        }
        
        self.brush_labels = {
            "solid": "实线", 
            "dashed": "虚线", 
            "dash": "虚线", 
            "glow": "发光", 
            "marker": "马克笔", 
            "rainbow": "彩虹"
        }

        # 交互状态
        self.hit_tolerance = 30
        self.hover_item = None
        self._hover_lock_frames = 0
        self._hover_frames = 0
        self._last_hover = None
        self._dwell_item = None
        self.dwell_frames = 12

    def toggle_visibility(self):
        self.visible = not self.visible

    # ================= 渲染逻辑 (PIL) =================
    
    def render(self, frame: np.ndarray, brush_manager, action_state: Optional[dict] = None):
        if not self.visible: return
        if action_state is None: action_state = {}

        overlay = frame.copy()
        img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, 'RGBA')

        # 1. 渲染左侧面板 (合并 Tools 和 Actions)
        # 先渲染工具 (前3个)
        for i, tool_name in enumerate(brush_manager.TOOLS):
            y = self.panel_start_y + i * (self.left_btn_height + self.left_spacing)
            self._render_button(draw, 
                                x=self.left_panel_x, y=y, 
                                w=self.left_btn_width, h=self.left_btn_height,
                                text=self.tool_labels.get(tool_name, tool_name),
                                is_selected=(i == brush_manager.current_tool_index),
                                is_hover=(self.hover_item == ("tool", i)),
                                active_color_key="bg_active")

        # 再渲染动作 (紧接着工具)
        tool_count = len(brush_manager.TOOLS)
        # [修改] 使用完整长文本
        action_items = [("clear", "清空画布"), ("particles", "粒子特效")]
        
        for i, (act_key, act_label) in enumerate(action_items):
            global_index = tool_count + i
            y = self.panel_start_y + global_index * (self.left_btn_height + self.left_spacing)
            
            is_active = False
            if act_key == "particles":
                is_active = bool(action_state.get("particles", False))
            
            is_selected = is_active
            is_hover = (self.hover_item == ("action", i))
            
            # 按钮样式逻辑
            active_key = "bg_active" # 默认高亮色
            if act_key == "particles": active_key = "bg_active" # 特效开启用选中色
            if act_key == "clear": active_key = "border_hover" # 清空只需闪烁边框色
            
            self._render_button(draw,
                                x=self.left_panel_x, y=y,
                                w=self.left_btn_width, h=self.left_btn_height,
                                text=act_label,
                                is_selected=is_selected if act_key != "clear" else False,
                                is_hover=is_hover,
                                active_color_key=active_key)

        # 2. 渲染中间颜色面板
        if brush_manager.tool == "pen":
            for i, color_name in enumerate(brush_manager.COLOR_NAMES):
                y_center = self.panel_start_y + i * (self.color_btn_size + self.color_spacing) + self.color_btn_size / 2
                
                color_bgr = brush_manager.COLORS[color_name]
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
                
                is_selected = (i == brush_manager.current_color_index)
                is_hover = (self.hover_item == ("color", i))
                
                radius = self.color_btn_size // 2
                if is_hover: radius += 3
                
                draw.ellipse((self.color_panel_center_x - radius, y_center - radius, 
                              self.color_panel_center_x + radius, y_center + radius), fill=color_rgb)
                
                if is_selected:
                    draw.ellipse((self.color_panel_center_x - radius - 3, y_center - radius - 3, 
                                  self.color_panel_center_x + radius + 3, y_center + radius + 3), 
                                 outline=(255, 255, 255, 255), width=3)

        # 3. 渲染右侧笔触面板
        if brush_manager.tool == "pen":
            for i, brush_type in enumerate(brush_manager.BRUSH_TYPES):
                y = self.panel_start_y + i * (self.right_btn_height + self.right_spacing)
                
                self._render_button(draw,
                                    x=self.right_panel_x, y=y,
                                    w=self.right_btn_width, h=self.right_btn_height,
                                    text=self.brush_labels.get(brush_type, brush_type),
                                    is_selected=(i == brush_manager.current_brush_type_index),
                                    is_hover=(self.hover_item == ("brush", i)),
                                    active_color_key="bg_active",
                                    has_preview=True, brush_type=brush_type)

        # 4. 渲染底部粗细 (简单居中排列)
        thickness_total_w = len(brush_manager.THICKNESSES) * self.thickness_btn_width + (len(brush_manager.THICKNESSES) - 1) * 20
        t_start_x = (self.width - thickness_total_w) // 2
        
        for i, thickness in enumerate(brush_manager.THICKNESSES):
            x = t_start_x + i * (self.thickness_btn_width + 20)
            y = self.thickness_panel_y
            
            is_selected = (i == brush_manager.current_thickness_index)
            is_hover = (self.hover_item == ("thickness", i))
            
            self._render_button(draw, x, y, self.thickness_btn_width, self.thickness_btn_height, 
                                "", is_selected, is_hover, "bg_active")
            
            line_y = y + self.thickness_btn_height // 2
            draw.line((x + 15, line_y, x + self.thickness_btn_width - 15, line_y), 
                      fill="white", width=thickness)

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