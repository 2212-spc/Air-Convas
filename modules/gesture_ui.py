# -*- coding: utf-8 -*-
"""
手势控制 UI 界面模块 (Gesture UI)

负责绘制屏幕上的 HUD (Head-Up Display) 和交互控件。
包含基于“悬停-驻留”机制 (Hover & Dwell) 的无接触交互逻辑。
"""

from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config

# [Type Hints] UI 专用类型定义
ColorRGB = Tuple[int, int, int]
ColorRGBA = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


@dataclass
class HUDStyle:
    """
    HUD 视觉样式配置。
    集中管理配色方案，方便后期统一调整 UI 主题。
    """
    bg_color: ColorRGBA = (20, 20, 20, 220)
    text_color_main: ColorRGB = (255, 255, 255)
    text_color_sub: ColorRGB = (180, 180, 180)
    
    # 强调色
    accent_color_pen: ColorRGB = (0, 255, 255)    # 青色
    accent_color_eraser: ColorRGB = (255, 80, 80) # 红色
    accent_color_laser: ColorRGB = (0, 100, 255)  # 橙色/深蓝
    
    # 字体与尺寸
    font_size_large: int = 28
    font_size_main: int = 20
    font_size_small: int = 14
    capsule_radius: int = 18
    margin: int = 15


class GestureUI:
    """
    手势 UI 管理器。

    核心职责：
    1. **渲染 (Render)**: 使用 PIL 绘制高质量抗锯齿文字和圆角矩形，再转换回 OpenCV 格式。
    2. **交互 (Interaction)**: 实现“悬停激活”逻辑。用户手指在按钮上停留一段时间 (Dwell) 即视为点击。
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.visible = True
        self.style = HUDStyle()

        # ================== 字体初始化 ==================
        self.font: ImageFont.FreeTypeFont
        self.font_small: ImageFont.FreeTypeFont
        self.font_large: ImageFont.FreeTypeFont
        self._init_fonts()

        # ================== 1. 布局配置 (Layout) ==================
        # 左侧工具栏 (Tools)
        self.tool_panel_x = 10
        self.tool_panel_y_start = 80
        self.tool_button_width = 120
        self.tool_button_height = 55
        self.tool_button_spacing = 65

        # 左侧动作栏 (Actions)
        self.action_panel_x = self.tool_panel_x
        self.action_panel_y_start = self.tool_panel_y_start + self.tool_button_spacing * 3 + 20
        self.action_button_width = 120
        self.action_button_height = 50
        self.action_button_spacing = 60
        
        self.action_items: List[Tuple[str, str]] = [
            ("clear", "清空画布"), 
            ("particles", "粒子特效"), 
            ("effects", "互动特效")
        ]

        # 顶部颜色栏 (Colors)
        self.color_panel_y = 80 
        self.color_panel_x_start = 160
        self.color_button_size = 36
        self.color_button_spacing = 50

        # 底部粗细栏 (Thickness)
        self.thickness_panel_y = height - 60
        self.thickness_button_width = 60
        self.thickness_button_height = 45
        self.thickness_button_spacing = 70

        # 右侧笔刷栏 (Brushes)
        self.brush_panel_x = width - 85
        self.brush_panel_y_start = 80
        self.brush_button_width = 75
        self.brush_button_height = 55
        self.brush_button_spacing = 65

        # ================== 2. 交互状态机 ==================
        self.hit_tolerance = 25  # 触控容差（像素），让按钮更容易点中
        
        # 当前悬停状态
        self.hover_item: Optional[Tuple[str, int]] = None
        self._hover_frames: int = 0
        self._last_hover: Optional[Tuple[str, int]] = None
        
        # 选中反馈
        self._selection_flash: int = 0
        self._last_selected: Optional[Tuple[str, int]] = None
        
        # 驻留确认配置
        self.dwell_frames: int = 18 
        self._pending_selection: Optional[Tuple[str, int]] = None

    def _init_fonts(self) -> None:
        """加载字体资源，若失败则回退到默认字体。"""
        self.font_path = getattr(config, 'UI_FONT_PATH', None)
        try:
            if self.font_path:
                self.font = ImageFont.truetype(self.font_path, self.style.font_size_main)
                self.font_small = ImageFont.truetype(self.font_path, self.style.font_size_small)
                self.font_large = ImageFont.truetype(self.font_path, self.style.font_size_large)
            else:
                raise IOError("Font path not set")
        except Exception:
            # Fallback
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def toggle_visibility(self) -> None:
        """切换 UI 显示/隐藏状态"""
        self.visible = not self.visible

    def render(
        self, 
        frame: np.ndarray, 
        brush_manager: Any, 
        action_state: Optional[Dict[str, bool]] = None, 
        hud_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        主渲染函数。
        
        将 UI 层绘制到传入的视频帧上（原地修改）。
        为了获得美观的半透明和圆角效果，内部使用 PIL 进行绘制，然后再转回 OpenCV。
        """
        if not self.visible:
            return
        
        if action_state is None: action_state = {}
        if hud_data is None: hud_data = {}

        # OpenCV (BGR) -> PIL (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, 'RGBA')

        if self._selection_flash > 0:
            self._selection_flash -= 1

        # 分层绘制
        self._render_hud(draw, brush_manager, hud_data, action_state)
        self._render_panels(draw, brush_manager, action_state)
        self._render_dwell_progress(draw)

        # PIL (RGB) -> OpenCV (BGR)
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ================== HUD 渲染子系统 ==================

    def _render_hud(
        self, 
        draw: ImageDraw.ImageDraw, 
        brush_manager: Any, 
        hud_data: Dict[str, Any], 
        action_state: Dict[str, bool]
    ) -> None:
        """绘制顶部状态栏 (HUD)。"""
        
        # 1. 左侧 FPS 显示
        fps = hud_data.get('fps', 0.0)
        mode = hud_data.get('mode', 'DRAW')
        
        # 根据帧率变色
        if fps > 20: fps_color = (50, 200, 100)
        elif fps > 10: fps_color = (255, 200, 0)
        else: fps_color = (255, 50, 50)
        
        self._draw_status_capsule(
            draw, x=20, y=15, w=140, h=40,
            text=f"FPS: {fps:.1f}", sub_text=mode,
            accent_color=fps_color
        )

        # 2. 中间动态胶囊 (当前工具状态)
        tool_name = brush_manager.tool.upper()
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

        # 3. 右侧历史记录
        right_start_x = self.width - 20
        history_info = hud_data.get('history', '0/0')
        
        # 解析 "History: X | Redo: Y" 字符串
        import re
        nums = re.findall(r'\d+', history_info)
        try:
            raw_total = int(nums[0]) if len(nums) > 0 else 0
            redo_val = int(nums[1]) if len(nums) > 1 else 0
            undo_display = max(0, raw_total - 1)
        except Exception:
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
        
        # 4. 状态小图标 (Feature Flags)
        status_icons = []
        if action_state.get('particles'): status_icons.append(("粒子特效", (255, 215, 0))) 
        if action_state.get('line_assist'): status_icons.append(("直线辅助", (100, 200, 255)))
        if action_state.get('pen_effect'): status_icons.append(("压感模拟", (255, 150, 200)))
        
        icon_start_x = right_start_x - capsule_w - 20 
        icon_width = 85
        
        for i, (label, color) in enumerate(status_icons):
            icon_x = icon_start_x - (i + 1) * (icon_width + 10)
            
            # 背景
            self._draw_rounded_rect_pil(draw, icon_x, 15, icon_width, 32, (40, 40, 40, 200), radius=10)
            
            # 文字
            bbox = draw.textbbox((0, 0), label, font=self.font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((icon_x + (icon_width - tw)//2, 15 + (32 - th)//2 - 2), label, font=self.font_small, fill=color)

        # 5. Toast 消息 (底部提示)
        message = hud_data.get('message', '')
        if message:
            self._draw_toast_message(draw, message)

    def _draw_status_capsule(
        self, draw, x, y, w, h, text, 
        sub_text=None, accent_color=(255,255,255), bg_color=None, is_active=False
    ):
        """绘制标准状态胶囊组件。"""
        if bg_color is None: bg_color = self.style.bg_color
        radius = h // 2
        
        # 背景与边框
        self._draw_rounded_rect_pil(draw, x, y, w, h, bg_color, radius=radius)
        if is_active:
             self._draw_rounded_rect_pil(draw, x, y, w, h, outline=accent_color, width=2, radius=radius)
        
        # 侧边色条
        bar_h = h - 16
        draw.line((x + 12, y + 8, x + 12, y + 8 + bar_h), fill=accent_color, width=3)
        
        # 文字内容
        text_x = x + 28
        title_y = y + 4 if sub_text else y + h//2 - 12
        draw.text((text_x, title_y), text, font=self.font, fill=self.style.text_color_main)
        if sub_text:
            draw.text((text_x, y + 24), sub_text, font=self.font_small, fill=self.style.text_color_sub)

    def _draw_toast_message(self, draw: ImageDraw.ImageDraw, message: str) -> None:
        """绘制底部 Toast 提示框。"""
        bbox = draw.textbbox((0, 0), message, font=self.font_large)
        text_w = bbox[2] - bbox[0]
        toast_w = text_w + 80
        toast_h = 50
        toast_x = (self.width - toast_w) // 2
        self._draw_rounded_rect_pil(draw, toast_x, 80, toast_w, toast_h, (0, 0, 0, 180), radius=25)
        draw.text((toast_x + 40, 88), message, font=self.font_large, fill=(255, 255, 255))

    # ================== 面板渲染系统 ==================

    def _render_panels(self, draw: ImageDraw.ImageDraw, brush_manager: Any, action_state: Dict) -> None:
        # 1. 左侧工具栏
        labels = ["画笔", "橡皮擦", "激光笔"]
        for i, label in enumerate(labels):
            y = self.tool_panel_y_start + i * self.tool_button_spacing
            self._render_button_pil(
                draw, self.tool_panel_x, y, self.tool_button_width, self.tool_button_height,
                label, (i == brush_manager.current_tool_index), 
                (self.hover_item == ("tool", i)), "tool", i
            )

        # 2. 左侧动作栏
        for i, (key, label) in enumerate(self.action_items):
            y = self.action_panel_y_start + i * self.action_button_spacing
            is_active = action_state.get(key, False)
            is_sel = is_active
            # 只有 "clear" 动作是瞬时的，需要闪烁反馈；其他是开关状态
            if key == "clear" and self._selection_flash > 0 and self._last_selected == ("action", i):
                is_sel = True
            
            self._render_button_pil(
                draw, self.action_panel_x, y, self.action_button_width, self.action_button_height,
                label, is_sel, (self.hover_item == ("action", i)), "action", i
            )

        # 3. 颜色选择栏 (仅在画笔模式显示)
        if brush_manager.tool == "pen":
            for i, c_name in enumerate(brush_manager.COLOR_NAMES):
                rgb = self._bgr_to_rgb(brush_manager.COLORS[c_name])
                x = self.color_panel_x_start + i * self.color_button_spacing
                y = self.color_panel_y
                r = self.color_button_size // 2
                
                # 悬停放大效果
                if self.hover_item == ("color", i): r += 3
                
                # 选中光圈
                if i == brush_manager.current_color_index:
                    draw.ellipse((x-r-3, y-r-3, x+r+3, y+r+3), outline=(255,255,255), width=2)
                
                draw.ellipse((x-r, y-r, x+r, y+r), fill=rgb)

        # 4. 粗细选择栏
        t_start_x = (self.width - (len(brush_manager.THICKNESSES) * self.thickness_button_spacing)) // 2
        for i, thick in enumerate(brush_manager.THICKNESSES):
            x = t_start_x + i * self.thickness_button_spacing
            y = self.thickness_panel_y
            self._render_button_pil(
                draw, x, y, self.thickness_button_width, self.thickness_button_height,
                "", (i == brush_manager.current_thickness_index), 
                (self.hover_item == ("thickness", i)), "thickness", i
            )
            # 绘制中间的线条示意粗细
            ly = y + self.thickness_button_height // 2
            draw.line((x+10, ly, x+self.thickness_button_width-10, ly), fill="white", width=thick)

        # 5. 右侧笔刷栏 (含预览)
        if brush_manager.tool == "pen":
            brush_labels = {"solid": "实线", "dashed": "虚线", "glow": "发光", "marker": "马克", "rainbow": "彩虹"}
            for i, b_type in enumerate(brush_manager.BRUSH_TYPES):
                y = self.brush_panel_y_start + i * self.brush_button_spacing
                label = brush_labels.get(b_type, b_type)
                
                self._render_button_pil(
                    draw, self.brush_panel_x, y, self.brush_button_width, self.brush_button_height,
                    label, (i == brush_manager.current_brush_type_index),
                    (self.hover_item == ("brush", i)), "brush", i,
                    hide_text=True
                )
                
                # 绘制笔刷效果预览图
                preview_y = y + 20
                start_x = self.brush_panel_x + 10
                end_x = self.brush_panel_x + self.brush_button_width - 10
                self._draw_brush_preview_pil(draw, b_type, start_x, end_x, preview_y)
                
                # 绘制小标签
                bbox = draw.textbbox((0, 0), label, font=self.font_small)
                tw = bbox[2] - bbox[0]
                text_x = self.brush_panel_x + (self.brush_button_width - tw) // 2
                draw.text((text_x, y + 35), label, font=self.font_small, fill=(200, 200, 200))

    def _render_button_pil(
        self, draw, x, y, w, h, text, is_selected, is_hover, item_type, index, hide_text=False
    ) -> None:
        """通用按钮绘制函数。"""
        is_flash = (self._selection_flash > 0 and self._last_selected == (item_type, index))
        
        # 背景色逻辑
        if is_flash:
            bg = (50, 200, 100, 255) # 闪烁绿
        elif is_selected:
            bg = (0, 120, 215, 230)  # 选中蓝
        elif is_hover:
            bg = (70, 70, 70, 220)   # 悬停灰
        else:
            bg = (40, 40, 40, 200)   # 默认黑
        
        self._draw_rounded_rect_pil(draw, x, y, w, h, bg, radius=10)
        
        # 边框
        if is_selected:
            self._draw_rounded_rect_pil(draw, x, y, w, h, outline=(255, 255, 255, 200), width=2, radius=10)
        elif is_hover:
            self._draw_rounded_rect_pil(draw, x, y, w, h, outline=(100, 200, 255, 150), width=1, radius=10)
            
        # 文字
        if text and not hide_text:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            tx = x + (w - (bbox[2] - bbox[0])) // 2
            ty = y + (h - (bbox[3] - bbox[1])) // 2 - 2
            draw.text((tx, ty), text, font=self.font, fill=(255, 255, 255))

    def _draw_brush_preview_pil(self, draw, brush_type, x1, x2, y) -> None:
        """绘制笔刷预览图案。"""
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

    def _render_dwell_progress(self, draw: ImageDraw.ImageDraw) -> None:
        """绘制驻留进度环 (Loading Circle)。"""
        if self.hover_item is None or self._hover_frames < 3:
            return
        
        progress = self.get_dwell_progress()
        if progress <= 0:
            return
            
        cx, cy = self._get_item_center_coords(self.hover_item)
        if cx is None:
            return
        
        radius = 28
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        # 背景轨
        draw.arc(bbox, 0, 360, fill=(100, 100, 100, 100), width=4)
        # 进度条
        draw.arc(bbox, -90, -90 + 360 * progress, fill=(0, 255, 200, 255), width=4)

    def _draw_rounded_rect_pil(self, draw, x, y, w, h, fill=None, outline=None, width=1, radius=10):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)

    def _bgr_to_rgb(self, bgr):
        return (bgr[2], bgr[1], bgr[0])

    def _get_item_center_coords(self, item_tuple) -> Tuple[Optional[int], Optional[int]]:
        """计算控件中心点坐标（用于绘制进度环）。"""
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

    # ================== 交互逻辑 (Interaction Logic) ==================

    def update_hover(self, point: Point, brush_manager: Any) -> Optional[Tuple[str, int]]:
        """
        更新当前的悬停状态。
        检测手指是否在任何 UI 控件的包围盒内。
        """
        if not self.visible:
            return None
            
        x, y = point
        tol = self.hit_tolerance
        new_hover = None
        
        # 1. 检测工具栏
        for i in range(len(brush_manager.TOOLS)):
            bx, by = self.tool_panel_x, self.tool_panel_y_start + i * self.tool_button_spacing
            if self._hit_rect(x, y, bx, by, self.tool_button_width, self.tool_button_height, tol):
                new_hover = ("tool", i)
                break
        
        # 2. 检测动作栏
        if not new_hover:
            for i in range(len(self.action_items)):
                bx, by = self.action_panel_x, self.action_panel_y_start + i * self.action_button_spacing
                if self._hit_rect(x, y, bx, by, self.action_button_width, self.action_button_height, tol):
                    new_hover = ("action", i)
                    break
        
        # 3. 检测颜色栏
        if not new_hover and brush_manager.tool == "pen":
            for i in range(len(brush_manager.COLOR_NAMES)):
                cx = self.color_panel_x_start + i * self.color_button_spacing
                cy = self.color_panel_y
                # 圆形碰撞检测
                if (x-cx)**2 + (y-cy)**2 <= (self.color_button_size//2 + tol)**2:
                    new_hover = ("color", i)
                    break

        # 4. 检测粗细栏
        if not new_hover:
            t_start_x = (self.width - (len(brush_manager.THICKNESSES) * self.thickness_button_spacing)) // 2
            for i in range(len(brush_manager.THICKNESSES)):
                bx = t_start_x + i * self.thickness_button_spacing
                by = self.thickness_panel_y
                if self._hit_rect(x, y, bx, by, self.thickness_button_width, self.thickness_button_height, tol):
                    new_hover = ("thickness", i)
                    break
        
        # 5. 检测笔刷栏
        if not new_hover and brush_manager.tool == "pen":
            for i in range(len(brush_manager.BRUSH_TYPES)):
                bx, by = self.brush_panel_x, self.brush_panel_y_start + i * self.brush_button_spacing
                if self._hit_rect(x, y, bx, by, self.brush_button_width, self.brush_button_height, tol):
                    new_hover = ("brush", i)
                    break
        
        # 驻留计时逻辑
        if new_hover == self._last_hover and new_hover:
            self._hover_frames += 1
        else:
            self._hover_frames = 0
            
        self._last_hover = new_hover
        self.hover_item = new_hover
        
        # 触发选择
        if self.hover_item and self._hover_frames >= self.dwell_frames:
            self._pending_selection = self.hover_item
            self._hover_frames = 0
            
        return self.hover_item

    def _hit_rect(self, mx, my, x, y, w, h, tol) -> bool:
        """矩形碰撞检测"""
        return (x - tol <= mx <= x + w + tol) and (y - tol <= my <= y + h + tol)
    
    def get_dwell_progress(self) -> float:
        """获取驻留进度 (0.0 - 1.0)"""
        if not self.hover_item: return 0.0
        return min(1.0, self._hover_frames / self.dwell_frames)

    def consume_pending_selection(self, brush_manager: Any) -> Dict[str, Any]:
        """
        消费并执行待处理的选择操作。
        在每帧调用，检查是否有通过驻留触发的命令。
        """
        result = {"selected": False, "item_type": None, "action": None}
        
        if self._pending_selection:
            t, i = self._pending_selection
            self._pending_selection = None
            
            result = {"selected": True, "item_type": t, "action": None}
            self._selection_flash = 5
            self._last_selected = (t, i)
            
            # 分发逻辑
            if t == "tool":
                brush_manager.current_tool_index = i
            elif t == "color":
                brush_manager.current_color_index = i
            elif t == "thickness":
                brush_manager.current_thickness_index = i
            elif t == "brush":
                brush_manager.current_brush_type_index = i
            elif t == "action":
                result["action"] = self.action_items[i][0]
                
        return result

    def select_hover_item(self, brush_manager: Any) -> Dict[str, Any]:
        """强制选择当前悬停项（用于鼠标点击或快速捏合）。"""
        if self.hover_item:
            self._pending_selection = self.hover_item
            return self.consume_pending_selection(brush_manager)
        return {"selected": False, "item_type": None, "action": None}
    
    def handle_mouse_click(self, point: Point, brush_manager: Any) -> bool:
        """处理鼠标点击事件。"""
        self.update_hover(point, brush_manager)
        if self.hover_item:
            self.select_hover_item(brush_manager)
            return True
        return False
    
    def is_in_dead_zone(self, point: Point, brush_manager: Any) -> bool:
        """检查坐标点是否位于 UI 控件区域内（防误触）。"""
        self.update_hover(point, brush_manager)
        return self.hover_item is not None