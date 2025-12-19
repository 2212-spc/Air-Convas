"""
手势控制的可视化UI界面
用于选择颜色、粗细、笔刷类型，无需键盘操作
[优化说明]
1. 修复了中文无法显示的问题，改为英文标签
2. 增加了背景遮罩，提高UI在视频上的可读性
3. 优化了布局间距和交互反馈
"""
from typing import Tuple, Optional
import cv2
import numpy as np


class GestureUI:
    """手势控制的UI管理器"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visible = True

        # === UI布局参数 ===
        # 左侧颜色面板
        self.color_panel_x = 50
        self.color_panel_y_start = 100
        self.color_button_size = 45
        self.color_button_spacing = 65
        self.color_panel_bg_width = 100 # 背景板宽度

        # 底部粗细面板
        self.thickness_panel_height = 90
        self.thickness_panel_y = height - self.thickness_panel_height + 20
        self.thickness_button_width = 80
        self.thickness_button_height = 50
        self.thickness_button_spacing = 100
        # 计算起始X以居中显示
        total_thickness_width = 6 * self.thickness_button_spacing
        self.thickness_panel_x_start = (width - total_thickness_width) // 2 + 10

        # 右侧笔刷面板
        self.brush_panel_bg_width = 130 # 背景板宽度
        self.brush_panel_x = width - self.brush_panel_bg_width + 25
        self.brush_panel_y_start = 100
        self.brush_button_width = 80
        self.brush_button_height = 65
        self.brush_button_spacing = 85

        self.hit_tolerance = 20
        self.hover_item = None
        self._hover_lock_frames = 0

    def toggle_visibility(self):
        self.visible = not self.visible

    def update_hover(self, point: Tuple[int, int], brush_manager) -> Optional[Tuple[str, int]]:
        """更新当前悬停的UI元素"""
        if not self.visible:
            return None

        x, y = point
        new_hover = None
        tol = self.hit_tolerance

        # 1. 检查颜色按钮 (圆形区域)
        for i in range(len(brush_manager.COLOR_NAMES)):
            btn_x = self.color_panel_x
            btn_y = self.color_panel_y_start + i * self.color_button_spacing
            if self._point_in_circle(x, y, btn_x, btn_y, self.color_button_size // 2 + tol):
                new_hover = ("color", i)
                break

        # 2. 检查粗细按钮 (矩形区域)
        if new_hover is None:
            for i in range(len(brush_manager.THICKNESSES)):
                btn_x = self.thickness_panel_x_start + i * self.thickness_button_spacing
                btn_y = self.thickness_panel_y
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.thickness_button_width + 2 * tol,
                                       self.thickness_button_height + 2 * tol):
                    new_hover = ("thickness", i)
                    break

        # 3. 检查笔刷类型按钮 (矩形区域)
        if new_hover is None:
            for i in range(len(brush_manager.BRUSH_TYPES)):
                btn_x = self.brush_panel_x
                btn_y = self.brush_panel_y_start + i * self.brush_button_spacing
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.brush_button_width + 2 * tol,
                                       self.brush_button_height + 2 * tol):
                    new_hover = ("brush", i)
                    break

        # 悬停状态防抖动处理
        if new_hover is not None:
            self.hover_item = new_hover
            self._hover_lock_frames = 5 # 锁定5帧
        else:
            if self._hover_lock_frames > 0:
                self._hover_lock_frames -= 1
            else:
                self.hover_item = None

        return self.hover_item

    def select_hover_item(self, brush_manager) -> bool:
        """选中当前悬停的元素"""
        if not self.hover_item:
            return False
        item_type, index = self.hover_item
        if item_type == "color":
            brush_manager.current_color_index = index
        elif item_type == "thickness":
            brush_manager.current_thickness_index = index
        elif item_type == "brush":
            brush_manager.current_brush_type_index = index
        return True

    def render(self, frame: np.ndarray, brush_manager):
        """渲染UI到该帧上"""
        if not self.visible:
            return

        # 创建一个覆盖层用于绘制半透明背景
        overlay = frame.copy()
        ui_layer = np.zeros_like(frame, dtype=np.uint8)

        # --- 1. 绘制半透明背景板 ---
        bg_color = (30, 30, 30)
        # 左侧背景
        cv2.rectangle(overlay, (0, 0), (self.color_panel_bg_width, self.height), bg_color, -1)
        # 右侧背景
        cv2.rectangle(overlay, (self.width - self.brush_panel_bg_width, 0), (self.width, self.height), bg_color, -1)
        # 底部背景
        cv2.rectangle(overlay, (0, self.height - self.thickness_panel_height), (self.width, self.height), (20, 20, 20), -1)
        
        # 应用半透明效果 (背景透明度 0.4)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # --- 2. 绘制UI元素 (不透明) ---
        # 直接在 frame 上绘制按钮和文字，确保清晰度
        self._render_color_panel(frame, brush_manager)
        self._render_thickness_panel(frame, brush_manager)
        self._render_brush_panel(frame, brush_manager)


    def _render_color_panel(self, canvas: np.ndarray, brush_manager):
        for i, color_name in enumerate(brush_manager.COLOR_NAMES):
            color = brush_manager.COLORS[color_name]
            x = self.color_panel_x
            y = self.color_panel_y_start + i * self.color_button_spacing
            is_selected = (i == brush_manager.current_color_index)
            is_hover = (self.hover_item == ("color", i))

            radius = self.color_button_size // 2
            if is_hover: radius += 4

            # 按钮主体
            cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            
            # 添加白色内圈增加质感
            cv2.circle(canvas, (x, y), radius - 3, (255,255,255), 1, lineType=cv2.LINE_AA)

            # 选中/悬停状态的高亮外圈
            if is_selected:
                cv2.circle(canvas, (x, y), radius + 6, (255, 255, 255), 3, lineType=cv2.LINE_AA)
            elif is_hover:
                cv2.circle(canvas, (x, y), radius + 6, (200, 200, 200), 2, lineType=cv2.LINE_AA)

    def _render_thickness_panel(self, canvas: np.ndarray, brush_manager):
        for i, thickness in enumerate(brush_manager.THICKNESSES):
            x = self.thickness_panel_x_start + i * self.thickness_button_spacing
            y = self.thickness_panel_y
            is_selected = (i == brush_manager.current_thickness_index)
            is_hover = (self.hover_item == ("thickness", i))

            # 按钮背景色
            bg_color = (60, 60, 60)
            if is_selected: bg_color = (100, 100, 100)
            elif is_hover: bg_color = (80, 80, 80)

            cv2.rectangle(canvas, (x, y), (x + self.thickness_button_width, y + self.thickness_button_height), bg_color, -1)

            # 绘制中间的示例线
            line_start = (x + 15, y + self.thickness_button_height // 2)
            line_end = (x + self.thickness_button_width - 15, y + self.thickness_button_height // 2)
            # 使用当前选中的颜色来绘制示例线
            cv2.line(canvas, line_start, line_end, brush_manager.color, thickness, lineType=cv2.LINE_AA)

            # 选中框
            if is_selected:
                cv2.rectangle(canvas, (x, y), (x + self.thickness_button_width, y + self.thickness_button_height), (255, 255, 255), 2)

    def _render_brush_panel(self, canvas: np.ndarray, brush_manager):
        # 使用英文标签以支持OpenCV渲染
        brush_labels = ["Solid", "Dot", "Glow", "Mark"]

        for i, (brush_type, label) in enumerate(zip(brush_manager.BRUSH_TYPES, brush_labels)):
            x = self.brush_panel_x
            y = self.brush_panel_y_start + i * self.brush_button_spacing
            is_selected = (i == brush_manager.current_brush_type_index)
            is_hover = (self.hover_item == ("brush", i))

            bg_color = (60, 60, 60)
            if is_selected: bg_color = (100, 100, 100)
            elif is_hover: bg_color = (80, 80, 80)
            
            cv2.rectangle(canvas, (x, y), (x + self.brush_button_width, y + self.brush_button_height), bg_color, -1)
            
            # 绘制文字标签 (底部居中)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (self.brush_button_width - text_size[0]) // 2
            text_y = y + self.brush_button_height - 10
            cv2.putText(canvas, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, lineType=cv2.LINE_AA)

            # 绘制图标预览 (文字上方)
            self._draw_brush_preview(canvas, brush_type, x, y - 5, brush_manager.color)

            # 选中框
            if is_selected:
                cv2.rectangle(canvas, (x, y), (x + self.brush_button_width, y + self.brush_button_height), (255, 255, 255), 2)

    def _draw_brush_preview(self, canvas: np.ndarray, brush_type: str, x: int, y: int, color: Tuple[int, int, int]):
        """在按钮上绘制笔刷效果预览"""
        line_y = y + 25
        start = (x + 15, line_y)
        end = (x + self.brush_button_width - 15, line_y)
        preview_thickness = 3
        
        if brush_type == "solid":
            cv2.line(canvas, start, end, color, preview_thickness, cv2.LINE_AA)
        elif brush_type == "dashed":
            # 简单画两段线表示虚线
            mid1 = (start[0] + 12, start[1])
            mid2 = (end[0] - 12, end[1])
            cv2.line(canvas, start, mid1, color, preview_thickness, cv2.LINE_AA)
            cv2.line(canvas, mid2, end, color, preview_thickness, cv2.LINE_AA)
        elif brush_type == "glow":
            # 画一个带光晕的线
             glow_color = tuple(int(c * 0.5) for c in color)
             cv2.line(canvas, start, end, glow_color, preview_thickness + 4, cv2.LINE_AA)
             cv2.line(canvas, start, end, color, preview_thickness, cv2.LINE_AA)
             cv2.line(canvas, start, end, (255, 255, 255), 1, cv2.LINE_AA) # 中心白光
        elif brush_type == "marker":
             # 画一个半透明的宽线
             overlay = canvas.copy()
             cv2.line(overlay, start, end, color, preview_thickness + 6, cv2.LINE_AA)
             cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    def _point_in_circle(self, px, py, cx, cy, r):
        return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2

    def _point_in_rect(self, px, py, rx, ry, rw, rh):
        return rx <= px <= rx + rw and ry <= py <= ry + rh