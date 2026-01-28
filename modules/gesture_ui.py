# -*- coding: utf-8 -*-
"""
手势控制的可视化UI界面 - 紧凑边缘式布局
所有UI元素贴近屏幕边缘，中间完全空出供绘画
使用悬停停留自动选择，无需捏合
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import math


class GestureUI:
    """手势控制的UI管理器 - 紧凑边缘式布局"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visible = True

        # ========== 紧凑边缘式布局参数 ==========
        
        # 1. 工具栏 (左侧边缘) - 缩小尺寸
        self.tool_panel_x = 8
        self.tool_panel_y_start = 60
        self.tool_button_width = 70
        self.tool_button_height = 50
        self.tool_button_spacing = 58

        # 1.1 快捷动作栏 (工具栏下方)
        self.action_panel_x = self.tool_panel_x
        self.action_panel_y_start = self.tool_panel_y_start + self.tool_button_spacing * 3 + 15
        self.action_button_width = 70
        self.action_button_height = 45
        self.action_button_spacing = 52
        self.action_items = [("clear", "CLR"), ("particles", "FX"), ("effects", "EFX")]

        # 2. 颜色栏 (顶部边缘 - 水平排列)
        self.color_panel_y = 12
        self.color_panel_x_start = 100  # 从左边工具栏右侧开始
        self.color_button_size = 32
        self.color_button_spacing = 42

        # 3. 粗细栏 (底部边缘 - 水平排列)
        self.thickness_panel_y = height - 55
        self.thickness_button_width = 55
        self.thickness_button_height = 40
        self.thickness_button_spacing = 65

        # 4. 笔刷类型栏 (右侧边缘)
        self.brush_panel_x = width - 78
        self.brush_panel_y_start = 60
        self.brush_button_width = 70
        self.brush_button_height = 50
        self.brush_button_spacing = 58

        # 碰撞检测容差
        self.hit_tolerance = 20

        # ========== 悬停和选择状态 ==========
        self.hover_item = None  # (type, index)
        self._hover_frames = 0
        self._last_hover = None
        self._selection_flash = 0  # 选中闪烁效果计数
        self._last_selected = None  # 最近选中的项
        
        # 悬停自动选择参数
        self.dwell_frames = 20  # 悬停20帧(~0.7秒)自动选中
        self._pending_selection = None  # 待执行的选择

    def toggle_visibility(self):
        """切换UI显示/隐藏"""
        self.visible = not self.visible

    def is_in_ui_area(self, point: Tuple[int, int], brush_manager) -> bool:
        """
        检查点是否在任何UI区域内
        用于在悬停UI时完全阻止绘画
        """
        if not self.visible:
            return False
        
        x, y = point
        margin = 30  # 边距
        
        # 1. 左侧工具+动作区域
        left_area_width = self.tool_button_width + margin * 2
        left_area_height = (self.action_panel_y_start + 
                          len(self.action_items) * self.action_button_spacing + margin)
        if x < left_area_width and y < left_area_height:
            return True
        
        # 2. 顶部颜色区域 (仅Pen模式)
        if brush_manager.tool == "pen":
            color_count = len(brush_manager.COLOR_NAMES)
            color_area_end = self.color_panel_x_start + color_count * self.color_button_spacing
            if y < self.color_panel_y + self.color_button_size + margin and x > self.color_panel_x_start - margin and x < color_area_end:
                return True
        
        # 3. 底部粗细区域
        thickness_count = len(brush_manager.THICKNESSES)
        thickness_total_w = thickness_count * self.thickness_button_spacing
        t_start_x = (self.width - thickness_total_w) // 2
        if y > self.thickness_panel_y - margin:
            if t_start_x - margin < x < t_start_x + thickness_total_w + margin:
                return True
        
        # 4. 右侧笔刷区域 (仅Pen模式)
        if brush_manager.tool == "pen":
            brush_count = len(brush_manager.BRUSH_TYPES)
            brush_area_height = brush_count * self.brush_button_spacing + margin
            if x > self.brush_panel_x - margin and y < self.brush_panel_y_start + brush_area_height:
                return True
        
        return False

    def is_in_dead_zone(self, point: Tuple[int, int], brush_manager) -> bool:
        """检查点是否在按钮死区内（用于防止误触）"""
        return self.is_in_ui_area(point, brush_manager)

    def update_hover(self, point: Tuple[int, int], brush_manager) -> Optional[Tuple[str, int]]:
        """
        更新悬停状态，返回当前悬停的项
        实现悬停停留自动选择
        """
        if not self.visible:
            self.hover_item = None
            return None

        x, y = point
        new_hover = None
        tol = self.hit_tolerance

        # 检查工具按钮
        for i in range(len(brush_manager.TOOLS)):
            btn_x = self.tool_panel_x
            btn_y = self.tool_panel_y_start + i * self.tool_button_spacing
            if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                   self.tool_button_width + 2 * tol,
                                   self.tool_button_height + 2 * tol):
                new_hover = ("tool", i)
                break

        # 检查动作按钮
        if new_hover is None:
            for i in range(len(self.action_items)):
                btn_x = self.action_panel_x
                btn_y = self.action_panel_y_start + i * self.action_button_spacing
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.action_button_width + 2 * tol,
                                       self.action_button_height + 2 * tol):
                    new_hover = ("action", i)
                    break

        # 检查颜色按钮 (仅Pen模式，顶部水平排列)
        if new_hover is None and brush_manager.tool == "pen":
            for i in range(len(brush_manager.COLOR_NAMES)):
                btn_x = self.color_panel_x_start + i * self.color_button_spacing
                btn_y = self.color_panel_y + self.color_button_size // 2
                if self._point_in_circle(x, y, btn_x, btn_y, self.color_button_size // 2 + tol):
                    new_hover = ("color", i)
                    break

        # 检查粗细按钮 (底部水平居中)
        if new_hover is None:
            thickness_count = len(brush_manager.THICKNESSES)
            thickness_total_w = thickness_count * self.thickness_button_spacing
            t_start_x = (self.width - thickness_total_w) // 2
            for i in range(thickness_count):
                btn_x = t_start_x + i * self.thickness_button_spacing
                btn_y = self.thickness_panel_y
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.thickness_button_width + 2 * tol,
                                       self.thickness_button_height + 2 * tol):
                    new_hover = ("thickness", i)
                    break

        # 检查笔刷按钮 (右侧，仅Pen模式)
        if new_hover is None and brush_manager.tool == "pen":
            for i in range(len(brush_manager.BRUSH_TYPES)):
                btn_x = self.brush_panel_x
                btn_y = self.brush_panel_y_start + i * self.brush_button_spacing
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.brush_button_width + 2 * tol,
                                       self.brush_button_height + 2 * tol):
                    new_hover = ("brush", i)
                    break

        # 更新悬停状态和计数
        if new_hover == self._last_hover and new_hover is not None:
            self._hover_frames += 1
        else:
            self._hover_frames = 0
        
        self._last_hover = new_hover
        self.hover_item = new_hover

        # 悬停停留自动选择检测
        if self.hover_item and self._hover_frames >= self.dwell_frames:
            self._pending_selection = self.hover_item
            self._hover_frames = 0  # 重置，避免重复触发

        return self.hover_item

    def get_dwell_progress(self) -> float:
        """获取当前悬停进度 (0.0 - 1.0)"""
        if self.hover_item is None:
            return 0.0
        return min(1.0, self._hover_frames / self.dwell_frames)

    def consume_pending_selection(self, brush_manager) -> dict:
        """
        消耗待执行的悬停选择
        返回: dict {selected, item_type, action}
        """
        result = {"selected": False, "item_type": None, "action": None}
        
        if self._pending_selection is None:
            return result
        
        item_type, index = self._pending_selection
        self._pending_selection = None
        result["item_type"] = item_type
        result["selected"] = True
        
        # 触发选中高亮效果（持续5帧，约0.17秒）
        self._selection_flash = 5
        self._last_selected = (item_type, index)

        if item_type == "tool":
            brush_manager.current_tool_index = index
        elif item_type == "color":
            brush_manager.current_color_index = index
        elif item_type == "thickness":
            brush_manager.current_thickness_index = index
        elif item_type == "brush":
            brush_manager.current_brush_type_index = index
        elif item_type == "action":
            action_name = self.action_items[index][0]
            result["action"] = action_name

        return result

    def select_hover_item(self, brush_manager) -> dict:
        """手动选择当前悬停项（兼容鼠标点击）"""
        result = {"selected": False, "item_type": None, "action": None}
        if not self.hover_item:
            return result

        item_type, index = self.hover_item
        result["item_type"] = item_type
        result["selected"] = True
        
        # 触发选中高亮效果（持续5帧，约0.17秒）
        self._selection_flash = 5
        self._last_selected = (item_type, index)

        if item_type == "tool":
            brush_manager.current_tool_index = index
        elif item_type == "color":
            brush_manager.current_color_index = index
        elif item_type == "thickness":
            brush_manager.current_thickness_index = index
        elif item_type == "brush":
            brush_manager.current_brush_type_index = index
        elif item_type == "action":
            action_name = self.action_items[index][0]
            result["action"] = action_name

        return result

    def handle_mouse_click(self, point: Tuple[int, int], brush_manager) -> bool:
        """处理鼠标点击"""
        if not self.visible:
            return False
        self.update_hover(point, brush_manager)
        if self.hover_item:
            self.select_hover_item(brush_manager)
            return True
        return False

    def render(self, frame: np.ndarray, brush_manager, action_state: Optional[dict] = None):
        """渲染UI到画面上"""
        if not self.visible:
            return
        if action_state is None:
            action_state = {}

        # 更新闪烁计数
        if self._selection_flash > 0:
            self._selection_flash -= 1

        overlay = frame.copy()

        # 1. 渲染左侧工具栏
        self._render_tool_panel(overlay, brush_manager)
        self._render_action_panel(overlay, action_state)

        # 2. 渲染顶部颜色栏 (仅Pen模式)
        if brush_manager.tool == "pen":
            self._render_color_panel(overlay, brush_manager)
            # 右侧笔刷栏
            self._render_brush_panel(overlay, brush_manager)

        # 3. 渲染底部粗细栏
        self._render_thickness_panel(overlay, brush_manager)

        # 4. 渲染悬停进度指示器
        self._render_dwell_progress(overlay)

        # 混合overlay到frame (半透明)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def _render_tool_panel(self, overlay: np.ndarray, brush_manager):
        """渲染工具选择面板 - 左侧边缘"""
        icons = ["✏", "⌫", "◎"]  # Pen, Eraser, Laser
        labels = ["Pen", "Ers", "Lsr"]
        
        for i, label in enumerate(labels):
            x = self.tool_panel_x
            y = self.tool_panel_y_start + i * self.tool_button_spacing
            
            is_selected = (i == brush_manager.current_tool_index)
            is_hover = (self.hover_item == ("tool", i))
            is_flash = (self._selection_flash > 0 and self._last_selected == ("tool", i))
            
            # 背景颜色 (选中时平滑高亮，不闪烁)
            if is_flash:
                bg_color = (100, 200, 100)  # 选中高亮绿色
            elif is_selected:
                bg_color = (120, 90, 60)  # 选中深蓝
            elif is_hover:
                bg_color = (80, 80, 80)
            else:
                bg_color = (50, 50, 50)
            
            # 绘制圆角矩形背景
            self._draw_rounded_rect(overlay, x, y, self.tool_button_width, self.tool_button_height, bg_color, 8)
            
            # 边框
            border_color = (255, 255, 255) if is_selected else (100, 100, 100)
            if is_hover:
                border_color = (0, 255, 255)
            self._draw_rounded_rect_border(overlay, x, y, self.tool_button_width, self.tool_button_height, border_color, 8, 2)
            
            # 文字
            text_color = (0, 255, 255) if is_selected else (200, 200, 200)
            cv2.putText(overlay, label, (x + 12, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, text_color, 1, cv2.LINE_AA)

    def _render_action_panel(self, overlay: np.ndarray, action_state: dict):
        """渲染快捷动作面板"""
        for i, (action_key, label) in enumerate(self.action_items):
            x = self.action_panel_x
            y = self.action_panel_y_start + i * self.action_button_spacing

            is_hover = (self.hover_item == ("action", i))
            is_active = bool(action_state.get(action_key, False))
            is_flash = (self._selection_flash > 0 and self._last_selected == ("action", i))

            if is_flash:
                bg_color = (100, 200, 100)  # 选中高亮绿色
            elif is_active:
                bg_color = (50, 100, 50)
            elif is_hover:
                bg_color = (80, 80, 80)
            else:
                bg_color = (50, 50, 50)

            self._draw_rounded_rect(overlay, x, y, self.action_button_width, self.action_button_height, bg_color, 6)
            
            border_color = (0, 255, 0) if is_active else ((0, 255, 255) if is_hover else (80, 80, 80))
            self._draw_rounded_rect_border(overlay, x, y, self.action_button_width, self.action_button_height, border_color, 6, 1)

            cv2.putText(overlay, label, (x + 15, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def _render_color_panel(self, overlay: np.ndarray, brush_manager):
        """渲染颜色选择面板 - 顶部水平排列"""
        for i, color_name in enumerate(brush_manager.COLOR_NAMES):
            color = brush_manager.COLORS[color_name]
            x = self.color_panel_x_start + i * self.color_button_spacing
            y = self.color_panel_y + self.color_button_size // 2

            is_selected = (i == brush_manager.current_color_index)
            is_hover = (self.hover_item == ("color", i))
            is_flash = (self._selection_flash > 0 and self._last_selected == ("color", i))

            radius = self.color_button_size // 2
            if is_hover:
                radius += 4
            if is_flash:
                radius += 6  # 选中时放大，不闪烁

            # 颜色圆
            cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

            # 选中外环
            if is_selected:
                cv2.circle(overlay, (x, y), radius + 4, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            # 悬停外环
            if is_hover:
                cv2.circle(overlay, (x, y), radius + 6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    def _render_thickness_panel(self, overlay: np.ndarray, brush_manager):
        """渲染粗细选择面板 - 底部水平居中"""
        thickness_count = len(brush_manager.THICKNESSES)
        thickness_total_w = thickness_count * self.thickness_button_spacing
        t_start_x = (self.width - thickness_total_w) // 2

        for i, thickness in enumerate(brush_manager.THICKNESSES):
            x = t_start_x + i * self.thickness_button_spacing
            y = self.thickness_panel_y

            is_selected = (i == brush_manager.current_thickness_index)
            is_hover = (self.hover_item == ("thickness", i))
            is_flash = (self._selection_flash > 0 and self._last_selected == ("thickness", i))

            if is_flash:
                bg_color = (100, 200, 100)  # 选中高亮绿色
            elif is_hover:
                bg_color = (80, 80, 80)
            else:
                bg_color = (50, 50, 50)

            self._draw_rounded_rect(overlay, x, y, self.thickness_button_width, self.thickness_button_height, bg_color, 6)

            border_color = (255, 255, 255) if is_selected else ((0, 255, 255) if is_hover else (80, 80, 80))
            self._draw_rounded_rect_border(overlay, x, y, self.thickness_button_width, self.thickness_button_height, border_color, 6, 2 if is_selected else 1)

            # 粗细预览线
            line_y = y + self.thickness_button_height // 2
            cv2.line(overlay, (x + 8, line_y), (x + self.thickness_button_width - 8, line_y), 
                    (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    def _render_brush_panel(self, overlay: np.ndarray, brush_manager):
        """渲染笔刷类型面板 - 右侧边缘"""
        brush_labels = ["Sld", "Dsh", "Glw", "Mrk", "Rnb"]

        for i, label in enumerate(brush_labels):
            x = self.brush_panel_x
            y = self.brush_panel_y_start + i * self.brush_button_spacing

            is_selected = (i == brush_manager.current_brush_type_index)
            is_hover = (self.hover_item == ("brush", i))
            is_flash = (self._selection_flash > 0 and self._last_selected == ("brush", i))

            if is_flash:
                bg_color = (100, 200, 100)  # 选中高亮绿色
            elif is_selected:
                bg_color = (120, 90, 60)
            elif is_hover:
                bg_color = (80, 80, 80)
            else:
                bg_color = (50, 50, 50)

            self._draw_rounded_rect(overlay, x, y, self.brush_button_width, self.brush_button_height, bg_color, 8)

            border_color = (255, 255, 255) if is_selected else ((0, 255, 255) if is_hover else (80, 80, 80))
            self._draw_rounded_rect_border(overlay, x, y, self.brush_button_width, self.brush_button_height, border_color, 8, 2 if is_selected else 1)

            # 笔刷预览
            self._draw_brush_preview(overlay, brush_manager.BRUSH_TYPES[i], x, y)

            # 标签
            text_color = (0, 255, 255) if is_selected else (180, 180, 180)
            cv2.putText(overlay, label, (x + 8, y + self.brush_button_height - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

    def _render_dwell_progress(self, overlay: np.ndarray):
        """渲染悬停进度指示器（圆形进度条）"""
        if self.hover_item is None or self._hover_frames < 3:
            return

        progress = self.get_dwell_progress()
        if progress <= 0:
            return

        # 获取当前悬停按钮的中心位置
        item_type, index = self.hover_item
        cx, cy = self._get_item_center(item_type, index)
        
        if cx is None:
            return

        # 绘制圆形进度条
        radius = 25
        angle = int(360 * progress)
        
        # 背景圆环
        cv2.circle(overlay, (cx, cy), radius, (50, 50, 50), 3, lineType=cv2.LINE_AA)
        
        # 进度圆弧
        if angle > 0:
            cv2.ellipse(overlay, (cx, cy), (radius, radius), -90, 0, angle, (0, 255, 200), 3, lineType=cv2.LINE_AA)

    def _get_item_center(self, item_type: str, index: int) -> Tuple[Optional[int], Optional[int]]:
        """获取按钮中心坐标"""
        if item_type == "tool":
            x = self.tool_panel_x + self.tool_button_width // 2
            y = self.tool_panel_y_start + index * self.tool_button_spacing + self.tool_button_height // 2
            return (x, y)
        elif item_type == "action":
            x = self.action_panel_x + self.action_button_width // 2
            y = self.action_panel_y_start + index * self.action_button_spacing + self.action_button_height // 2
            return (x, y)
        elif item_type == "color":
            x = self.color_panel_x_start + index * self.color_button_spacing
            y = self.color_panel_y + self.color_button_size // 2
            return (x, y)
        elif item_type == "thickness":
            thickness_total_w = 5 * self.thickness_button_spacing  # 假设5个粗细选项
            t_start_x = (self.width - thickness_total_w) // 2
            x = t_start_x + index * self.thickness_button_spacing + self.thickness_button_width // 2
            y = self.thickness_panel_y + self.thickness_button_height // 2
            return (x, y)
        elif item_type == "brush":
            x = self.brush_panel_x + self.brush_button_width // 2
            y = self.brush_panel_y_start + index * self.brush_button_spacing + self.brush_button_height // 2
            return (x, y)
        return (None, None)

    def _draw_rounded_rect(self, img, x, y, w, h, color, radius):
        """绘制填充圆角矩形"""
        # 使用多边形近似圆角
        pts = []
        # 左上角
        pts.append((x + radius, y))
        pts.append((x + w - radius, y))
        # 右上角
        for angle in range(-90, 0, 15):
            px = int(x + w - radius + radius * math.cos(math.radians(angle)))
            py = int(y + radius + radius * math.sin(math.radians(angle)))
            pts.append((px, py))
        pts.append((x + w, y + radius))
        pts.append((x + w, y + h - radius))
        # 右下角
        for angle in range(0, 90, 15):
            px = int(x + w - radius + radius * math.cos(math.radians(angle)))
            py = int(y + h - radius + radius * math.sin(math.radians(angle)))
            pts.append((px, py))
        pts.append((x + w - radius, y + h))
        pts.append((x + radius, y + h))
        # 左下角
        for angle in range(90, 180, 15):
            px = int(x + radius + radius * math.cos(math.radians(angle)))
            py = int(y + h - radius + radius * math.sin(math.radians(angle)))
            pts.append((px, py))
        pts.append((x, y + h - radius))
        pts.append((x, y + radius))
        # 左上角
        for angle in range(180, 270, 15):
            px = int(x + radius + radius * math.cos(math.radians(angle)))
            py = int(y + radius + radius * math.sin(math.radians(angle)))
            pts.append((px, py))
        
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)

    def _draw_rounded_rect_border(self, img, x, y, w, h, color, radius, thickness):
        """绘制圆角矩形边框"""
        # 四条边
        cv2.line(img, (x + radius, y), (x + w - radius, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w - radius, y + h), (x + radius, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + h - radius), (x, y + radius), color, thickness, cv2.LINE_AA)
        # 四个角的弧
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_brush_preview(self, overlay: np.ndarray, brush_type: str, x: int, y: int):
        """绘制笔刷预览线"""
        line_y = y + 18
        line_start = (x + 8, line_y)
        line_end = (x + self.brush_button_width - 8, line_y)

        if brush_type == "solid":
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        elif brush_type == "dashed":
            for j in range(3):
                seg_start = (x + 8 + j * 12, line_y)
                seg_end = (x + 8 + j * 12 + 6, line_y)
                cv2.line(overlay, seg_start, seg_end, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        elif brush_type == "glow":
            cv2.line(overlay, line_start, line_end, (150, 150, 255), 4, lineType=cv2.LINE_AA)
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        elif brush_type == "marker":
            cv2.line(overlay, line_start, line_end, (200, 200, 200), 4, lineType=cv2.LINE_AA)
        elif brush_type == "rainbow":
            seg_len = (line_end[0] - line_start[0]) // 3
            cv2.line(overlay, line_start, (line_start[0] + seg_len, line_y), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(overlay, (line_start[0] + seg_len, line_y), (line_start[0] + 2*seg_len, line_y), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(overlay, (line_start[0] + 2*seg_len, line_y), line_end, (255, 0, 0), 2, lineType=cv2.LINE_AA)

    def _point_in_circle(self, px: int, py: int, cx: int, cy: int, radius: int) -> bool:
        """检查点是否在圆内"""
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2

    def _point_in_rect(self, px: int, py: int, rx: int, ry: int, rw: int, rh: int) -> bool:
        """检查点是否在矩形内"""
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    # 兼容旧接口
    def consume_dwell_item(self) -> Optional[Tuple[str, int]]:
        """兼容旧接口"""
        return None
