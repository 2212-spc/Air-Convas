# -*- coding: utf-8 -*-
"""
手势控制的可视化UI界面
用于选择工具、颜色、粗细、笔刷类型，无需键盘操作
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class GestureUI:
    """手势控制的UI管理器"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visible = True  # UI是否显示

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
        self.dwell_frames = 12  # 停留多少帧自动选中（仅工具/动作）

    def toggle_visibility(self):
        """切换UI显示/隐藏"""
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

        # 悬停锁定机制
        if new_hover is not None:
            self.hover_item = new_hover
            self._hover_lock_frames = 5
        else:
            if self._hover_lock_frames > 0:
                self._hover_lock_frames -= 1
            else:
                self.hover_item = None

        # 悬停停留计数（用于自动选择工具/动作）
        if self.hover_item and self.hover_item == self._last_hover:
            self._hover_frames += 1
        else:
            self._hover_frames = 0
        self._last_hover = self.hover_item

        if self.hover_item and self.hover_item[0] in ("tool", "action"):
            if self._hover_frames >= self.dwell_frames:
                self._dwell_item = self.hover_item
                self._hover_frames = 0

        return self.hover_item

    def select_hover_item(self, brush_manager) -> dict:
        """
        选择当前悬停的项
        返回: dict {selected, item_type, action}
        """
        result = {"selected": False, "item_type": None, "action": None}
        if not self.hover_item:
            return result

        item_type, index = self.hover_item
        result["item_type"] = item_type

        if item_type == "tool":
            brush_manager.current_tool_index = index
            result["selected"] = True
            return result
        elif item_type == "color":
            brush_manager.current_color_index = index
            result["selected"] = True
            return result
        elif item_type == "thickness":
            brush_manager.current_thickness_index = index
            result["selected"] = True
            return result
        elif item_type == "brush":
            brush_manager.current_brush_type_index = index
            result["selected"] = True
            return result
        elif item_type == "action":
            action_name = self.action_items[index][0]
            result["action"] = action_name
            result["selected"] = True
            return result

        return result

    def consume_dwell_item(self) -> Optional[Tuple[str, int]]:
        """消耗自动选择项（仅工具/动作）"""
        if self._dwell_item:
            item = self._dwell_item
            self._dwell_item = None
            return item
        return None

    def render(self, frame: np.ndarray, brush_manager, action_state: Optional[dict] = None):
        """渲染UI到画面上"""
        if not self.visible:
            return
        if action_state is None:
            action_state = {}

        overlay = frame.copy()

        # 1. 渲染工具栏 (最左侧)
        self._render_tool_panel(overlay, brush_manager)
        self._render_action_panel(overlay, action_state)

        # 2. 渲染颜色选择器 (仅当工具为Pen时)
        if brush_manager.tool == "pen":
            self._render_color_panel(overlay, brush_manager)
            self._render_brush_panel(overlay, brush_manager)

        # 3. 渲染粗细选择器
        self._render_thickness_panel(overlay, brush_manager)

        # 混合overlay到frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def _render_tool_panel(self, overlay: np.ndarray, brush_manager):
        """渲染工具选择面板"""
        labels = ["Pen", "Eraser", "Laser"]
        
        for i, (tool_name, label) in enumerate(zip(brush_manager.TOOLS, labels)):
            x = self.tool_panel_x
            y = self.tool_panel_y_start + i * self.tool_button_spacing
            
            is_selected = (i == brush_manager.current_tool_index)
            is_hover = (self.hover_item == ("tool", i))
            
            # 背景框
            bg_color = (60, 60, 60)
            if is_selected:
                bg_color = (100, 100, 180) # 选中高亮
            elif is_hover:
                bg_color = (100, 100, 100)
                
            cv2.rectangle(overlay, (x, y), (x + self.tool_button_width, y + self.tool_button_height),
                         bg_color, -1, cv2.LINE_AA)
            
            # 文字
            text_color = (255, 255, 255)
            if is_selected:
                text_color = (0, 255, 255)
                
            cv2.putText(overlay, label, (x + 5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, text_color, 1, cv2.LINE_AA)
            
            # 选中边框
            if is_selected:
                cv2.rectangle(overlay, (x, y), (x + self.tool_button_width, y + self.tool_button_height),
                             (255, 255, 255), 2, cv2.LINE_AA)

    def _render_action_panel(self, overlay: np.ndarray, action_state: dict):
        """渲染快捷动作面板"""
        for i, (action_key, label) in enumerate(self.action_items):
            x = self.action_panel_x
            y = self.action_panel_y_start + i * self.action_button_spacing

            is_hover = (self.hover_item == ("action", i))
            is_active = False
            if action_key == "particles":
                is_active = bool(action_state.get("particles", False))

            bg_color = (60, 60, 60)
            if is_active:
                bg_color = (70, 120, 70)
            elif is_hover:
                bg_color = (100, 100, 100)

            cv2.rectangle(overlay, (x, y), (x + self.action_button_width, y + self.action_button_height),
                         bg_color, -1, cv2.LINE_AA)
            cv2.putText(overlay, label, (x + 8, y + 35), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if is_active:
                cv2.rectangle(overlay, (x, y), (x + self.action_button_width, y + self.action_button_height),
                             (0, 255, 0), 2, cv2.LINE_AA)
            elif is_hover:
                cv2.rectangle(overlay, (x, y), (x + self.action_button_width, y + self.action_button_height),
                             (255, 255, 0), 2, cv2.LINE_AA)

    def _render_color_panel(self, overlay: np.ndarray, brush_manager):
        """渲染颜色选择面板"""
        for i, color_name in enumerate(brush_manager.COLOR_NAMES):
            color = brush_manager.COLORS[color_name]
            x = self.color_panel_x
            y = self.color_panel_y_start + i * self.color_button_spacing

            is_selected = (i == brush_manager.current_color_index)
            is_hover = (self.hover_item == ("color", i))

            radius = self.color_button_size // 2
            if is_hover:
                radius += 5

            cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

            if is_selected:
                cv2.circle(overlay, (x, y), radius + 5, (255, 255, 255), 3, lineType=cv2.LINE_AA)

            if is_hover:
                cv2.circle(overlay, (x, y), radius + 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    def _render_thickness_panel(self, overlay: np.ndarray, brush_manager):
        """渲染粗细选择面板"""
        for i, thickness in enumerate(brush_manager.THICKNESSES):
            x = self.thickness_panel_x_start + i * self.thickness_button_spacing
            y = self.thickness_panel_y

            is_selected = (i == brush_manager.current_thickness_index)
            is_hover = (self.hover_item == ("thickness", i))

            bg_color = (80, 80, 80)
            if is_hover:
                bg_color = (120, 120, 120)
            cv2.rectangle(overlay,
                         (x, y),
                         (x + self.thickness_button_width, y + self.thickness_button_height),
                         bg_color, -1, lineType=cv2.LINE_AA)

            if is_selected:
                cv2.rectangle(overlay,
                             (x, y),
                             (x + self.thickness_button_width, y + self.thickness_button_height),
                             (255, 255, 255), 3, lineType=cv2.LINE_AA)

            line_start = (x + 10, y + self.thickness_button_height // 2)
            line_end = (x + self.thickness_button_width - 10, y + self.thickness_button_height // 2)
            cv2.line(overlay, line_start, line_end, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

            if is_hover:
                cv2.rectangle(overlay,
                             (x - 3, y - 3),
                             (x + self.thickness_button_width + 3, y + self.thickness_button_height + 3),
                             (0, 255, 255), 2, lineType=cv2.LINE_AA)

    def _render_brush_panel(self, overlay: np.ndarray, brush_manager):
        """渲染笔刷类型面板"""
        brush_labels = ["solid", "dash", "glow", "marker", "rainbow"]

        for i, (brush_type, label) in enumerate(zip(brush_manager.BRUSH_TYPES, brush_labels)):
            x = self.brush_panel_x
            y = self.brush_panel_y_start + i * self.brush_button_spacing

            is_selected = (i == brush_manager.current_brush_type_index)
            is_hover = (self.hover_item == ("brush", i))

            bg_color = (80, 80, 80)
            if is_hover:
                bg_color = (120, 120, 120)
            cv2.rectangle(overlay,
                         (x, y),
                         (x + self.brush_button_width, y + self.brush_button_height),
                         bg_color, -1, lineType=cv2.LINE_AA)

            if is_selected:
                cv2.rectangle(overlay,
                             (x, y),
                             (x + self.brush_button_width, y + self.brush_button_height),
                             (255, 255, 255), 3, lineType=cv2.LINE_AA)

            self._draw_brush_preview(overlay, brush_type, x, y)

            cv2.putText(overlay, label,
                       (x + 5, y + self.brush_button_height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            if is_hover:
                cv2.rectangle(overlay,
                             (x - 3, y - 3),
                             (x + self.brush_button_width + 3, y + self.brush_button_height + 3),
                             (0, 255, 255), 2, lineType=cv2.LINE_AA)

    def _draw_brush_preview(self, overlay: np.ndarray, brush_type: str, x: int, y: int):
        """绘制笔刷预览"""
        line_y = y + 25
        line_start = (x + 10, line_y)
        line_end = (x + self.brush_button_width - 10, line_y)

        if brush_type == "solid":
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        elif brush_type == "dashed":
            for i in range(3):
                seg_start = (x + 10 + i * 15, line_y)
                seg_end = (x + 10 + i * 15 + 8, line_y)
                cv2.line(overlay, seg_start, seg_end, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        elif brush_type == "glow":
            cv2.line(overlay, line_start, line_end, (200, 200, 255), 5, lineType=cv2.LINE_AA)
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        elif brush_type == "marker":
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 5, lineType=cv2.LINE_AA)
        elif brush_type == "rainbow":
            seg_len = (line_end[0] - line_start[0]) // 3
            cv2.line(overlay, line_start, (line_start[0] + seg_len, line_y), (0, 0, 255), 3, lineType=cv2.LINE_AA)
            cv2.line(overlay, (line_start[0] + seg_len, line_y), (line_start[0] + 2*seg_len, line_y), (0, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(overlay, (line_start[0] + 2*seg_len, line_y), line_end, (255, 0, 0), 3, lineType=cv2.LINE_AA)

    def _point_in_circle(self, px: int, py: int, cx: int, cy: int, radius: int) -> bool:
        """检查点是否在圆内"""
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2

    def _point_in_rect(self, px: int, py: int, rx: int, ry: int, rw: int, rh: int) -> bool:
        """检查点是否在矩形内"""
        return rx <= px <= rx + rw and ry <= py <= ry + rh
