"""
手势控制的可视化UI界面
用于选择颜色、粗细、笔刷类型，无需键盘操作
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

        # UI布局参数 - 增大按钮尺寸
        self.color_panel_x = 40  # 稍微向右移动
        self.color_panel_y_start = 120
        self.color_button_size = 50  # 从40增大到50
        self.color_button_spacing = 70  # 从60增大到70

        self.thickness_panel_y = height - 100  # 稍微上移
        self.thickness_panel_x_start = 200
        self.thickness_button_width = 100  # 从80增大到100
        self.thickness_button_height = 60  # 从50增大到60
        self.thickness_button_spacing = 120  # 从100增大到120

        self.brush_panel_x = width - 120  # 稍微向左移动
        self.brush_panel_y_start = 120
        self.brush_button_width = 90  # 从70增大到90
        self.brush_button_height = 70  # 从60增大到70
        self.brush_button_spacing = 90  # 从80增大到90

        # 碰撞检测容差（额外增加的检测范围）
        self.hit_tolerance = 15

        # 悬停和选择状态
        self.hover_item = None  # (type, index) 例如 ("color", 2)
        self.pinch_ready = False  # 是否准备捏合选择
        self._hover_lock_frames = 0  # 悬停锁定计数器，防止抖动

    def toggle_visibility(self):
        """切换UI显示/隐藏"""
        self.visible = not self.visible

    def update_hover(self, point: Tuple[int, int], brush_manager) -> Optional[Tuple[str, int]]:
        """
        更新悬停状态
        返回: (类型, 索引) 如果悬停在某个按钮上
        """
        if not self.visible:
            return None

        x, y = point
        new_hover = None
        tol = self.hit_tolerance  # 容差

        # 检查颜色按钮
        for i in range(len(brush_manager.COLOR_NAMES)):
            btn_x = self.color_panel_x
            btn_y = self.color_panel_y_start + i * self.color_button_spacing
            # 增加容差的圆形检测
            if self._point_in_circle(x, y, btn_x, btn_y, self.color_button_size // 2 + tol):
                new_hover = ("color", i)
                break

        # 检查粗细按钮
        if new_hover is None:
            for i in range(len(brush_manager.THICKNESSES)):
                btn_x = self.thickness_panel_x_start + i * self.thickness_button_spacing
                btn_y = self.thickness_panel_y
                # 增加容差的矩形检测
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.thickness_button_width + 2 * tol,
                                       self.thickness_button_height + 2 * tol):
                    new_hover = ("thickness", i)
                    break

        # 检查笔刷类型按钮
        if new_hover is None:
            for i in range(len(brush_manager.BRUSH_TYPES)):
                btn_x = self.brush_panel_x
                btn_y = self.brush_panel_y_start + i * self.brush_button_spacing
                # 增加容差的矩形检测
                if self._point_in_rect(x, y, btn_x - tol, btn_y - tol,
                                       self.brush_button_width + 2 * tol,
                                       self.brush_button_height + 2 * tol):
                    new_hover = ("brush", i)
                    break

        # 悬停锁定机制：减少因手部抖动导致的悬停丢失
        if new_hover is not None:
            self.hover_item = new_hover
            self._hover_lock_frames = 5  # 悬停后保持5帧
        else:
            # 如果之前有悬停且还在锁定期内，保持之前的悬停
            if self._hover_lock_frames > 0:
                self._hover_lock_frames -= 1
            else:
                self.hover_item = None

        return self.hover_item

    def select_hover_item(self, brush_manager) -> bool:
        """
        选择当前悬停的项
        返回: 是否选择了某个项
        """
        if not self.hover_item:
            return False

        item_type, index = self.hover_item

        if item_type == "color":
            brush_manager.current_color_index = index
            return True
        elif item_type == "thickness":
            brush_manager.current_thickness_index = index
            return True
        elif item_type == "brush":
            brush_manager.current_brush_type_index = index
            return True

        return False

    def render(self, frame: np.ndarray, brush_manager):
        """渲染UI到画面上"""
        if not self.visible:
            return

        overlay = frame.copy()

        # 1. 渲染颜色选择器（左侧）
        self._render_color_panel(overlay, brush_manager)

        # 2. 渲染粗细选择器（底部）
        self._render_thickness_panel(overlay, brush_manager)

        # 3. 渲染笔刷类型选择器（右侧）
        self._render_brush_panel(overlay, brush_manager)

        # 混合overlay到frame（半透明效果）
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def _render_color_panel(self, overlay: np.ndarray, brush_manager):
        """渲染颜色选择面板"""
        for i, color_name in enumerate(brush_manager.COLOR_NAMES):
            color = brush_manager.COLORS[color_name]
            x = self.color_panel_x
            y = self.color_panel_y_start + i * self.color_button_spacing

            # 当前选中的颜色显示大圆圈
            is_selected = (i == brush_manager.current_color_index)
            is_hover = (self.hover_item == ("color", i))

            radius = self.color_button_size // 2
            if is_hover:
                radius += 5  # 悬停时稍大

            # 绘制圆形按钮
            cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

            # 选中状态：白色外圈
            if is_selected:
                cv2.circle(overlay, (x, y), radius + 5, (255, 255, 255), 3, lineType=cv2.LINE_AA)

            # 悬停状态：黄色外圈
            if is_hover:
                cv2.circle(overlay, (x, y), radius + 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    def _render_thickness_panel(self, overlay: np.ndarray, brush_manager):
        """渲染粗细选择面板"""
        for i, thickness in enumerate(brush_manager.THICKNESSES):
            x = self.thickness_panel_x_start + i * self.thickness_button_spacing
            y = self.thickness_panel_y

            is_selected = (i == brush_manager.current_thickness_index)
            is_hover = (self.hover_item == ("thickness", i))

            # 背景框
            bg_color = (80, 80, 80)
            if is_hover:
                bg_color = (120, 120, 120)
            cv2.rectangle(overlay,
                         (x, y),
                         (x + self.thickness_button_width, y + self.thickness_button_height),
                         bg_color, -1, lineType=cv2.LINE_AA)

            # 选中状态：白色边框
            if is_selected:
                cv2.rectangle(overlay,
                             (x, y),
                             (x + self.thickness_button_width, y + self.thickness_button_height),
                             (255, 255, 255), 3, lineType=cv2.LINE_AA)

            # 绘制示例线条（显示粗细）
            line_start = (x + 10, y + self.thickness_button_height // 2)
            line_end = (x + self.thickness_button_width - 10, y + self.thickness_button_height // 2)
            cv2.line(overlay, line_start, line_end, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

            # 悬停状态：黄色外框
            if is_hover:
                cv2.rectangle(overlay,
                             (x - 3, y - 3),
                             (x + self.thickness_button_width + 3, y + self.thickness_button_height + 3),
                             (0, 255, 255), 2, lineType=cv2.LINE_AA)

    def _render_brush_panel(self, overlay: np.ndarray, brush_manager):
        """渲染笔刷类型面板"""
        brush_labels = ["实线", "虚线", "发光", "马克"]

        for i, (brush_type, label) in enumerate(zip(brush_manager.BRUSH_TYPES, brush_labels)):
            x = self.brush_panel_x
            y = self.brush_panel_y_start + i * self.brush_button_spacing

            is_selected = (i == brush_manager.current_brush_type_index)
            is_hover = (self.hover_item == ("brush", i))

            # 背景框
            bg_color = (80, 80, 80)
            if is_hover:
                bg_color = (120, 120, 120)
            cv2.rectangle(overlay,
                         (x, y),
                         (x + self.brush_button_width, y + self.brush_button_height),
                         bg_color, -1, lineType=cv2.LINE_AA)

            # 选中状态：白色边框
            if is_selected:
                cv2.rectangle(overlay,
                             (x, y),
                             (x + self.brush_button_width, y + self.brush_button_height),
                             (255, 255, 255), 3, lineType=cv2.LINE_AA)

            # 绘制示例笔刷效果
            self._draw_brush_preview(overlay, brush_type, x, y)

            # 文字标签
            cv2.putText(overlay, label,
                       (x + 5, y + self.brush_button_height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # 悬停状态：黄色外框
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
            # 简化虚线预览
            for i in range(3):
                seg_start = (x + 10 + i * 15, line_y)
                seg_end = (x + 10 + i * 15 + 8, line_y)
                cv2.line(overlay, seg_start, seg_end, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        elif brush_type == "glow":
            # 发光效果
            cv2.line(overlay, line_start, line_end, (200, 200, 255), 5, lineType=cv2.LINE_AA)
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        elif brush_type == "marker":
            cv2.line(overlay, line_start, line_end, (255, 255, 255), 5, lineType=cv2.LINE_AA)

    def _point_in_circle(self, px: int, py: int, cx: int, cy: int, radius: int) -> bool:
        """检查点是否在圆内"""
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2

    def _point_in_rect(self, px: int, py: int, rx: int, ry: int, rw: int, rh: int) -> bool:
        """检查点是否在矩形内"""
        return rx <= px <= rx + rw and ry <= py <= ry + rh
