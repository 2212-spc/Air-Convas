# -*- coding: utf-8 -*-
"""视口管理器 - 管理大画布上的可视区域和滚动"""

from typing import Tuple, Optional
import numpy as np
import cv2


class Viewport:
    """
    视口管理器 - 管理大画布上的可视区域
    
    支持：
    - 视口在大画布上的位置控制
    - 边缘自动滚动
    - 缩放控制
    - 坐标转换
    """
    
    # 缩放级别
    ZOOM_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (4000, 3000),
        view_size: Tuple[int, int] = (1280, 720),
        edge_margin: int = 80,
        scroll_speed: float = 15.0,
    ):
        """
        初始化视口管理器
        
        Args:
            canvas_size: 大画布尺寸 (宽, 高)
            view_size: 视口尺寸 (宽, 高)
            edge_margin: 边缘检测区域宽度（像素）
            scroll_speed: 最大滚动速度
        """
        self.canvas_w, self.canvas_h = canvas_size
        self.view_w, self.view_h = view_size
        self.edge_margin = edge_margin
        self.scroll_speed = scroll_speed
        
        # 视口位置（左上角在大画布上的坐标）
        # 默认居中
        self.x = (self.canvas_w - self.view_w) // 2
        self.y = (self.canvas_h - self.view_h) // 2
        
        # 缩放
        self._zoom_index = 2  # 默认 1.0x
        self.scale = self.ZOOM_LEVELS[self._zoom_index]
        
        # 平滑滚动状态
        self._scroll_velocity_x = 0.0
        self._scroll_velocity_y = 0.0
        self._scroll_smoothing = 0.3  # EMA 平滑系数
    
    def reset(self) -> None:
        """重置视口到画布中心"""
        self.x = (self.canvas_w - self.view_w) // 2
        self.y = (self.canvas_h - self.view_h) // 2
        self._scroll_velocity_x = 0.0
        self._scroll_velocity_y = 0.0
    
    def scroll(self, dx: int, dy: int) -> None:
        """
        移动视口
        
        Args:
            dx: X方向偏移
            dy: Y方向偏移
        """
        self.x = max(0, min(self.canvas_w - self.view_w, self.x + dx))
        self.y = max(0, min(self.canvas_h - self.view_h, self.y + dy))
    
    def check_edge_scroll(self, pen_pos: Tuple[int, int], roi_rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        检测笔尖是否靠近ROI窗口边缘，返回滚动量
        
        Args:
            pen_pos: 笔尖在视口中的位置 (x, y)
            roi_rect: ROI窗口矩形 (x, y, width, height)
        
        Returns:
            (dx, dy) 滚动量
        """
        x, y = pen_pos
        rx, ry, rw, rh = roi_rect
        margin = self.edge_margin
        
        target_dx, target_dy = 0.0, 0.0
        
        # 计算到各边缘的距离
        dist_left = x - rx
        dist_right = (rx + rw) - x
        dist_top = y - ry
        dist_bottom = (ry + rh) - y
        
        # 左边缘
        if dist_left < margin and dist_left > 0:
            target_dx = -self.scroll_speed * (1 - dist_left / margin)
        # 右边缘
        elif dist_right < margin and dist_right > 0:
            target_dx = self.scroll_speed * (1 - dist_right / margin)
        
        # 上边缘
        if dist_top < margin and dist_top > 0:
            target_dy = -self.scroll_speed * (1 - dist_top / margin)
        # 下边缘
        elif dist_bottom < margin and dist_bottom > 0:
            target_dy = self.scroll_speed * (1 - dist_bottom / margin)
        
        # EMA 平滑
        alpha = self._scroll_smoothing
        self._scroll_velocity_x = alpha * target_dx + (1 - alpha) * self._scroll_velocity_x
        self._scroll_velocity_y = alpha * target_dy + (1 - alpha) * self._scroll_velocity_y
        
        # 速度衰减（当不在边缘时）
        if target_dx == 0:
            self._scroll_velocity_x *= 0.8
        if target_dy == 0:
            self._scroll_velocity_y *= 0.8
        
        return int(self._scroll_velocity_x), int(self._scroll_velocity_y)
    
    def zoom_in(self) -> float:
        """放大，返回新的缩放因子"""
        if self._zoom_index < len(self.ZOOM_LEVELS) - 1:
            self._zoom_index += 1
            self.scale = self.ZOOM_LEVELS[self._zoom_index]
        return self.scale
    
    def zoom_out(self) -> float:
        """缩小，返回新的缩放因子"""
        if self._zoom_index > 0:
            self._zoom_index -= 1
            self.scale = self.ZOOM_LEVELS[self._zoom_index]
        return self.scale
    
    def get_view(self, canvas: np.ndarray) -> np.ndarray:
        """
        从大画布提取当前视口区域
        
        Args:
            canvas: 大画布数组
        
        Returns:
            视口区域的副本
        """
        return canvas[self.y:self.y + self.view_h, self.x:self.x + self.view_w].copy()
    
    def to_canvas_coords(self, view_point: Tuple[int, int]) -> Tuple[int, int]:
        """
        将视口坐标转换为画布坐标
        
        Args:
            view_point: 视口中的坐标 (x, y)
        
        Returns:
            画布上的坐标 (x, y)
        """
        vx, vy = view_point
        # 考虑缩放
        canvas_x = int(self.x + vx / self.scale)
        canvas_y = int(self.y + vy / self.scale)
        return canvas_x, canvas_y
    
    def to_view_coords(self, canvas_point: Tuple[int, int]) -> Tuple[int, int]:
        """
        将画布坐标转换为视口坐标
        
        Args:
            canvas_point: 画布上的坐标 (x, y)
        
        Returns:
            视口中的坐标 (x, y)
        """
        cx, cy = canvas_point
        view_x = int((cx - self.x) * self.scale)
        view_y = int((cy - self.y) * self.scale)
        return view_x, view_y
    
    def get_position_info(self) -> str:
        """获取当前位置信息字符串"""
        x_percent = self.x / max(1, self.canvas_w - self.view_w) * 100
        y_percent = self.y / max(1, self.canvas_h - self.view_h) * 100
        return f"Pos: ({x_percent:.0f}%, {y_percent:.0f}%) | Zoom: {self.scale:.0%}"


def create_grid_background(
    width: int,
    height: int,
    grid_size: int = 25,
    bg_color: Tuple[int, int, int] = (250, 250, 250),
    line_color: Tuple[int, int, int] = (220, 220, 220),
    major_line_color: Tuple[int, int, int] = (200, 200, 200),
    major_interval: int = 5,
) -> np.ndarray:
    """
    创建网格背景
    
    Args:
        width: 宽度
        height: 高度
        grid_size: 网格大小（像素）
        bg_color: 背景颜色 (B, G, R)
        line_color: 普通网格线颜色
        major_line_color: 主网格线颜色
        major_interval: 主网格线间隔（每N条线加粗）
    
    Returns:
        网格背景数组
    """
    # 创建背景
    bg = np.ones((height, width, 3), dtype=np.uint8)
    bg[:, :] = bg_color
    
    # 绘制垂直线
    line_count = 0
    for x in range(0, width, grid_size):
        color = major_line_color if line_count % major_interval == 0 else line_color
        thickness = 1
        cv2.line(bg, (x, 0), (x, height), color, thickness)
        line_count += 1
    
    # 绘制水平线
    line_count = 0
    for y in range(0, height, grid_size):
        color = major_line_color if line_count % major_interval == 0 else line_color
        thickness = 1
        cv2.line(bg, (0, y), (width, y), color, thickness)
        line_count += 1
    
    return bg


class ROIWindow:
    """
    ROI悬浮窗口 - 摄像头画面裁剪区域
    """
    
    def __init__(
        self,
        view_size: Tuple[int, int] = (1280, 720),
        roi_scale: float = 0.6,  # ROI窗口占视口的比例
        padding: float = 0.1,    # 摄像头画面边缘裁剪比例
        opacity: float = 0.7,    # 透明度
    ):
        """
        初始化ROI窗口
        
        Args:
            view_size: 视口尺寸
            roi_scale: ROI窗口占视口的比例
            padding: 摄像头画面边缘裁剪比例
            opacity: 透明度 (0-1)
        """
        self.view_w, self.view_h = view_size
        self.roi_scale = roi_scale
        self.padding = padding
        self.opacity = opacity
        
        # 计算ROI窗口尺寸和位置
        self.roi_w = int(self.view_w * roi_scale)
        self.roi_h = int(self.view_h * roi_scale)
        # 居中
        self.roi_x = (self.view_w - self.roi_w) // 2
        self.roi_y = (self.view_h - self.roi_h) // 2
    
    def get_rect(self) -> Tuple[int, int, int, int]:
        """获取ROI矩形 (x, y, width, height)"""
        return (self.roi_x, self.roi_y, self.roi_w, self.roi_h)
    
    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        裁剪摄像头画面边缘
        
        Args:
            frame: 原始摄像头画面
        
        Returns:
            裁剪后的画面
        """
        h, w = frame.shape[:2]
        x1 = int(w * self.padding)
        y1 = int(h * self.padding)
        x2 = int(w * (1 - self.padding))
        y2 = int(h * (1 - self.padding))
        return frame[y1:y2, x1:x2]
    
    def render(self, background: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        将ROI窗口渲染到背景上
        
        Args:
            background: 背景画面（视口内容）
            frame: 摄像头原始画面
        
        Returns:
            合成后的画面
        """
        result = background.copy()
        
        # 裁剪并缩放摄像头画面
        cropped = self.crop_frame(frame)
        roi_content = cv2.resize(cropped, (self.roi_w, self.roi_h), interpolation=cv2.INTER_LINEAR)
        
        # 半透明混合
        roi_region = result[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w]
        blended = cv2.addWeighted(roi_content, self.opacity, roi_region, 1 - self.opacity, 0)
        result[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w] = blended
        
        # 绘制边框
        cv2.rectangle(
            result,
            (self.roi_x, self.roi_y),
            (self.roi_x + self.roi_w, self.roi_y + self.roi_h),
            (100, 100, 100),
            2
        )
        
        # 绘制边缘指示（四个角）
        corner_len = 20
        corner_color = (0, 200, 200)
        corners = [
            # 左上
            ((self.roi_x, self.roi_y), (self.roi_x + corner_len, self.roi_y)),
            ((self.roi_x, self.roi_y), (self.roi_x, self.roi_y + corner_len)),
            # 右上
            ((self.roi_x + self.roi_w, self.roi_y), (self.roi_x + self.roi_w - corner_len, self.roi_y)),
            ((self.roi_x + self.roi_w, self.roi_y), (self.roi_x + self.roi_w, self.roi_y + corner_len)),
            # 左下
            ((self.roi_x, self.roi_y + self.roi_h), (self.roi_x + corner_len, self.roi_y + self.roi_h)),
            ((self.roi_x, self.roi_y + self.roi_h), (self.roi_x, self.roi_y + self.roi_h - corner_len)),
            # 右下
            ((self.roi_x + self.roi_w, self.roi_y + self.roi_h), (self.roi_x + self.roi_w - corner_len, self.roi_y + self.roi_h)),
            ((self.roi_x + self.roi_w, self.roi_y + self.roi_h), (self.roi_x + self.roi_w, self.roi_y + self.roi_h - corner_len)),
        ]
        for p1, p2 in corners:
            cv2.line(result, p1, p2, corner_color, 3)
        
        return result
    
    def map_to_roi(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """
        将归一化坐标(0-1)映射到ROI窗口内的像素坐标
        
        考虑了边缘裁剪，所以归一化坐标需要调整
        
        Args:
            norm_point: 归一化坐标 (0-1)
        
        Returns:
            ROI窗口内的像素坐标
        """
        nx, ny = norm_point
        
        # 调整裁剪区域的归一化坐标
        # 原始 0-1 中，padding~(1-padding) 映射到 0-1
        adj_x = (nx - self.padding) / (1 - 2 * self.padding)
        adj_y = (ny - self.padding) / (1 - 2 * self.padding)
        
        # 限制在 0-1 范围内
        adj_x = max(0.0, min(1.0, adj_x))
        adj_y = max(0.0, min(1.0, adj_y))
        
        # 映射到ROI窗口像素坐标
        px = int(self.roi_x + adj_x * self.roi_w)
        py = int(self.roi_y + adj_y * self.roi_h)
        
        return px, py


class FloatingROI:
    """
    可移动的ROI书写窗口 - 阿里云美效SDK风格
    
    核心设计：
    - 白板完整/缩放显示
    - ROI窗口可以在白板上自由移动
    - 笔迹映射到ROI对应的白板位置
    - 移动ROI窗口 = 改变书写位置
    
    参考：阿里云视频云美颜特效SDK的AR隔空书写能力
    """
    
    def __init__(
        self,
        whiteboard_size: Tuple[int, int] = (1920, 1080),  # 白板尺寸
        display_size: Tuple[int, int] = (1280, 720),      # 显示尺寸
        roi_scale: float = 0.5,                            # ROI占显示区域的比例
        camera_padding: float = 0.1,                       # 摄像头边缘裁剪比例
        edge_margin: int = 50,                             # 边缘触发移动的区域宽度
        move_speed: float = 8.0,                           # ROI移动速度
    ):
        """
        初始化可移动ROI
        
        Args:
            whiteboard_size: 白板画布尺寸
            display_size: 显示窗口尺寸
            roi_scale: ROI窗口占显示区域的比例
            camera_padding: 摄像头画面边缘裁剪比例
            edge_margin: 边缘触发移动的区域宽度
            move_speed: ROI移动速度
        """
        self.whiteboard_w, self.whiteboard_h = whiteboard_size
        self.display_w, self.display_h = display_size
        self.roi_scale = roi_scale
        self.camera_padding = camera_padding
        self.edge_margin = edge_margin
        self.move_speed = move_speed
        
        # 计算缩放比例（白板缩放到显示尺寸）
        self.scale_x = self.display_w / self.whiteboard_w
        self.scale_y = self.display_h / self.whiteboard_h
        self.scale = min(self.scale_x, self.scale_y)  # 保持宽高比
        
        # 缩放后的白板尺寸
        self.scaled_wb_w = int(self.whiteboard_w * self.scale)
        self.scaled_wb_h = int(self.whiteboard_h * self.scale)
        
        # 白板在显示区域中的偏移（居中）
        self.wb_offset_x = (self.display_w - self.scaled_wb_w) // 2
        self.wb_offset_y = (self.display_h - self.scaled_wb_h) // 2
        
        # ROI窗口尺寸（在显示坐标系中）
        self.roi_display_w = int(self.display_w * roi_scale)
        self.roi_display_h = int(self.display_h * roi_scale)
        
        # ROI在白板上的位置（白板坐标系，不是显示坐标系）
        # 对应的白板区域尺寸
        self.roi_wb_w = int(self.roi_display_w / self.scale)
        self.roi_wb_h = int(self.roi_display_h / self.scale)
        
        # ROI在白板上的位置（默认居中）
        self.roi_x = (self.whiteboard_w - self.roi_wb_w) // 2
        self.roi_y = (self.whiteboard_h - self.roi_wb_h) // 2
        
        # 平滑移动
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._smoothing = 0.3
    
    def reset(self) -> None:
        """重置ROI位置到白板中心"""
        self.roi_x = (self.whiteboard_w - self.roi_wb_w) // 2
        self.roi_y = (self.whiteboard_h - self.roi_wb_h) // 2
        self._velocity_x = 0.0
        self._velocity_y = 0.0
    
    def move(self, dx: int, dy: int) -> None:
        """
        移动ROI窗口
        
        Args:
            dx, dy: 移动量（白板坐标系）
        """
        self.roi_x = max(0, min(self.whiteboard_w - self.roi_wb_w, self.roi_x + dx))
        self.roi_y = max(0, min(self.whiteboard_h - self.roi_wb_h, self.roi_y + dy))
    
    def check_edge_move(self, pen_pos_in_roi: Tuple[float, float]) -> Tuple[int, int]:
        """
        检测笔尖是否靠近ROI边缘，返回移动量
        
        当笔尖靠近ROI窗口边缘时，ROI窗口在白板上移动
        
        Args:
            pen_pos_in_roi: 笔尖在ROI窗口内的归一化位置 (0-1, 0-1)
        
        Returns:
            (dx, dy) ROI移动量
        """
        nx, ny = pen_pos_in_roi
        
        # 边缘区域比例
        edge_ratio = self.edge_margin / min(self.roi_display_w, self.roi_display_h)
        
        target_dx, target_dy = 0.0, 0.0
        
        # 左边缘
        if nx < edge_ratio:
            target_dx = -self.move_speed * (1 - nx / edge_ratio)
        # 右边缘
        elif nx > 1 - edge_ratio:
            target_dx = self.move_speed * (1 - (1 - nx) / edge_ratio)
        
        # 上边缘
        if ny < edge_ratio:
            target_dy = -self.move_speed * (1 - ny / edge_ratio)
        # 下边缘
        elif ny > 1 - edge_ratio:
            target_dy = self.move_speed * (1 - (1 - ny) / edge_ratio)
        
        # 平滑移动
        self._velocity_x = self._smoothing * target_dx + (1 - self._smoothing) * self._velocity_x
        self._velocity_y = self._smoothing * target_dy + (1 - self._smoothing) * self._velocity_y
        
        # 衰减
        if target_dx == 0:
            self._velocity_x *= 0.8
        if target_dy == 0:
            self._velocity_y *= 0.8
        
        return int(self._velocity_x), int(self._velocity_y)
    
    def map_to_whiteboard(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """
        将归一化坐标映射到白板坐标
        
        Args:
            norm_point: 摄像头归一化坐标 (0-1)，已考虑裁剪
        
        Returns:
            白板上的坐标
        """
        nx, ny = norm_point
        
        # 调整裁剪区域的归一化坐标
        adj_x = (nx - self.camera_padding) / (1 - 2 * self.camera_padding)
        adj_y = (ny - self.camera_padding) / (1 - 2 * self.camera_padding)
        
        # 限制在 0-1 范围内
        adj_x = max(0.0, min(1.0, adj_x))
        adj_y = max(0.0, min(1.0, adj_y))
        
        # 映射到ROI在白板上的位置
        wb_x = int(self.roi_x + adj_x * self.roi_wb_w)
        wb_y = int(self.roi_y + adj_y * self.roi_wb_h)
        
        return wb_x, wb_y
    
    def get_roi_norm_position(self, norm_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        获取归一化坐标在ROI内的相对位置 (0-1)
        
        Args:
            norm_point: 摄像头归一化坐标
        
        Returns:
            在ROI内的归一化位置
        """
        nx, ny = norm_point
        
        # 调整裁剪区域
        adj_x = (nx - self.camera_padding) / (1 - 2 * self.camera_padding)
        adj_y = (ny - self.camera_padding) / (1 - 2 * self.camera_padding)
        
        return max(0.0, min(1.0, adj_x)), max(0.0, min(1.0, adj_y))
    
    def whiteboard_to_display(self, wb_point: Tuple[int, int]) -> Tuple[int, int]:
        """
        将白板坐标转换为显示坐标
        
        Args:
            wb_point: 白板坐标
        
        Returns:
            显示坐标
        """
        wx, wy = wb_point
        dx = int(wx * self.scale) + self.wb_offset_x
        dy = int(wy * self.scale) + self.wb_offset_y
        return dx, dy
    
    def get_roi_display_rect(self) -> Tuple[int, int, int, int]:
        """
        获取ROI在显示坐标系中的矩形
        
        Returns:
            (x, y, width, height)
        """
        dx = int(self.roi_x * self.scale) + self.wb_offset_x
        dy = int(self.roi_y * self.scale) + self.wb_offset_y
        dw = int(self.roi_wb_w * self.scale)
        dh = int(self.roi_wb_h * self.scale)
        return dx, dy, dw, dh
    
    def render(
        self,
        whiteboard: np.ndarray,
        camera_frame: np.ndarray,
        draw_layer: np.ndarray,
        opacity: float = 0.7
    ) -> np.ndarray:
        """
        渲染完整画面
        
        Args:
            whiteboard: 白板背景（原始尺寸）
            camera_frame: 摄像头画面
            draw_layer: 绘图层（原始尺寸）
            opacity: 摄像头画面透明度
        
        Returns:
            渲染后的显示画面
        """
        # 1. 缩放白板和绘图层
        # 优化：不再先合成，而是分开缩放，以便后续分层渲染
        scaled_wb = cv2.resize(whiteboard, (self.scaled_wb_w, self.scaled_wb_h), interpolation=cv2.INTER_LINEAR)
        scaled_draw = cv2.resize(draw_layer, (self.scaled_wb_w, self.scaled_wb_h), interpolation=cv2.INTER_NEAREST) # 绘图层用最近邻保持锐利，或者线性也行
        
        # 2. 创建显示画面（带边距的居中）
        display = np.ones((self.display_h, self.display_w, 3), dtype=np.uint8) * 200  # 灰色背景
        
        # 3. 放置缩放后的白板背景
        wb_x1 = self.wb_offset_x
        wb_y1 = self.wb_offset_y
        wb_x2 = wb_x1 + self.scaled_wb_w
        wb_y2 = wb_y1 + self.scaled_wb_h
        
        display[wb_y1:wb_y2, wb_x1:wb_x2] = scaled_wb
        
        # 4. 在背景上叠加非ROI区域的笔迹（可选，为了让全图都能看到笔迹，但被ROI覆盖的部分稍后会重绘）
        # 这里先简单叠加，会被后续的摄像头画面覆盖
        mask_draw = np.any(scaled_draw != 0, axis=2)
        # 注意坐标偏移
        display_roi_full = display[wb_y1:wb_y2, wb_x1:wb_x2]
        display_roi_full[mask_draw] = scaled_draw[mask_draw]
        
        # 5. 裁剪并缩放摄像头画面
        h, w = camera_frame.shape[:2]
        pad = self.camera_padding
        x1, y1 = int(w * pad), int(h * pad)
        x2, y2 = int(w * (1 - pad)), int(h * (1 - pad))
        cropped = camera_frame[y1:y2, x1:x2]
        
        # ROI在显示坐标系中的位置和尺寸
        roi_dx, roi_dy, roi_dw, roi_dh = self.get_roi_display_rect()
        
        # 确保ROI在显示区域内
        roi_dx = max(0, min(self.display_w - roi_dw, roi_dx))
        roi_dy = max(0, min(self.display_h - roi_dh, roi_dy))
        
        # 缩放摄像头画面到ROI尺寸
        roi_camera = cv2.resize(cropped, (roi_dw, roi_dh), interpolation=cv2.INTER_LINEAR)
        
        # 6. 半透明叠加摄像头画面 (Overlay Camera)
        roi_region = display[roi_dy:roi_dy + roi_dh, roi_dx:roi_dx + roi_dw]
        blended = cv2.addWeighted(roi_camera, opacity, roi_region, 1 - opacity, 0)
        display[roi_dy:roi_dy + roi_dh, roi_dx:roi_dx + roi_dw] = blended
        
        # 7. 【关键改进】在ROI区域再次叠加笔迹（Glow Effect / Additive Blend）
        # 计算ROI在scaled_draw中的对应位置
        draw_roi_x = roi_dx - self.wb_offset_x
        draw_roi_y = roi_dy - self.wb_offset_y
        
        # 边界检查
        if 0 <= draw_roi_x < self.scaled_wb_w and 0 <= draw_roi_y < self.scaled_wb_h:
            # 获取对应的绘图层区域
            # 注意：roi_dw, roi_dh 可能因为边界裁剪而需要调整，这里简化假设完整
            # 更严谨的做法是计算交集
            
            # 实际可用的宽高
            act_w = min(roi_dw, self.scaled_wb_w - draw_roi_x)
            act_h = min(roi_dh, self.scaled_wb_h - draw_roi_y)
            
            if act_w > 0 and act_h > 0:
                draw_roi_content = scaled_draw[draw_roi_y:draw_roi_y+act_h, draw_roi_x:draw_roi_x+act_w]
                
                # 找出非黑像素
                mask_roi = np.any(draw_roi_content != 0, axis=2)
                
                if np.any(mask_roi):
                    # 获取当前显示区域（包含了背景+摄像头）
                    current_display_roi = display[roi_dy:roi_dy+act_h, roi_dx:roi_dx+act_w]
                    
                    # 使用加法混合（Glow Effect），让字迹发亮且清晰
                    # 仅在有笔迹的地方进行混合
                    
                    # 提取笔迹和背景
                    fg = draw_roi_content[mask_roi]
                    bg = current_display_roi[mask_roi]
                    
                    # 加法混合
                    added = cv2.add(bg, fg)
                    
                    # 更新回去
                    current_display_roi[mask_roi] = added

        # 8. 绘制ROI边框
        cv2.rectangle(display, (roi_dx, roi_dy), (roi_dx + roi_dw, roi_dy + roi_dh), (0, 200, 200), 2)
        
        # 9. 绘制四角指示
        corner_len = 15
        corner_color = (0, 255, 255)
        corners = [
            ((roi_dx, roi_dy), (roi_dx + corner_len, roi_dy), (roi_dx, roi_dy + corner_len)),
            ((roi_dx + roi_dw, roi_dy), (roi_dx + roi_dw - corner_len, roi_dy), (roi_dx + roi_dw, roi_dy + corner_len)),
            ((roi_dx, roi_dy + roi_dh), (roi_dx + corner_len, roi_dy + roi_dh), (roi_dx, roi_dy + roi_dh - corner_len)),
            ((roi_dx + roi_dw, roi_dy + roi_dh), (roi_dx + roi_dw - corner_len, roi_dy + roi_dh), (roi_dx + roi_dw, roi_dy + roi_dh - corner_len)),
        ]
        for corner, h_end, v_end in corners:
            cv2.line(display, corner, h_end, corner_color, 3)
            cv2.line(display, corner, v_end, corner_color, 3)
        
        return display
    
    def get_position_info(self) -> str:
        """获取当前位置信息字符串"""
        x_percent = self.roi_x / max(1, self.whiteboard_w - self.roi_wb_w) * 100
        y_percent = self.roi_y / max(1, self.whiteboard_h - self.roi_wb_h) * 100
        return f"ROI: ({x_percent:.0f}%, {y_percent:.0f}%)"

