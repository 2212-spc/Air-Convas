from typing import Tuple, Optional
import cv2
import numpy as np
from datetime import datetime


class PalmHUD:
    """掌心HUD - 在手掌上显示信息"""

    def __init__(
        self,
        still_threshold: float = 0.02,  # 静止判定阈值
        still_frames: int = 30,  # 需要静止的帧数（约1秒）
        box_width: int = 200,
        box_height: int = 100
    ):
        self.still_threshold = still_threshold
        self.still_frames = still_frames
        self.box_width = box_width
        self.box_height = box_height

        # 跟踪手掌位置历史
        self.palm_history = []
        self.is_still = False
        self.start_time = datetime.now()

    def _is_palm_still(self, palm_pos: Tuple[float, float]) -> bool:
        """判断手掌是否静止"""
        self.palm_history.append(palm_pos)

        # 保持足够的历史记录
        if len(self.palm_history) > self.still_frames:
            self.palm_history.pop(0)

        # 需要足够的历史数据
        if len(self.palm_history) < self.still_frames:
            return False

        # 计算位置方差
        positions = np.array(self.palm_history)
        variance = np.var(positions, axis=0)
        total_variance = np.sum(variance)

        # 如果方差很小，说明手掌静止
        return total_variance < self.still_threshold

    def reset(self) -> None:
        """重置状态"""
        self.palm_history.clear()
        self.is_still = False

    def update(self, palm_pos: Optional[Tuple[float, float]]) -> None:
        """更新手掌状态"""
        if palm_pos is None:
            self.reset()
            return

        self.is_still = self._is_palm_still(palm_pos)

    def render(
        self,
        frame: np.ndarray,
        palm_pos: Tuple[int, int],
        custom_text: Optional[str] = None
    ) -> None:
        """渲染HUD信息"""
        if not self.is_still:
            return

        x, y = palm_pos

        # 边界检查并调整位置
        x = max(self.box_width // 2, min(x, frame.shape[1] - self.box_width // 2))
        y = max(self.box_height // 2, min(y, frame.shape[0] - self.box_height // 2))

        # 创建半透明背景
        overlay = frame.copy()

        # HUD框的左上角和右下角
        x1 = x - self.box_width // 2
        y1 = y - self.box_height // 2
        x2 = x + self.box_width // 2
        y2 = y + self.box_height // 2

        # 绘制半透明黑色背景
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # 绘制边框
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # Alpha混合
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 准备文本信息
        current_time = datetime.now()
        time_str = current_time.strftime("%H:%M:%S")

        # 计算演讲时长
        elapsed = current_time - self.start_time
        duration_str = f"{int(elapsed.total_seconds() // 60):02d}:{int(elapsed.total_seconds() % 60):02d}"

        # 文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 255, 255)  # 青色

        # 绘制时间
        time_text = f"Time: {time_str}"
        cv2.putText(
            frame,
            time_text,
            (x1 + 10, y1 + 30),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

        # 绘制计时器
        duration_text = f"Duration: {duration_str}"
        cv2.putText(
            frame,
            duration_text,
            (x1 + 10, y1 + 55),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

        # 如果有自定义文本，显示它
        if custom_text:
            cv2.putText(
                frame,
                custom_text,
                (x1 + 10, y1 + 80),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA
            )

    def reset_timer(self) -> None:
        """重置计时器"""
        self.start_time = datetime.now()
