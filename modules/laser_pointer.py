from typing import Tuple
import cv2
import numpy as np


class LaserPointer:
    """激光笔效果 - 在食指指尖显示激光指示器"""

    def __init__(
        self,
        color: Tuple[int, int, int] = (0, 0, 255),  # BGR红色
        inner_radius: int = 4,
        middle_radius: int = 8,
        outer_radius: int = 15
    ):
        self.color = color
        self.inner_radius = inner_radius
        self.middle_radius = middle_radius
        self.outer_radius = outer_radius

    def render(self, frame: np.ndarray, position: Tuple[int, int]) -> None:
        """在指定位置渲染激光笔效果"""
        x, y = position

        # 边界检查（包括最大半径）
        h, w = frame.shape[:2]
        if x < -self.outer_radius or y < -self.outer_radius or \
           x >= w + self.outer_radius or y >= h + self.outer_radius:
            return

        try:
            # 创建一个临时图层用于混合
            overlay = frame.copy()

            # 外层 - 半透明红色光晕
            b, g, r = self.color
            outer_color = (int(b * 0.3), int(g * 0.3), int(r * 0.5))
            cv2.circle(overlay, (x, y), self.outer_radius, outer_color, -1, lineType=cv2.LINE_AA)

            # 中层 - 较亮的红色
            middle_color = (int(b * 0.5), int(g * 0.5), int(r * 0.7))
            cv2.circle(overlay, (x, y), self.middle_radius, middle_color, -1, lineType=cv2.LINE_AA)

            # 内层 - 亮红色实心圆
            cv2.circle(overlay, (x, y), self.inner_radius, self.color, -1, lineType=cv2.LINE_AA)

            # Alpha混合
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 添加一个小的白色高光点
            highlight_offset = 1
            cv2.circle(
                frame,
                (x - highlight_offset, y - highlight_offset),
                1,
                (255, 255, 255),
                -1,
                lineType=cv2.LINE_AA
            )
        except Exception:
            # 忽略渲染错误
            pass

    def render_with_trail(
        self,
        frame: np.ndarray,
        position: Tuple[int, int],
        trail_positions: list
    ) -> None:
        """渲染激光笔和拖尾效果"""
        try:
            # 先绘制拖尾
            if len(trail_positions) > 1:
                for i in range(len(trail_positions) - 1):
                    # 计算透明度（越旧越透明）
                    alpha = (i + 1) / len(trail_positions) * 0.3

                    pt1 = trail_positions[i]
                    pt2 = trail_positions[i + 1]

                    # 绘制淡淡的拖尾线
                    overlay = frame.copy()
                    cv2.line(overlay, pt1, pt2, self.color, 2, lineType=cv2.LINE_AA)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 然后绘制主激光点
            self.render(frame, position)
        except Exception:
            # 如果拖尾渲染失败，至少渲染主激光点
            self.render(frame, position)
