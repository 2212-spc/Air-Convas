from typing import List, Optional, Tuple

import cv2
import numpy as np


class ShapeRecognizer:
    def __init__(
        self,
        closedness_thresh: float = 0.2,
        circle_score_thresh: float = 0.75,
    ) -> None:
        self.closedness_thresh = closedness_thresh
        self.circle_score_thresh = circle_score_thresh

    def _closedness(self, pts: np.ndarray) -> float:
        perim = float(cv2.arcLength(pts, False))
        if perim < 1e-6:
            return 1.0
        start = pts[0, 0]
        end = pts[-1, 0]
        dist = np.linalg.norm(start - end)
        return dist / perim

    def _circle_score(self, pts: np.ndarray) -> float:
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        area_circle = np.pi * (radius**2)
        area_contour = float(cv2.contourArea(pts))
        if area_circle < 1e-6:
            return 0.0
        return area_contour / area_circle

    def recognize(self, points: List[Tuple[int, int]]) -> Optional[str]:
        if len(points) < 5:
            return None
        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        closedness = self._closedness(contour)
        closed = closedness < self.closedness_thresh

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        circle_score = self._circle_score(contour)

        if closed and vertices == 3:
            return "triangle"
        if closed and vertices == 4:
            return "rectangle"
        if closed and circle_score > self.circle_score_thresh:
            return "circle"
        return None

    def beautify(
        self,
        points: List[Tuple[int, int]],
        canvas_img: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> Optional[str]:
        shape = self.recognize(points)
        if not shape:
            return None

        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

        if shape == "triangle":
            hull = cv2.convexHull(contour)
            cv2.polylines(canvas_img, [hull], True, color, thickness, lineType=cv2.LINE_AA)
        elif shape == "rectangle":
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                canvas_img,
                (x, y),
                (x + w, y + h),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        elif shape == "circle":
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(
                canvas_img,
                (int(cx), int(cy)),
                int(radius),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        return shape
