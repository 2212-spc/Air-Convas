# -*- coding: utf-8 -*-
"""Temporary Ink Module - Implements GoodNotes-style disappearing laser ink"""

import time
import cv2
import numpy as np
from typing import List, Tuple, Deque
from collections import deque


class TemporaryStroke:
    """A single stroke of temporary ink that fades over time"""
    
    def __init__(self, color: Tuple[int, int, int], thickness: int, lifetime: float = 1.0):
        self.points: List[Tuple[int, int]] = []
        self.creation_time = time.time()
        self.color = color
        self.thickness = thickness
        self.lifetime = lifetime
        self.is_active = True
        
    def add_point(self, point: Tuple[int, int]):
        self.points.append(point)
        
    def is_expired(self) -> bool:
        return (time.time() - self.creation_time) > self.lifetime


class TemporaryInkManager:
    """Manages temporary ink strokes (laser pointer trails)"""
    
    def __init__(self, default_lifetime: float = 1.0):
        self.strokes: Deque[TemporaryStroke] = deque()
        self.current_stroke: TemporaryStroke = None
        self.default_lifetime = default_lifetime
        
    def start_stroke(self, color: Tuple[int, int, int], thickness: int):
        """Start a new temporary stroke"""
        self.current_stroke = TemporaryStroke(color, thickness, self.default_lifetime)
        self.strokes.append(self.current_stroke)
        
    def add_point(self, point: Tuple[int, int]):
        """Add point to current stroke"""
        if self.current_stroke:
            self.current_stroke.add_point(point)
            # Reset creation time to keep the stroke alive while drawing
            self.current_stroke.creation_time = time.time()
            
    def end_stroke(self):
        """Finish current stroke"""
        self.current_stroke = None
        
    def update(self):
        """Update strokes (remove expired ones)"""
        current_time = time.time()
        
        # Remove expired strokes
        while self.strokes:
            stroke = self.strokes[0]
            # Calculate remaining life
            age = current_time - stroke.creation_time
            if age > stroke.lifetime:
                self.strokes.popleft()
            else:
                break
                
    def render(self, frame: np.ndarray):
        """Render all active strokes with fading effect"""
        current_time = time.time()
        
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
                
            age = current_time - stroke.creation_time
            remaining_life = max(0, stroke.lifetime - age)
            
            # Fade out based on remaining life
            alpha = min(1.0, remaining_life / 0.5)  # Fade out in last 0.5s
            
            if alpha <= 0:
                continue
                
            # Convert points to numpy array for polylines
            pts = np.array(stroke.points, np.int32).reshape((-1, 1, 2))
            
            # Draw with alpha blending
            if alpha < 1.0:
                overlay = frame.copy()
                cv2.polylines(overlay, [pts], False, stroke.color, stroke.thickness, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                cv2.polylines(frame, [pts], False, stroke.color, stroke.thickness, cv2.LINE_AA)
                
    def clear(self):
        self.strokes.clear()
        self.current_stroke = None


