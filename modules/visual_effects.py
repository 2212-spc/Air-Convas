# -*- coding: utf-8 -*-
"""Visual Effects Module - Implements gesture feedback effects like ripples"""

import cv2
import numpy as np
from typing import List, Tuple
import math


class RippleEffect:
    """Expanding ripple effect (concentric circles)"""
    
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int], max_radius: int = 50, speed: int = 2):
        self.position = position
        self.color = color
        self.max_radius = max_radius
        self.current_radius = 0
        self.speed = speed
        self.is_active = True
        
    def update(self):
        self.current_radius += self.speed
        if self.current_radius >= self.max_radius:
            self.is_active = False
            
    def render(self, frame: np.ndarray):
        if not self.is_active:
            return
            
        # Alpha fades as radius grows
        alpha = 1.0 - (self.current_radius / self.max_radius)
        if alpha <= 0:
            return
            
        # Draw concentric circle
        overlay = frame.copy()
        cv2.circle(overlay, self.position, int(self.current_radius), self.color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)


class EffectManager:
    """Manages active visual effects"""
    
    def __init__(self):
        self.effects: List[RippleEffect] = []
        
    def add_ripple(self, position: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 255)):
        """Trigger a ripple effect at position"""
        self.effects.append(RippleEffect(position, color))
        
    def update(self):
        """Update all active effects"""
        self.effects = [e for e in self.effects if e.is_active]
        for effect in self.effects:
            effect.update()
            
    def render(self, frame: np.ndarray):
        """Render all active effects"""
        for effect in self.effects:
            effect.render(frame)
            
    def clear(self):
        self.effects.clear()


