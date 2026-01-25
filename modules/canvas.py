# -*- coding: utf-8 -*-
"""画布模块 - 管理绘图画布，支持撤销/重做功能"""

import cv2
import numpy as np
from typing import List, Optional
from collections import deque


class StrokeHistory:
    """
    笔画历史管理器 - 支持撤销/重做功能
    
    实现原理：
    - 每完成一笔，保存当前画布快照到历史栈
    - 撤销：恢复到上一个快照
    - 重做：恢复到下一个快照
    """
    
    def __init__(self, max_history: int = 50):
        """
        初始化历史管理器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self._history: List[np.ndarray] = []  # 历史快照栈
        self._redo_stack: List[np.ndarray] = []  # 重做栈
        self._current_index = -1  # 当前位置
    
    def push(self, canvas_snapshot: np.ndarray) -> None:
        """
        保存画布快照
        
        Args:
            canvas_snapshot: 画布的副本
        """
        # 清空重做栈（新操作会使重做历史失效）
        self._redo_stack.clear()
        
        # 添加到历史
        self._history.append(canvas_snapshot.copy())
        
        # 限制历史长度
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        self._current_index = len(self._history) - 1
    
    def can_undo(self) -> bool:
        """是否可以撤销"""
        return len(self._history) > 0
    
    def can_redo(self) -> bool:
        """是否可以重做"""
        return len(self._redo_stack) > 0
    
    def undo(self, current_canvas: np.ndarray) -> Optional[np.ndarray]:
        """
        撤销操作
        
        Args:
            current_canvas: 当前画布（用于保存到重做栈）
        
        Returns:
            上一个画布快照，如果无法撤销则返回 None
        """
        if not self.can_undo():
            return None
        
        # 将当前状态保存到重做栈
        self._redo_stack.append(current_canvas.copy())
        
        # 弹出并返回上一个状态
        return self._history.pop()
    
    def redo(self, current_canvas: np.ndarray) -> Optional[np.ndarray]:
        """
        重做操作
        
        Args:
            current_canvas: 当前画布（用于保存到历史栈）
        
        Returns:
            下一个画布快照，如果无法重做则返回 None
        """
        if not self.can_redo():
            return None
        
        # 将当前状态保存到历史栈
        self._history.append(current_canvas.copy())
        
        # 弹出并返回重做状态
        return self._redo_stack.pop()
    
    def clear(self) -> None:
        """清空所有历史"""
        self._history.clear()
        self._redo_stack.clear()
        self._current_index = -1
    
    @property
    def history_count(self) -> int:
        """返回历史记录数量"""
        return len(self._history)
    
    @property
    def redo_count(self) -> int:
        """返回重做记录数量"""
        return len(self._redo_stack)


class Canvas:
    """
    画布类 - 管理绘图画布，支持撤销/重做
    
    特点：
    - 基础绘图功能（画线、擦除、清空）
    - 集成 StrokeHistory 实现撤销/重做
    - 自动在笔画结束时保存快照
    """
    
    def __init__(self, width: int, height: int, max_history: int = 50) -> None:
        """
        初始化画布
        
        Args:
            width: 画布宽度
            height: 画布高度
            max_history: 最大历史记录数
        """
        self.width = width
        self.height = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._history = StrokeHistory(max_history=max_history)
        
        # 保存初始空白状态
        self._history.push(self._canvas)

    def draw_line(self, pt1, pt2, color, thickness: int) -> None:
        """绘制线条"""
        cv2.line(self._canvas, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def erase(self, center, radius: int) -> None:
        """擦除区域"""
        cv2.circle(self._canvas, center, radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    def clear(self) -> None:
        """清空画布"""
        # 保存清空前的状态
        self._history.push(self._canvas.copy())
        self._canvas[:] = 0

    def get_canvas(self) -> np.ndarray:
        """获取画布数组"""
        return self._canvas
    
    def set_canvas(self, canvas: np.ndarray) -> None:
        """设置画布内容（用于撤销/重做）"""
        self._canvas[:] = canvas

    def save(self, filename: str) -> None:
        """保存画布到文件"""
        cv2.imwrite(filename, self._canvas)
    
    # ========== 撤销/重做功能 ==========
    
    def save_stroke(self) -> None:
        """
        保存当前笔画（在笔画结束时调用）
        
        这会将当前画布状态保存到历史栈，使其可以被撤销
        """
        self._history.push(self._canvas.copy())
    
    def undo(self) -> bool:
        """
        撤销上一步操作
        
        Returns:
            是否成功撤销
        """
        snapshot = self._history.undo(self._canvas)
        if snapshot is not None:
            self._canvas[:] = snapshot
            return True
        return False
    
    def redo(self) -> bool:
        """
        重做上一步撤销的操作
        
        Returns:
            是否成功重做
        """
        snapshot = self._history.redo(self._canvas)
        if snapshot is not None:
            self._canvas[:] = snapshot
            return True
        return False
    
    def can_undo(self) -> bool:
        """是否可以撤销"""
        return self._history.can_undo()
    
    def can_redo(self) -> bool:
        """是否可以重做"""
        return self._history.can_redo()
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self._history.clear()
    
    def get_history_info(self) -> str:
        """获取历史状态信息"""
        return f"History: {self._history.history_count} | Redo: {self._history.redo_count}"
