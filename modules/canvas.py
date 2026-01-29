# -*- coding: utf-8 -*-
"""画布模块 - 提供绘图核心逻辑与历史状态管理"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from collections import deque


class StrokeHistory:
    """
    笔画历史管理器 (StrokeHistory)
    
    维护画布的历史快照栈，支持撤销 (Undo) 和重做 (Redo) 功能。
    采用“栈”结构管理状态：
    1. Undo Stack: 存储过去的操作状态。
    2. Redo Stack: 存储被撤销的操作状态（一旦有新操作，Redo Stack 即被清空）。

    Attributes:
        max_history (int): 最大允许保存的历史记录步数。
        _history (List[np.ndarray]): 历史快照列表（Undo Stack）。
        _redo_stack (List[np.ndarray]): 重做快照列表（Redo Stack）。
        _current_index (int): 当前历史指针索引。
    """
    
    # [Type Hints] 显式声明属性类型
    max_history: int
    _history: List[np.ndarray]
    _redo_stack: List[np.ndarray]
    _current_index: int

    def __init__(self, max_history: int = 50) -> None:
        """
        初始化历史管理器。

        Args:
            max_history (int, optional): 最大历史步数，超过该值将丢弃最早的记录。默认为 50。
        """
        self.max_history = max_history
        self._history = []  # 历史快照栈
        self._redo_stack = []  # 重做栈
        self._current_index = -1  # 当前位置
    
    def push(self, canvas_snapshot: np.ndarray) -> None:
        """
        压入新的画布快照。
        
        当用户完成一次笔画或清空画布时调用。
        注意：此操作会清空重做栈 (Redo Stack)。

        Args:
            canvas_snapshot (np.ndarray): 当前画布的图像数组副本。
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
        """检查是否有可撤销的操作。"""
        return len(self._history) > 0
    
    def can_redo(self) -> bool:
        """检查是否有可重做的操作。"""
        return len(self._redo_stack) > 0
    
    def undo(self, current_canvas: np.ndarray) -> Optional[np.ndarray]:
        """
        执行撤销操作。

        将当前画布状态保存到重做栈，并弹出历史栈顶的状态返回。

        Args:
            current_canvas (np.ndarray): 执行撤销前的画布状态。

        Returns:
            Optional[np.ndarray]: 撤销后的目标画布状态。如果无法撤销，返回 None。
        """
        if not self.can_undo():
            return None
        
        # 将当前状态保存到重做栈
        self._redo_stack.append(current_canvas.copy())
        
        # 弹出并返回上一个状态
        return self._history.pop()
    
    def redo(self, current_canvas: np.ndarray) -> Optional[np.ndarray]:
        """
        执行重做操作。

        将当前画布状态保存回历史栈，并弹出重做栈顶的状态返回。

        Args:
            current_canvas (np.ndarray): 执行重做前的画布状态。

        Returns:
            Optional[np.ndarray]: 重做后的目标画布状态。如果无法重做，返回 None。
        """
        if not self.can_redo():
            return None
        
        # 将当前状态保存到历史栈
        self._history.append(current_canvas.copy())
        
        # 弹出并返回重做状态
        return self._redo_stack.pop()
    
    def clear(self) -> None:
        """清空所有历史记录和重做记录。"""
        self._history.clear()
        self._redo_stack.clear()
        self._current_index = -1
    
    @property
    def history_count(self) -> int:
        """int: 当前 Undo 栈中的记录数量。"""
        return len(self._history)
    
    @property
    def redo_count(self) -> int:
        """int: 当前 Redo 栈中的记录数量。"""
        return len(self._redo_stack)


class Canvas:
    """
    画布核心类 (Canvas)
    
    封装了 OpenCV 图像矩阵作为绘图表面，提供高层绘图接口（画线、擦除）。
    内部集成了 `StrokeHistory`，自动管理绘图状态的撤销与重做。

    Attributes:
        width (int): 画布宽度。
        height (int): 画布高度。
        _canvas (np.ndarray): 实际存储像素数据的 NumPy 数组 (BGR 格式)。
        _history (StrokeHistory): 历史状态管理器。
    """

    # [Type Hints] 显式声明属性类型
    width: int
    height: int
    _canvas: np.ndarray
    _history: StrokeHistory
    
    def __init__(self, width: int, height: int, max_history: int = 50) -> None:
        """
        初始化空白画布。

        Args:
            width (int): 画布宽度。
            height (int): 画布高度。
            max_history (int, optional): 最大历史步数。默认为 50。
        """
        self.width = width
        self.height = height
        # 初始化黑色背景 (BGR)
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._history = StrokeHistory(max_history=max_history)
        
        # 保存初始空白状态
        self._history.push(self._canvas)

    def draw_line(
        self, 
        pt1: Tuple[int, int], 
        pt2: Tuple[int, int], 
        color: Tuple[int, int, int], 
        thickness: int
    ) -> None:
        """
        在画布上绘制抗锯齿线条。

        Args:
            pt1 (Tuple[int, int]): 起点 (x, y)。
            pt2 (Tuple[int, int]): 终点 (x, y)。
            color (Tuple[int, int, int]): 线条颜色 (B, G, R)。
            thickness (int): 线条粗细。
        """
        cv2.line(self._canvas, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def erase(self, center: Tuple[int, int], radius: int) -> None:
        """
        在画布上擦除圆形区域（填充黑色）。

        Args:
            center (Tuple[int, int]): 擦除中心点 (x, y)。
            radius (int): 擦除半径。
        """
        cv2.circle(self._canvas, center, radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    def clear(self) -> None:
        """
        清空整张画布。
        
        此操作会自动保存当前状态到历史栈，以便撤销。
        """
        # 保存清空前的状态
        self._history.push(self._canvas.copy())
        self._canvas[:] = 0

    def get_canvas(self) -> np.ndarray:
        """
        获取当前画布的图像数组。

        Returns:
            np.ndarray: BGR 格式的图像数组。
        """
        return self._canvas
    
    def set_canvas(self, canvas: np.ndarray) -> None:
        """
        覆盖当前画布内容。

        通常用于撤销/重做时恢复状态。

        Args:
            canvas (np.ndarray): 新的画布图像数组。
        """
        self._canvas[:] = canvas

    def save(self, filename: str) -> None:
        """
        将当前画布保存为图片文件。

        Args:
            filename (str): 输出文件路径（如 "output.png"）。
        """
        cv2.imwrite(filename, self._canvas)
    
    # ========== 撤销/重做功能 ==========
    
    def save_stroke(self) -> None:
        """
        保存当前笔画状态。
        
        应当在每次笔画结束（如 Pinch End）或重要操作后调用。
        """
        self._history.push(self._canvas.copy())
    
    def undo(self) -> bool:
        """
        撤销上一步。

        Returns:
            bool: 如果撤销成功返回 True，否则返回 False。
        """
        snapshot = self._history.undo(self._canvas)
        if snapshot is not None:
            self._canvas[:] = snapshot
            return True
        return False
    
    def redo(self) -> bool:
        """
        重做上一步。

        Returns:
            bool: 如果重做成功返回 True，否则返回 False。
        """
        snapshot = self._history.redo(self._canvas)
        if snapshot is not None:
            self._canvas[:] = snapshot
            return True
        return False
    
    def can_undo(self) -> bool:
        """bool: 是否有可撤销的历史。"""
        return self._history.can_undo()
    
    def can_redo(self) -> bool:
        """bool: 是否有可重做的历史。"""
        return self._history.can_redo()
    
    def clear_history(self) -> None:
        """强制清空所有撤销历史（慎用）。"""
        self._history.clear()
    
    def get_history_info(self) -> str:
        """
        获取格式化的历史状态字符串。

        Returns:
            str: 例如 "History: 5 | Redo: 0"
        """
        return f"History: {self._history.history_count} | Redo: {self._history.redo_count}"