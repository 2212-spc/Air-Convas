"""
透明叠加画板 - 可以悬浮在任何应用（包括PPT放映）上方绘画
使用 tkinter 实现透明窗口 + OpenCV 处理画布
"""

import tkinter as tk
from tkinter import Canvas
import numpy as np
from collections import deque
import threading
import time


class TransparentOverlay:
    """
    透明全屏叠加画板
    - 悬浮在所有窗口上方
    - 背景完全透明，只显示笔迹
    - 支持多种颜色和橡皮擦
    """
    
    def __init__(self, pen_color="#FF0000", pen_width=4):
        self.pen_color = pen_color
        self.pen_width = pen_width
        self.eraser_width = 30
        
        self.root = None
        self.canvas = None
        self.is_running = False
        self.is_drawing = False
        self.is_erasing = False
        
        # 笔迹存储：每条线是一个点列表
        self.strokes = []  # [(color, width, [(x1,y1), (x2,y2), ...]), ...]
        self.current_stroke = []
        self.current_color = pen_color
        self.current_width = pen_width
        
        # 线条 ID 映射（用于橡皮擦删除）
        self.line_ids = []  # 存储所有 canvas line 的 id
        
        # 上一个点（用于连续画线）
        self.prev_point = None
        
        # 光标指示器
        self.cursor_id = None  # 光标圆圈的 canvas id
        self.cursor_size = 15  # 光标大小
        
        # 线程控制
        self._thread = None
        self._lock = threading.Lock()
        
        # 屏幕尺寸
        self.screen_width = 0
        self.screen_height = 0
    
    def start(self):
        """在新线程中启动透明窗口"""
        if self.is_running:
            return
        self._thread = threading.Thread(target=self._run_window, daemon=True)
        self._thread.start()
        # 等待窗口初始化
        time.sleep(0.5)
    
    def _run_window(self):
        """运行 tkinter 主循环（在单独线程）"""
        self.root = tk.Tk()
        self.root.title("AirCanvas Overlay")
        
        # 获取屏幕尺寸
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # 设置全屏无边框
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        
        # 设置透明背景
        self.root.configure(bg='white')
        self.root.attributes('-transparentcolor', 'white')
        self.root.attributes('-topmost', True)  # 始终置顶
        self.root.attributes('-alpha', 0.99)  # 几乎不透明（让线条清晰）
        
        # 创建画布
        self.canvas = Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # 允许鼠标穿透（不拦截鼠标事件）
        # 这样 PPT 等应用仍可接收鼠标点击
        self._set_click_through(True)
        
        self.is_running = True
        
        # 定期检查是否需要关闭 + 强制置顶
        self.root.after(100, self._check_alive)
        self.root.after(200, self._force_topmost)
        
        self.root.mainloop()
        self.is_running = False

    def _force_topmost(self):
        """定期强制置顶（对抗 PPT 全屏）"""
        if self.is_running and self.root:
            try:
                # 方法1: tkinter 原生置顶
                self.root.attributes('-topmost', True)
                
                # 方法2: Win32 API 强制置顶
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                HWND_TOPMOST = -1
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_NOACTIVATE = 0x0010
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
                )
            except Exception:
                pass
            
            # 每 200ms 重复
            self.root.after(200, self._force_topmost)
    
    def _set_click_through(self, enable: bool):
        """设置窗口是否允许鼠标穿透"""
        try:
            import ctypes
            from ctypes import wintypes
            
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            
            if enable:
                style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            else:
                style = style & ~WS_EX_TRANSPARENT
            
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception as e:
            print(f"设置鼠标穿透失败: {e}")
    
    def _check_alive(self):
        """定期检查"""
        if self.is_running and self.root:
            self.root.after(100, self._check_alive)
    
    def stop(self):
        """关闭透明窗口"""
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
            self.root = None
    
    def set_pen_color(self, color: str):
        """设置画笔颜色 (如 '#FF0000')"""
        self.current_color = color
    
    def set_pen_width(self, width: int):
        """设置画笔宽度"""
        self.current_width = width
    
    def start_stroke(self, x: int, y: int):
        """开始一笔"""
        with self._lock:
            self.is_drawing = True
            self.prev_point = (x, y)
            self.current_stroke = [(x, y)]
    
    def draw_to(self, x: int, y: int):
        """画线到指定点"""
        if not self.is_drawing or not self.canvas:
            return
        
        with self._lock:
            if self.prev_point:
                px, py = self.prev_point
                # 在 canvas 上画线（需要在主线程）
                def _draw():
                    if self.canvas:
                        line_id = self.canvas.create_line(
                            px, py, x, y,
                            fill=self.current_color,
                            width=self.current_width,
                            capstyle=tk.ROUND,
                            smooth=True
                        )
                        self.line_ids.append(line_id)
                
                if self.root:
                    self.root.after(0, _draw)
            
            self.prev_point = (x, y)
            self.current_stroke.append((x, y))
    
    def end_stroke(self):
        """结束一笔"""
        with self._lock:
            if self.current_stroke:
                self.strokes.append((
                    self.current_color,
                    self.current_width,
                    self.current_stroke.copy()
                ))
            self.is_drawing = False
            self.prev_point = None
            self.current_stroke = []

    def update_cursor(self, x: int, y: int, is_drawing: bool = False, is_erasing: bool = False):
        """
        更新光标位置 - 始终显示一个圆圈指示笔的位置
        - 未画时：绿色空心圆（导航状态）
        - 画笔时：红色实心圆
        - 橡皮时：蓝色大圆
        """
        if not self.canvas or not self.root:
            return
        
        def _update():
            if not self.canvas:
                return
            
            # 【关键】先删除所有旧光标元素（圆圈 + 十字）
            try:
                self.canvas.delete("cursor_circle")
                self.canvas.delete("cursor_cross")
            except:
                pass
            
            # 根据状态选择样式
            if is_erasing:
                # 橡皮擦：蓝色大圆
                size = 30
                color = "#00BFFF"
                width = 3
                fill = ""  # 空心
            elif is_drawing:
                # 画笔：红色小圆
                size = 6
                color = self.current_color
                width = 2
                fill = ""
            else:
                # 导航：亮绿色圆 + 十字
                size = 10
                color = "#00FF00"
                width = 2
                fill = ""
            
            # 画圆圈（带 tag 便于删除）
            self.canvas.create_oval(
                x - size, y - size,
                x + size, y + size,
                outline=color,
                width=width,
                fill=fill,
                tags="cursor_circle"
            )
            
            # 导航模式额外画十字线（更容易看到位置）
            if not is_drawing and not is_erasing:
                cross_size = 25
                # 横线
                self.canvas.create_line(
                    x - cross_size, y, x + cross_size, y,
                    fill=color, width=2, tags="cursor_cross"
                )
                # 竖线
                self.canvas.create_line(
                    x, y - cross_size, x, y + cross_size,
                    fill=color, width=2, tags="cursor_cross"
                )
        
        self.root.after(0, _update)

    def hide_cursor(self):
        """隐藏光标"""
        if not self.canvas or not self.root:
            return
        
        def _hide():
            if self.cursor_id:
                try:
                    self.canvas.delete(self.cursor_id)
                    self.canvas.delete("cursor_cross")
                except:
                    pass
                self.cursor_id = None
        
        self.root.after(0, _hide)
    
    def erase_at(self, x: int, y: int, radius: int = 30):
        """橡皮擦：删除指定位置附近的线条（但不删光标）"""
        if not self.canvas:
            return
        
        def _erase():
            if not self.canvas:
                return
            # 找到范围内的元素
            items = self.canvas.find_overlapping(
                x - radius, y - radius,
                x + radius, y + radius
            )
            for item in items:
                # 跳过光标元素（带 cursor_ tag 的）
                tags = self.canvas.gettags(item)
                if "cursor_circle" in tags or "cursor_cross" in tags:
                    continue
                self.canvas.delete(item)
        
        if self.root:
            self.root.after(0, _erase)
    
    def clear(self):
        """清除所有笔迹"""
        with self._lock:
            self.strokes = []
            self.current_stroke = []
            self.line_ids = []
        
        def _clear():
            if self.canvas:
                self.canvas.delete("all")
        
        if self.root:
            self.root.after(0, _clear)
    
    def set_visible(self, visible: bool):
        """显示/隐藏叠加层"""
        if self.root:
            def _set_vis():
                if visible:
                    self.root.deiconify()
                    self.root.attributes('-topmost', True)
                else:
                    self.root.withdraw()
            self.root.after(0, _set_vis)


# 单例模式
_overlay_instance = None

def get_overlay() -> TransparentOverlay:
    """获取透明叠加层单例"""
    global _overlay_instance
    if _overlay_instance is None:
        _overlay_instance = TransparentOverlay()
    return _overlay_instance


if __name__ == "__main__":
    # 测试：启动透明叠加层，手动画几条线
    overlay = get_overlay()
    overlay.start()
    
    print("透明叠加层已启动，按 Ctrl+C 退出")
    print("当前窗口会悬浮在所有应用上方")
    
    # 测试画线
    time.sleep(1)
    overlay.start_stroke(100, 100)
    for i in range(50):
        overlay.draw_to(100 + i * 10, 100 + i * 5)
        time.sleep(0.02)
    overlay.end_stroke()
    
    # 换颜色再画
    overlay.set_pen_color("#00FF00")
    overlay.start_stroke(500, 100)
    for i in range(50):
        overlay.draw_to(500 + i * 10, 100 + i * 3)
        time.sleep(0.02)
    overlay.end_stroke()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        overlay.stop()
        print("已退出")
