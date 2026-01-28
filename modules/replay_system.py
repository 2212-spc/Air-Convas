# -*- coding: utf-8 -*-
"""
绘画回放系统 (Replay System) - 修复版
修复内容：
1. _execute_action 中增加对 "save_stroke" 事件的处理，解决回放时撤销失效/痕迹残留问题。
2. 修复 Canvas 属性访问错误。
"""

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

# ==================== 数据模型层 ====================

@dataclass
class PaintAction:
    timestamp: float
    action_type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PaintAction':
        return PaintAction(
            timestamp=data['timestamp'],
            action_type=data['action_type'],
            data=data['data']
        )

# ==================== 逻辑控制层：录像机 ====================

class ReplayRecorder:
    def __init__(self):
        self.is_recording = False
        self.start_time = 0.0
        self.actions: List[PaintAction] = []
        self.current_session_id = ""

    def start_recording(self):
        self.is_recording = True
        self.start_time = time.time()
        self.actions = []
        self.current_session_id = time.strftime("%Y%m%d_%H%M%S")
        print(f"[Recorder] Started: {self.current_session_id}")

    def stop_recording(self) -> str:
        if not self.is_recording: return ""
        self.is_recording = False
        return self.save_to_file()

    def record_stroke_segment(self, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int, tool: str):
        if not self.is_recording: return
        rel_time = time.time() - self.start_time
        action = PaintAction(
            timestamp=rel_time,
            action_type="stroke",
            data={"tool": tool, "pt1": pt1, "pt2": pt2, "color": color, "thickness": thickness}
        )
        self.actions.append(action)

    def record_event(self, event_type: str, details: Optional[Dict] = None):
        if not self.is_recording: return
        rel_time = time.time() - self.start_time
        action = PaintAction(
            timestamp=rel_time,
            action_type=event_type,
            data=details if details else {}
        )
        self.actions.append(action)

    def save_to_file(self, output_dir: str = "recordings") -> str:
        if not self.actions: return ""
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"replay_{self.current_session_id}.json")
        payload = {
            "meta": {"version": "1.1", "session_id": self.current_session_id, "total_actions": len(self.actions)},
            "timeline": [action.to_dict() for action in self.actions]
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f: json.dump(payload, f, indent=2)
            print(f"[Recorder] Saved: {filename}")
            return filename
        except Exception as e:
            print(f"[Recorder] Error: {e}")
            return ""

# ==================== 逻辑控制层：播放器 ====================

class ReplayPlayer:
    def __init__(self, canvas_manager):
        self.canvas_manager = canvas_manager
        self.actions: List[PaintAction] = []
        self.is_playing = False
        self.is_paused = False
        self.current_index = 0
        self.playback_start_time = 0.0
        self.playback_speed = 1.0

    def load_file(self, filepath: str) -> bool:
        if not os.path.exists(filepath): return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            self.actions = [PaintAction.from_dict(item) for item in data.get("timeline", [])]
            return True
        except Exception: return False

    def play(self, speed: float = 1.0):
        if not self.actions: return
        self.is_playing = True; self.is_paused = False
        self.playback_speed = speed; self.current_index = 0
        self.playback_start_time = time.time()
        self.canvas_manager.clear() # 播放前清空画布

    def stop(self):
        self.is_playing = False; self.is_paused = False; self.current_index = 0

    def update(self):
        if not self.is_playing or self.is_paused: return
        if self.current_index >= len(self.actions): self.stop(); return
        
        elapsed = time.time() - self.playback_start_time
        current_play_time = elapsed * self.playback_speed

        while self.current_index < len(self.actions):
            action = self.actions[self.current_index]
            if action.timestamp <= current_play_time:
                self._execute_action(action)
                self.current_index += 1
            else: break

    def _execute_action(self, action: PaintAction):
        """执行动作 (修复版)"""
        d = action.data
        if action.action_type == "stroke":
            pt1, pt2 = tuple(d["pt1"]), tuple(d["pt2"])
            color, thickness = tuple(d["color"]), d["thickness"]
            tool = d.get("tool", "pen")
            target_canvas = self.canvas_manager.get_canvas() # 使用 get_canvas()
            
            c = (0,0,0) if tool == "eraser" else color
            cv2.line(target_canvas, pt1, pt2, c, thickness, cv2.LINE_AA)

        elif action.action_type == "save_stroke":
            # [关键修复] 执行保存笔画，确保回放时的 Undo 有效
            self.canvas_manager.save_stroke()

        elif action.action_type == "clear": self.canvas_manager.clear()
        elif action.action_type == "undo": self.canvas_manager.undo()
        elif action.action_type == "redo": self.canvas_manager.redo()

    def get_progress(self) -> float:
        if not self.actions: return 0.0
        return min(1.0, self.current_index / len(self.actions))