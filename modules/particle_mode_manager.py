# -*- coding: utf-8 -*-
"""
粒子模式管理器 - 3D版本
整合3D粒子系统和UI面板
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from modules.particle_system_3d import ParticleSystem3D
from modules.particle_mode_ui import ParticleModeUI
from modules.particle_models_3d import ParticleModel3DLibrary


class ParticleModeManager:
    """粒子模式管理器 - 整合3D粒子系统"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # 状态
        self.is_active = False
        self.is_ui_selecting = True  # 是否在UI选择阶段
        
        # 3D粒子系统
        self.particle_system_3d = ParticleSystem3D()
        
        # UI面板
        self.ui = ParticleModeUI(width, height)
        self.ui.on_model_change = self.on_model_change
        self.ui.on_color_change = self.on_color_change
        self.ui.on_confirm = self.on_confirm
        self.ui.on_cancel = self.on_cancel
        
        # 粒子数量
        self.particle_count = 5000
    
    def activate(self):
        """激活粒子模式"""
        self.is_active = True
        self.is_ui_selecting = True
        self.ui.show()
        print(">>> 进入粒子特效模式（UI选择）")
    
    def deactivate(self):
        """退出粒子模式"""
        self.is_active = False
        self.is_ui_selecting = False
        self.ui.hide()
        self.particle_system_3d.reset()
        print(">>> 退出粒子特效模式")
    
    def on_model_change(self, model_name: str):
        """模型选择回调"""
        self.particle_system_3d.current_model = model_name
        self.ui.set_active_model(model_name)
        print(f"选择模型: {model_name}")
    
    def on_color_change(self, color: Tuple[int, int, int]):
        """颜色选择回调"""
        self.particle_system_3d.set_color(color)
        print(f"选择颜色: {color}")
    
    def on_confirm(self):
        """确认选择，初始化粒子"""
        print("确认选择，初始化3D粒子...")
        self.particle_system_3d.initialize_particles(self.particle_count)
        self.is_ui_selecting = False
        self.ui.hide()
        print(f"3D粒子特效已启动！粒子数: {self.particle_count}")
    
    def on_cancel(self):
        """取消/退出"""
        self.deactivate()
    
    def handle_click(self, point: Tuple[int, int]) -> bool:
        """
        处理点击事件（手势或鼠标）
        返回: 是否点击到UI
        """
        if not self.is_active or not self.is_ui_selecting:
            return False
        
        # 使用UI的鼠标处理函数
        self.ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, point[0], point[1], 0, None)
        return True
    
    def handle_keyboard(self, key: int) -> bool:
        """
        处理键盘输入
        返回: 是否处理了按键
        """
        if not self.is_active:
            return False
        
        if key == ord('1'):
            # 确认
            if self.is_ui_selecting:
                self.on_confirm()
            return True
        elif key == ord('2'):
            # 退出
            self.on_cancel()
            return True
        
        return False
    
    def get_hand_spread_factor(self, hand) -> float:
        """
        计算手掌张开程度（基于手指间距）
        返回: 0.0-1.0的张开度
        - 0.0 = 手完全合拢（拳头）
        - 1.0 = 手完全张开（五指展开）
        """
        if not hand or not hasattr(hand, 'landmarks_norm'):
            return 0.0
        
        landmarks = hand.landmarks_norm
        if len(landmarks) < 21:
            return 0.0
        
        from core.hand_detector import THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST
        from core.hand_detector import distance as point_distance
        
        # 获取关键点
        thumb_tip = landmarks[THUMB_TIP]
        index_tip = landmarks[INDEX_TIP]
        middle_tip = landmarks[MIDDLE_TIP]
        ring_tip = landmarks[RING_TIP]
        pinky_tip = landmarks[PINKY_TIP]
        wrist = landmarks[WRIST]
        
        # 方法1：计算拇指和小指之间的距离（主要指标）
        thumb_pinky_dist = point_distance(thumb_tip, pinky_tip)
        
        # 方法2：计算所有指尖到手腕的平均距离（辅助指标）
        finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        avg_finger_distance = sum(point_distance(tip, wrist) for tip in finger_tips) / 5.0
        
        # 综合两个指标
        # 拇指-小指距离：闭合约0.03，张开约0.20（严格判定）
        thumb_pinky_factor = min(1.0, max(0.0, (thumb_pinky_dist - 0.03) / 0.17))
        
        # 指尖-手腕距离：闭合约0.15，张开约0.30（严格判定）
        finger_wrist_factor = min(1.0, max(0.0, (avg_finger_distance - 0.15) / 0.15))
        
        # 综合两个因子（拇指-小指距离占80%权重，更依赖主要指标）
        spread_factor = thumb_pinky_factor * 0.8 + finger_wrist_factor * 0.2
        
        # 额外处理：如果拇指-小指距离非常小，直接判定为完全合拢
        if thumb_pinky_dist < 0.04:
            spread_factor = 0.0
        
        return spread_factor
    
    def update(self, hands: list = None):
        """
        更新粒子系统
        :param hands: 检测到的手部列表
        
        逻辑：
        - 没有手 → 正常大小（scale=1.0）
        - 有手且张开 → 放大（scale>1.0）
        - 有手且合拢 → 缩小回原样（scale=1.0）
        """
        if not self.is_active or self.is_ui_selecting:
            return
        
        # 更新手势控制
        has_hand = False
        if hands and len(hands) > 0:
            hand_spread = self.get_hand_spread_factor(hands[0])
            self.particle_system_3d.update_hand_control(hand_spread)
            has_hand = True
        else:
            # 没有手时，恢复正常大小（不呼吸）
            self.particle_system_3d.reset_to_normal_size()
        
        # 更新粒子状态
        self.particle_system_3d.update(self.width, self.height, has_hand)
    
    def render(self, frame: np.ndarray, hands: list = None):
        """
        渲染粒子特效或UI
        :param frame: 视频帧
        :param hands: 检测到的手部列表
        """
        if not self.is_active:
            return
        
        if self.is_ui_selecting:
            # 渲染UI面板
            ui_frame = self.ui.render(frame)
            frame[:] = ui_frame
            
            # 底部提示
            cv2.putText(frame, "Press '1' to Confirm | Press '2' to Exit", 
                       (self.width // 2 - 250, self.height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # 渲染3D粒子特效
            self.particle_system_3d.render(frame)
            
            # 底部退出提示
            cv2.putText(frame, "Press '2' to Exit Particle Mode", 
                       (self.width // 2 - 200, self.height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    def render_camera_feed(self, frame: np.ndarray, camera_frame: np.ndarray):
        """
        渲染摄像头画面到左下角
        :param frame: 主画面
        :param camera_frame: 摄像头画面
        """
        if not self.is_active or self.is_ui_selecting:
            return
        
        # 缩小到1/5
        cam_h = self.height // 5
        cam_w = self.width // 5
        
        camera_resized = cv2.resize(camera_frame, (cam_w, cam_h))
        
        # 放置到左下角
        y_offset = self.height - cam_h - 20
        x_offset = 20
        
        frame[y_offset:y_offset + cam_h, x_offset:x_offset + cam_w] = camera_resized
