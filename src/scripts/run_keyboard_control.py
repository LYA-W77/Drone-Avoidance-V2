import airsim
import numpy as np
import math
import time
import sys, os
import keyboard
from enum import Enum, auto

# 导入三个模块（需将父目录加入路径）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.flight_controller import FlightController
from perception.sensor_fusion import SensorFusion
from decision.advanced_avoidance import AdvancedAvoidance

class FlightMode(Enum):
    MANUAL = auto()
    AUTO = auto()
    EMERGENCY_STOP = auto()

class DroneRunner:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✓ 无人机已连接")

        self.flight = FlightController(self.client)
        self.perception = SensorFusion(self.client)
        self.decision = AdvancedAvoidance(safe_distance=5.0, cruise_speed=3.0)
        self.mode = FlightMode.AUTO
        self.last_key_time = 0
        self.target_position = None
        self.TARGET_DIST = 200.0
        self.ARRIVE_DIST = 2.0
        self.manual_speed = 2.0
        self.manual_vx, self.manual_vy = 0.0, 0.0

        # 注册热键
        try:
            keyboard.add_hotkey('p', self._on_p)
            keyboard.add_hotkey('space', self._on_space)
            keyboard.add_hotkey('esc', self._on_space)
            keyboard.add_hotkey('r', self._on_r)
            print("✓ 热键注册成功 (P/空格/ESC/R)")
        except:
            print("⚠ 请以管理员身份运行以启用热键")

    def _on_p(self):
        if time.time() - self.last_key_time < 0.3: return
        self.mode = FlightMode.MANUAL if self.mode == FlightMode.AUTO else FlightMode.AUTO
        print(f"🔄 切换至: {'手动' if self.mode == FlightMode.MANUAL else '自动'}模式")
        self.last_key_time = time.time()

    def _on_space(self):
        if time.time() - self.last_key_time < 0.3: return
        if self.mode != FlightMode.EMERGENCY_STOP:
            self.mode = FlightMode.EMERGENCY_STOP
            print("🛑 紧急停止")
        self.last_key_time = time.time()

    def _on_r(self):
        if time.time() - self.last_key_time < 0.3: return
        if self.mode == FlightMode.EMERGENCY_STOP:
            self.mode = FlightMode.AUTO
            print("✅ 已恢复自动")
        self.last_key_time = time.time()

    def run(self, max_time=120):
        try:
            # 起飞并设定目标（不左转90°）
            self.flight.takeoff(-5.0)
            #self.flight.rotate_in_place(-90)
            pos = self.flight.get_position()
            yaw = self.flight.get_yaw()
            self.target_position = np.array([
                pos[0] + self.TARGET_DIST * math.cos(yaw),
                pos[1] + self.TARGET_DIST * math.sin(yaw),
                pos[2]
            ])
            print(f"目标点: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f})")
            print("="*40)

            start, step = time.time(), 0
            while time.time() - start < max_time:
                # 1. 处理紧急模式
                if self.mode == FlightMode.EMERGENCY_STOP:
                    self.client.hoverAsync()
                    time.sleep(0.1)
                    continue

                # 2. 手动模式
                if self.mode == FlightMode.MANUAL:
                    vx = vy = 0.0
                    if keyboard.is_pressed('w'): vx = self.manual_speed
                    if keyboard.is_pressed('s'): vx = -self.manual_speed
                    if keyboard.is_pressed('a'): vy = self.manual_speed
                    if keyboard.is_pressed('d'): vy = -self.manual_speed
                    # 平滑
                    self.manual_vx += (vx - self.manual_vx) * 0.3
                    self.manual_vy += (vy - self.manual_vy) * 0.3
                    self.flight.move_by_velocity(self.manual_vx, self.manual_vy, duration=0.2)
                    self.flight.reset_camera_follow()
                    time.sleep(0.05)
                    continue

                # 3. 自动模式（核心）
                pos = self.flight.get_position()
                if np.linalg.norm(pos[:2] - self.target_position[:2]) < self.ARRIVE_DIST:
                    print("🎯 到达目标！")
                    break

                if step % 5 == 0: self.flight.reset_camera_follow()
                yaw = self.flight.get_yaw()
                distances = self.perception.get_distances(yaw)

                target_vec = self.target_position[:2] - pos[:2]
                target_dist = np.linalg.norm(target_vec)
                target_dir = target_vec / target_dist if target_dist > 0.1 else np.array([1.0, 0.0])

                safest, fwd, scores = self.decision.analyze(distances, target_dir, yaw)

                if self.decision.is_stuck(scores):
                    self.decision.increment_stuck()
                    if self.decision.should_force_escape():
                        self.flight.force_escape()
                        self.decision.reset_stuck()
                        continue
                    else:
                        self.flight.rotate_in_place(self.decision.stuck_rotate_angle)
                        vx, vy = self.decision.decide_action(safest, fwd)
                        self.flight.move_by_velocity(vx, vy, duration=0.5)
                        continue
                else:
                    self.decision.reset_stuck()

                vx, vy = self.decision.decide_action(safest, fwd)
                self.flight.move_by_velocity(vx, vy)
                step += 1

                if step % 5 == 0:
                    act = "back" if vx<0 else ("avoid" if vy!=0 else "cruise")
                    print(f"[{step:3d}] {act:6s} vx:{vx:+4.1f} vy:{vy:+4.1f} "
                          f"距目标:{np.linalg.norm(pos[:2] - self.target_position[:2]):.1f}m")

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.flight.land()

if __name__ == "__main__":
    runner = DroneRunner()
    runner.run(max_time=120)