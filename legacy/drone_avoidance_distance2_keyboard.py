"""
无人机自主避障系统（距离传感器 + 起飞后左转 + 键盘控制）
- 需要 settings.json 中配置 5 个距离传感器
- 起飞后原地左转90°，然后朝正前方目标点飞行
- 支持三种模式：AUTO（自动避障）、MANUAL（键盘控制）、EMERGENCY_STOP（紧急悬停）
- 按 M 切换 AUTO <-> MANUAL，按 空格/ESC 紧急停止，按 R 恢复 AUTO
"""

import airsim
import numpy as np
import math
import time
from enum import Enum, auto

# ==================== 飞行模式枚举 ====================
class FlightMode(Enum):
    MANUAL = auto()          # 手动键盘控制
    AUTO = auto()            # 自动避障（原有逻辑）
    EMERGENCY_STOP = auto()  # 紧急停止（悬停）

# ==================== 主类 ====================
class ObstacleAvoidanceDrone:
    def __init__(self, verbose=True):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✓ 无人机已连接")

        # 参数
        self.SAFE_DISTANCE = 5.0
        self.CRUISE_SPEED = 3.0
        self.FLY_HEIGHT = -5.0
        self.CONTROL_DURATION = 0.25
        self.GOAL_WEIGHT = 0.7
        self.stuck_threshold = 1.5
        self.stuck_rotate_angle = 45
        self.max_stuck_rotates = 3

        # 目标点（将在左转后自动设定）
        self.target_position = None
        self.TARGET_FORWARD_DIST = 50.0   # 向前飞行的距离
        self.ARRIVE_DIST = 2.0

        # 传感器名称（必须与 settings.json 中一致）
        self.sensor_names = [
            "MyDistanceLeftLarge",
            "MyDistanceLeftSmall",
            "MyDistanceCenter",
            "MyDistanceRightSmall",
            "MyDistanceRightLarge"
        ]

        self.verbose = verbose
        self.is_flying = False
        self.obstacle_count = 0
        self.flight_distance = 0
        self.last_position = None
        self.stuck_counter = 0
        self.sector_angles = [30, 15, 0, -15, -30]  # 相对机头角度

        # ===== 新增：飞行模式与键盘防抖 =====
        self.mode = FlightMode.AUTO
        self.last_key_time = 0

        # 尝试导入键盘库，如果失败则标记不可用
        self.keyboard_available = False
        try:
            import keyboard
            self.keyboard_module = keyboard
            self.keyboard_available = True
        except ImportError:
            print("⚠ 未安装 keyboard 库，键盘控制不可用。请运行: pip install keyboard")

    def connect_and_takeoff(self):
        print("正在初始化...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("起飞中...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        self.client.moveToZAsync(self.FLY_HEIGHT, 2).join()
        print(f"当前高度: {abs(self.FLY_HEIGHT)} 米")
        self.is_flying = True
        self.last_position = self.get_position()
        print("✓ 起飞成功")

    def get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_distance_sensors(self):
        """读取五个距离传感器，返回米为单位的列表"""
        dists = []
        for name in self.sensor_names:
            try:
                data = self.client.getDistanceSensorData(name)
                d = data.distance
                if d < 0.1:
                    d = 30.0
                dists.append(d)
            except:
                dists.append(30.0)
                if self.verbose:
                    print(f"⚠ 传感器 {name} 未找到，返回30m")
        return dists

    def analyze_environment(self, distances, target_dir_world, current_yaw):
        section_names = ["大左", "小左", "正中", "小右", "大右"]
        safety_scores = []
        goal_scores = []

        for i, dist in enumerate(distances):
            if dist >= self.SAFE_DISTANCE:
                obs_score = dist
            else:
                obs_score = dist * (dist / self.SAFE_DISTANCE)
            safety_scores.append(obs_score)

            if target_dir_world is not None:
                local_dir = np.array([math.cos(math.radians(self.sector_angles[i])),
                                      math.sin(math.radians(self.sector_angles[i]))])
                rot = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                                [math.sin(current_yaw), math.cos(current_yaw)]])
                world_dir = rot @ local_dir
                cos_sim = np.dot(world_dir, target_dir_world)
                goal_score = (cos_sim + 1) / 2
            else:
                goal_score = 1.0
            goal_scores.append(goal_score)

        combined = [s * ((1 - self.GOAL_WEIGHT) + self.GOAL_WEIGHT * g)
                    for s, g in zip(safety_scores, goal_scores)]

        safest_index = np.argmax(combined)
        forward_dist = distances[2]  # 正前方

        if self.verbose:
            print("\n--- 传感器扇区 ---")
            print(f"偏航: {math.degrees(current_yaw):.0f}°  目标方向: ({target_dir_world[0]:.2f}, {target_dir_world[1]:.2f})")
            for i in range(5):
                print(f"[{section_names[i]:3s}] 距离:{distances[i]:5.1f}m  安全:{safety_scores[i]:5.1f}  目标:{goal_scores[i]:.2f}  综合:{combined[i]:5.1f}{' <-' if i == safest_index else ''}")
            print("-----------------\n")

        return safest_index, forward_dist, combined

    def is_stuck(self, safety_scores):
        return max(safety_scores) < self.stuck_threshold

    def decide_action(self, safest_index, forward_dist):
        side_map = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}
        if forward_dist < 2.0:
            vx, vy = -1.5, side_map[safest_index] * 1.5
            self.obstacle_count += 1
        elif forward_dist < self.SAFE_DISTANCE * 0.5:
            vx, vy = self.CRUISE_SPEED * 0.3, side_map[safest_index] * self.CRUISE_SPEED * 0.8
            self.obstacle_count += 1
        elif forward_dist < self.SAFE_DISTANCE:
            vx, vy = self.CRUISE_SPEED * 0.6, side_map[safest_index] * self.CRUISE_SPEED * 0.4
        else:
            vx, vy = self.CRUISE_SPEED, 0.0
        return vx, vy

    def execute_flight(self, vx_body, vy_body, duration=None):
        if duration is None:
            duration = self.CONTROL_DURATION
        self.client.moveByVelocityBodyFrameAsync(
            vx_body, vy_body, 0, duration,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0)
        )
        new_pos = self.get_position()
        if self.last_position is not None:
            self.flight_distance += np.linalg.norm(new_pos - self.last_position)
        self.last_position = new_pos

    def rotate_in_place(self, angle_deg):
        state = self.client.getMultirotorState()
        current_yaw_deg = math.degrees(
            airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        )
        target_yaw = (current_yaw_deg + angle_deg) % 360
        print(f"  ↻ 旋转 {angle_deg}°（{current_yaw_deg:.0f}° → {target_yaw:.0f}°）")
        self.client.rotateToYawAsync(target_yaw, timeout_sec=1.5).join()
        self.last_position = self.get_position()

    def force_escape(self):
        print("  ⚡ 强制脱离：后退 + 上升")
        pos = self.get_position()
        state = self.client.getMultirotorState()
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        back_x = pos[0] - 2.0 * math.cos(yaw)
        back_y = pos[1] - 2.0 * math.sin(yaw)
        up_z = pos[2] - 3.0
        self.client.moveToPositionAsync(back_x, back_y, up_z, 2.0,
                                         drivetrain=airsim.DrivetrainType.ForwardOnly,
                                         yaw_mode=airsim.YawMode(True, math.degrees(yaw) % 360)).join()
        self.last_position = self.get_position()
        self.stuck_counter = 0

    # ==================== 键盘控制方法（超级防崩溃版） ====================
    def update_mode_by_keyboard(self):
        """轮询键盘，更新飞行模式（非阻塞），任何异常都吃掉，保证飞行不中断"""
        current_time = time.time()
        if current_time - self.last_key_time < 0.3:  # 防抖
            return

        if not self.keyboard_available:
            return

        kb = self.keyboard_module
        
        try:
            # 1. M 键循环切换
            if kb.is_pressed('m'):
                if self.mode == FlightMode.AUTO:
                    self.mode = FlightMode.MANUAL
                    print("🔄 切换至：手动控制模式 (WASD控制)")
                elif self.mode == FlightMode.MANUAL:
                    self.mode = FlightMode.AUTO
                    print("🔄 切换至：自动避障模式")
                self.last_key_time = current_time
        except Exception:
            pass  # 忽略键盘读取错误

        try:
            # 2. 空格/ESC 紧急停止
            if kb.is_pressed('space') or kb.is_pressed('esc'):
                if self.mode != FlightMode.EMERGENCY_STOP:
                    self.mode = FlightMode.EMERGENCY_STOP
                    print("🛑 紧急停止！按 R 键恢复自动模式")
                self.last_key_time = current_time
        except Exception:
            pass

        try:
            # 3. R 键恢复
            if kb.is_pressed('r') and self.mode == FlightMode.EMERGENCY_STOP:
                self.mode = FlightMode.AUTO
                print("✅ 已恢复，回到自动避障模式")
                self.last_key_time = current_time
        except Exception:
            pass

    def get_manual_velocity(self):
        """获取键盘 WASD 映射的速度指令，任何异常都返回 (0,0) 保证安全"""
        vx, vy = 0.0, 0.0
        speed = 3.0
        
        if not self.keyboard_available:
            return 0.0, 0.0

        kb = self.keyboard_module
        try:
            if kb.is_pressed('w'): vx = speed
            if kb.is_pressed('s'): vx = -speed
            if kb.is_pressed('a'): vy = speed
            if kb.is_pressed('d'): vy = -speed
        except Exception:
            pass  # 键盘读取失败，直接悬停
        
        return vx, vy

    # ==================== 原有方法 ====================
    def land(self):
        print("降落中...")
        self.client.hoverAsync().join()
        time.sleep(0.5)
        self.client.landAsync().join()
        while True:
            if abs(self.get_position()[2]) < 0.5:
                break
            time.sleep(0.2)
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("✓ 安全降落")
        print(f"飞行距离: {self.flight_distance:.2f} m  避障: {self.obstacle_count} 次")

    def run(self, max_time=120):
        try:
            print("🚀 脚本开始执行... (请确保 AirSim 已启动)")
            self.connect_and_takeoff()

            # ========= 关键：起飞后原地左转90° =========
            print("起飞后左转90°...")
            self.rotate_in_place(-90)
            time.sleep(0.5)

            # 设定目标点：当前机头正前方 TARGET_FORWARD_DIST 米
            pos = self.get_position()
            state = self.client.getMultirotorState()
            yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
            target_x = pos[0] + self.TARGET_FORWARD_DIST * math.cos(yaw)
            target_y = pos[1] + self.TARGET_FORWARD_DIST * math.sin(yaw)
            target_z = pos[2]
            self.target_position = np.array([target_x, target_y, target_z])
            print(f"目标点: ({target_x:.1f}, {target_y:.1f})")
            print("=" * 40)

            start_time = time.time()
            step = 0

            # ==================== 主循环 ====================
            while time.time() - start_time < max_time:
                # 1. 先处理键盘指令（更新状态）
                self.update_mode_by_keyboard()

                # 2. 根据当前模式执行动作
                if self.mode == FlightMode.EMERGENCY_STOP:
                    self.client.hoverAsync()
                    time.sleep(0.1)
                    continue

                elif self.mode == FlightMode.MANUAL:
                    vx, vy = self.get_manual_velocity()
                    self.client.moveByVelocityBodyFrameAsync(
                        vx, vy, 0, 0.1,
                        airsim.DrivetrainType.ForwardOnly,
                        airsim.YawMode(False, 0)
                    )
                    time.sleep(0.05)
                    continue

                elif self.mode == FlightMode.AUTO:
                    # ---------- 原有自动避障逻辑 ----------
                    pos = self.get_position()
                    dist_to_target = np.linalg.norm(pos[:2] - self.target_position[:2])
                    if dist_to_target < self.ARRIVE_DIST:
                        print("已到达目标！")
                        self.land()
                        return

                    state = self.client.getMultirotorState()
                    current_yaw = airsim.to_eularian_angles(
                        state.kinematics_estimated.orientation
                    )[2]

                    distances = self.get_distance_sensors()

                    target_vector = self.target_position[:2] - pos[:2]
                    target_dist = np.linalg.norm(target_vector)
                    if target_dist > 0.1:
                        target_dir = target_vector / target_dist
                    else:
                        target_dir = np.array([1.0, 0.0])

                    safest, forward_dist, scores = self.analyze_environment(
                        distances, target_dir, current_yaw
                    )

                    if self.is_stuck(scores):
                        self.stuck_counter += 1
                        if self.stuck_counter > self.max_stuck_rotates:
                            self.force_escape()
                            continue
                        else:
                            self.rotate_in_place(self.stuck_rotate_angle)
                            vx_test, vy_test = self.decide_action(safest, forward_dist)
                            self.execute_flight(vx_test, vy_test, duration=0.5)
                            continue
                    else:
                        self.stuck_counter = 0

                    vx, vy = self.decide_action(safest, forward_dist)
                    self.execute_flight(vx, vy)

                    step += 1
                    if step % 5 == 0:
                        act = "back" if vx < 0 else ("avoid" if vy != 0 else "cruise")
                        print(f"[步{step:3d}] {act} | vx:{vx:+4.1f} vy:{vy:+4.1f} | 距目标:{dist_to_target:.1f}m")

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.land()

if __name__ == "__main__":
    drone = ObstacleAvoidanceDrone(verbose=True)
    drone.run(max_time=120)