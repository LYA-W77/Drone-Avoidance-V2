"""
无人机自主避障系统（激光雷达 + 距离传感器融合 + 键盘控制）
- 激光雷达为主，距离传感器为辅（冗余容错）
- 三种模式：AUTO / MANUAL / EMERGENCY_STOP
- 按 P 切换 AUTO/MANUAL，空格/ESC 紧急停止，R 恢复
- 手动模式：长按 WASD 连续控制，松开即停
- 必须 以管理员身份运行！
"""

import airsim
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import keyboard
from enum import Enum, auto

# ==================== 飞行模式 ====================
class FlightMode(Enum):
    MANUAL = auto()
    AUTO = auto()
    EMERGENCY_STOP = auto()

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
        self.target_position = None
        self.target_distance = 50.0
        self.ARRIVE_DIST = 2.0
        self.sector_angles = [30, 15, 0, -15, -30]

        self.verbose = verbose
        self.is_flying = False
        self.obstacle_count = 0
        self.flight_distance = 0
        self.last_position = None
        self.stuck_counter = 0
        self.initial_yaw = 0.0
        self.latest_pc_x = np.array([])
        self.latest_pc_y = np.array([])

        # 键盘控制
        self.mode = FlightMode.AUTO
        self.last_key_time = 0
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        self.manual_speed = 2.0

        # 注册热键
        try:
            keyboard.add_hotkey('p', self._on_p)
            keyboard.add_hotkey('space', self._on_space)
            keyboard.add_hotkey('esc', self._on_space)
            keyboard.add_hotkey('r', self._on_r)
            print("✓ 热键注册成功 (P/空格/ESC/R)")
        except Exception as e:
            print(f"⚠ 热键注册失败，请以管理员身份运行: {e}")

    # ---------- 基础功能 ----------
    def connect_and_takeoff(self):
        print("初始化...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("起飞...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        self.client.moveToZAsync(self.FLY_HEIGHT, 2).join()
        print(f"高度: {abs(self.FLY_HEIGHT)} 米")
        state = self.client.getMultirotorState()
        self.initial_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        self.is_flying = True
        self.last_position = self.get_position()
        print("✓ 起飞成功")
        pos = self.get_position()
        left_yaw = self.initial_yaw + math.radians(90)
        self.target_position = np.array([
            pos[0] + self.target_distance * math.cos(left_yaw),
            pos[1] + self.target_distance * math.sin(left_yaw),
            pos[2]
        ])
        print(f"目标点: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f})")

    def get_position(self):
        p = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

    # ---------- 传感器融合 ----------
    def get_lidar_distances(self, current_yaw):
        lidar_dists = [30.0] * 5
        try:
            lidar = self.client.getLidarData(vehicle_name="Drone1", lidar_name="")
            if len(lidar.point_cloud) >= 3:
                pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
                pts = pts[pts[:, 2] < 0.5]  # 滤除地面
                if len(pts) > 0:
                    cos_y = math.cos(current_yaw)
                    sin_y = math.sin(current_yaw)
                    pos = self.get_position()
                    dx = pts[:, 0] - pos[0]
                    dy = pts[:, 1] - pos[1]
                    x_b = dx * cos_y + dy * sin_y
                    y_b = -dx * sin_y + dy * cos_y
                    valid = x_b > 0.5
                    x_b = x_b[valid]
                    y_b = y_b[valid]
                    if len(x_b) > 0:
                        dists = np.sqrt(x_b**2 + y_b**2)
                        angles = np.degrees(np.arctan2(y_b, x_b))
                        for i, ang in enumerate(self.sector_angles):
                            mask = (angles > ang - 10) & (angles < ang + 10)
                            if np.any(mask):
                                lidar_dists[i] = np.min(dists[mask])
                        if self.verbose:
                            self.latest_pc_x = x_b
                            self.latest_pc_y = y_b
        except Exception as e:
            if self.verbose:
                print(f"激光雷达错误: {e}")

        # 距离传感器
        names = ["MyDistanceLeftLarge", "MyDistanceLeftSmall", "MyDistanceCenter",
                 "MyDistanceRightSmall", "MyDistanceRightLarge"]
        sensor_dists = [30.0] * 5
        for i, name in enumerate(names):
            try:
                d = self.client.getDistanceSensorData(name).distance
                if d > 0.1:
                    sensor_dists[i] = d
            except:
                pass

        # 融合
        fused = []
        for i in range(5):
            if lidar_dists[i] < 29.9:
                fused.append(min(lidar_dists[i], sensor_dists[i]))
            else:
                fused.append(sensor_dists[i])
        return fused

    # ---------- 决策 ----------
    def analyze_environment(self, distances, target_dir, yaw):
        names = ["大左", "小左", "正中", "小右", "大右"]
        safety, goal, combined = [], [], []
        for i, d in enumerate(distances):
            safety.append(d if d >= self.SAFE_DISTANCE else d * (d / self.SAFE_DISTANCE))
            if target_dir is not None:
                local = np.array([math.cos(math.radians(self.sector_angles[i])),
                                  math.sin(math.radians(self.sector_angles[i]))])
                rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                                [math.sin(yaw), math.cos(yaw)]])
                world = rot @ local
                goal.append((np.dot(world, target_dir) + 1) / 2)
            else:
                goal.append(1.0)
        for i in range(5):
            combined.append(safety[i] * ((1 - self.GOAL_WEIGHT) + self.GOAL_WEIGHT * goal[i]))
        safest = np.argmax(combined)
        forward = distances[2]
        if self.verbose:
            print("\n--- 融合扇区 ---")
            print(f"偏航: {math.degrees(yaw):.0f}°")
            for i in range(5):
                print(f"[{names[i]:3s}] {distances[i]:5.1f}m 安全:{safety[i]:5.1f} 目标:{goal[i]:.2f} 综合:{combined[i]:5.1f} {'<-' if i==safest else ''}")
            print(f"正前方: {forward:.1f}m")
            print("-----------------")
        return safest, forward, combined

    def is_stuck(self, scores):
        return max(scores) < self.stuck_threshold

    def decide_action(self, safest, forward):
        side = {0:1.0, 1:0.5, 2:0.0, 3:-0.5, 4:-1.0}
        if forward < 2.0:
            vx, vy = -1.5, side[safest] * 1.5
            self.obstacle_count += 1
        elif forward < self.SAFE_DISTANCE * 0.5:
            vx, vy = self.CRUISE_SPEED * 0.3, side[safest] * self.CRUISE_SPEED * 0.8
            self.obstacle_count += 1
        elif forward < self.SAFE_DISTANCE:
            vx, vy = self.CRUISE_SPEED * 0.6, side[safest] * self.CRUISE_SPEED * 0.4
        else:
            vx, vy = self.CRUISE_SPEED, 0.0
        return vx, vy

    def execute_flight(self, vx, vy, dur=None):
        dur = dur if dur else self.CONTROL_DURATION
        self.client.moveByVelocityBodyFrameAsync(vx, vy, 0, dur,
                airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
        new_pos = self.get_position()
        if self.last_position is not None:
            self.flight_distance += np.linalg.norm(new_pos - self.last_position)
        self.last_position = new_pos

    def rotate_in_place(self, deg):
        state = self.client.getMultirotorState()
        cur = math.degrees(airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2])
        target = (cur + deg) % 360
        print(f"  ↻ 旋转 {deg}° ({cur:.0f}° → {target:.0f}°)")
        self.client.rotateToYawAsync(target, 1.5).join()
        self.last_position = self.get_position()

    def force_escape(self):
        print("  ⚡ 强制脱离")
        pos = self.get_position()
        yaw = airsim.to_eularian_angles(self.client.getMultirotorState().kinematics_estimated.orientation)[2]
        back = pos[:2] - 2.0 * np.array([math.cos(yaw), math.sin(yaw)])
        self.client.moveToPositionAsync(back[0], back[1], pos[2]-3.0, 2.0,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(True, math.degrees(yaw)%360)).join()
        self.last_position = self.get_position()
        self.stuck_counter = 0

    def reset_camera_follow(self):
        try:
            pose = self.client.simGetVehiclePose()
            pos = pose.position
            orient = pose.orientation
            _, _, yaw = airsim.to_eularian_angles(orient)
            back_x = -8 * math.cos(yaw)
            back_y = -8 * math.sin(yaw)
            cam = airsim.Vector3r(pos.x_val + back_x, pos.y_val + back_y, pos.z_val - 3.0)
            self.client.simSetCameraPose("0", airsim.Pose(cam, orient), True)
        except:
            pass

    # ---------- 键盘回调 ----------
    def _on_p(self):
        if time.time() - self.last_key_time < 0.3: return
        if self.mode == FlightMode.AUTO:
            self.mode = FlightMode.MANUAL
            print("🔄 手动模式 (WASD长按控制)")
        else:
            self.mode = FlightMode.AUTO
            print("🔄 自动模式")
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

    def get_manual_velocity(self):
        vx = vy = 0.0
        if keyboard.is_pressed('w'): vx = self.manual_speed
        if keyboard.is_pressed('s'): vx = -self.manual_speed
        if keyboard.is_pressed('a'): vy = self.manual_speed
        if keyboard.is_pressed('d'): vy = -self.manual_speed
        smooth = 0.3
        self.manual_vx += (vx - self.manual_vx) * smooth
        self.manual_vy += (vy - self.manual_vy) * smooth
        return self.manual_vx, self.manual_vy

    # ---------- 降落 ----------
    def land(self):
        print("降落...")
        try:
            self.client.hoverAsync().join()
            time.sleep(0.5)
            self.client.landAsync().join()
            while abs(self.get_position()[2]) > 0.5:
                time.sleep(0.2)
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print(f"✓ 降落完成  飞行距离:{self.flight_distance:.1f}m  避障次数:{self.obstacle_count}")
        except Exception as e:
            print(f"降落忽略错误: {e}")

    # ---------- 主循环 ----------
    def run(self, max_time=120):
        try:
            self.connect_and_takeoff()
            print("="*40)
            print("【热键】P切换模式  空格/ESC急停  R恢复")
            print("【手动模式】长按WASD控制，松开即停")
            print("【重要】请以管理员身份运行！")
            print("="*40)
            start = time.time()
            step = 0

            while time.time() - start < max_time:
                # 模式执行
                if self.mode == FlightMode.EMERGENCY_STOP:
                    self.client.hoverAsync()
                    time.sleep(0.1)
                    continue

                elif self.mode == FlightMode.MANUAL:
                    vx, vy = self.get_manual_velocity()
                    self.client.moveByVelocityBodyFrameAsync(vx, vy, 0, 0.2,
                            airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
                    if step % 5 == 0:
                        self.reset_camera_follow()
                    step += 1
                    time.sleep(0.05)
                    continue

                # AUTO 模式
                pos = self.get_position()
                if np.linalg.norm(pos[:2] - self.target_position[:2]) < self.ARRIVE_DIST:
                    print("到达目标！")
                    self.land()
                    return

                if step % 5 == 0:
                    self.reset_camera_follow()

                state = self.client.getMultirotorState()
                yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
                distances = self.get_lidar_distances(yaw)

                target_vec = self.target_position[:2] - pos[:2]
                target_dist = np.linalg.norm(target_vec)
                target_dir = target_vec / target_dist if target_dist > 0.1 else np.array([1.0, 0.0])

                safest, fwd, scores = self.analyze_environment(distances, target_dir, yaw)

                if self.is_stuck(scores):
                    self.stuck_counter += 1
                    if self.stuck_counter > self.max_stuck_rotates:
                        self.force_escape()
                        continue
                    else:
                        self.rotate_in_place(self.stuck_rotate_angle)
                        vx, vy = self.decide_action(safest, fwd)
                        self.execute_flight(vx, vy, 0.5)
                        continue
                else:
                    self.stuck_counter = 0

                vx, vy = self.decide_action(safest, fwd)
                self.execute_flight(vx, vy)
                step += 1

                # 画图
                if step % 50 == 0 and hasattr(self, 'latest_pc_x') and len(self.latest_pc_x) > 0:
                    plt.clf()
                    plt.scatter(self.latest_pc_x, self.latest_pc_y, s=1, c='blue')
                    for ang in self.sector_angles:
                        rad = math.radians(ang)
                        plt.plot([0, 30*math.cos(rad)], [0, 30*math.sin(rad)], 'r--', lw=0.5)
                    plt.xlim(-15,35); plt.ylim(-25,25)
                    plt.xlabel('X Body'); plt.ylabel('Y Body')
                    plt.title(f'LiDAR扇区 step{step}')
                    plt.grid(True)
                    plt.savefig(f'lidar_sectors_step_{step}.png')
                    print(f"  📸 点云图保存")

                if step % 5 == 0:
                    act = "back" if vx<0 else ("avoid" if vy!=0 else "cruise")
                    print(f"[{step:3d}] {act:6s} vx:{vx:+4.1f} vy:{vy:+4.1f} 距目标:{np.linalg.norm(pos[:2] - self.target_position[:2]):.1f}m")

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.land()

# ==================== 运行 ====================
if __name__ == "__main__":
    drone = ObstacleAvoidanceDrone(verbose=True)
    drone.run(max_time=120)