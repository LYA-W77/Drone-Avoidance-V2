"""
无人机自主避障系统（激光雷达 + 距离传感器融合 + 键盘控制）
- 激光雷达为主，距离传感器为辅（冗余容错）
- 支持三种模式：AUTO（自动避障）、MANUAL（键盘控制）、EMERGENCY_STOP（紧急悬停）
- 按 P 切换 AUTO <-> MANUAL，按 空格/ESC 紧急停止，按 R 恢复 AUTO
"""

import airsim
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from enum import Enum, auto

# ==================== 飞行模式枚举 ====================
class FlightMode(Enum):
    MANUAL = auto()          # 手动键盘控制
    AUTO = auto()            # 自动避障（融合传感器）
    EMERGENCY_STOP = auto()  # 紧急停止（悬停）

# ==================== 主类 ====================
class ObstacleAvoidanceDrone:
    def __init__(self, verbose=True):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✓ 无人机已连接")

        # 避障参数
        self.SAFE_DISTANCE = 5.0
        self.CRUISE_SPEED = 3.0
        self.FLY_HEIGHT = -5.0
        self.CONTROL_DURATION = 0.25
        self.GOAL_WEIGHT = 0.7

        # 死胡同
        self.stuck_threshold = 1.5
        self.stuck_rotate_angle = 45
        self.max_stuck_rotates = 3

        # 目标点（将在起飞后自动计算）
        self.target_position = None
        self.target_distance = 50.0       # 目标距离（米）
        self.ARRIVE_DIST = 2.0

        # 扇区角度（左正右负，相对机头）
        self.sector_angles = [30, 15, 0, -15, -30]

        self.verbose = verbose
        self.is_flying = False
        self.obstacle_count = 0
        self.flight_distance = 0
        self.last_position = None
        self.stuck_counter = 0
        self.initial_yaw = 0.0

        # 点云存储（用于画图）
        self.latest_pc_x = np.array([])
        self.latest_pc_y = np.array([])

        # ===== 键盘控制相关 =====
        self.mode = FlightMode.AUTO
        self.last_key_time = 0

        # 手动控制平滑参数
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        self.manual_speed_max = 1.0
        self.manual_smooth_factor = 0.4

    # ==================== 基础功能 ====================
    def connect_and_takeoff(self):
        print("正在初始化...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("起飞中...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        self.client.moveToZAsync(self.FLY_HEIGHT, 2).join()
        print(f"当前高度: {abs(self.FLY_HEIGHT)} 米")

        state = self.client.getMultirotorState()
        self.initial_yaw = airsim.to_eularian_angles(
            state.kinematics_estimated.orientation
        )[2]

        self.is_flying = True
        self.last_position = self.get_position()
        print("✓ 起飞成功")

        # 自动设置目标点：左转90°方向 target_distance 米处
        pos = self.get_position()
        left_yaw = self.initial_yaw + math.radians(90)
        target_x = pos[0] + self.target_distance * math.cos(left_yaw)
        target_y = pos[1] + self.target_distance * math.sin(left_yaw)
        target_z = pos[2]
        self.target_position = np.array([target_x, target_y, target_z])
        print(f"目标点（左转90°方向 {self.target_distance}米）: ({target_x:.1f}, {target_y:.1f})")

    def get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    # ==================== 传感器数据获取（融合） ====================
    def get_lidar_distances(self, current_yaw):
        """
        融合激光雷达 + 距离传感器，返回5个扇区距离
        """
        # 第一步：读取激光雷达
        lidar_dists = [30.0] * 5
        try:
            lidar = self.client.getLidarData(vehicle_name="Drone1", lidar_name="")
            if len(lidar.point_cloud) >= 3:
                points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
                points = points[points[:, 2] < 0.5]
                if len(points) > 0:
                    cos_yaw = math.cos(current_yaw)
                    sin_yaw = math.sin(current_yaw)
                    pos = self.get_position()
                    dx = points[:, 0] - pos[0]
                    dy = points[:, 1] - pos[1]
                    x_body = dx * cos_yaw + dy * sin_yaw
                    y_body = -dx * sin_yaw + dy * cos_yaw
                    valid = x_body > 0.5
                    x_body = x_body[valid]
                    y_body = y_body[valid]
                    if len(x_body) > 0:
                        dists = np.sqrt(x_body**2 + y_body**2)
                        angles = np.degrees(np.arctan2(y_body, x_body))
                        for i, ang in enumerate(self.sector_angles):
                            mask = (angles > ang - 10) & (angles < ang + 10)
                            if np.any(mask):
                                lidar_dists[i] = np.min(dists[mask])
                        if self.verbose:
                            self.latest_pc_x = x_body
                            self.latest_pc_y = y_body
        except Exception as e:
            if self.verbose:
                print(f"激光雷达读取错误: {e}")

        # 第二步：读取距离传感器
        sensor_names = [
            "MyDistanceLeftLarge",
            "MyDistanceLeftSmall",
            "MyDistanceCenter",
            "MyDistanceRightSmall",
            "MyDistanceRightLarge"
        ]
        sensor_dists = [30.0] * 5
        for i, name in enumerate(sensor_names):
            try:
                data = self.client.getDistanceSensorData(name)
                d = data.distance
                if d > 0.1:
                    sensor_dists[i] = d
            except Exception as e:
                if self.verbose:
                    print(f"距离传感器 {name} 读取错误: {e}")

        # 第三步：融合
        fused_dists = []
        for i in range(5):
            lidar_val = lidar_dists[i]
            sensor_val = sensor_dists[i]
            if lidar_val < 29.9:
                fused_dists.append(min(lidar_val, sensor_val))
            else:
                fused_dists.append(sensor_val)
        return fused_dists
    
    # ==================== 环境分析与决策 ====================
    def analyze_environment(self, distances, target_dir_world, current_yaw):
        section_names = ["大左", "小左", "正中", "小右", "大右"]
        safety_scores = []
        goal_scores = []

        for i, dist in enumerate(distances):
            if dist >= self.SAFE_DISTANCE:
                obstacle_score = dist
            else:
                obstacle_score = dist * (dist / self.SAFE_DISTANCE)
            safety_scores.append(obstacle_score)

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
        forward_distance = distances[2]

        if self.verbose:
            print("\n--- 融合扇区 ---")
            print(f"当前偏航: {math.degrees(current_yaw):.0f}°")
            if target_dir_world is not None:
                print(f"目标方向(世界): ({target_dir_world[0]:.2f}, {target_dir_world[1]:.2f})")
            for i in range(5):
                print(f"[{section_names[i]:3s}] 距离:{distances[i]:5.1f}m "
                      f"安全分:{safety_scores[i]:5.1f} "
                      f"目标分:{goal_scores[i]:.2f} "
                      f"综合:{combined[i]:5.1f} {' <-- 选择' if i == safest_index else ''}")
            print(f"正前方距离: {forward_distance:.1f}m")
            print("-----------------\n")

        return safest_index, forward_distance, combined

    def is_stuck(self, safety_scores):
        return max(safety_scores) < self.stuck_threshold

    def decide_action(self, safest_index, forward_dist):
        side_map = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}
        if forward_dist < 2.0:
            vx = -1.5
            vy = side_map[safest_index] * 1.5
            self.obstacle_count += 1
        elif forward_dist < self.SAFE_DISTANCE * 0.5:
            vx = self.CRUISE_SPEED * 0.3
            vy = side_map[safest_index] * self.CRUISE_SPEED * 0.8
            self.obstacle_count += 1
        elif forward_dist < self.SAFE_DISTANCE:
            vx = self.CRUISE_SPEED * 0.6
            vy = side_map[safest_index] * self.CRUISE_SPEED * 0.4
        else:
            vx = self.CRUISE_SPEED
            vy = 0.0
        return vx, vy

    # ==================== 执行控制 ====================
    def execute_flight(self, vx_body, vy_body, duration=None):
        if duration is None:
            duration = self.CONTROL_DURATION
        self.client.moveByVelocityBodyFrameAsync(
            vx_body, vy_body, 0, duration,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0)
        ).join()
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

    # ==================== 键盘控制 ====================
    def update_mode_by_keyboard(self):
        current_time = time.time()
        if current_time - self.last_key_time < 0.3:
            return

        # 读取按键状态（使用 AirSim API）
        try:
            state = self.client.simGetInputState()
            kb = state.get('keyboard', {})
        except:
            return

        # P 键切换
        if kb.get('p', False):
            if self.mode == FlightMode.AUTO:
                self.mode = FlightMode.MANUAL
                print("🔄 切换至：手动控制模式 (WASD控制)[按 P 切回]")
            elif self.mode == FlightMode.MANUAL:
                self.mode = FlightMode.AUTO
                print("🔄 切换至：自动避障模式")
            self.last_key_time = current_time
            return  # 防止同一按键触发多个操作

        # 空格或 ESC 紧急停止
        if kb.get('space', False) or kb.get('escape', False):
            if self.mode != FlightMode.EMERGENCY_STOP:
                self.mode = FlightMode.EMERGENCY_STOP
                print("🛑 紧急停止！按 R 键恢复自动模式")
            self.last_key_time = current_time
            return

        # R 键恢复
        if kb.get('r', False) and self.mode == FlightMode.EMERGENCY_STOP:
            self.mode = FlightMode.AUTO
            print("✅ 已恢复，回到自动避障模式")
            self.last_key_time = current_time

    def reset_camera_follow(self):
        """强制将 UE4 摄像机置于无人机后上方，实现跟随效果"""
        try:
            # 获取无人机姿态
            pose = self.client.simGetVehiclePose()
            pos = pose.position
            orient = pose.orientation
            # 计算机体后方偏移（相对机体坐标）
            # 简单做法：直接在世界坐标系中偏移
            # 获取偏航角
            _, _, yaw = airsim.to_eularian_angles(orient)
            # 后方 8 米，上方 3 米
            back_x = -8 * math.cos(yaw)
            back_y = -8 * math.sin(yaw)
            cam_pos = airsim.Vector3r(pos.x_val + back_x, pos.y_val + back_y, pos.z_val - 3.0)  # 注意 Z 向下为负，所以 -3 是上升
            # 摄像机看向无人机
            # 设置摄像机姿态
            self.client.simSetCameraPose(
                camera_name="0",  # 默认主相机
                pose=airsim.Pose(cam_pos, orient),
                ignore_collision=True
            )
        except Exception as e:
            # 如果连接断开或出错，忽略
            pass

    def get_manual_velocity(self):
        try:
            state = self.client.simGetInputState()
            kb = state.get('keyboard', {})
        except:
            return 0.0, 0.0

        speed = self.manual_speed_max
        vx = 0.0
        vy = 0.0
        if kb.get('w', False):
            vx = speed
        if kb.get('s', False):
            vx = -speed
        if kb.get('a', False):
            vy = speed
        if kb.get('d', False):
            vy = -speed

        # 可选平滑（保留原平滑逻辑）
        alpha = self.manual_smooth_factor
        self.manual_vx += (vx - self.manual_vx) * alpha
        self.manual_vy += (vy - self.manual_vy) * alpha
        return self.manual_vx, self.manual_vy

    # ==================== 降落 ====================
    def land(self):
        print("降落中...")
        try:
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
        except Exception as e:
            print(f"⚠ 降落时连接已断开，忽略错误: {e}")

    # ==================== 主循环 ====================
    def run(self, max_time=360):
        try:
            self.connect_and_takeoff()
            print("=" * 40)

            start_time = time.time()
            step = 0

            while time.time() - start_time < max_time:
                # 1. 处理键盘输入
                self.update_mode_by_keyboard()

                # 2. 根据模式执行动作
                if self.mode == FlightMode.EMERGENCY_STOP:
                    self.client.hoverAsync()
                    time.sleep(0.1)
                    continue

                elif self.mode == FlightMode.MANUAL:
                    vx, vy = self.get_manual_velocity()
                    self.client.moveByVelocityBodyFrameAsync(
                        vx, vy, 0, 0.3,
                        airsim.DrivetrainType.ForwardOnly,
                        airsim.YawMode(False, 0)
                    )
                    time.sleep(0.1)
                    self.reset_camera_follow()
                    continue

                # ----- AUTO 模式（融合避障） -----
                pos = self.get_position()
                dist_to_target = np.linalg.norm(pos[:2] - self.target_position[:2])
                if dist_to_target < self.ARRIVE_DIST:
                    print("已到达目标！")
                    self.land()
                    return

                if step % 5 == 0:
                    self.reset_camera_follow()

                state = self.client.getMultirotorState()
                current_yaw = airsim.to_eularian_angles(
                    state.kinematics_estimated.orientation
                )[2]

                # 获取融合距离数据
                distances = self.get_lidar_distances(current_yaw)

                # 计算目标方向
                target_vector = self.target_position[:2] - pos[:2]
                target_dist = np.linalg.norm(target_vector)
                if target_dist > 0.1:
                    target_dir = target_vector / target_dist
                else:
                    target_dir = np.array([1.0, 0.0])

                safest, forward_dist, scores = self.analyze_environment(
                    distances, target_dir, current_yaw
                )

                # 死胡同检测
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
                # 每50步保存点云图（可选）
                if step % 50 == 0 and hasattr(self, 'latest_pc_x') and len(self.latest_pc_x) > 0:
                    plt.clf()
                    plt.scatter(self.latest_pc_x, self.latest_pc_y, s=1, c='blue', label='LiDAR Points')
                    for ang in self.sector_angles:
                        rad = math.radians(ang)
                        x_line = [0, 30 * math.cos(rad)]
                        y_line = [0, 30 * math.sin(rad)]
                        plt.plot(x_line, y_line, 'r--', linewidth=0.5)
                    plt.xlim(-15, 35)
                    plt.ylim(-25, 25)
                    plt.xlabel('X Body (机头前方)')
                    plt.ylabel('Y Body (机身左侧)')
                    plt.title(f'LiDAR 扇区 (步数 {step})')
                    plt.grid(True)
                    plt.savefig(f'lidar_sectors_step_{step}.png')
                    print(f"  📸 点云图已保存为 lidar_sectors_step_{step}.png")

                if step % 5 == 0:
                    act = "back" if vx < 0 else ("avoid" if vy != 0 else "cruise")
                    print(f"[步{step:3d}] {act} | vx:{vx:+4.1f} vy:{vy:+4.1f} | 距目标:{dist_to_target:.1f}m")

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.land()

# ==================== 运行入口 ====================
if __name__ == "__main__":
    drone = ObstacleAvoidanceDrone(verbose=True)
    drone.run(max_time=120)