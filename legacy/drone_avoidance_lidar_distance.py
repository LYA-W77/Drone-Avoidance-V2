"""
无人机自主避障系统（激光雷达版）
- 使用默认激光雷达获取360°点云，分割为5个前方扇区
- 支持目标点导航（可设置左转90°方向）
- 机头方向自由，通过侧向速度实现平移避障
"""

import airsim
import numpy as np
import math
import time
import matplotlib.pyplot as plt

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

        # 目标点（将在起飞后自动计算）
        self.target_position = None
        self.target_distance = 50.0       # 目标距离（米）
        self.ARRIVE_DIST = 2.0
        self.GOAL_WEIGHT = 0.7

        # 死胡同
        self.stuck_threshold = 1.5
        self.stuck_rotate_angle = 45
        self.max_stuck_rotates = 3

        self.verbose = verbose
        self.is_flying = False
        self.obstacle_count = 0
        self.flight_distance = 0
        self.last_position = None
        self.stuck_counter = 0
        self.initial_yaw = 0.0

        # 扇区角度（左正右负，相对机头）
        self.sector_angles = [30, 15, 0, -15, -30]
        self.latest_pc_x = np.array([])
        self.latest_pc_y = np.array([])

    def connect_and_takeoff(self):
        print("正在初始化...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("起飞中...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        self.client.moveToZAsync(self.FLY_HEIGHT, 2).join()
        print(f"当前高度: {abs(self.FLY_HEIGHT)} 米")

        # 记录初始偏航
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
        target_z = pos[2]  # 保持高度
        self.target_position = np.array([target_x, target_y, target_z])
        print(f"目标点（左转90°方向 {self.target_distance}米）: ({target_x:.1f}, {target_y:.1f})")

    def get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_lidar_distances(self, current_yaw):
        """
        获取激光雷达点云，并提取5个扇区的最小距离
        返回: [左大, 左小, 正中, 小右, 大右] 各方向距离（米）
        """
        #第一步读取激光雷达数据
        lidar_dists = [30.0] * 5 #默认值
        try:
            lidar = self.client.getLidarData(vehicle_name="Drone1", lidar_name="")
            if len(lidar.point_cloud) >= 3:
                # 解析点云为 Nx3 数组
                points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
                # 过滤地面点（z > 0 表示下方，可根据实际调整）
                # 这里简单过滤掉 z > -0.5 的点（NED坐标系，向下为正，地面在 z>0）
                points = points[points[:, 2] < 0.5]  # 保留高度 -0.5 以下的点（避免地面干扰）

                if len(points) > 0:
                    # 计算机体坐标系下的点
                    # 激光雷达点云是世界坐标系，需要转换到机体坐标系才能按角度分割
                    cos_yaw = math.cos(current_yaw)
                    sin_yaw = math.sin(current_yaw)
                    # 世界 → 机体：先平移后旋转？实际上直接用点与无人机位置差，再旋转到机体
                    pos = self.get_position()
                    dx = points[:,0]-pos[0]
                    dy = points[:,1]-pos[1]
                    # 旋转到机体坐标系（机头方向为 x 轴，左为 y 轴）
                    x_body = dx * cos_yaw + dy * sin_yaw
                    y_body = -dx * sin_yaw + dy * cos_yaw

                    # 过滤掉无人机后方的点（x_body < 0）
                    valid = x_body > 0.5
                    x_body = x_body[valid]
                    y_body = y_body[valid]
                    if len(x_body) > 0:
                        dists = np.sqrt(x_body**2+y_body**2)
                        angles = np.degrees(np.arctan2(y_body,x_body))
                        #计算5个扇区
                        for i,ang in enumerate(self.sector_angles):
                            mask = (angles > ang -10)&(angles<ang+10)
                            if np.any(mask):
                                lidar_dists[i]=np.min(dists[mask])
                            #否则保留30.0
                        #存储点云供可视化
                        if self.verbose:
                            self.latest_pc_x = x_body
                            self.latest_pc_y = y_body
        except Exception as e:
            if self.verbose:
                print(f"激光雷达读取错误：{e}")

        #第二步：读取传感器数据
        sensor_names = [
            "MyDistanceLeftLarge",
            "MyDistanceLeftSmall",
            "MyDistanceCenter",
            "MyDistanceRightSmall",
            "MyDistanceRightLarge"
        ]
        sensor_dists = [30.0] * 5
        for i,name in enumerate(sensor_names):
            try:
                data = self.client.getDistanceSensorData(name)
                d = data.distance
                if d > 0.1: #有效距离
                    sensor_dists[i] = d
                #否则保留30.0
            except Exception as e:
                if self.verbose:
                    print(f"距离传感器 {name} 读取错误: {e}")
        #第三步：融合两个传感器数据
        fused_dists = []
        for i in range(5):
            # 策略：如果激光雷达有效（< 29.9），取两者最小值（更保守）
            #       如果激光雷达无效（≈30.0），则用距离传感器的值
            lidar_val = lidar_dists[i]
            sensor_val =sensor_dists[i]
            if lidar_val <29.9:
                fused_dists.append(min(lidar_val, sensor_val))
            else:
                fused_dists.append(sensor_val)
        return fused_dists

    def analyze_environment(self, distances, target_dir_world, current_yaw):
        section_names = ["大左", "小左", "正中", "小右", "大右"]
        safety_scores = []
        goal_scores = []

        for i, dist in enumerate(distances):
            # 安全分：距离越远分数越高，低于安全距离时快速降低
            if dist >= self.SAFE_DISTANCE:
                obstacle_score = dist
            else:
                obstacle_score = dist * (dist / self.SAFE_DISTANCE)
            safety_scores.append(obstacle_score)

            # 目标对准分
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
        forward_distance = distances[2]  # 正前方

        if self.verbose:
            print("\n--- 激光雷达扇区 ---")
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
        """
        【我对这个决策逻辑的理解——2026.7.20】
        1. 如果正前方小于2米（极度危险）：立刻倒车（vx=-1.5），并以最大幅度向安全扇区侧移（vy=±1.5）。
        2. 如果正前方在2.5米到5米之间（警戒区）：减速至0.3倍速度，并以中等幅度向安全扇区侧移（vy=±0.8）。
        3. 如果正前方在5米以内但大于2.5米：轻微减速（0.6倍速度），小幅侧移（vy=±0.4）。
        4. 如果正前方大于5米（安全）：全速前进（vx=3.0），完全不侧移。
        
        侧移方向解释：
        - safest_index=0（大左扇区最安全）→ 说明右边有障碍，所以向左移（vy = +1.0）
        - safest_index=4（大右扇区最安全）→ 说明左边有障碍，所以向右移（vy = -1.0）
        - safest_index=2（正中扇区最安全）→ 直走，不偏移（vy = 0.0）
        """
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
            self.connect_and_takeoff()
            print("=" * 40)

            start_time = time.time()
            step = 0

            while time.time() - start_time < max_time:
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

                # 使用激光雷达获取距离
                distances = self.get_lidar_distances(current_yaw)
                # ========== 新增：检测死胡同（所有扇区都有障碍） ==========
                if all(d < self.SAFE_DISTANCE for d in distances):
                    print("🔄 四面受阻，原地左转30°尝试脱困")
                    self.rotate_in_place(30)  # 左转30度
                    continue  # 跳过本次决策，重新循环

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
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.land()

if __name__ == "__main__":
    drone = ObstacleAvoidanceDrone(verbose=True)
    # 目标点会自动计算：起飞后左转90°方向50米处
    drone.run(max_time=120)