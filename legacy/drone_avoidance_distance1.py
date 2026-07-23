"""
无人机自主避障系统（距离传感器版 + 自动左转目标）
- 使用5个距离传感器代替深度图
- 起飞后自动将目标设置为左转90°方向50米处
- 支持目标导航、死胡同检测与强制脱离
"""

import airsim
import numpy as np
import math
import time

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

        # 目标点（会在起飞后自动设置）
        self.target_position = None
        self.ARRIVE_DIST = 2.0
        self.GOAL_WEIGHT = 0.7
        self.TARGET_DISTANCE = 50.0      # 目标距离（米）

        # 死胡同
        self.stuck_threshold = 1.5
        self.stuck_rotate_angle = 45
        self.max_stuck_rotates = 3

        # 距离传感器名称（需与 AirSim settings.json 一致）
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
        self.initial_yaw = None

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
        # 记录初始偏航（用于设置目标）
        state = self.client.getMultirotorState()
        self.initial_yaw = airsim.to_eularian_angles(
            state.kinematics_estimated.orientation
        )[2]
        print("✓ 起飞成功")

    def get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_distance_sensors(self):
        """读取五个方向距离（米），无效值返回30.0"""
        distances = []
        for name in self.sensor_names:
            try:
                data = self.client.getDistanceSensorData(name)
                dist = data.distance
                if dist < 0.1:
                    dist = 30.0
                distances.append(dist)
            except:
                distances.append(30.0)
        return distances

    def analyze_environment(self, distances, target_dir_world, current_yaw):
        section_angles = [30, 15, 0, -15, -30]
        section_names = ["大左", "小左", "正中", "小右", "大右"]
        safety_scores = []
        goal_scores = []

        for i, dist in enumerate(distances):
            # 安全分：越远越高，低于安全距离时快速衰减
            if dist > self.SAFE_DISTANCE:
                obstacle_score = dist
            else:
                obstacle_score = dist * (dist / self.SAFE_DISTANCE)
            safety_scores.append(obstacle_score)

            # 目标对准分
            if target_dir_world is not None:
                local_dir = np.array([math.cos(math.radians(section_angles[i])),
                                      math.sin(math.radians(section_angles[i]))])
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
            print("\n--- 传感器分析 ---")
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

    def execute_flight(self, vx_body, vy_body, duration=None):
        if duration is None:
            duration = self.CONTROL_DURATION
        self.client.moveByVelocityBodyFrameAsync(
            vx_body, vy_body, 0, duration,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0)
        )
        time.sleep(duration)
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

            # 起飞后自动设置目标点：左转90°方向，距离 self.TARGET_DISTANCE 米
            pos = self.get_position()
            yaw = self.initial_yaw
            left_yaw = yaw + math.radians(90)
            target_x = pos[0] + self.TARGET_DISTANCE * math.cos(left_yaw)
            target_y = pos[1] + self.TARGET_DISTANCE * math.sin(left_yaw)
            target_z = pos[2]  # 保持高度
            self.target_position = np.array([target_x, target_y, target_z])
            print(f"目标点（左转90°方向 {self.TARGET_DISTANCE}米）: ({target_x:.1f}, {target_y:.1f})")
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

                distances = self.get_distance_sensors()

                target_vector = self.target_position[:2] - pos[:2]
                target_dist = np.linalg.norm(target_vector)
                if target_dist > 0.1:
                    target_dir = target_vector / target_dist
                else:
                    target_dir = np.array([1.0, 0.0])

                state = self.client.getMultirotorState()
                current_yaw = airsim.to_eularian_angles(
                    state.kinematics_estimated.orientation
                )[2]

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
    drone.TARGET_DISTANCE = 50.0   # 可修改目标距离
    drone.run(max_time=120)