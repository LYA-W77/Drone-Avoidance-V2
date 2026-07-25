import numpy as np
import math

class AdvancedAvoidance:
    def __init__(self, safe_distance=5.0, cruise_speed=3.0):
        self.safe_distance = safe_distance
        self.cruise_speed = cruise_speed
        self.sector_angles = [30, 15, 0, -15, -30]
        
        # 死胡同参数（复用阈值版的逻辑）
        self.stuck_counter = 0
        self.max_stuck_rotates = 3
        self.stuck_threshold = 1.5
        self.stuck_rotate_angle = 45
        
        # 存储上一次的距离和目标方向（供 decide_action 使用）
        self.last_distances = None
        self.last_target_dir = None
        self.obstacle_count = 0

    def analyze(self, distances, target_dir_world, current_yaw):
        """分析环境，存储数据，返回兼容格式"""
        self.last_distances = distances
        self.last_target_dir = target_dir_world
        self.last_yaw = current_yaw
        
        # 计算安全分（用于死胡同检测，沿用阈值版逻辑）
        safety_scores = []
        for d in distances:
            safety_scores.append(d if d >= self.safe_distance else d * (d / self.safe_distance))
        
        # 返回：safest_index(随便填), forward_dist, safety_scores
        return 2, distances[2], safety_scores

    def is_stuck(self, scores):
        """死胡同判断：综合分低于阈值"""
        return max(scores) < self.stuck_threshold

    def decide_action(self, safest_index, forward_dist):
        """
        核心决策：目标方向 + 左右空旷度 加权避障
        """
        if self.last_distances is None or self.last_target_dir is None:
            return 0.0, 0.0

        distances = self.last_distances
        target_dir = self.last_target_dir

        # ========== 1. 前向速度（永不归零） ==========
        if forward_dist < 2.0:
            vx = 0.8
        elif forward_dist < self.safe_distance:
            vx = self.cruise_speed * (forward_dist / self.safe_distance)
        else:
            vx = self.cruise_speed

        # ========== 2. 计算左右空旷度 ==========
        left_avg = (distances[0] + distances[1]) / 2.0
        right_avg = (distances[3] + distances[4]) / 2.0
        center = distances[2]

        # ========== 3. 计算目标方向在当前机头坐标系下的偏转 ==========
        # target_dir 是世界坐标系下的方向，需要转换到机体系
        yaw = self.last_yaw
        # 目标方向相对机头的角度（弧度）
        target_angle = math.atan2(target_dir[1], target_dir[0]) - yaw
        # 归一化到 [-pi, pi]
        target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))
    
        # 目标偏转角度（度），正=左，负=右
        target_deg = math.degrees(target_angle)

        # ========== 4. 加权决策：目标方向占 60%，障碍物占 40% ==========
        alpha = 0.6   # 目标方向权重（越大越倾向直飞目标）

        # 目标方向产生的侧向指令（正=左转，负=右转）
        # 当目标在左侧时，vy_target 为正；右侧时为负
        vy_target = math.sin(target_angle) * self.cruise_speed * 0.5   # 幅度控制

        # 障碍物产生的侧向指令（原逻辑，但降低权重）
        if left_avg > right_avg:
            vy_obs = 0.3 * (left_avg - right_avg) / self.safe_distance
            vy_obs = min(vy_obs, 1.5)
        else:
            vy_obs = -0.3 * (right_avg - left_avg) / self.safe_distance
            vy_obs = max(vy_obs, -1.5)

        # 加权合成最终侧向速度
        vy_raw = alpha * vy_target + (1 - alpha) * vy_obs

        # ========== 5. 正前方堵死时强制转向（覆盖加权） ==========
        if center < 3.0:
            if left_avg > right_avg:
                vy_raw = max(vy_raw, 0.8)
            else:
                vy_raw = min(vy_raw, -0.8)

        # ========== 6. 一阶低通滤波（平滑） ==========
        filter_alpha = 0.3
        if not hasattr(self, 'vy_filtered'):
            self.vy_filtered = 0.0
        self.vy_filtered = self.vy_filtered * (1 - filter_alpha) + vy_raw * filter_alpha
        vy = self.vy_filtered

        # ========== 7. 统计 ==========
        if abs(vy) > 0.3:
            self.obstacle_count += 1

        return vx, vy

    def increment_stuck(self):
        self.stuck_counter += 1

    def reset_stuck(self):
        self.stuck_counter = 0

    def should_force_escape(self):
        return self.stuck_counter > self.max_stuck_rotates