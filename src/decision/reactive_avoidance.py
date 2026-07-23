import numpy as np
import math

class ReactiveAvoidance:
    def __init__(self, safe_distance=5.0, cruise_speed=3.0, goal_weight=0.7):
        self.safe_distance = safe_distance
        self.cruise_speed = cruise_speed
        self.goal_weight = goal_weight
        self.sector_angles = [30, 15, 0, -15, -30]
        self.obstacle_count = 0
        self.stuck_counter = 0
        self.max_stuck_rotates = 3
        self.stuck_threshold = 1.5
        self.stuck_rotate_angle = 45

    def analyze(self, distances, target_dir_world, current_yaw):
        safety_scores, goal_scores, combined = [], [], []
        for i, d in enumerate(distances):
            # 安全分
            safety_scores.append(d if d >= self.safe_distance else d * (d / self.safe_distance))
            # 目标分
            if target_dir_world is not None:
                local = np.array([math.cos(math.radians(self.sector_angles[i])),
                                  math.sin(math.radians(self.sector_angles[i]))])
                rot = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                                [math.sin(current_yaw), math.cos(current_yaw)]])
                world = rot @ local
                goal_scores.append((np.dot(world, target_dir_world) + 1) / 2)
            else:
                goal_scores.append(1.0)
        # 综合分
        for i in range(5):
            combined.append(safety_scores[i] * ((1 - self.goal_weight) + self.goal_weight * goal_scores[i]))
        safest_index = np.argmax(combined)
        forward_dist = distances[2]
        return safest_index, forward_dist, combined

    def is_stuck(self, scores):
        return max(scores) < self.stuck_threshold

    def decide_action(self, safest_index, forward_dist):
        side_map = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}
        if forward_dist < 2.0:
            vx, vy = -1.5, side_map[safest_index] * 1.5
            self.obstacle_count += 1
        elif forward_dist < self.safe_distance * 0.5:
            vx, vy = self.cruise_speed * 0.3, side_map[safest_index] * self.cruise_speed * 0.8
            self.obstacle_count += 1
        elif forward_dist < self.safe_distance:
            vx, vy = self.cruise_speed * 0.6, side_map[safest_index] * self.cruise_speed * 0.4
        else:
            vx, vy = self.cruise_speed, 0.0
        return vx, vy

    def increment_stuck(self): self.stuck_counter += 1
    def reset_stuck(self): self.stuck_counter = 0
    def should_force_escape(self): return self.stuck_counter > self.max_stuck_rotates