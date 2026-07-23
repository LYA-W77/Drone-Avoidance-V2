import airsim
import numpy as np
import math

class SensorFusion:
    def __init__(self, client, sector_angles=[30, 15, 0, -15, -30]):
        self.client = client
        self.sector_angles = sector_angles
        self.latest_pc_x = np.array([])
        self.latest_pc_y = np.array([])

    def get_distances(self, current_yaw):
        """融合激光雷达 + 5个距离传感器，返回5个扇区距离[m]"""
        lidar_dists = [30.0] * 5
        try:
            lidar = self.client.getLidarData(vehicle_name="Drone1", lidar_name="")
            if len(lidar.point_cloud) >= 3:
                pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
                pts = pts[pts[:, 2] < 0.5]  # 滤除地面
                if len(pts) > 0:
                    cos_y = math.cos(current_yaw)
                    sin_y = math.sin(current_yaw)
                    pos = self.client.simGetVehiclePose().position
                    dx = pts[:, 0] - pos.x_val
                    dy = pts[:, 1] - pos.y_val
                    x_b = dx * cos_y + dy * sin_y
                    y_b = -dx * sin_y + dy * cos_y
                    valid = x_b > 0.5
                    x_b, y_b = x_b[valid], y_b[valid]
                    if len(x_b) > 0:
                        dists = np.sqrt(x_b**2 + y_b**2)
                        angles = np.degrees(np.arctan2(y_b, x_b))
                        for i, ang in enumerate(self.sector_angles):
                            mask = (angles > ang - 10) & (angles < ang + 10)
                            if np.any(mask):
                                lidar_dists[i] = np.min(dists[mask])
                        self.latest_pc_x, self.latest_pc_y = x_b, y_b
        except Exception as e:
            pass

        # 距离传感器备份
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

        # 保守融合（取较小值）
        fused = []
        for i in range(5):
            fused.append(min(lidar_dists[i], sensor_dists[i]))
        return fused