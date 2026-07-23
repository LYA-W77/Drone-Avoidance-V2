import airsim
import numpy as np
import math
import time

class FlightController:
    def __init__(self, client):
        self.client = client
        self.last_position = None
        self.flight_distance = 0.0
        self.is_flying = False

    def takeoff(self, height=-5.0):
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(2)
        self.client.moveToZAsync(height, 2).join()
        self.is_flying = True
        self.last_position = self.get_position()
        print(f"✓ 起飞成功，高度: {abs(height)} 米")

    def get_position(self):
        p = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

    def get_yaw(self):
        state = self.client.getMultirotorState()
        return airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]

    def move_by_velocity(self, vx, vy, vz=0, duration=0.25):
        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, duration,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0)
        ).join()
        new_pos = self.get_position()
        if self.last_position is not None:
            self.flight_distance += np.linalg.norm(new_pos - self.last_position)
        self.last_position = new_pos

    def rotate_in_place(self, deg):
        cur = math.degrees(self.get_yaw())
        target = (cur + deg) % 360
        print(f"  ↻ 旋转 {deg}° ({cur:.0f}° → {target:.0f}°)")
        self.client.rotateToYawAsync(target, 1.5).join()
        self.last_position = self.get_position()

    def force_escape(self):
        print("  ⚡ 强制脱离")
        pos = self.get_position()
        yaw = self.get_yaw()
        back = pos[:2] - 2.0 * np.array([math.cos(yaw), math.sin(yaw)])
        self.client.moveToPositionAsync(back[0], back[1], pos[2]-3.0, 2.0,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(True, math.degrees(yaw)%360)).join()
        self.last_position = self.get_position()

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
            print(f"✓ 降落完成  飞行距离:{self.flight_distance:.1f}m")
        except Exception as e:
            print(f"降落忽略错误: {e}")