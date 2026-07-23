import airsim
import keyboard
import time

print("连接无人机...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("按 K 起飞，按 L 降落")

while True:
    if keyboard.is_pressed('k'):
        print("起飞！")
        client.takeoffAsync().join()
        time.sleep(1)
    elif keyboard.is_pressed('l'):
        print("降落...")
        client.landAsync().join()
        time.sleep(1)
    elif keyboard.is_pressed('w'):
        client.moveByVelocityBodyFrameAsync(2, 0, 0, 0.5).join()
    elif keyboard.is_pressed('s'):
        client.moveByVelocityBodyFrameAsync(-2, 0, 0, 0.5).join()
    elif keyboard.is_pressed('a'):
        client.rotateByYawRateAsync(-15, 0.5).join()
    elif keyboard.is_pressed('d'):
        client.rotateByYawRateAsync(15, 0.5).join()
    elif keyboard.is_pressed('up'):
        client.moveByVelocityBodyFrameAsync(0, 0, -1, 0.5).join()
    elif keyboard.is_pressed('down'):
        client.moveByVelocityBodyFrameAsync(0, 0, 1, 0.5).join()
    time.sleep(0.05)