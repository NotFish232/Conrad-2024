import xarm
import time

arm = xarm.Controller("USB")

print(arm.getBatteryVoltage())

arm.setPosition(0, 20, wait=False)

while True:
    time.sleep(0.1)
