import xarm
import time

arm = xarm.Controller("USB")

print(arm.getBatteryVoltage())

# 1: claw pincers
# 2: claw rotate
# 3: arm top
# 4: arm middle
# 5: arm bottom
# 6: base

LIMITS = {
    1: (1150, 2300, 1500),  # close
    2: (400, 2600, 520),   # counterclockwise
    3: (400, 1900, 400),    # down
    4: (400, 2600, 1500),   # down
    5: (1200, 2600, 2300),  # up
    6: (400, 2600, 1900),   # counterclockwise
}


input("Moving servos to default positions. Press enter to continue...")

for servo, limits in LIMITS.items():
    upper, lower, default = limits
    arm.setPosition(servo, default, wait=False)


while True:
    s = input("Enter servo number: ")
    arm.setPosition(int(s), int(input("Enter position: ")))
    print()
