import xarm
import time
from scipy.optimize import least_squares
import math

# 1: claw pincers
# 2: claw rotate
# 3: arm top
# 4: arm middle
# 5: arm bottom
# 6: base

LIMITS = {
    1: (1150, 2300, 1500),  # close
    2: (400, 2600, 1500),  # counterclockwise
    3: (400, 1900, 400),  # down
    4: (400, 2600, 1500),  # down
    5: (1200, 2600, 2300),  # up
    6: (400, 2600, 1900),  # counterclockwise
}

H = 5
L1 = 4
L2 = 4
L3 = 4
M3_BOUNDS = (math.pi / 12, 3 / 4 * math.pi)
M4_BOUNDS = (math.pi / 12, 7 / 12 * math.pi)
M5_BOUNDS = (-math.pi / 2, math.pi / 2)
M6_BOUNDS = (-math.pi / 2, math.pi / 2)
BOUNDS = [M3_BOUNDS, M4_BOUNDS, M5_BOUNDS, M6_BOUNDS]


def calc_angles(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    def equations(p: tuple[float, float, float, float]) -> tuple[float, float, float]:
        M3, M4, M5, M6 = p
        # fmt: off
        eq_1 = math.cos(M6) * (math.sin(M5) * L1 + math.sin(M4 + M5) * L2 + math.sin(M3 + M4 + M5) * L3 ) - x
        eq_2 = math.sin(M6) * (math.sin(M5) * L1 + math.sin(M4 + M5) * L2 + math.sin(M3 + M4 + M5) * L3 ) - y
        eq_3 = H + math.cos(M5) * L1 + math.cos(M4 + M5) * L2 + math.cos(M3 + M4 + M5) * L3 - z
        # fmt: on
        return (eq_1, eq_2, eq_3, 0)
    result = least_squares(
        equations,
        [sum(b) / 2 for b in BOUNDS],
        bounds=[*zip(*BOUNDS)],
    )
    M3, M4, M5, M6 = result.x
    return M3, M4, M5, M6




arm = xarm.Controller("USB")

print(arm.getBatteryVoltage())


input("Moving servos to default positions. Press enter to continue...")

for servo, limits in LIMITS.items():
    upper, lower, default = limits
    arm.setPosition(servo, default, wait=False)


while True:
    s = input("Enter servo number: ")
    arm.setPosition(int(s), int(input("Enter position: ")))
    print()
