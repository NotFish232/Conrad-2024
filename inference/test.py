import math
import xarm
import time
from scipy.optimize import least_squares


# 1: claw pincers
# 2: claw rotate
# 3: arm top
# 4: arm middle
# 5: arm bottom
# 6: base

LIMITS = {
    1: ((1150, None), (2300, None), 1500),  # close
    2: ((400, None), (2600, None), 500),  # counterclockwise
    3: ((400, 10), (1900, 140), 400),  # down
    4: ((400, -100), (2600, 100), 1500),  # down
    5: ((1200, 95), (2600, -30), 2300),  # up
    6: ((400, 120), (2600, -50), 1900),  # counterclockwise
}

H = 7.8  # cm
L1 = 13
L2 = 9.5
L3 = 13
M3_BOUNDS = (math.pi / 12, 3 / 4 * math.pi)
M4_BOUNDS = (math.pi / 12, 7 / 12 * math.pi)
M5_BOUNDS = (-math.pi / 2, math.pi / 2)
M6_BOUNDS = (-math.pi / 2, math.pi / 2)
BOUNDS = [M3_BOUNDS, M4_BOUNDS, M5_BOUNDS, M6_BOUNDS]


def angle_to_position(servo: int, angle: float) -> int:
    (lower_pos, lower_angle), (upper_pos, upper_angle), _ = LIMITS[servo]
    slope = (upper_pos - lower_pos) / (upper_angle - lower_angle)
    position = int((angle - lower_angle) * slope + lower_pos)
    return position


def calc_angles(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    def equations(p: tuple[float, float, float, float]) -> tuple[float, float, float]:
        M3, M4, M5, M6 = p
        # fmt: off
        eq_1 = math.cos(M6) * (math.sin(M5) * L1 + math.sin(M4 + M5) * L2 + math.sin(M3 + M4 + M5) * L3 ) - x
        eq_2 = math.sin(M6) * (math.sin(M5) * L1 + math.sin(M4 + M5) * L2 + math.sin(M3 + M4 + M5) * L3 ) - y
        eq_3 = H + math.cos(M5) * L1 + math.cos(M4 + M5) * L2 + math.cos(M3 + M4 + M5) * L3 - z
        eq_4 = abs(M3 + M4 + M5) - math.pi
        # fmt: on
        return (eq_1, eq_2, eq_3, eq_4)

    result = least_squares(
        equations,
        [sum(b) / 2 for b in BOUNDS],
        bounds=[*zip(*BOUNDS)],
    )
    # M3, M4, M5, M6 = result.x
    return tuple(math.degrees(m) for m in result.x)


def move_to_default(arm: xarm.Controller) -> None:
    for servo, (*_, default) in LIMITS.items():
        arm.setPosition(servo, default, wait=False)
    time.sleep(1)  # idk if a sleep is necessary here


def move_to_position(arm: xarm.Controller, pos: tuple[float, float, float]) -> None:
    # TODO: Alan do the intervals kekw 
    m3, m4, m5, m6 = calc_angles(*pos)
    for servo, angle in zip(range(3, 7), (m3, m4, m5, m6)):
        position = angle_to_position(servo, angle)
        arm.setPosition(servo, position, wait=False)


def main() -> None:
    arm = xarm.Controller("USB")
    print("Arm successfully set up")
    input(" enter to continue...")

    while True:
        move_to_default(arm)
        pos = input("Enter a position (seperate x, y, z with spaces): ").split(" ")
        move_to_position(arm, pos)


if __name__ == "__main__":
    main()
