import math
import time
import xarm
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import torch as T
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression


# 1: claw pincers
# 2: claw rotate
# 3: arm top
# 4: arm middle
# 5: arm bottom
# 6: base

LIMITS = {
    1: ((1150, 90), (2300, 180), 1150),  # close
    2: ((400, -180), (2600, 200), 1500),  # counterclockwise
    3: ((410, 10), (1900, 140), 410),  # down
    4: ((400, -100), (2600, 100), 1500),  # down
    5: ((1200, 95), (2600, -30), 2300),  # up
    6: ((400, 120), (2600, -50), 1900),  # counterclockwise
}


# Position, angle
# fmt: off
POINTS = {
    1: ( (1150, 90),   (2300, 180),              ),
    2: ( (400, -180),  (2600, 200),              ),
    3: ( (1275, 90),   (750, 45),   (1750, 135), ),
    4: ( (1500, 0),    (1975, 45),  (2475, 90),  ),
    5: ( (2325, 0),    (1850, 45),  (1325, 90),  ),
    6: ( (1900, 0),    (875, 90),                ),
}
# fmt: on


H = 14.5  # cm
L1 = 10.5
L_3_1 = 1.7
L2 = 9
L3 = 17

# width, height of Area that camera sees
# As well as the location of the arm origin relative to the bottom left of the camera vision
AREA_W = 52
AREA_H = 41
ARM_X = -3
ARM_Y = 41 - 31

M3_BOUNDS = (0, 3 / 4 * math.pi)
M4_BOUNDS = (0, 7 / 12 * math.pi)
M5_BOUNDS = (-math.pi / 2, math.pi / 2)
M6_BOUNDS = (-5 / 9 * math.pi, 2 / 3 * math.pi)
BOUNDS = [M3_BOUNDS, M4_BOUNDS, M5_BOUNDS, M6_BOUNDS]

CURRENT_POSITIONS = {i: None for i in range(1, 7)}

# fmt: off
EQ_X = lambda m3, m4, m5, m6: math.cos(m6) * (math.sin(m5) * L1 + math.sin(m4 + m5) * L2 + math.sin(m3 + m4 + m5) * L3 - math.cos(m3 + m4 + m5) * L_3_1)
EQ_Y = lambda m3, m4, m5, m6: math.sin(m6) * (math.sin(m5) * L1 + math.sin(m4 + m5) * L2 + math.sin(m3 + m4 + m5) * L3 - math.cos(m3 + m4 + m5) * L_3_1)
EQ_Z = lambda m3, m4, m5: H + math.cos(m5) * L1 + math.cos(m4 + m5) * L2 + math.cos(m3 + m4 + m5) * L3 + math.sin(m3 + m4 + m5) * L_3_1
EQ_THETA = lambda m3, m4, m5: m3 + m4 + m5
# fmt: on


def calc_regressions() -> (
    tuple[dict[int, LinearRegression], dict[int, LinearRegression]]
):
    pta_regressions, atp_regressions = {}, {}
    for servo, pts in POINTS.items():
        servo_positions = []
        angles = []
        for servo_position, angle in pts:
            servo_positions.append([servo_position])
            angles.append([angle])
        pta_regression = LinearRegression().fit(servo_positions, angles)
        atp_regression = LinearRegression().fit(angles, servo_positions)
        pta_regressions[servo] = pta_regression
        atp_regressions[servo] = atp_regression
    return pta_regressions, atp_regressions


# use like REGRESSIONS[servo].predict([[angle]])[0, 0]
PTA_REGRESSIONS, ATP_REGRESSIONS = calc_regressions()


def angle_to_position(servo: int, angle: float) -> int:
    return int(ATP_REGRESSIONS[servo].predict([[angle]])[0, 0])


def position_to_angle(servo: int, position: int) -> float:
    return PTA_REGRESSIONS[servo].predict([[position]])[0, 0]


def open_claw(arm):
    move(arm, 1, 1150)


def close_claw(arm):
    move(arm, 1, 2200)


def vertical_claw(arm):
    move(arm, 2, 1500)


def horizontal_claw(arm):
    move(arm, 2, 500)


def bounding_box_to_position(bbox: T.Tensor) -> tuple[float, float]:
    cx, cy, *_ = bbox.tolist()
    return cx * AREA_W - ARM_X, cy * AREA_H - ARM_Y


def calc_angles(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    def equations(p: tuple[float, float, float, float]) -> tuple[float, float, float]:
        m3, m4, m5, m6 = p

        eq_1 = EQ_X(m3, m4, m5, m6) - x
        eq_2 = EQ_Y(m3, m4, m5, m6) - y
        eq_3 = EQ_Z(m3, m4, m5) - z
        eq_4 = abs(EQ_THETA(m3, m4, m5)) - math.pi

        return (10 * eq_1, 10 * eq_2, 10 * eq_3, eq_4)

    result = least_squares(
        equations,
        [sum(b) / 2 for b in BOUNDS],
        bounds=[*zip(*BOUNDS)],
    )

    return tuple(math.degrees(m) for m in result.x)


def move_to_default(arm: xarm.Controller, open_claw: bool = True) -> None:
    for servo, (*_, default) in LIMITS.items():
        CURRENT_POSITIONS[servo] = default

        if servo == 1 and not open_claw:
            continue

        arm.setPosition(servo, default, wait=False)
        time.sleep(0.5)


X_OFFSET = -2
X_COEFFICIENT = 1
Y_OFFSET = 0
Y_COEFFICIENT = 0.5
Z_OFFSET = 0
Z_COEFFICIENT = 1

ADJUSTMENTS = (
    (X_OFFSET, X_COEFFICIENT),
    (Y_OFFSET, Y_COEFFICIENT),
    (Z_OFFSET, Z_COEFFICIENT),
)

CLAW_ADJUST_TIME = 0.95


def move_to_position(
    arm: xarm.Controller, pos: tuple[float, float, float], open=False, close=False
) -> None:
    # pos = tuple(c * p + o for (o, c), p in zip(ADJUSTMENTS, pos))
    m3, m4, m5, m6 = calc_angles(*pos)
    m3r, m4r, m5r, m6r = tuple(math.radians(x) for x in (m3, m4, m5, m6))
    x = EQ_X(m3r, m4r, m5r, m6r)
    y = EQ_Y(m3r, m4r, m5r, m6r)
    z = EQ_Z(m3r, m4r, m5r)
    theta = math.degrees(EQ_THETA(m3r, m4r, m5r))
    print(f"({x:.2f}, {y:.2f}, {z:.2f}) {theta:.2f}")
    servos = list(zip(range(3, 7), (m3, m4, m5, m6)))

    durations = []

    for servo, angle in servos:
        durations.append(move(arm, servo, angle_to_position(servo, angle)))

    max_duration = max(durations) / 1000

    if close:
        time.sleep(CLAW_ADJUST_TIME * max_duration)
        close_claw(arm)
        time.sleep(1)
    elif open:
        time.sleep(max_duration + 0.1)
        open_claw(arm)
    else:
        time.sleep(max_duration + 0.1)


MIN_DURATION = 500
MAX_DURATION = 1000
SPEED = 60  # degrees per second


def move(arm, servo, target_pos):
    current_pos = CURRENT_POSITIONS[servo]
    current_angle = position_to_angle(servo, current_pos)
    target_angle = position_to_angle(servo, target_pos)
    delta_angle = abs(target_angle - current_angle)
    duration = min(
        max(int(delta_angle * (1 / SPEED) * 1000), MIN_DURATION), MAX_DURATION
    )
    # target_pos = max(target_pos, 0)
    print(
        f"{servo=}, {current_pos=}, {target_pos=}, {current_angle=}, {target_angle=}, {duration=}"
    )

    CURRENT_POSITIONS[servo] = target_pos
    arm.setPosition(servo, target_pos, duration=duration, wait=False)

    return duration


BINS = [
    (-30, 25, 25),
    (-15, 15, 25),
    (-20, 3, 25),
]
arm = xarm.Controller("USB")

# move_to_default(arm)
# for bin in BINS:
#     input()
#     move_to_position(arm, bin)

# while True:
#     inp = input("Enter a position to move to: ")
#     if inp == "default":
#         move_to_default(arm)
#         continue
#     pos = tuple(float(i) for i in inp.split())


def pickup(arm, pos, bin_num):
    move_to_position(arm, (pos[0], pos[1], pos[2] + 10))
    move_to_position(arm, pos, close=True)
    move_to_position(arm, (pos[0], pos[1], pos[2] + 10))
    move_to_default(arm, open_claw=False)
    move_to_position(arm, BINS[bin_num], open=True)
    move_to_default(arm)
    

def main() -> None:
    print("Video capture set up")
    print("Arm successfully set up")
    model = YOLO("yolov8N.pt")
    print("Model loaded")

    # input("Press enter to move to default position...")
    move_to_default(arm)

    while True:

        # input("Press enter to capture a frame...")
        ret, frame = cv2.VideoCapture(1).read()
        if not ret:
            break

        result = model.predict(frame, verbose=False)[0]
        annotator = Annotator(frame)
        boxes = [box for box in result.boxes if box.xywhn[0, 0] > 0.25]

        if len(boxes) == 0:
            cv2.imshow("Detection Results", frame)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
            continue
        box = boxes[0]
        annotator.box_label(
            box.xyxy[0],
            f"{model.names[ box.cls.item()]}: {box.conf.item():.2f}",
            color=(0, 0, 255),
        )
        # cv2.imshow("Detection Results", frame)

        # if cv2.waitKey(50) & 0xFF == ord("q"):
        #    break

        input("Press enter to move to selected piece...")
        pos = (*bounding_box_to_position(box.xywhn[0]), 0)
        pickup(arm, pos, 0)


if __name__ == "__main__":
    main()
