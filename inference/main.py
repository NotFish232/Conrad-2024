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
    2: ((400, 0), (2600, 90), 1500),  # counterclockwise
    3: ((410, 10), (1900, 140), 410),  # down
    4: ((400, -100), (2600, 100), 1500),  # down
    5: ((1200, 95), (2600, -30), 2300),  # up
    6: ((400, 120), (2600, -50), 1900),  # counterclockwise
}


# Position, angle
# fmt: off
POINTS = {
    1: ( (1150, 0),    (2300, 45),               ),
    2: ( (400,  0),    (2600, 45),               ),
    3: ( (1275, 90),   (750, 45),   (1750, 135), ),
    4: ( (1500, 0),    (1975, 45),  (2475, 90),  ),
    5: ( (2325, 0),    (1850, 45),  (1325, 90),  ),
    6: ( (1900, 0),    (875, 90),                ),
}
# fmt: on

### Measurements of arm joints
H = 14.5  # cm
L1 = 10.5
L_3_1 = 1.7
L2 = 9
L3 = 17
###

# width, height of Area that camera sees
# As well as the location of the arm origin relative to the bottom left of the camera vision
AREA_W = 52
AREA_H = 41
ARM_X = 0
ARM_Y = 41 - 10.5

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


### Adjustments for Servo position calculations
# X_OFFSET = -2
# X_COEFFICIENT = 1
# Y_OFFSET = 0
# Y_COEFFICIENT = 0.5
# Z_OFFSET = 0
# Z_COEFFICIENT = 1

# ADJUSTMENTS = (
#     (X_OFFSET, X_COEFFICIENT),
#     (Y_OFFSET, Y_COEFFICIENT),
#     (Z_OFFSET, Z_COEFFICIENT),
# )
###

### Bin Constants
BINS = [
    (-30, 25, 25),
    (-15, 15, 25),
    # (-20, 3, 25),
]
###

### Claw close measuement for each label
CLASS_NAME_TO_SERVO_POS = {"Paper": 2200, "Other plastic": 2150, "Bottle cap": 2000, "Can": 2100}
###

# Movement speeds
MIN_DURATION = 500
MAX_DURATION = 1000
SPEED = 60  # degrees per second
###


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


def move_to_default(arm: xarm.Controller, open: bool = True) -> None:
    for servo, (*_, default) in LIMITS.items():
        CURRENT_POSITIONS[servo] = default

        if servo == 1 and not open:
            continue

        arm.setPosition(servo, default, wait=False)

    time.sleep(0.5)


def open_claw(arm: xarm.Controller) -> None:
    move(arm, 1, 1150, wait=True)


def close_claw(arm: xarm.Controller, pos: int | None = None) -> None:
    move(arm, 1, pos or 1840, wait=True)


def vertical_claw(arm: xarm.Controller) -> None:
    move(arm, 2, 1500, wait=True)


def horizontal_claw(arm: xarm.Controller) -> None:
    move(arm, 2, 500, wait=True)


def move_to_position(
    arm: xarm.Controller,
    pos: tuple[float, float, float],
    open: bool = False,
    close: bool = False,
    duration: int | None = None,
) -> None:
    # pos = tuple(c * p + o for (o, c), p in zip(ADJUSTMENTS, pos))
    m3, m4, m5, m6 = calc_angles(*pos)
    """ m3r, m4r, m5r, m6r = tuple(math.radians(x) for x in (m3, m4, m5, m6))
    x = EQ_X(m3r, m4r, m5r, m6r)
    y = EQ_Y(m3r, m4r, m5r, m6r)
    z = EQ_Z(m3r, m4r, m5r)
    theta = math.degrees(EQ_THETA(m3r, m4r, m5r))
    print(f"({x:.2f}, {y:.2f}, {z:.2f}) {theta:.2f}")"""
    servos = list(zip(range(3, 7), (m3, m4, m5, m6)))

    durations = []

    for servo, angle in servos:
        durations.append(move(arm, servo, angle_to_position(servo, angle)))

    max_duration = (duration or max(durations)) / 1000

    time.sleep(max_duration)

    if close:
        close_claw(arm)
    if open:
        horizontal_claw(arm)
        open_claw(arm)


def move(
    arm: xarm.Controller,
    servo: int,
    target_pos: tuple[float, float, float],
    wait: bool = False,
) -> float:
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
    arm.setPosition(servo, target_pos, duration=duration, wait=wait)

    return duration


arm = xarm.Controller("USB")


def pickup_detected(
    arm: xarm.Controller, pos: tuple[float, float, float], pred_class: str, bin_num: int
) -> None:
    move_to_position(arm, (pos[0], pos[1], pos[2] + 10))
    move_to_position(arm, pos)
    close_claw(arm, CLASS_NAME_TO_SERVO_POS.get(pred_class, None))

    move_to_default(arm, open=False)
    move_to_position(arm, BINS[bin_num], open=True)
    move_to_default(arm)


def main_ml() -> None:
    model = YOLO("yolov8N.pt")
    # equivalant of allocating tensors
    model.predict(cv2.VideoCapture(0).read()[1], verbose=False)

    move_to_default(arm)

    # capture = cv2.VideoCapture(0)
    #capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    while True:
        ret, frame = cv2.VideoCapture(0).read()
        if not ret:
            break

        result = model.predict(frame, verbose=False, conf=0.15)[0]
        annotator = Annotator(frame)
        boxes = [box for box in result.boxes if 0.25 < box.xywhn[0, 0] < 0.6]
     
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                annotator.box_label(
                    box.xyxy[0],
                    f"{model.names[box.cls.item()]}: {box.conf.item():.2f}",
                    color=color,
                )

        cv2.imshow("Detection Results", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
        if len(boxes) == 0:
            continue

        box = boxes[0]
        pos = (*bounding_box_to_position(box.xywhn[0]), 0)
        pickup_detected(arm, pos, model.names[box.cls.item()], int(box.cls.item() < 6))

        # for i in range(4):
        #     capture.grab()

CLOSE = 0
OPEN = 1
DEFAULT = 2
BIN = (-15, 15, 25)
MOVE_SEQUENCE = [
    (27, 0, 20),
    (27, 0, 7),
    CLOSE, DEFAULT, BIN, OPEN, DEFAULT, 
    (10, 20, 20),
    (20, 0, 6),
    DEFAULT,
    (20, -20, 20),
    (20, -21, 8),
    CLOSE, DEFAULT, BIN, OPEN, DEFAULT,
    (20, -30, 20),
    (0, 0, 50),
    (20, 30, 50),
    DEFAULT, 
    (-10, 20, 20),
    DEFAULT,
    (10, 30, 10),
    DEFAULT
]


def main_hardcode():
    input()
    move_to_default(arm)
    time.sleep(0.5)

    for pos in MOVE_SEQUENCE:
        if pos == CLOSE:
            close_claw(arm)
        elif pos == OPEN:
            open_claw(arm)
        elif pos == DEFAULT:
            move_to_default(arm, open=False)
            time.sleep(1)
        else:
            move_to_position(arm, pos)

    time.sleep(5)
    move_to_default(arm)

    while True:
        a, b, c = map(float, input("Enter position: ").split())
        move_to_position(arm, (a, b, c))

if __name__ == "__main__":
    main_hardcode()
