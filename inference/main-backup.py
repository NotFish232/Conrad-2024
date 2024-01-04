import math
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
    1: ((1150, 90), (2300, 180), 2300),  # close
    2: ((400, -180), (2600, 200), 500),  # counterclockwise
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
    4: ( (1500, 0),    (1975, 45),  (2425, 90),  ),
    5: ( (2325, 0),    (1850, 45),  (1325, 90),  ),
    6: ( (1900, 0),    (875, 45),                ),
}
# fmt: on


H = 9.5  # cm
L1 = 10.5
L2 = 9
L3 = 17

# width, height of Area that camera sees
# As well as the location of the arm origin relative to the bottom left of the camera vision
AREA_W = 50
AREA_H = 50
ARM_X = 10
ARM_Y = 10

M3_BOUNDS = (math.pi / 12, 3 / 4 * math.pi)
M4_BOUNDS = (math.pi / 12, 7 / 12 * math.pi)
M5_BOUNDS = (-math.pi / 2, math.pi / 2)
M6_BOUNDS = (-math.pi / 2, math.pi / 2)
BOUNDS = [M3_BOUNDS, M4_BOUNDS, M5_BOUNDS, M6_BOUNDS]

CURRENT_POSITIONS = {i: None for i in range(1, 7)}


def calc_regressions() -> (tuple[dict[int, LinearRegression], dict[int, LinearRegression]]):
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
    return cx * AREA_W - ARM_X, (1 - cy) * AREA_H - ARM_Y


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
        CURRENT_POSITIONS[servo] = default
        arm.setPosition(servo, default, wait=False)


def move_to_position(arm: xarm.Controller, pos: tuple[float, float, float]) -> None:
    m3, m4, m5, m6 = calc_angles(*pos)
    servos = list(zip(range(3, 7), (m3, m4, m5, m6)))

    for servo, angle in servos:
        move(arm, servo, angle_to_position(servo, angle))


MIN_DURATION = 800
MAX_DURATION = 5000
SPEED = 30  # degrees per second


def move(arm, servo, target_pos):
    current_pos = CURRENT_POSITIONS[servo]
    current_angle = position_to_angle(servo, current_pos)
    target_angle = position_to_angle(servo, target_pos)
    delta_angle = abs(target_angle - current_angle)
    duration = min(
        max(int(delta_angle * (1 / SPEED) * 1000), MIN_DURATION), MAX_DURATION
    )
    print(servo, delta_angle, duration / 1000)

    CURRENT_POSITIONS[servo] = target_pos
    arm.setPosition(servo, target_pos, duration=duration, wait=False)



arm = xarm.Controller("USB")
print("Arm successfully set up")
input("Press enter to move to default position...")
move_to_default(arm)

while True:
    inp = input("Enter a position to move to: ")
    if inp == "default":
        move_to_default(arm)
        continue
    pos = tuple(float(i) for i in inp.split())
    move_to_position(arm, pos)


model = YOLO("yolov8N.pt")
input_video = cv2.VideoCapture(0)
while True:
    ret, frame = input_video.read()
    if not ret:
        break

    result = model.predict(frame, verbose=False)[0]
    annotator = Annotator(frame)
    box = result.boxes[0]
    annotator.box_label(
        box.xyxy[0],
        f"{model.names[ box.cls.item()]}: {box.conf.item():.2f}",
        color=(0, 0, 255),
    )
    cv2.imshow("Detection Results", frame)

    input("Press enter to move to selected piece...")
    move_to_default(arm)
    pos = bounding_box_to_position(box.xywhn[0])
    move_to_position(arm, pos)

