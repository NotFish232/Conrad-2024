from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2
import time

INPUT_FILE = 4
OUTPUT_FILE = "output_video.mp4"
IMG_SIZE = 640


def main():
    model = YOLO("yolov8N.pt")

    input_video = cv2.VideoCapture(INPUT_FILE)
    output_video = cv2.VideoWriter(
        OUTPUT_FILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_video.get(cv2.CAP_PROP_FPS),
        (
            int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    last_frame_time = time.time()

    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        result = model.predict(frame, verbose=False, half=False)[0]
        annotator = Annotator(frame)
        for box in result.boxes:
            b = box.xyxy[0]
            cls = box.cls.item()
            lbl = model.names[cls]
            prob = box.conf.item()
            annotator.box_label(b, f"{lbl}: {prob:.2f}", color=(0, 0, 255))

        frame_time = time.time()
        fps = 1 / (frame_time - last_frame_time)
        last_frame_time = frame_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            2,
        )
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        output_video.write(np.array(frame))
        
    input_video.release()
    output_video.release()


if __name__ == "__main__":
    main()
