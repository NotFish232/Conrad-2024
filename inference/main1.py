from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
from PIL import Image, ImageDraw
import numpy as np
import cv2

INPUT_FILE =  "input_video.mp4"
OUTPUT_FILE = "output_video.mp4"
IMG_SIZE = 640
FPS = 30




def main():
    model = YOLO("yolov8N.pt")

    input_video = cv2.VideoCapture(0)
    output_video = cv2.VideoWriter(
        OUTPUT_FILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (
            int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break

        result = model.predict(frame, verbose=False)[0]
        annotator = Annotator(frame)
        for box in result.boxes:
            b = box.xyxy[0] 
            cls = box.cls.item()
            lbl = model.names[cls]
            prob = box.conf.item()
            annotator.box_label(b, f"{lbl}: {prob:.2f}")


        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        output_video.write(np.array(frame))
    input_video.release()
    output_video.release()


if __name__ == "__main__":
    main()
