import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw
import numpy as np
import cv2

INPUT_FILE = "input_video.mp4"
OUTPUT_FILE = "output_video.mp4"
IMG_SIZE = 640
FPS = 30


def process_image(image, interpreter, input_size=(IMG_SIZE, IMG_SIZE)):
    scale, zero_point = interpreter.get_input_details()[0]["quantization"]
    image = image.resize(input_size)
    image = np.array(image, dtype=np.float32)
    image /= 255
    image = np.round(image / scale) + zero_point
    image = np.clip(image, -127, 128)
    image = image.astype(np.int8)
    image = image[np.newaxis, ...]
    return image


def post_process(
    output_tensor, interpreter, confidence_threshold=0.2, iou_threshold=0.75
):
    scale, zero_point = interpreter.get_output_details()[0]["quantization"]
    output_tensor = output_tensor.squeeze().transpose(1, 0).astype(np.float32)
    output_tensor = scale * (output_tensor - zero_point)
    boxes, scores = np.split(output_tensor, [4], axis=1)

    mask = np.any(scores > confidence_threshold, axis=1)
    boxes = boxes[mask]
    scores = scores[mask]

    sort_idxs = np.argsort(np.max(scores, axis=1), axis=0)[::-1]
    boxes = boxes[sort_idxs]
    scores = scores[sort_idxs]

    good_idxs = []
    for idx, box in enumerate(boxes):
        found_overlap = False
        for idx in good_idxs:
            if calc_iou(box, boxes[idx]) > iou_threshold:
                found_overlap = True
                break
        if not found_overlap:
            good_idxs.append(idx)

    return boxes[good_idxs], scores[good_idxs]


def calc_iou(box_A, box_B):
    xA = max(box_A[0], box_B[0])
    yA = max(box_A[1], box_B[1])
    xB = min(box_A[2], box_B[2])
    yB = min(box_A[3], box_B[3])

    intersection_area = (xB - xA) * (yB - yA)

    box_A_area = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
    box_B_area = (box_B[2] - box_B[0]) * (box_B[3] - box_B[1])

    iou = intersection_area / (box_A_area + box_B_area - intersection_area)

    return iou


def draw_box(image, box, label, score):
    draw = ImageDraw.Draw(image)
    cx = box[0] * image.size[0]
    cy = box[1] * image.size[1]
    w = box[2] * image.size[0]
    h = box[3] * image.size[1]
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    draw.rectangle([(x1, y1), (x2, y2)], outline="red")
    draw.text(
        (x1 + 10, y1 + 10),
        f"{label}: {score:.2f}",
        fill="red",
    )


def main():
    # Load the Edge TPU model
    interpreter = Interpreter(
        model_path="model.edge.tflite",
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()

    labels = [l.strip() for l in open("labels.txt")]

    input_video = cv2.VideoCapture(INPUT_FILE)
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

        inp = process_image(Image.fromarray(frame), interpreter)

        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], inp)
        interpreter.invoke()

        detection_results = interpreter.get_tensor(
            interpreter.get_output_details()[0]["index"]
        )
        boxes, scores = post_process(detection_results, interpreter)

        image = Image.fromarray(frame)

        for box, score in zip(boxes, scores):
            score_idx = score.argmax()
            label = labels[score_idx]
            score = score[score_idx]
            draw_box(image, box, label, score)

        output_video.write(np.array(image))
    input_video.release()
    output_video.release()


if __name__ == "__main__":
    main()
