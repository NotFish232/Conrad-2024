import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw
import numpy as np
import argparse


def process_image(image, interpreter, input_size=(640, 640)):
    scale, zero_point = interpreter.get_input_details()[0]["quantization"]
    image = image.resize(input_size)
    image = np.array(image, dtype=np.float32)
    image /= 255
    image = image / scale + zero_point
    image = image.astype(np.int8)
    image = image[np.newaxis, ...]
    return image


def post_process(output_tensor, interpreter, confidence_threshold=0.25):
    scale, zero_point = interpreter.get_output_details()[0]["quantization"]
    output_tensor = output_tensor.squeeze().transpose(1, 0).astype(np.float32)
    output_tensor = scale * (output_tensor - zero_point)
    boxes, scores = np.split(output_tensor, [4], axis=1)
    mask = scores > confidence_threshold
    mask = np.any(mask, axis=1)
    boxes = boxes[mask]
    scores = scores[mask]
    return boxes, scores


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--labels")
    args = arg_parser.parse_args()

    # Load the Edge TPU model
    interpreter = Interpreter(
        model_path="model.edge.tflite",
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()

    labels = [l.strip() for l in open(args.labels)]

    image = Image.open("input_image.png")
    input_image = process_image(image, interpreter)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
    interpreter.invoke()

    detection_results = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )
    boxes, scores = post_process(detection_results, interpreter)

    i = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        label_idx = score.argmax()
        cx = box[0] * image.size[0]
        cy = box[1] * image.size[1]
        w = box[2] * image.size[0]
        h = box[3] * image.size[1]
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = cx + w // 2
        y2 = cy + h // 2
        i.rectangle([(x1, y1), (x2, y2)], outline="red")
        i.text(
            (x1 + 10, y1 + 10),
            f"{labels[label_idx]}: {score[label_idx].item():.2f}",
            fill="red",
        )
    image.save("out.png")


if __name__ == "__main__":
    main()
