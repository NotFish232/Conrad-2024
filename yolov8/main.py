from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(
    data="/export/home/2025jlee/code/conrad_2023/yolov8/taco/data.yaml",
    batch=32,
    device=4,
    epochs=10000,
    patience=200,
    project="/export/home/2025jlee/code/conrad_2023/yolov8/runs",
)  # train the model
path = model.export(format="onnx")
