from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="trash_dataset.yaml",
            batch=32,
            device=4,
            epochs=1000,
             project="/export/home/2025jlee/conrad_2023/yolov7/runs/train",
             )  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx") 