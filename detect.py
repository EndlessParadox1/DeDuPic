from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # load an official model

# Predict with the model
results = model("0HcOHnRoSfKxlsPjDeLC.jpg")  # predict on an image
results[0].show()
