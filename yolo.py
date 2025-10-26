from ultralytics import YOLO

# Load the YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Print all class names
print(model.names)
