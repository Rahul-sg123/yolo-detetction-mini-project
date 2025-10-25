from ultralytics import YOLO

# This will automatically download YOLOv8 nano model
model = YOLO('yolov8n.pt')
print("YOLO model loaded successfully!")