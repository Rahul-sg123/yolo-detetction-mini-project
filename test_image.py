import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Test on sample image
image_path = 'sample_image.jpeg'  # Put your image file here
results = model(image_path)

# Count and display people
people_count = 0
print("Detection Results:")

for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == 0:  # 0 = person class
                people_count += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"Person {people_count}: confidence={confidence:.2f}, location=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

print(f"\nTotal people detected: {people_count}")

# Optional: Save image with bounding boxes
annotated_frame = results[0].plot()
cv2.imwrite('detected_people.jpg', annotated_frame)
print("Saved result image as 'detected_people.jpg'")