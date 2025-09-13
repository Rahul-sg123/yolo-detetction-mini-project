import cv2
from ultralytics import YOLO

def count_people_in_video(video_path):
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    total_people = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame to save time
        if frame_count % 30 == 0:
            results = model(frame)
            
            # Count people in this frame
            people_in_frame = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # person class
                            people_in_frame += 1
            
            print(f"Frame {frame_count}: {people_in_frame} people detected")
            total_people = max(total_people, people_in_frame)
        
        frame_count += 1
    
    cap.release()
    return total_people

# Test with your video
if __name__ == "__main__":
    video_path = "test_video.mp4"  # Put your video file here
    count = count_people_in_video(video_path)
    print(f"Maximum people detected: {count}")