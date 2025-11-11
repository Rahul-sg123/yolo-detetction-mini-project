import cv2
import os
import math
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ======= CONFIG =======
YOLO_MODEL_PATH = "yolov8n.pt"

# --- BEV LOGIC ---
# 1. These are the 4 points you clicked from the 'get_coordinates.py' script
SOURCE_POINTS = [
    (11, 162),  # Top-Left
    (352, 161), # Top-Right
    (357, 325), # Bottom-Right
    (15, 327)   # Bottom-Left
]

# 2. We define a new, flat 2D rectangle.
#    The width (341px) is from your points (352-11)
#    The height (166px) is from your points (327-161)
DEST_WIDTH_BEV = 341
DEST_HEIGHT_BEV = 166

# 3. We define the 4 corners of this new flat rectangle
DEST_POINTS_BEV = [
    (0, 0),
    (DEST_WIDTH_BEV - 1, 0),
    (DEST_WIDTH_BEV - 1, DEST_HEIGHT_BEV - 1),
    (0, DEST_HEIGHT_BEV - 1)
]

# 4. We assume the *width* of the area you clicked (341px) is 5 meters in real life.
REAL_WORLD_WIDTH_METERS = 5.0 
PIXELS_PER_METER_BEV = DEST_WIDTH_BEV / REAL_WORLD_WIDTH_METERS # This is our new, accurate constant
# --- END BEV LOGIC ---


# Global in-memory stats store (filename -> latest stats dict)
STATS_STORE = {}

def is_in_queue(x, y, queue_zone):
    """Check if a point is inside the queue ROI polygon"""
    return cv2.pointPolygonTest(np.array(queue_zone, np.int32), (x, y), False) >= 0

def draw_stylish_box(frame, x1, y1, x2, y2, track_id):
    """Draws corner style blinking bounding boxes with label."""
    color = (0, 255, 0) if int(time.time() * 2) % 2 == 0 else (0, 0, 255)
    thickness = 2
    corner_len = 20

    # Draw corners
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    label = f"ID:{track_id} | DETECTED"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 8)
    x0 = max(0, x1)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x0 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def _init_stats_for(filename):
    """Initializes or resets the stats for a given video file."""
    if filename not in STATS_STORE:
        STATS_STORE[filename] = {
            'people_count': 0,
            'avg_speed': 0.0,
            'wait_time': 0.0,
            'processing_time': 0.0,
            'accuracy': 72, # Default accuracy
            'last_update': time.time()
        }

def generate_processed_video(video_path):
    """
    Streams processed frames, updates STATS_STORE, and uses BEV for speed calculation.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model file not found: {YOLO_MODEL_PATH}")

    filename = os.path.basename(video_path)
    _init_stats_for(filename)

    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # This dynamic zone is still used to decide *if* a person is in the area
    margin_x = int(frame_width * 0.1)
    margin_y = int(frame_height * 0.1)
    queue_zone = [
        (margin_x, margin_y),
        (frame_width - margin_x, margin_y),
        (frame_width - margin_x, frame_height - margin_y),
        (margin_x, frame_height - margin_y)
    ]
    
    # --- BEV LOGIC ---
    # 5. Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_POINTS), np.float32(DEST_POINTS_BEV))
    # --- END BEV LOGIC ---

    tracker = DeepSort(max_age=30)
    last_positions = {} # This will now store (bev_x, bev_y)
    speeds = []

    # --- DWELL TIME LOGIC ---
    dwell_tracker = {} 
    completed_wait_times = [] 
    # --------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.04
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Video stream finished for {filename}")
            break

        results = model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if is_in_queue(cx, cy, queue_zone):
                        detections.append(([x1, y1, x2, y2], float(box.conf[0]), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        people_count = 0
        current_tracks_in_zone = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if is_in_queue(cx, cy, queue_zone):
                people_count += 1
                current_tracks_in_zone.add(track_id)
                
                if track_id not in dwell_tracker:
                    dwell_tracker[track_id] = time.time()

                draw_stylish_box(frame, x1, y1, x2, y2, track_id)

                # --- BEV LOGIC ---
                # 6. Transform the person's *foot* position to the BEV
                #    We use (cx, y2) as the foot position
                foot_point = np.array([[[cx, y2]]], dtype=np.float32)
                bev_point = cv2.perspectiveTransform(foot_point, M)
                bev_x, bev_y = bev_point[0][0]
                
                # 7. Calculate speed using the new, accurate BEV coordinates
                if track_id in last_positions:
                    # Get distance in the flat, 2D BEV space
                    dist_px_bev = np.linalg.norm(np.array([bev_x, bev_y]) - last_positions[track_id])
                    # Convert to meters using our new, accurate constant
                    dist_m = dist_px_bev / PIXELS_PER_METER_BEV
                    
                    speed = dist_m / frame_time
                    if 0.1 < speed < 5: # Filter noise
                        speeds.append(speed)
                        
                # 8. Store the BEV coordinate for the next frame
                last_positions[track_id] = (bev_x, bev_y)
                # --- END BEV LOGIC ---

        # --- DWELL TIME LOGIC ---
        for track_id in list(dwell_tracker.keys()):
            if track_id not in current_tracks_in_zone:
                start_time_of_wait = dwell_tracker.pop(track_id)
                total_time_spent = time.time() - start_time_of_wait
                if total_time_spent > 3.0:
                    completed_wait_times.append(total_time_spent)
        
        if completed_wait_times:
            avg_wait_seconds = np.mean(completed_wait_times[-10:])
        else:
            avg_wait_seconds = 0.0
        
        avg_wait_minutes = round(avg_wait_seconds / 60, 1)
        # --------------------------

        avg_speed = float(np.mean(speeds[-50:])) if speeds else 0.0
        
        # Update shared stats
        STATS_STORE[filename]['people_count'] = people_count
        STATS_STORE[filename]['avg_speed'] = round(avg_speed, 2)
        STATS_STORE[filename]['wait_time'] = avg_wait_minutes # Use new Dwell Time
        STATS_STORE[filename]['processing_time'] = round(time.time() - start_time, 2)
        STATS_STORE[filename]['last_update'] = time.time()

        # Draw ROI (we'll draw the *original* clicked zone, not the dynamic one)
        cv2.polylines(frame, [np.array(SOURCE_POINTS, np.int32)], True, (255, 255, 0), 2)
        cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} m/s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Avg. Wait: {avg_wait_minutes:.1f} min", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    STATS_STORE[filename]['people_count'] = 0
    STATS_STORE[filename]['wait_time'] = 0
    print(f"Stats cleared for {filename}")

def get_video_statistics(video_path):
    """
    Process entire video and return final statistics (legacy / batch mode).
    This function is also updated with BEV logic.
    """
    print("Running legacy batch processing for statistics...")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    filename = os.path.basename(video_path)
    _init_stats_for(filename)

    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # --- BEV LOGIC ---
    # We use the same matrix 'M' here
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_POINTS), np.float32(DEST_POINTS_BEV))
    # --- END BEV LOGIC ---
    
    # We'll use the SOURCE_POINTS as the queue_zone for this batch function
    queue_zone = SOURCE_POINTS
    
    tracker = DeepSort(max_age=30)
    max_people = 0
    speeds = []
    frame_count = 0
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.04
    last_positions = {}
    
    dwell_tracker = {} 
    completed_wait_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- DWELL/BEV LOGIC FOR SKIPPED FRAMES ---
        # We need to update trackers even on skipped frames
        tracks = tracker.update_tracks([], frame=frame) # Update tracker without new detections
        current_tracks_in_zone = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_in_queue(cx, cy, queue_zone):
                current_tracks_in_zone.add(track_id)
        
        for track_id in list(dwell_tracker.keys()):
            if track_id not in current_tracks_in_zone:
                start_time_of_wait = dwell_tracker.pop(track_id)
                total_time_spent = time.time() - start_time_of_wait
                if total_time_spent > 3.0:
                    completed_wait_times.append(total_time_spent)
        
        if frame_count % 10 != 0: # Process 1 in 10 frames
            continue
        # ---------------------------------------------

        # --- This code runs only for 1 in 10 frames ---
        results = model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if is_in_queue(cx, cy, queue_zone):
                        detections.append(([x1, y1, x2, y2], float(box.conf[0]), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_people = 0
        current_tracks_in_zone = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_in_queue(cx, cy, queue_zone):
                current_people += 1
                current_tracks_in_zone.add(track_id)
                
                if track_id not in dwell_tracker:
                    dwell_tracker[track_id] = time.time()
                
                # --- BEV LOGIC ---
                foot_point = np.array([[[cx, y2]]], dtype=np.float32)
                bev_point = cv2.perspectiveTransform(foot_point, M)
                bev_x, bev_y = bev_point[0][0]
                    
                if track_id in last_positions:
                    dist_px_bev = np.linalg.norm(np.array([bev_x, bev_y]) - last_positions[track_id])
                    dist_m = dist_px_bev / PIXELS_PER_METER_BEV
                    speed = dist_m / (frame_time * 10) # Adjust for frame skipping
                    if 0.1 < speed < 5:
                        speeds.append(speed)
                last_positions[track_id] = (bev_x, bev_y)
                # --- END BEV LOGIC ---
        
        max_people = max(max_people, current_people)
        
        for track_id in list(dwell_tracker.keys()):
            if track_id not in current_tracks_in_zone:
                start_time_of_wait = dwell_tracker.pop(track_id)
                total_time_spent = time.time() - start_time_of_wait
                if total_time_spent > 3.0:
                    completed_wait_times.append(total_time_spent)

    cap.release()
    processing_time = time.time() - start_time
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    
    if completed_wait_times:
        avg_wait_seconds = np.mean(completed_wait_times)
    else:
        avg_wait_seconds = 0.0
    
    avg_wait_minutes = round(avg_wait_seconds / 60, 1)

    final_stats = {
        'people_count': max_people,
        'avg_speed': round(avg_speed, 2),
        'wait_time': avg_wait_minutes,
        'processing_time': round(processing_time, 2),
        'accuracy': 72
    }
    
    STATS_STORE[filename].update(final_stats)
    return final_stats

def calculate_wait_time(people_count, processing_rate=2.0):
    """
    Return wait time in minutes as a float with one decimal place.
    Example: 2 people with rate 2.0 -> 1.0, 1 person -> 0.5 (rounded to 0.5 -> 0.5)
    """
    if people_count <= 0:
        return 0.0
    wait = people_count / float(processing_rate)
    # Round to one decimal for chart granularity (0.1 steps)
    return round(wait, 1)

def update_stats(frame_detections, video_filename):
    """Update statistics based on current frame detections"""
    if video_filename not in STATS_STORE:
        STATS_STORE[video_filename] = {
            'people_count': 0,
            'wait_time': 0.0,
            'processing_rate': 2.0,  # default rate (people per minute)
            'accuracy': 95.0,
            'avg_speed': 0.2,
            'last_update': time.time()
        }

    # Count people with confidence > 0.5 (adjust index if your detection format differs)
    current_count = len([d for d in frame_detections if d[5] > 0.5])

    # Optionally smooth people count (simple exponential smoothing)
    alpha = 0.35
    prev = STATS_STORE[video_filename].get('people_count', 0)
    smoothed = int(round(alpha * current_count + (1 - alpha) * prev))

    processing_rate = STATS_STORE[video_filename].get('processing_rate', 2.0)
    wait_time = calculate_wait_time(smoothed, processing_rate)

    STATS_STORE[video_filename].update({
        'people_count': smoothed,
        'wait_time': wait_time,   # float (one decimal)
        'processing_rate': processing_rate,
        'last_update': time.time()
    })

    return STATS_STORE[video_filename]
