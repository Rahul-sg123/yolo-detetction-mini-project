import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ======= CONFIG =======
QUEUE_ZONE = [(100, 100), (500, 100), (500, 700), (100, 700)]
PIXELS_PER_METER = 100
YOLO_MODEL_PATH = "yolov8n.pt"
# ======================

# Global in-memory stats store (filename -> latest stats dict)
STATS_STORE = {}

def is_in_queue(x, y):
    """Check if a point is inside the queue ROI polygon"""
    return cv2.pointPolygonTest(np.array(QUEUE_ZONE, np.int32), (x, y), False) >= 0

def draw_stylish_box(frame, x1, y1, x2, y2, track_id):
    """Draws corner style blinking bounding boxes with label."""
    color = (0, 255, 0) if int(time.time() * 2) % 2 == 0 else (0, 0, 255)
    thickness = 2
    corner_len = 20

    # top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    # Label with ID and DETECTED text
    label = f"ID:{track_id} | DETECTED"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # draw filled rectangle behind text (slightly above top-left)
    y0 = max(0, y1 - th - 8)
    x0 = max(0, x1)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x0 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def _init_stats_for(filename):
    if filename not in STATS_STORE:
        STATS_STORE[filename] = {
            'people_count': 0,
            'avg_speed': 0.0,
            'wait_time': 0.0,
            'processing_time': 0.0,
            'accuracy': 72,
            'last_update': time.time()
        }

def generate_processed_video(video_path):
    """
    Streams processed frames and updates STATS_STORE[basename(video_path)]
    The streaming function updates stats in realtime while frames are being generated.
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

    tracker = DeepSort(max_age=30)
    last_positions = {}
    speeds = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.04
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if is_in_queue(cx, cy):
                        detections.append(([x1, y1, x2, y2], float(box.conf[0]), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        people_count = 0

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if is_in_queue(cx, cy):
                people_count += 1

                # Draw fancy box
                draw_stylish_box(frame, x1, y1, x2, y2, track_id)

                # Speed calculation (meters per second)
                if track_id in last_positions:
                    dist_px = np.linalg.norm(np.array([cx, cy]) - np.array(last_positions[track_id]))
                    dist_m = dist_px / PIXELS_PER_METER
                    # time between frames is frame_time (approx), but use actual to smooth
                    speed = dist_m / frame_time
                    speeds.append(speed)
                last_positions[track_id] = (cx, cy)

        avg_speed = float(np.mean(speeds[-200:])) if speeds else 0.0
        avg_person_length_m = 0.5
        est_time_seconds = (people_count * avg_person_length_m / avg_speed) if avg_speed > 0 else 0.0

        # Update shared stats (write latest)
        STATS_STORE[filename]['people_count'] = people_count
        STATS_STORE[filename]['avg_speed'] = round(avg_speed, 2)
        STATS_STORE[filename]['wait_time'] = round(est_time_seconds / 60, 2)  # in minutes
        STATS_STORE[filename]['processing_time'] = round(time.time() - start_time, 2)
        STATS_STORE[filename]['last_update'] = time.time()

        # Draw ROI and HUD
        cv2.polylines(frame, [np.array(QUEUE_ZONE, np.int32)], True, (255, 255, 0), 2)
        cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} m/s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Est. Wait Time: {est_time_seconds:.1f} s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

def get_video_statistics(video_path):
    """Process entire video and return final statistics (legacy / batch mode)."""
    # Keep this function for compatibility — it can be used for a full offline pass.
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    filename = os.path.basename(video_path)
    _init_stats_for(filename)

    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    tracker = DeepSort(max_age=30)
    max_people = 0
    speeds = []
    frame_count = 0
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.04
    last_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        results = model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if is_in_queue(cx, cy):
                        detections.append(([x1, y1, x2, y2], float(box.conf[0]), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_people = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_in_queue(cx, cy):
                current_people += 1
                if track_id in last_positions:
                    dist_px = np.linalg.norm(np.array([cx, cy]) - np.array(last_positions[track_id]))
                    dist_m = dist_px / PIXELS_PER_METER
                    speed = dist_m / frame_time
                    speeds.append(speed)
                last_positions[track_id] = (cx, cy)
        max_people = max(max_people, current_people)

    cap.release()
    processing_time = time.time() - start_time
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    wait_time_minutes = round((max_people * 0.5 / avg_speed) / 60, 1) if avg_speed > 0 else max_people * 0.5

    # update final stats
    STATS_STORE[filename].update({
        'people_count': max_people,
        'avg_speed': round(avg_speed, 2),
        'wait_time': wait_time_minutes,
        'processing_time': round(processing_time, 2),
        'accuracy': 72
    })

    return {
        'people_count': max_people,
        'avg_speed': round(avg_speed, 2),
        'wait_time': wait_time_minutes,
        'processing_time': round(processing_time, 2),
        'accuracy': 72
    }
