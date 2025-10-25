import cv2
import os
import numpy as np

# --- CONFIG ---
VIDEO_FILE = os.path.join(os.path.dirname(__file__), 'uploads', 'default_queue.mp4')
# --------------

# Global list to store clicked points
points = []

def click_event(event, x, y, flags, params):
    global points
    
    # Check for left mouse-click event
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the point to our list
        points.append((x, y))
        
        # Draw a circle on the frame
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(frame, f'({x},{y})', (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        print(f"Point {len(points)}: ({x}, {y})")
        
        # If we have 4 points, draw the polygon
        if len(points) == 4:
            print("\nGot 4 points. Press 'q' to quit and use these values.")
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
        
        cv2.imshow("Click 4 Points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)", frame)

# --- Main ---
if not os.path.exists(VIDEO_FILE):
    print(f"Error: Video not found at {VIDEO_FILE}")
    print("Please make sure 'default_queue.mp4' is in the 'uploads' folder.")
    exit()

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_FILE}")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

print("="*50)
print("Click the 4 corners of the QUEUE FLOOR in this order:")
print("1. Top-Left")
print("2. Top-Right")
print("3. Bottom-Right")
print("4. Bottom-Left")
print("\nPress 'r' to reset points.")
print("Press 'q' to quit when done.")
print("="*50)

# Create a window and set the mouse callback
cv2.namedWindow("Click 4 Points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
cv2.setMouseCallback("Click 4 Points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)", click_event)

while True:
    cv2.imshow("Click 4 Points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Quit
    if key == ord('q'):
        break
    
    # Reset
    if key == ord('r'):
        print("Resetting points. Please click 4 points again.")
        points = []
        # Re-read the first frame to clear old drawings
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cv2.imshow("Click 4 Points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)", frame)

cap.release()
cv2.destroyAllWindows()

if len(points) == 4:
    print("\n--- COPY THESE VALUES ---")
    print("SOURCE_POINTS = [ \\")
    print(f"    {points[0]}, # Top-Left")
    print(f"    {points[1]}, # Top-Right")
    print(f"    {points[2]}, # Bottom-Right")
    print(f"    {points[3]}  # Bottom-Left")
    print("]")
else:
    print("\nDid not get 4 points. Please try again.")