import cv2
from ultralytics import YOLO
import math

# --- Configuration ---

# Path to your custom-trained YOLOv8 model
MODEL_PATH = 'best.pt'  # Make sure this is in the same folder as the script

# Video source: 
# - Use a path to a video file, e.g., 'my_test_video.mp4'
# - Use 0 for your primary webcam, 1 for a secondary webcam, etc.
VIDEO_SOURCE = 'test5.mp4'  # Change to 0 to use webcam

# Confidence threshold: Only show detections with confidence > this value
CONF_THRESHOLD = 0.5 

# --- Main Inference Logic ---

# 1. Load the trained YOLO model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the 'best.pt' file is in the same directory as the script.")
    exit()

# Get the class names from the model
class_names = model.names

# 2. Open the video source
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
    exit()

print("Starting live inference... Press 'q' to quit.")

# 3. Loop through video frames
while True:
    # Read a new frame from the video
    success, frame = cap.read()

    if not success:
        print("End of video stream or cannot read frame. Exiting.")
        break

    # 4. Run YOLOv8 inference on the frame
    # The 'stream=True' argument is efficient for video processing
    results = model(frame, conf=CONF_THRESHOLD, stream=True)

    # Initialize a dictionary to count objects for the dashboard
    object_counts = {name: 0 for name in class_names.values()}

    # 5. Process results and draw on the frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # --- Get Bounding Box Coordinates ---
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- Get Confidence Score and Class Name ---
            confidence = math.ceil(box.conf[0] * 100) / 100
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            
            # Increment the count for the detected class
            object_counts[class_name] += 1

            # --- Draw Bounding Box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # --- Draw Label (Class Name and Confidence) ---
            label = f"{class_name.upper()} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # --- Task 4: Pick Point Generation ---
            # Calculate the center of the bounding box
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            
            # Draw a crosshair at the center point
            crosshair_size = 10
            cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 0, 255), 2) # Horizontal line
            cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 0, 255), 2) # Vertical line

    # --- Task 5: Basic Dashboard ---
    # Display the counts of each detected object
    y_offset = 30
    for class_name, count in object_counts.items():
        text = f"{class_name.upper()}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30

    # 6. Display the annotated frame
    cv2.imshow("Real-Time Scrap Sorter Simulation", frame)

    # 7. Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 8. Release resources
cap.release()
cv2.destroyAllWindows()

print("Inference stopped.")