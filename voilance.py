import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# Load the YOLOv8 model (adjust path if needed)
model = YOLO(r'C:\Project_version_1\Yolo_nano_weights.pt')

# Directory to save recorded clips
output_dir = r'C:\Project_version_1\project\static\detected_clips'
os.makedirs(output_dir, exist_ok=True)

# Open webcam (use 0 for default, 1 if external)
cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Recording state
recording = False
out = None
recording_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Flag to check if violence is detected in current frame
    violence_detected = False

    for detection in results[0].boxes:
        class_id = int(detection.cls)
        confidence = float(detection.conf)

        # If class is Violence (ID = 1)
        if class_id == 1:
            violence_detected = True
            x1, y1, x2, y2 = map(int, detection.xyxy[0])

            # Draw red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Violence {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Start recording if violence detected
    if violence_detected and not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'violence_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        recording = True
        recording_start_time = time.time()
        print(f"[INFO] Started recording: {output_path}")

    # Write frame if currently recording
    if recording:
        out.write(frame)

        # Optional: stop recording after N seconds of inactivity
        if not violence_detected and time.time() - recording_start_time > 5:
            recording = False
            out.release()
            print(f"[INFO] Stopped recording: {output_path}")

    # Display the frame
    cv2.imshow("Real-Time Violence Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
