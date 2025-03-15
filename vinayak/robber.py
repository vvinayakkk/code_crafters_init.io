import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from roboflow import Roboflow
import time

# Initialize YOLO model for person detection
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 nano model for person detection

# Initialize Roboflow mask detection model
# Replace "YOUR_ROBOFLOW_API_KEY" with your actual Roboflow API key
rf = Roboflow(api_key="7lNTHsBWxOAEwCP8wIvq")
project = rf.workspace("muhammad-al-azis-firdaus-vbkie").project("face-mask-detection-isvki")
mask_model = project.version(8).model  # Version 8 of the face mask detection model

# Initialize MediaPipe Pose for hand and body tracking
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Optional: Set lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize FPS counter
prev_time = 0

while cap.isOpened():
    # Read frame from webcam
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Convert frame to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # YOLO person detection
    object_results = model(rgb_frame)
    
    # Roboflow mask detection
    mask_results = mask_model.predict(rgb_frame, confidence=40, overlap=30).json()
    
    # MediaPipe pose detection
    pose_results = pose.process(rgb_frame)
    
    # Process person detections
    for result in object_results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates and details
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # Handle person detection
            if class_name == "person":
                # Default: green box for person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Flag for suspicious behavior
                suspicious = False
                
                # Check mask status with Roboflow
                for detection in mask_results["predictions"]:
                    mx = detection["x"] - detection["width"] / 2
                    my = detection["y"] - detection["height"] / 2
                    mw, mh = detection["width"], detection["height"]
                    mx1, my1 = int(mx), int(my)
                    mx2, my2 = int(mx + mw), int(my + mh)
                    
                    # Check if mask detection overlaps with person
                    if (mx1 >= x1 and mx2 <= x2 and my1 >= y1 and my2 <= y2):
                        if detection["class"] == "no-mask":
                            suspicious = True
                            cv2.putText(frame, "SUSPICIOUS: NO MASK", (x1, y2 + 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check hand position with MediaPipe Pose
                if pose_results.pose_landmarks:
                    # Get shoulder and wrist coordinates
                    shoulder_y = pose_results.pose_landmarks.landmark[11].y * frame.shape[0]  # Left shoulder
                    for idx in [15, 16]:  # Left and right wrist
                        hand_y = pose_results.pose_landmarks.landmark[idx].y * frame.shape[0]
                        if hand_y < shoulder_y:  # Hand above shoulder
                            suspicious = True
                            cv2.putText(frame, "SUSPICIOUS: HANDS RAISED", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # If suspicious, change person box to red
                if suspicious:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
    
    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Robber Detection", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()