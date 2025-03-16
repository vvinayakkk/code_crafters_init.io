from flask import Flask, request, send_file, jsonify, after_this_request
import cv2
import tempfile
import os
import numpy as np
from flask_cors import CORS  # Add CORS support for cross-origin requests
import io
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
import time
import collections
import uuid
import tempfile
from werkzeug.utils import secure_filename
from openvino.runtime import Core
import io
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a temporary output file
    output_path = tempfile.mktemp(suffix='.mp4')
    
    # Define the codec and create VideoWriter object
    # Use H.264 codec which is more widely supported
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1' (H.264)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize video processing components
    # HOG (Histogram of Oriented Gradients) person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Background subtractor for motion detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=25,
        detectShadows=True
    )
    
    # Dictionary to store person IDs and trajectories
    person_tracks = {}
    next_id = 0
    
    # Create individual trackers dictionary
    trackers = {}
    
    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy of the frame for drawing
        frame_display = frame.copy()
        
        # Create segmentation mask layer
        segmentation_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background subtraction for motion detection
        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.erode(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        
        # Draw contours on segmentation mask
        cv2.drawContours(segmentation_mask, filtered_contours, -1, (0, 255, 0), -1)
        
        # Person detection using HOG every 15 frames
        if frame_idx % 15 == 0:
            boxes, weights = hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05
            )
            
            # Reset trackers with new detections
            trackers = {}
            
            for i, box in enumerate(boxes):
                x, y, w, h = box
                tracker_id = next_id + i
                
                # Try different tracker types
                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    try:
                        tracker = cv2.TrackerKCF_create()
                    except AttributeError:
                        continue
                
                tracker.init(frame, (x, y, w, h))
                trackers[tracker_id] = tracker
                
                person_tracks[tracker_id] = {
                    'trajectory': [(x + w//2, y + h)],
                    'last_seen': frame_idx,
                    'bbox': (x, y, w, h)
                }
            
            if len(boxes) > 0:
                next_id += len(boxes)
        
        # Update trackers
        to_delete = []
        for person_id, tracker in trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Update trajectory
                person_tracks[person_id]['trajectory'].append((x + w//2, y + h))
                person_tracks[person_id]['last_seen'] = frame_idx
                person_tracks[person_id]['bbox'] = (x, y, w, h)
                
                # Draw bounding box
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(
                    frame_display,
                    f"Person {person_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # Enhance segmentation based on bounding box
                person_roi = segmentation_mask[y:y+h, x:x+w]
                center = (w//2, h//2)
                axes = (w//3, h//2)
                mask = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.ellipse(mask, center, axes, 0, 0, 360, (0, 255, 0), -1)
                person_roi = cv2.bitwise_or(person_roi, mask)
                segmentation_mask[y:y+h, x:x+w] = person_roi
            else:
                to_delete.append(person_id)
        
        # Remove failed trackers
        for pid in to_delete:
            if pid in trackers:
                del trackers[pid]
        
        # Draw trajectories
        for person_id, data in person_tracks.items():
            if frame_idx - data['last_seen'] < 30:
                traj = data['trajectory']
                if len(traj) > 1:
                    for i in range(1, len(traj)):
                        cv2.line(frame_display, traj[i-1], traj[i], (0, 0, 255), 2)
        
        # Blend segmentation mask with original frame
        frame_display = cv2.addWeighted(frame_display, 0.6, segmentation_mask, 0.4, 0)
        
        # Draw motion detection mask in corner
        h, w = frame_display.shape[:2]
        small_mask = cv2.resize(fg_mask, (w//4, h//4))
        small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        frame_display[0:h//4, 0:w//4] = small_mask
        
        # Write the frame to output video
        out.write(frame_display)
        
        frame_idx += 1
    
    # Release video objects
    cap.release()
    out.release()
    
    return output_path

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    # Check if request contains binary data (blob)
    if request.content_type.startswith('application/octet-stream'):
        # Get binary data directly from request
        video_data = request.get_data()
        
        # Create temporary files with unique names
        input_path = tempfile.mktemp(suffix='.mp4')
        
        # Save uploaded video to temporary file
        with open(input_path, 'wb') as f:
            f.write(video_data)
    
    # Also keep support for form-data submissions
    elif 'video' in request.files:
        video_file = request.files['video']
        input_path = tempfile.mktemp(suffix='.mp4')
        video_file.save(input_path)
    
    else:
        return jsonify({'error': 'No video data provided'}), 400
    
    try:
        # Process the video
        output_path = process_video(input_path)
        
        # Register cleanup function
        @after_this_request
        def clean_up(response):
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up files: {e}")
            return response
        
        # Instead of directly sending the file, read it into memory first
        # This ensures we're sending the complete file with proper headers
        with open(output_path, 'rb') as video_file:
            video_binary = video_file.read()
        
        # Create a proper response with correct headers
        response = app.response_class(
            response=video_binary,
            status=200,
            mimetype='video/mp4'
        )
        
        # Add proper headers for download
        response.headers['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response.headers['Content-Length'] = len(video_binary)
        response.headers['Accept-Ranges'] = 'bytes'
        
        return response
    
    except Exception as e:
        # Clean up input file in case of error
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500
    



# Add an endpoint to check if video format is supported by client
@app.route('/check_video_format', methods=['GET'])
def check_video_format():
    return jsonify({
        'supported_formats': ['mp4'],
        'recommended_codec': 'H.264 (avc1)',
        'status': 'ok'
    })






if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)