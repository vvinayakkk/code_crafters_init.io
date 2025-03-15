import os
import cv2
import numpy as np
import tempfile
import base64
import urllib.request
import zipfile
import tarfile
import shutil
import sys
from flask import Flask, render_template, request, jsonify, send_file, url_for
import threading
import time
import uuid

app = Flask(__name__)

# Global variables to track processing status
processing_tasks = {}

def get_binary_file_encoded(bin_file):
    """
    Generate base64 encoding of a file for download
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return bin_str

def download_mask_rcnn_model():
    """
    Downloads the Mask R-CNN model files if they don't exist
    Returns a tuple (success, message, progress_callback)
    """
    model_dir = 'mask_rcnn_inception_v2_coco_2018_01_28'
    model_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
    config_path = os.path.join(model_dir, 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
    
    # Check if model files already exist
    if os.path.exists(model_path) and os.path.exists(config_path):
        return True, "Mask R-CNN model already exists.", 100
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # URLs for model files
        model_url = "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
        config_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        
        # Download progress 
        progress = {"value": 0}
        
        def progress_callback(count, block_size, total_size):
            progress["value"] = min(int(count * block_size / total_size * 100), 100)
            return progress["value"]
        
        # Download and extract model
        tar_file = os.path.join(model_dir, 'mask_rcnn.tar.gz')
        urllib.request.urlretrieve(
            model_url, 
            tar_file,
            progress_callback
        )
        
        # Extract tar file
        with tarfile.open(tar_file, 'r:gz') as tar:
            # Extract only the frozen_inference_graph.pb file
            for member in tar.getmembers():
                if member.name.endswith('frozen_inference_graph.pb'):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=model_dir)
        
        # Download config file
        urllib.request.urlretrieve(config_url, config_path)
        
        # Clean up
        if os.path.exists(tar_file):
            os.remove(tar_file)
            
        return True, "Mask R-CNN model downloaded successfully.", 100
        
    except Exception as e:
        error_msg = f"Failed to download Mask R-CNN model: {str(e)}"
        return False, error_msg, 0

def check_mask_rcnn_availability():
    """
    Check if the Mask R-CNN model files are available
    Returns a tuple (available, message)
    """
    model_dir = 'mask_rcnn_inception_v2_coco_2018_01_28'
    model_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
    config_path = os.path.join(model_dir, 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        return False, "Model files not found. Please download the Mask R-CNN model."
    
    return True, "Mask R-CNN model files found."

def process_video_task(video_path, segmentation_method, color_option, task_id):
    """
    Background task to process video
    """
    try:
        # Update task status
        processing_tasks[task_id]['status'] = 'processing'
        
        # Process the video
        output_path = process_video(video_path, segmentation_method, color_option, task_id)
        
        # Update task status on completion
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['output_path'] = output_path
        processing_tasks[task_id]['filename'] = os.path.basename(output_path)
        
    except Exception as e:
        # Update task status on error
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

def process_video(video_path, segmentation_method, color_option, task_id):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a temporary output file
    output_path = tempfile.mktemp(suffix='.mp4')
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize video processing components
    
    # 1. HOG (Histogram of Oriented Gradients) person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # 2. Background subtractor for motion detection - improved parameters
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300, 
        varThreshold=25, 
        detectShadows=True
    )
    
    # 3. Initialize Mask R-CNN if selected
    if segmentation_method == "Mask R-CNN (Best but slower)":
        # Check if model files exist before trying to load
        model_available, _ = check_mask_rcnn_availability()
        
        if model_available:
            try:
                # Model directory path
                model_dir = 'mask_rcnn_inception_v2_coco_2018_01_28'
                model_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
                config_path = os.path.join(model_dir, 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
                
                # Load pre-trained Mask R-CNN model
                net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                
                # Set preferable backend and target
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                # Class ID for person in COCO dataset
                person_class_id = 1
                
                processing_tasks[task_id]['log'] = "Loaded Mask R-CNN model successfully"
            except Exception as e:
                processing_tasks[task_id]['log'] = f"Error loading Mask R-CNN model: {str(e)}"
                processing_tasks[task_id]['log'] += "\nFalling back to background subtraction method"
                segmentation_method = "Background Subtraction (Fast)"
        else:
            processing_tasks[task_id]['log'] = "Mask R-CNN model files not found."
            processing_tasks[task_id]['log'] += "\nFalling back to background subtraction method"
            segmentation_method = "Background Subtraction (Fast)"
    
    # Dictionary to store person IDs and trajectories
    person_tracks = {}
    next_id = 0
    
    # Create individual trackers dictionary
    trackers = {}
    
    # Watershed algorithm components
    if segmentation_method == "Watershed Segmentation (Better)":
        kernel = np.ones((3,3),np.uint8)
        kernel_large = np.ones((7,7),np.uint8)
    
    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = int(frame_idx / frame_count * 100)
        processing_tasks[task_id]['progress'] = progress
        frame_idx += 1
        
        # Create a copy of the frame for drawing
        frame_display = frame.copy()
        
        # Create segmentation mask layer
        segmentation_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply the selected segmentation method
        if segmentation_method == "Background Subtraction (Fast)":
            # Background subtraction for motion detection
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
            fg_mask = cv2.erode(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
            fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)
            
            # Improved contour finding for better segmentation
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size to reduce noise
            filtered_contours = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:  # Adjust threshold as needed
                    filtered_contours.append(cnt)
            
            # Draw the contours on segmentation mask
            cv2.drawContours(segmentation_mask, filtered_contours, -1, (0, 255, 0), -1)
            
        elif segmentation_method == "Watershed Segmentation (Better)":
            # Background subtraction
            fg_mask = bg_subtractor.apply(frame)
            threshold = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
            
            # Noise removal
            opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel_large, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labeling
            ret, markers = cv2.connectedComponents(sure_fg)
            
            # Add one to all labels so that background is not 0, but 1
            markers = markers + 1
            
            # Mark the unknown region with zero
            markers[unknown == 255] = 0
            
            # Apply watershed algorithm
            markers = cv2.watershed(frame, markers)
            
            # Mark watershed boundaries
            frame_display[markers == -1] = [0, 0, 255]
            
            # Create mask for segmentation visualization
            for i in range(2, np.max(markers) + 1):
                segmentation_mask[markers == i] = [0, 255, 0]
            
        elif segmentation_method == "Mask R-CNN (Best but slower)":
            try:
                # Create a blob from the frame
                blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
                
                # Set the input to the network
                net.setInput(blob)
                
                # Run the forward pass
                boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
                
                # Process Mask R-CNN detections
                detection_count = boxes.shape[2]
                
                for i in range(detection_count):
                    box = boxes[0, 0, i]
                    class_id = int(box[1])
                    score = box[2]
                    
                    # Filter for person class with good confidence
                    if class_id == person_class_id and score > 0.75:
                        # Get box coordinates
                        left = int(box[3] * width)
                        top = int(box[4] * height)
                        right = int(box[5] * width)
                        bottom = int(box[6] * height)
                        
                        # Ensure coordinates are within frame boundaries
                        left = max(0, min(left, width - 1))
                        top = max(0, min(top, height - 1))
                        right = max(0, min(right, width - 1))
                        bottom = max(0, min(bottom, height - 1))
                        
                        # Calculate width and height of the box
                        bbox_width = right - left + 1
                        bbox_height = bottom - top + 1
                        
                        # Get the mask
                        mask = masks[i, class_id]
                        mask = cv2.resize(mask, (bbox_width, bbox_height))
                        mask = (mask > 0.5)
                        
                        # Create person instance mask
                        roi = segmentation_mask[top:bottom+1, left:right+1]
                        roi[mask] = [0, 255, 0]  # Green mask for person
                        
                        # Assign ID to detection
                        person_id = next_id
                        next_id += 1
                        
                        # Store track information
                        person_tracks[person_id] = {
                            'trajectory': [(left + bbox_width//2, bottom)],
                            'last_seen': frame_idx,
                            'bbox': (left, top, bbox_width, bbox_height)
                        }
                        
                        # Draw ID
                        cv2.putText(
                            frame_display, 
                            f"Person {person_id}", 
                            (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
            except Exception as e:
                processing_tasks[task_id]['log'] = f"Error in Mask R-CNN processing: {str(e)}"
                # Fall back to background subtraction
                fg_mask = bg_subtractor.apply(frame)
                fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
                fg_mask = cv2.erode(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
                fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)
            # Person detection using HOG every 15 frames (for non Mask R-CNN methods)
        if segmentation_method != "Mask R-CNN (Best but slower)" and frame_idx % 15 == 0:
            # Detect people in the frame
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
                # Create a tracker for each person
                tracker_id = next_id + i
                
                # Create individual trackers based on available implementations
                try:
                    # Try CSRT tracker first
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    try:
                        # Fall back to KCF tracker if CSRT is not available
                        tracker = cv2.TrackerKCF_create()
                    except AttributeError:
                        # Use a simpler approach if no trackers are available
                        continue
                
                tracker.init(frame, (x, y, w, h))
                trackers[tracker_id] = tracker
                
                # Assign ID to new detections
                person_tracks[tracker_id] = {
                    'trajectory': [(x + w//2, y + h)],
                    'last_seen': frame_idx,
                    'bbox': (x, y, w, h)
                }
            
            # Update next_id
            if len(boxes) > 0:
                next_id += len(boxes)
        
        # Update trackers for non Mask R-CNN methods
        if segmentation_method != "Mask R-CNN (Best but slower)":
            to_delete = []
            for person_id, tracker in trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # Update trajectory
                    person_tracks[person_id]['trajectory'].append((x + w//2, y + h))
                    person_tracks[person_id]['last_seen'] = frame_idx
                    person_tracks[person_id]['bbox'] = (x, y, w, h)
                    
                    # Draw bounding box if selected
                    if color_option in ["Bounding Boxes", "Combined View"]:
                        cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw ID
                    cv2.putText(
                        frame_display, 
                        f"Person {person_id}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX,0.5, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # For non-MaskRCNN approaches, enhance segmentation based on bounding box
                    if segmentation_method != "Mask R-CNN (Best but slower)":
                        person_roi = segmentation_mask[y:y+h, x:x+w]
                        # Create a simple ellipse-shaped mask to approximate human shape
                        center = (w//2, h//2)
                        axes = (w//3, h//2)
                        mask = np.zeros((h, w, 3), dtype=np.uint8)
                        cv2.ellipse(mask, center, axes, 0, 0, 360, (0, 255, 0), -1)
                        
                        # Apply the mask to the person ROI
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
            # Only draw trajectories for recently seen people
            if frame_idx - data['last_seen'] < 30:
                traj = data['trajectory']
                if len(traj) > 1:
                    for i in range(1, len(traj)):
                        cv2.line(
                            frame_display, 
                            traj[i-1], 
                            traj[i], 
                            (0, 0, 255), 
                            2
                        )
        
        # Combine original frame with segmentation mask based on selected view
        if color_option == "Segmentation Masks":
            # Show only segmentation masks
            frame_display = segmentation_mask
        elif color_option == "Combined View":
            # Blend segmentation mask with original frame for combined view
            alpha = 0.4  # Transparency factor
            frame_display = cv2.addWeighted(frame_display, 1 - alpha, segmentation_mask, alpha, 0)
        
        # Draw motion detection mask in corner
        h, w = frame_display.shape[:2]
        
        if segmentation_method in ["Background Subtraction (Fast)", "Watershed Segmentation (Better)"]:
            small_mask = cv2.resize(fg_mask, (w//4, h//4))
            small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
            frame_display[0:h//4, 0:w//4] = small_mask
        
        # Add frame information
        cv2.putText(
            frame_display,
            f"Frame: {frame_idx}/{frame_count}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add person count
        cv2.putText(
            frame_display,
            f"People: {len([p for p in person_tracks.values() if frame_idx - p['last_seen'] < 30])}",
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add segmentation method info
        cv2.putText(
            frame_display,
            f"Method: {segmentation_method}",
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Write the frame to output video
        out.write(frame_display)
    
    # Release video objects
    cap.release()
    out.release()
    
    return output_path

# Flask routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/check_mask_rcnn')
def check_mask_rcnn():
    """Check if Mask R-CNN model is available"""
    available, message = check_mask_rcnn_availability()
    return jsonify({
        'available': available,
        'message': message
    })

@app.route('/download_mask_rcnn', methods=['POST'])
def download_mask_rcnn_route():
    """Download Mask R-CNN model"""
    task_id = str(uuid.uuid4())
    
    # Create a background thread to download the model
    def download_task():
        success, message, progress = download_mask_rcnn_model()
        return {'success': success, 'message': message, 'progress': progress}
    
    # Start download in background
    thread = threading.Thread(target=lambda: download_task())
    thread.daemon = True
    thread.start()
    
    # Return immediately with a task ID
    return jsonify({
        'status': 'downloading',
        'task_id': task_id,
        'message': 'Started downloading Mask R-CNN model'
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload video file and start processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Get processing parameters
    segmentation_method = request.form.get('segmentation_method', 'Background Subtraction (Fast)')
    color_option = request.form.get('color_option', 'Combined View')
    
    # Save video to temporary file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(video_path)
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task entry
    processing_tasks[task_id] = {
        'status': 'starting',
        'progress': 0,
        'input_path': video_path,
        'segmentation_method': segmentation_method,
        'color_option': color_option,
        'temp_dir': temp_dir,
        'start_time': time.time(),
        'log': ''
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video_task, 
                             args=(video_path, segmentation_method, color_option, task_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'processing',
        'task_id': task_id,
        'message': 'Video upload successful. Processing started.'
    })

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Get status of a processing task"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    
    response = {
        'status': task['status'],
        'progress': task['progress'],
        'segmentation_method': task['segmentation_method'],
        'color_option': task['color_option'],
        'elapsed_time': int(time.time() - task['start_time']),
        'log': task.get('log', '')
    }
    
    # Include output path if processing is complete
    if task['status'] == 'completed':
        response['output_url'] = url_for('download_video', task_id=task_id)
        response['filename'] = task.get('filename', 'processed_video.mp4')
    
    # Include error if processing failed
    if task['status'] == 'failed':
        response['error'] = task.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/download_video/<task_id>')
def download_video(task_id):
    """Download processed video file"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    
    if task['status'] != 'completed' or 'output_path' not in task:
        return jsonify({'error': 'Video processing not completed'}), 400
    
    # Send the file
    return send_file(
        task['output_path'],
        as_attachment=True,
        download_name=task.get('filename', 'processed_video.mp4')
    )

@app.route('/clear_task/<task_id>', methods=['POST'])
def clear_task(task_id):
    """Clear a completed task and its temporary files"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    
    # Clean up files
    try:
        if 'temp_dir' in task and os.path.exists(task['temp_dir']):
            shutil.rmtree(task['temp_dir'])
        
        if 'output_path' in task and os.path.exists(task['output_path']):
            os.unlink(task['output_path'])
    except Exception as e:
        return jsonify({'error': f'Error clearing files: {str(e)}'}), 500
    
    # Remove task from dictionary
    del processing_tasks[task_id]
    
    return jsonify({'status': 'success', 'message': 'Task cleared successfully'})

# Needed for development
if __name__ == "__main__":
    # Create static and templates folder if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html with a basic interface
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Person Tracking and Segmentation</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        .hidden { display: none; }
        .processing-container { margin-top: 20px; }
        .results-container { margin-top: 20px; }
        #progressBar { height: 25px; }
        .log-container {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            height: 100px;
            overflow-y: auto;
            font-family: monospace;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1>Person Tracking and Segmentation</h1>
        <hr>
        
        <!-- Settings Panel -->
        <div class="card mb-4">
            <div class="card-header">Processing Options</div>
            <div class="card-body">
                <form id="videoForm" enctype="multipart/form-data">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="segmentation_method" class="form-label">Segmentation Method</label>
                            <select class="form-select" id="segmentation_method" name="segmentation_method">
                                <option value="Background Subtraction (Fast)">Background Subtraction (Fast)</option>
                                <option value="Watershed Segmentation (Better)">Watershed Segmentation (Better)</option>
                                <option value="Mask R-CNN (Best but slower)">Mask R-CNN (Best but slower)</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label for="color_option" class="form-label">Visualization Style</label>
                            <select class="form-select" id="color_option" name="color_option">
                                <option value="Bounding Boxes">Bounding Boxes</option>
                                <option value="Segmentation Masks">Segmentation Masks</option>
                                <option value="Combined View" selected>Combined View</option>
                            </select>
                        </div>
                    </div>
                    
                    <div id="maskRcnnStatus" class="alert alert-warning hidden">
                        Checking Mask R-CNN availability...
                    </div>
                    
                    <div id="maskRcnnDownload" class="alert alert-info hidden">
                        <p>Mask R-CNN model files are not available. You need to download them to use this feature.</p>
                        <button type="button" id="downloadModelBtn" class="btn btn-primary">Download Mask R-CNN Model</button>
                        <div class="progress mt-2 hidden" id="downloadProgress">
                            <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="video" class="form-label">Select Video File</label>
                        <input class="form-control" type="file" id="video" name="video" accept=".mp4,.avi,.mov">
                    </div>
                    
                    <button type="submit" class="btn btn-primary" id="processBtn">Process Video</button>
                </form>
            </div>
        </div>
        
        <!-- Processing Status -->
        <div class="processing-container hidden" id="processingContainer">
            <div class="card">
                <div class="card-header">Processing Status</div>
                <div class="card-body">
                    <h5 id="statusMessage">Processing your video...</h5>
                    <div class="progress mb-3" id="progressBar">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span id="progressText">0%</span>
                        <span id="timeElapsed">Time: 0s</span>
                    </div>
                    <div class="log-container" id="logContainer">
                        <div id="logContent"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Results -->
        <div class="results-container hidden" id="resultsContainer">
            <div class="card">
                <div class="card-header">Results</div>
                <div class="card-body">
                    <div class="mb-3">
                        <h5>Processed Video</h5>
                        <video id="processedVideo" controls class="img-fluid">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    <div class="d-flex justify-content-between">
                        <a id="downloadLink" class="btn btn-success" download>Download Processed Video</a>
                        <button id="clearBtn" class="btn btn-danger">Clear Results</button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Error Display -->
        <div class="alert alert-danger mt-3 hidden" id="errorContainer">
            <h5>Error</h5>
            <p id="errorMessage"></p>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            let currentTaskId = null;
            let statusInterval = null;
            
            // Check Mask R-CNN availability if option is selected
            $("#segmentation_method").change(function() {
                if ($(this).val() === "Mask R-CNN (Best but slower)") {
                    checkMaskRcnnAvailability();
                } else {
                    $("#maskRcnnStatus").addClass("hidden");
                    $("#maskRcnnDownload").addClass("hidden");
                }
            });
            
            // Initial check if Mask R-CNN is selected
            if ($("#segmentation_method").val() === "Mask R-CNN (Best but slower)") {
                checkMaskRcnnAvailability();
            }
            
            // Download Mask R-CNN model
            $("#downloadModelBtn").click(function(e) {
                e.preventDefault();
                downloadMaskRcnnModel();
            });
            
            // Form submission
            $("#videoForm").submit(function(e) {
                e.preventDefault();
                
                // Validate input
                const videoFile = $("#video")[0].files[0];
                if (!videoFile) {
                    showError("Please select a video file.");
                    return;
                }
                
                // Clear previous results
                clearResults();
                
                // Show processing container
                $("#processingContainer").removeClass("hidden");
                
                // Create form data
                const formData = new FormData(this);
                
                // Upload video
                $.ajax({
                    url: '/upload',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        currentTaskId = response.task_id;
                        startStatusPolling(currentTaskId);
                    },
                    error: function(xhr) {
                        showError("Error uploading video: " + xhr.responseJSON.error);
                        $("#processingContainer").addClass("hidden");
                    }
                });
            });
            
            // Clear results
            $("#clearBtn").click(function() {
                if (currentTaskId) {
                    $.post('/clear_task/' + currentTaskId, function() {
                        clearResults();
                    });
                }
            });
            
            // Function to check Mask R-CNN availability
            function checkMaskRcnnAvailability() {
                $("#maskRcnnStatus").removeClass("hidden").text("Checking Mask R-CNN availability...");
                $("#maskRcnnDownload").addClass("hidden");
                
                $.ajax({
                    url: '/check_mask_rcnn',
                    type: 'GET',
                    success: function(response) {
                        $("#maskRcnnStatus").addClass("hidden");
                        
                        if (!response.available) {
                            $("#maskRcnnDownload").removeClass("hidden");
                        }
                    },
                    error: function() {
                        $("#maskRcnnStatus").text("Error checking Mask R-CNN availability.");
                    }
                });
            }
            
            // Function to download Mask R-CNN model
            function downloadMaskRcnnModel() {
                $("#downloadModelBtn").prop("disabled", true).text("Downloading...");
                $("#downloadProgress").removeClass("hidden");
                
                $.ajax({
                    url: '/download_mask_rcnn',
                    type: 'POST',
                    success: function(response) {
                        // Poll for download status
                        let downloadInterval = setInterval(function() {
                            $.ajax({
                                url: '/check_mask_rcnn',
                                type: 'GET',
                                success: function(checkResponse) {
                                    if (checkResponse.available) {
                                        clearInterval(downloadInterval);
                                        $("#maskRcnnDownload").addClass("hidden");
                                        $("#downloadModelBtn").prop("disabled", false).text("Download Mask R-CNN Model");
                                        $("#downloadProgress").addClass("hidden");
                                        $("#maskRcnnStatus").removeClass("hidden").text("Mask R-CNN model downloaded successfully!");
                                    }
                                }
                            });
                        }, 2000);
                    },
                    error: function() {
                        $("#downloadModelBtn").prop("disabled", false).text("Download Failed - Retry");
                        $("#downloadProgress").addClass("hidden");
                    }
                });
            }
            
            // Function to start polling for task status
            function startStatusPolling(taskId) {
                if (statusInterval) {
                    clearInterval(statusInterval);
                }
                
                statusInterval = setInterval(function() {
                    $.ajax({
                        url: '/task_status/' + taskId,
                        type: 'GET',
                        success: function(response) {
                            updateStatus(response);
                            
                            // If processing is complete or failed, stop polling
                            if (response.status === 'completed' || response.status === 'failed') {
                                clearInterval(statusInterval);
                            }
                        },
                        error: function() {
                            showError("Error getting task status.");
                            clearInterval(statusInterval);
                        }
                    });
                }, 1000);
            }
            
            // Function to update status display
            function updateStatus(response) {
                // Update progress bar
                const progressPercent = response.progress + "%";
                $("#progressBar .progress-bar").css("width", progressPercent);
                $("#progressText").text(progressPercent);
                
                // Update time elapsed
                $("#timeElapsed").text("Time: " + response.elapsed_time + "s");
                
                // Update log
                if (response.log) {
                    $("#logContent").text(response.log);
                    $("#logContainer").scrollTop($("#logContainer")[0].scrollHeight);
                }
                
                // Handle different statuses
                if (response.status === 'completed') {
                    $("#statusMessage").text("Processing completed!");
                    
                    // Show results
                    $("#resultsContainer").removeClass("hidden");
                    $("#processedVideo").attr("src", response.output_url);
                    $("#downloadLink").attr("href", response.output_url).text("Download " + response.filename);
                    
                } else if (response.status === 'failed') {
                    $("#statusMessage").text("Processing failed!");
                    showError(response.error || "Unknown error occurred during processing.");
                    
                } else {
                    $("#statusMessage").text("Processing your video (" + response.status + ")...");
                }
            }
            
            // Function to show error message
            function showError(message) {
                $("#errorContainer").removeClass("hidden");
                $("#errorMessage").text(message);
            }
            
            // Function to clear results and reset UI
            function clearResults() {
                $("#resultsContainer").addClass("hidden");
                $("#processingContainer").addClass("hidden");
                $("#errorContainer").addClass("hidden");
                $("#progressBar .progress-bar").css("width", "0%");
                $("#progressText").text("0%");
                $("#timeElapsed").text("Time: 0s");
                $("#logContent").text("");
                
                if (statusInterval) {
                    clearInterval(statusInterval);
                    statusInterval = null;
                }
                
                currentTaskId = null;
            }
        });
    </script>
</body>
</html>
        """)
    
    # Run the Flask app
    app.run(debug=True)