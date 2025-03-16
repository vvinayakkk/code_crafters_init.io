import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import time
import base64
import urllib.request
import zipfile
import tarfile
import shutil
import sys

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """
    Generate a link to download a file in Streamlit
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def download_mask_rcnn_model():
    """
    Downloads the Mask R-CNN model files if they don't exist
    Returns a tuple (success, message)
    """
    model_dir = 'mask_rcnn_inception_v2_coco_2018_01_28'
    model_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
    config_path = os.path.join(model_dir, 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
    
    # Check if model files already exist
    if os.path.exists(model_path) and os.path.exists(config_path):
        return True, "Mask R-CNN model already exists."
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # URLs for model files
        model_url = "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
        config_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        
        # Download progress for streamlit
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text(f"Downloading Mask R-CNN model files...")
        
        # Download and extract model
        tar_file = os.path.join(model_dir, 'mask_rcnn.tar.gz')
        urllib.request.urlretrieve(
            model_url, 
            tar_file,
            lambda x, y, z: progress_bar.progress(min(int(x * y / z * 100), 100))
        )
        
        # Extract tar file
        progress_text.text("Extracting model files...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            # Extract only the frozen_inference_graph.pb file
            for member in tar.getmembers():
                if member.name.endswith('frozen_inference_graph.pb'):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=model_dir)
        
        # Download config file
        progress_text.text("Downloading model configuration...")
        urllib.request.urlretrieve(config_url, config_path)
        
        # Clean up
        if os.path.exists(tar_file):
            os.remove(tar_file)
            
        progress_text.text("Mask R-CNN model downloaded successfully!")
        progress_bar.progress(100)
        
        return True, "Mask R-CNN model downloaded successfully."
        
    except Exception as e:
        error_msg = f"Failed to download Mask R-CNN model: {str(e)}"
        return False, error_msg

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

def main():
    st.title("Person Tracking and Segmentation with OpenCV")
    st.write("Upload a video to track and segment people")
    
    # Add segmentation method selection
    segmentation_method = st.selectbox(
        "Choose segmentation method",
        ["Background Subtraction (Fast)", "Watershed Segmentation (Better)", "Mask R-CNN (Best but slower)"]
    )
    
    # If Mask R-CNN is selected, check if model is available and offer download
    if segmentation_method == "Mask R-CNN (Best but slower)":
        available, msg = check_mask_rcnn_availability()
        if not available:
            st.warning(msg)
            if st.button("Download Mask R-CNN Model"):
                success, download_msg = download_mask_rcnn_model()
                if not success:
                    st.error(download_msg)
                    st.warning("Falling back to Background Subtraction method")
                    segmentation_method = "Background Subtraction (Fast)"
                else:
                    st.success(download_msg)
                    # Need to reload after downloading
                    st.rerun()
    
    # Color options for visualization
    color_option = st.selectbox(
        "Choose visualization style",
        ["Bounding Boxes", "Segmentation Masks", "Combined View"]
    )
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Process the video
        with st.spinner("Processing video..."):
            output_path = process_video(video_path, segmentation_method, color_option)
        
        # Display the processed video
        st.success("Video processed successfully!")
        st.video(output_path)
        
        # Create download button for processed video
        st.markdown(get_binary_file_downloader_html(output_path, 'Processed Video'), unsafe_allow_html=True)
        
        # Clean up temporary files (only delete input file, keep output for downloading)
        os.unlink(video_path)
        
        # Add a button to delete the output file when done
        if st.button("Clear processed video"):
            os.unlink(output_path)
            st.success("Temporary files cleared successfully!")
            st.experimental_rerun()

def process_video(video_path, segmentation_method, color_option):
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
                
                # Display status
                st.text("Loaded Mask R-CNN model successfully")
            except Exception as e:
                st.error(f"Error loading Mask R-CNN model: {str(e)}")
                st.text("Falling back to background subtraction method")
                segmentation_method = "Background Subtraction (Fast)"
        else:
            st.error("Mask R-CNN model files not found.")
            st.text("Falling back to background subtraction method")
            segmentation_method = "Background Subtraction (Fast)"
    
    # Dictionary to store person IDs and trajectories
    person_tracks = {}
    next_id = 0
    
    # Create individual trackers dictionary
    trackers = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    
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
        
        # Update progress bar
        progress = int(frame_idx / frame_count * 100)
        progress_bar.progress(progress)
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
                st.error(f"Error in Mask R-CNN processing: {str(e)}")
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

if __name__ == "__main__":
    main()