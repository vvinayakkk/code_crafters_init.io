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

# Constants
IMG_SIZE = (299, 224)
MAX_SEQ_LENGTH = 32
NUM_FEATURES = 2048
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv'}

# Initialize models and classes
class_vocab = ['Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting']

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load models globally
def load_models():
    global decoder, compiled_model_ir, output_layer_ir, height_en, width_en, frames2decode
    
    # Update these paths to your model locations
    model_path = "C:/Users/vinay/OneDrive/Desktop/official_code_crafters/code_crafters_init.io/vinayak/classifier_lstm_e19.h5"
    ir_model_path = "C:/Users/vinay/OneDrive/Desktop/official_code_crafters/code_crafters_init.io/vinayak/saved_model.xml"
    
    decoder = load_model(model_path, compile=False)
    
    # Load the OpenVINO model
    ie = Core()
    model_ir = ie.read_model(model=ir_model_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
    
    # Get output layer
    output_layer_ir = compiled_model_ir.output(0)
    
    # Get input dimensions
    try:
        height_en, width_en = list(decoder.inputs[0].shape)[1:3]
    except:
        height_en, width_en = list(compiled_model_ir.inputs[0].shape)[1:3]
    
    # Get frames to decode
    frames2decode = list(decoder.inputs[0].shape)[1] if len(decoder.inputs) == 1 else 16  # Default to 16 if not specified
    
    return "Models loaded successfully"

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility functions for video processing
def display_text_fnc(frame, display_text, index):
    # Configuration for displaying images with text
    FONT_COLOR = (255, 255, 255)
    FONT_COLOR2 = (0, 0, 0)
    FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
    FONT_SIZE = 0.5
    TEXT_VERTICAL_INTERVAL = 25
    TEXT_LEFT_MARGIN = 15
    
    # Put text over frame
    text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (index + 1))
    text_loc2 = (TEXT_LEFT_MARGIN + 1, TEXT_VERTICAL_INTERVAL * (index + 1) + 1)
    frame2 = frame.copy()
    _ = cv2.putText(frame2, display_text, text_loc2, FONT_STYLE, FONT_SIZE, FONT_COLOR2)
    _ = cv2.putText(frame2, display_text, text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)
    return frame2

# Process video function
def process_video(input_path, output_path):
    # Initialize variables
    fps = 30
    final_inf_counter = 0
    final_infer_time = time.time()
    final_infer_duration = 0
    frames_idx = []
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up output video writer
    # Use H.264 codec if available, otherwise fall back to XVID
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (620, 350))
    except:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Alternative H.264 fourcc
            out = cv2.VideoWriter(output_path, fourcc, fps, (620, 350))
        except:
            # Fall back to XVID if H.264 isn't available
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (620, 350))
    
    processing_times = collections.deque()
    processing_time = 0
    fps_calc = 0  # Initialize fps_calc here
    encoder_output = []
    decoded_labels = ["", "", ""]
    decoded_top_probs = [0, 0, 0]
    counter = 0
    
    # Text templates
    text_inference_template = "Infer Time:{Time:.1f}ms, FPS:{fps:.1f}"
    text_template = "{label},{conf:.2f}%"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        counter += 1
        frame_counter = counter  # Simple frame counter
        
        # Preprocess frame
        preprocessed = cv2.resize(frame, IMG_SIZE)
        preprocessed = preprocessed[:, :, [2, 1, 0]]  # BGR -> RGB
        
        # Process every second frame
        if counter % 2 == 0:
            frames_idx.append((counter, frame_counter, 'Yes'))
            
            # Measure processing time
            start_time = time.time()
            
            # Encoder inference per frame
            encoder_output.append(compiled_model_ir([preprocessed[None, ...]])[output_layer_ir][0])
            
            if len(encoder_output) == frames2decode:
                # Run decoder on collected frames
                encoder_output_array = np.array(encoder_output)[None, ...]
                probabilities = decoder.predict(encoder_output_array)[0]
                
                for idx, i in enumerate(np.argsort(probabilities)[::-1][:3]):
                    decoded_labels[idx] = class_vocab[i]
                    decoded_top_probs[idx] = probabilities[i]
                
                encoder_output = []
                final_inf_counter += 1
                final_infer_duration = (time.time() - final_infer_time)
                final_infer_time = time.time()
            
            # Inference has finished
            stop_time = time.time()
            
            # Calculate processing time
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()
            
            processing_time = np.mean(processing_times) * 1000 if processing_times else 0
            fps_calc = 1000 / processing_time if processing_time > 0 else 0
            
        else:
            frames_idx.append((counter, frame_counter, 'No'))
        
        # Resize frame for display
        frame = cv2.resize(frame, (620, 350))
        
        # Add text overlays
        for i in range(0, 3):
            display_text = text_template.format(
                label=decoded_labels[i],
                conf=decoded_top_probs[i] * 100,
            )
            frame = display_text_fnc(frame, display_text, i)
        
        display_text = text_inference_template.format(Time=processing_time, fps=fps_calc)
        frame = display_text_fnc(frame, display_text, 3)
        frame = display_text_fnc(frame, f"Infer Count: {final_inf_counter}", 4)
        
        # Write frame to output video
        out.write(frame)
    
    # Clean up
    cap.release()
    out.release()
    
    return output_path

import json
# API endpoint to process video - UPDATED to return binary data directly
@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    # Check if video file is in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        base_name, extension = os.path.splitext(filename)
        input_filename = f"{base_name}_{unique_id}{extension}"
        output_filename = f"processed_{base_name}_{unique_id}.mp4"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save the uploaded file
        file.save(input_path)
        
        try:
            # Process the video and get classification results
            output_path, classification_results = process_video_with_results(input_path, output_path)
            
            # Read the processed video file as binary data
            with open(output_path, 'rb') as video_file:
                video_binary = video_file.read()
            
            # Return the video binary directly as a response
            response = Response(
                video_binary,
                status=200,
                mimetype='video/mp4',
                headers={
                    'Content-Disposition': f'attachment; filename={output_filename}',
                    'X-Classification-Results': json.dumps(classification_results)
                }
            )
            
            return response
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up files
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up files: {e}")
    
    return jsonify({'error': 'Invalid file format'}), 400

# Modified video processing function that returns the classification results
def process_video_with_results(input_path, output_path):
    # Initialize variables
    fps = 30
    final_inf_counter = 0
    final_infer_time = time.time()
    final_infer_duration = 0
    frames_idx = []
    
    # For storing final results
    final_results = {
        'top_classes': [],
        'confidences': [],
        'inference_stats': {}
    }
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up output video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (620, 350))
    except:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Alternative H.264 fourcc
            out = cv2.VideoWriter(output_path, fourcc, fps, (620, 350))
        except:
            # Fall back to XVID if H.264 isn't available
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (620, 350))
    
    processing_times = collections.deque()
    processing_time = 0
    fps_calc = 0
    encoder_output = []
    decoded_labels = ["", "", ""]
    decoded_top_probs = [0, 0, 0]
    counter = 0
    
    # Text templates
    text_inference_template = "Infer Time:{Time:.1f}ms, FPS:{fps:.1f}"
    text_template = "{label},{conf:.2f}%"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        counter += 1
        frame_counter = counter
        
        # Preprocess frame
        preprocessed = cv2.resize(frame, IMG_SIZE)
        preprocessed = preprocessed[:, :, [2, 1, 0]]  # BGR -> RGB
        
        # Process every second frame
        if counter % 2 == 0:
            frames_idx.append((counter, frame_counter, 'Yes'))
            
            # Measure processing time
            start_time = time.time()
            
            # Encoder inference per frame
            encoder_output.append(compiled_model_ir([preprocessed[None, ...]])[output_layer_ir][0])
            
            if len(encoder_output) == frames2decode:
                # Run decoder on collected frames
                encoder_output_array = np.array(encoder_output)[None, ...]
                probabilities = decoder.predict(encoder_output_array)[0]
                
                for idx, i in enumerate(np.argsort(probabilities)[::-1][:3]):
                    decoded_labels[idx] = class_vocab[i]
                    decoded_top_probs[idx] = probabilities[i]
                
                # Update final results with current top predictions
                final_results['top_classes'] = decoded_labels.copy()
                final_results['confidences'] = [float(prob * 100) for prob in decoded_top_probs]
                
                encoder_output = []
                final_inf_counter += 1
                final_infer_duration = (time.time() - final_infer_time)
                final_infer_time = time.time()
            
            # Inference has finished
            stop_time = time.time()
            
            # Calculate processing time
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()
            
            processing_time = np.mean(processing_times) * 1000 if processing_times else 0
            fps_calc = 1000 / processing_time if processing_time > 0 else 0
            
        else:
            frames_idx.append((counter, frame_counter, 'No'))
        
        # Resize frame for display
        frame = cv2.resize(frame, (620, 350))
        
        # Add text overlays
        for i in range(0, 3):
            display_text = text_template.format(
                label=decoded_labels[i],
                conf=decoded_top_probs[i] * 100,
            )
            frame = display_text_fnc(frame, display_text, i)
        
        display_text = text_inference_template.format(Time=processing_time, fps=fps_calc)
        frame = display_text_fnc(frame, display_text, 3)
        frame = display_text_fnc(frame, f"Infer Count: {final_inf_counter}", 4)
        
        # Write frame to output video
        out.write(frame)
    
    # Update final statistics before finishing
    final_results['inference_stats'] = {
        'total_inferences': final_inf_counter,
        'avg_processing_time_ms': float(processing_time),
        'final_fps': float(fps_calc)
    }
    
    # Clean up
    cap.release()
    out.release()
    
    return output_path, final_results
# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Load models on startup
    load_models()
    app.run(host='0.0.0.0', port=5001, debug=False)