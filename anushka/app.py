from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import tempfile
import mimetypes
from google.api_core import retry

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from frontend on port 3000

# Set up your API key
# Replace with your actual API key or set as environment variable
api_key = os.environ.get('GOOGLE_API_KEY', 'AIzaSyBunysL99MXlydkvuDmFWOwcgNKA-fcZX4')
genai.configure(api_key=api_key)

# Load Gemini Pro Vision model
model = genai.GenerativeModel('gemini-2.0-flash')

def get_video_description(video_data, mime_type="video/mp4"):
    """
    Send a video file to Gemini and get description of activities
    
    Args:
        video_data: Binary data of the video file
        mime_type: MIME type of the video
    
    Returns:
        String description of activities in the video
    """
    try:
        # Check file size - Gemini has limits
        file_size = len(video_data) / (1024 * 1024)  # Size in MB
        if file_size > 20:  # Adjust based on actual Gemini limits
            return f"Error: File size ({file_size:.1f}MB) exceeds limit"
        
        # Create prompt for video analysis
        prompt = """
        Reply in 15-20 words for any activity is happening like robbery or burglary or explosion or any of these 'Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting'
        """
        
        # Create a Part object for the video
        video_part = {
            "mime_type": mime_type,
            "data": video_data
        }
        
        # Make request with retry for transient errors
        @retry.Retry(predicate=retry.if_transient_error)
        def generate_with_retry():
            response = model.generate_content([prompt, video_part])
            return response.text
        
        # Get response
        description = generate_with_retry()
        return description
        
    except Exception as e:
        return f"Error processing video: {str(e)}"

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    try:
        # Check if the request has the file part
        if 'video' not in request.files:
            return jsonify({'error': 'No video file in request'}), 400
        
        video_file = request.files['video']
        
        # If user doesn't select a file, browser might send an empty file
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save the uploaded file to a temporary location
        video_data = video_file.read()
        
        # Get the MIME type
        mime_type = video_file.content_type or "video/mp4"
        
        # Process the video
        result = get_video_description(video_data, mime_type)
        
        return jsonify({
            'description': result,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)