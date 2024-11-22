from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
from datetime import datetime
import json
from PIL import Image
import io
import requests
import socket
import base64
from werkzeug.utils import secure_filename
import time
import hashlib

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
    OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change this in production

app = Flask(__name__)
app.config.from_object(Config)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_ollama_url():
    """Get Ollama server URL from environment variables"""
    return f"http://{app.config['OLLAMA_HOST']}:{app.config['OLLAMA_PORT']}"

def is_valid_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return False, "Unable to read image file"
        
        # Get original dimensions
        height, width = image.shape[:2]
        print(f"Original image dimensions: {width}x{height}")
        
        # If image is small, resize it to minimum dimensions
        min_dimension = 100  # Reduced minimum dimension
        if height < min_dimension or width < min_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            if width < height:
                new_width = min_dimension
                new_height = int(height * (min_dimension / width))
            else:
                new_height = min_dimension
                new_width = int(width * (min_dimension / height))
            
            # Resize image
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"Resized image to: {new_width}x{new_height}")
            
            # Save the resized image
            cv2.imwrite(image_path, image)
            
            # Update dimensions
            height, width = image.shape[:2]

        # Convert to grayscale for blur detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate the Laplacian variance (less strict threshold)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 50  # Reduced from previous value for more tolerance
        
        if laplacian_var < blur_threshold:
            print(f"Image blur score: {laplacian_var} (threshold: {blur_threshold})")
            return False, "Image is too blurry. Please upload a clearer photo."

        # Basic content validation (less strict)
        # Check if the image has enough green or plant-like colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for plant colors (broader range)
        lower_green = np.array([20, 20, 20])  # More tolerant lower bound
        upper_green = np.array([180, 255, 255])  # Broader upper bound
        
        # Create a mask for plant-like colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate the percentage of plant-like pixels
        plant_pixel_ratio = np.count_nonzero(mask) / (height * width)
        
        # More tolerant threshold for plant content
        if plant_pixel_ratio < 0.1:  # Reduced from previous value
            print(f"Plant pixel ratio: {plant_pixel_ratio}")
            return False, "Image doesn't appear to contain enough plant content"

        return True, "Image is valid"
    except Exception as e:
        print(f"Error in image validation: {str(e)}")
        return False, str(e)

def get_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_duplicate_image(file_path, history_data):
    """Check if image is already in history based on file hash"""
    new_file_hash = get_file_hash(file_path)
    
    # Load existing history
    try:
        with open('history.json', 'r') as f:
            history = json.load(f)
            for item in history:
                if 'file_hash' in item and item['file_hash'] == new_file_hash:
                    return True
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return False

def make_ollama_request(url, data, max_retries=3, timeout=120):
    """Make a request to Ollama with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}")
            response = requests.post(
                url,
                json=data,
                timeout=timeout
            )
            return response
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:  # Last attempt
                raise
            print(f"Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(5)  # Wait 5 seconds before retrying
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:  # Last attempt
                raise
            print(f"Connection error on attempt {attempt + 1}, retrying...")
            time.sleep(5)
    raise Exception("Max retries exceeded")

def analyze_image():
    try:
        # Debug logging
        print("Request Files:", request.files)
        print("Request Form:", request.form)
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PNG and JPEG images are allowed'}), 400

        # Generate a unique filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Ensure uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file temporarily
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500

        # Check for duplicate before proceeding
        history_file = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
        if is_duplicate_image(filepath, history_file):
            os.remove(filepath)  # Remove the duplicate file
            return jsonify({'error': 'This image has already been analyzed'}), 400

        # Validate image with more lenient checks
        is_valid, message = is_valid_image(filepath)
        if not is_valid:
            os.remove(filepath)
            print(f"Invalid image: {message}")
            return jsonify({'error': message}), 400

        try:
            # Prepare the image for analysis
            with open(filepath, 'rb') as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')

            prompt = """Analyze this plant image in detail. If you see both leaves and fruits/vegetables:
1. Identify the plant species
2. Describe the condition of both the leaves and any fruits/vegetables present
3. Note any visible signs of disease or stress
4. Provide a diagnosis if issues are found
5. Suggest treatment steps

Keep the analysis concise but thorough. Focus on the most important observations."""

            # Use configured Ollama URL with increased timeout and retry logic
            ollama_url = f"{get_ollama_url()}/api/generate"
            print(f"Calling Ollama at: {ollama_url}")
            
            response = make_ollama_request(
                ollama_url,
                {
                    'model': 'llava',
                    'prompt': prompt,
                    'stream': False,
                    'images': [base64_image]
                },
                max_retries=3,
                timeout=120  # Increased timeout to 120 seconds
            )
            
            print(f"Ollama response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Ollama error response: {response.text}")
                raise Exception(f"Ollama API error: {response.text}")

            # Parse the response
            result = response.json()
            if 'response' not in result:
                print("Invalid Ollama response format:", result)
                raise Exception("Invalid response format from Ollama")

            analysis_text = result['response'].strip()
            print("Analysis completed successfully")

            # Format the analysis text without the AI header
            analysis_with_disclaimer = f"""{analysis_text}

---
Note: This analysis is provided as a general guide only. Please consult with a plant health professional for critical decisions."""

            # Extract plant identification for history
            plant_name = "Unknown Plant"
            lines = analysis_text.split('\n')
            for line in lines:
                if "Plant identification:" in line.lower() or "plant:" in line.lower():
                    plant_name = line.split(':', 1)[1].strip()
                    break

            # Save to history
            history_item = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': unique_filename,
                'analysis': analysis_with_disclaimer,
                'plant_name': plant_name,
                'file_hash': get_file_hash(filepath)
            }
            
            save_to_history(history_item)

            return jsonify({
                'success': True,
                'analysis': analysis_with_disclaimer,
                'history_item': history_item
            })

        except requests.exceptions.ConnectionError:
            print("Connection error to Ollama server")
            return jsonify({'error': 'Could not connect to the analysis server. Please ensure Ollama is running.'}), 503
        except requests.exceptions.Timeout:
            print("Timeout error from Ollama server")
            return jsonify({'error': 'Analysis server timeout. Please try again.'}), 504
        except Exception as e:
            error_msg = str(e)
            print(f"Error in analyze_image: {error_msg}")
            return jsonify({'error': f'Analysis failed: {error_msg}'}), 500

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    return analyze_image()

@app.route('/api/history', methods=['GET'])
def get_history():
    history_file = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                # Return history in reverse chronological order
                return jsonify(list(reversed(history)))
        return jsonify([])
    except Exception as e:
        print(f"Error reading history: {str(e)}")
        return jsonify([])

@app.route('/api/history/delete/<timestamp>', methods=['DELETE'])
def delete_history_item(timestamp):
    try:
        history_file = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            updated_history = [item for item in history if item['timestamp'] != timestamp]
            with open(history_file, 'w') as f:
                json.dump(updated_history, f)
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "History file not found"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def save_to_history(history_item):
    history_file = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
    try:
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Add new item to history
        history.append(history_item)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
            
    except Exception as e:
        print(f"Error saving to history: {str(e)}")

if __name__ == '__main__':
    # In production, use environment variables for host and port
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)
