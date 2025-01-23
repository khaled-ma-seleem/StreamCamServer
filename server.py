import ssl
import cv2
import base64
import numpy as np
import os
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from datetime import datetime
from deepface import DeepFace

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create a directory for saving frames
SAVE_DIR = 'saved_frames'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")

def detect_emotion(frame):
    """
    Detect emotion using DeepFace
    """
    try:
        # Analyze emotions
        result = DeepFace.analyze(frame, 
                                actions=['emotion'],
                                enforce_detection=False,
                                silent=True)
        
        if isinstance(result, list):
            result = result[0]
            
        emotion = result['dominant_emotion']
        return emotion.title()
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return "No face detected"

def add_text_to_frame(frame, text=None):
    """
    Add text overlay to the video frame
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White color
    
    # Add timestamp at the top left
    cv2.putText(frame, timestamp, (10, 30), font, font_scale, color, thickness)
    
    # Add emotion text if provided
    if text:
        # Position the text at the bottom of the frame
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2  # Center horizontally
        text_y = height - 20  # 20 pixels from bottom
        
        # Add dark background for better readability
        bg_padding = 10
        cv2.rectangle(frame, 
                     (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                     (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Error: Failed to decode frame")
            return
        
        # Detect emotion
        emotion = detect_emotion(frame)
        
        # Add text overlay with emotion
        frame = add_text_to_frame(frame, emotion)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        # Save the processed frame
        success = cv2.imwrite(filepath, frame)
        
        if success:
            print(f"Successfully saved frame as: {filepath}")
            
            # Convert the processed frame back to base64 to send to client
            _, buffer = cv2.imencode('.jpg', frame)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('processed_frame', f'data:image/jpeg;base64,{processed_frame}')
        else:
            print(f"Failed to save frame: {filepath}")
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

if __name__ == '__main__':
    # Define paths to your SSL certificate and key
    cert_file = 'certificates/certificate.crt'
    key_file = 'certificates/private.key'

    # Create an SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    print("Starting server with DeepFace emotion detection...")
    
    # Use Werkzeug server with SSL
    socketio.run(app, host='0.0.0.0', port=5000, ssl_context=ssl_context)
