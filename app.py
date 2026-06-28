from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import joblib
from datetime import datetime
import threading
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# --- LOAD MODEL AND SCALER ---
try:
    model = joblib.load('Dataset/drowsiness_model.pkl')
    scaler = joblib.load('Dataset/scaler.pkl')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# --- CONFIGURATION ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Alert settings
DROWSY_THRESHOLD = 3
drowsy_counter = 0
alert_active = False

# Thresholds
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.55

# Global state
detection_state = {
    'status': 'No Face Detected',
    'ear': 0.25,
    'mar': 0.3,
    'drowsy_counter': 0,
    'is_drowsy': False,
    'ml_probability': 0.0,
    'drowsy_reason': '',
    'fps': 0,
    'alert_triggered': False,
    'timestamp': datetime.now().isoformat()
}

def calculate_ear_simple(eye_region):
    if eye_region is None or eye_region.size == 0:
        return 0.25
    height, width = eye_region.shape[:2]
    if width == 0:
        return 0.25
    return height / width

def calculate_mar_simple(mouth_region):
    if mouth_region is None or mouth_region.size == 0:
        return 0.3
    height, width = mouth_region.shape[:2]
    if width == 0:
        return 0.3
    return height / width

def process_frame(frame):
    global drowsy_counter, alert_active
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    detection_state['status'] = 'No Face Detected'
    detection_state['ear'] = 0.25
    detection_state['mar'] = 0.3
    detection_state['is_drowsy'] = False
    detection_state['drowsy_reason'] = ''
    detection_state['ml_probability'] = 0.0
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_roi = gray[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(face_roi[0:h//2, :], 1.1, 3)
        
        ear_values = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            ear_values.append(calculate_ear_simple(eye_region))
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
        
        if len(ear_values) > 0:
            ear = np.mean(ear_values)
        else:
            ear = 0.25
        
        mouth_region = face_roi[h//2:, :]
        mar = calculate_mar_simple(mouth_region)
        
        ear = max(0.1, min(0.5, ear))
        mar = max(0.1, min(1.0, mar))
        
        detection_state['ear'] = float(ear)
        detection_state['mar'] = float(mar)
        
        if model and scaler:
            features = scaler.transform([[ear, mar]])
            ml_prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            detection_state['ml_probability'] = float(probability[1] * 100)
        else:
            ml_prediction = 0
        
        eyes_closed = ear < EAR_THRESHOLD
        mouth_open = mar > MAR_THRESHOLD
        
        is_drowsy = ml_prediction == 1 or eyes_closed or mouth_open
        
        if is_drowsy:
            drowsy_counter += 1
            
            reasons = []
            if eyes_closed:
                reasons.append("EYES CLOSED")
            if mouth_open:
                reasons.append("YAWNING")
            if ml_prediction == 1 and not reasons:
                reasons.append("ML DETECTED")
            
            drowsy_reason = " + ".join(reasons) if reasons else "DROWSY"
            detection_state['drowsy_reason'] = drowsy_reason
            detection_state['status'] = f"DROWSY! ({drowsy_reason})"
            detection_state['is_drowsy'] = True
            color = (0, 0, 255)
        else:
            drowsy_counter = 0
            detection_state['status'] = "ALERT"
            detection_state['is_drowsy'] = False
            alert_active = False
            color = (0, 255, 0)
        
        detection_state['drowsy_counter'] = drowsy_counter
        
        if drowsy_counter >= DROWSY_THRESHOLD and not alert_active:
            alert_active = True
            detection_state['alert_triggered'] = True
        else:
            detection_state['alert_triggered'] = False
        
        status_box_height = 50
        cv2.rectangle(frame, (x, y-status_box_height), (x+w, y), color, -1)
        cv2.putText(frame, detection_state['status'], (x+5, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        metrics_y = y + h + 30
        ear_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        mar_color = (0, 0, 255) if mouth_open else (0, 255, 0)
        
        cv2.putText(frame, f"EAR: {ear:.3f}", (x, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (x, metrics_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mar_color, 2)
        cv2.putText(frame, f"ML: {detection_state['ml_probability']:.1f}%", (x, metrics_y+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        break
    
    detection_state['timestamp'] = datetime.now().isoformat()
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame)
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n' 
               + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection-state')
def get_detection_state():
    return jsonify(detection_state)

@app.route('/api/start')
def start_detection():
    return jsonify({'status': 'Detection started'})

@app.route('/api/stop')
def stop_detection():
    return jsonify({'status': 'Detection stopped'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
