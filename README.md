# Drowsiness Detection System - Frontend

A modern, real-time drowsiness detection web application built with Flask and vanilla JavaScript with a sleek, dark-themed UI.

## Features

✨ **Modern Dark UI** - Responsive, professional interface with smooth animations
📹 **Real-time Video Stream** - Live webcam feed with overlay information
📊 **Live Metrics Display** - Real-time EAR, MAR, and ML probability scores
⚠️ **Smart Alerts** - Audio and visual alerts when drowsiness is detected
📈 **Progress Tracking** - Visual progress bars for all key metrics
🎯 **Threshold Monitoring** - Clear visual indicators when thresholds are exceeded
📱 **Responsive Design** - Works on desktop, tablet, and mobile devices

## Project Structure

```
├── app.py                 # Flask backend API
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main frontend HTML
└── static/
    ├── style.css         # Modern styling with dark theme
    └── script.js         # Frontend logic and real-time updates
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam access
- Modern web browser

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure model files exist:**
   - `Dataset/drowsiness_model.pkl`
   - `Dataset/scaler.pkl`

3. **Run the application:**
```bash
python app.py
```

4. **Open in browser:**
   Navigate to `http://localhost:5000`

## How It Works

### Backend (Flask)
- Captures video from webcam
- Processes each frame to detect faces and eyes
- Calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
- Runs ML model prediction for drowsiness
- Streams video with annotations back to frontend
- Provides detection state via REST API

### Frontend (HTML/CSS/JS)
- Displays live video stream from backend
- Fetches detection metrics every 500ms
- Updates UI with real-time metrics and status
- Triggers audio/visual alerts when drowsiness is detected
- Shows comprehensive metrics dashboard:
  - **EAR Metric**: Eye Aspect Ratio (threshold: 0.21)
  - **MAR Metric**: Mouth Aspect Ratio (threshold: 0.55)
  - **ML Prediction**: Drowsiness probability from ML model
  - **Alert Counter**: Consecutive drowsy frame count

## Key Metrics Explained

### Eye Aspect Ratio (EAR)
- Measures the ratio of eye height to width
- Lower values indicate closed eyes
- **Threshold**: 0.21 (below = eyes likely closed)

### Mouth Aspect Ratio (MAR)
- Measures the ratio of mouth height to width
- Higher values indicate open mouth (yawning)
- **Threshold**: 0.55 (above = mouth likely open)

### ML Probability
- Confidence score from trained machine learning model
- Combines both facial features and pattern recognition
- Provides additional layer of detection accuracy

## Alert System

The system triggers an alert when:
1. Eyes are closed (EAR < 0.21), OR
2. Mouth is open (MAR > 0.55), OR
3. ML model predicts drowsiness with high confidence

**Alert Conditions**: Alert triggers after 3 consecutive frames of detected drowsiness

## API Endpoints

- `GET /` - Main dashboard page
- `GET /video_feed` - Live video stream (MJPEG)
- `GET /api/detection-state` - Current detection metrics (JSON)

## Configuration

Edit thresholds in `app.py`:
```python
DROWSY_THRESHOLD = 3      # Frames before alert
EAR_THRESHOLD = 0.21      # Eye closure threshold
MAR_THRESHOLD = 0.55      # Yawn threshold
```

## Performance

- **Real-time Processing**: 30+ FPS on modern hardware
- **Low Latency**: ~500ms update interval
- **Lightweight**: No heavy dependencies, pure OpenCV-based detection
- **Browser Compatible**: Works on all modern browsers

## Troubleshooting

### "Could not open webcam"
- Check if another application is using your webcam
- Grant browser permission to access camera
- Try a different camera index in `app.py` (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

### No detections showing
- Ensure proper lighting conditions
- Face should be clearly visible to camera
- Check if Haar Cascade files are loaded correctly

### Low FPS
- Close other resource-intensive applications
- Reduce video resolution
- Use GPU acceleration if available

## Training the Model

To train a new drowsiness detection model:
1. Use provided training script: `Dataset/train_model.py`
2. Requires labeled data in `Dataset/drowsy_data.csv`
3. Generates `drowsiness_model.pkl` and `scaler.pkl`

## Future Enhancements

- [ ] Database logging of detection events
- [ ] Daily/weekly reports generation
- [ ] Multi-face detection support
- [ ] Custom alert sounds
- [ ] Settings customization in UI
- [ ] Export detection logs
- [ ] Mobile app support
- [ ] Advanced analytics dashboard

## Notes

- The system uses Haar Cascades for face/eye detection (fast, reliable)
- ML model uses EAR and MAR as primary features
- Audio alerts play system sounds or text-to-speech
- All processing happens in real-time without cloud dependencies

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, ensure:
1. All dependencies are installed correctly
2. Webcam access is properly granted
3. Python version is 3.8+
4. Model files are in correct location
