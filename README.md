# Driver Drowsiness Protection System

A real-time driver drowsiness monitoring prototype built with Python, OpenCV, MediaPipe, and TensorFlow/Keras. The system checks eye closure, yawning, and model confidence to detect drowsiness and trigger an audible alarm.

## Project structure

- `src/preprocess.py` – dataset handling, augmentation, and image preprocessing
- `src/facial_landmarks.py` – EAR/MAR calculations and face landmark utilities
- `src/train.py` – CNN model construction and training
- `src/detection.py` – temporal drowsiness logic and inference helpers
- `src/alarm.py` – audio warning manager
- `main.py` – real-time webcam detection application
- `evaluate.py` – model evaluation on the test split
- `dataset/` – train/validation/test images
- `models/` – saved trained model
- `assets/` – alarm sound file

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. For Apple Silicon macOS, install the Metal-enabled TensorFlow build instead:

```bash
python3 -m pip install tensorflow-macos==2.16.2 tensorflow-metal==1.1.0
```

3. Prepare the dataset.

4. Train the model:

```bash
python src/train.py --data-root dataset --epochs 25 --batch-size 32 --output-model models/drowsiness_model.keras
```

This saves the trained model and generates evaluation graphs in the `reports/` folder.

## Model evaluation

```bash
python evaluate.py --model models/drowsiness_model.keras --test-dir dataset/test
```

The evaluation script prints accuracy, precision, recall, F1-score, and confusion matrix metrics. It also saves a confusion matrix plot and reports.

## Real-time system

```bash
python main.py --model models/drowsiness_model.keras --camera-index 0 --alarm assets/alarm.wav
```

Press `q` to quit the webcam window.

## Notes

- The system uses EAR and MAR to reduce false alarms and to trigger only after several consecutive frames.
- Thresholds are configurable in `src/detection.py`.
- This is a research prototype for academic/experimental use only, not a certified automotive safety device.

