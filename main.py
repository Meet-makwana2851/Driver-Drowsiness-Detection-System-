from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.alarm import AudioAlarm
from src.detection import DrowsinessConfig, DrowsinessDetector
from src.facial_landmarks import FaceMeshAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Driver drowsiness protection system")
    parser.add_argument("--model", type=str, default="models/drowsiness_model.keras", help="Path to trained model.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--alarm", type=str, default="assets/alarm.wav", help="Alarm sound file.")
    parser.add_argument("--show-landmarks", action="store_true", help="Display facial landmarks.")
    return parser.parse_args()


def draw_status(frame, status, confidence, ear, mar, yawn_count, alarm_on, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w - 10, 155), (0, 0, 0), -1)
    cv2.putText(frame, "DRIVER DROWSINESS PROTECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.3f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Yawns: {yawn_count}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Alarm: {'ON' if alarm_on else 'OFF'}", (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if alarm_on else (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Warning banner when drowsy
    if status == "DROWSY":
        cv2.putText(frame, "WARNING: DRIVER DROWSINESS DETECTED!", (60, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif status == "WARNING":
        cv2.putText(frame, "WARNING: MONITOR DRIVER", (80, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "SYSTEM ACTIVE", (150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first using src/train.py.")

    detector = DrowsinessDetector(model_path=model_path)
    alarm = AudioAlarm(args.alarm)
    face_mesh_analyzer = FaceMeshAnalyzer()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}. Check the camera is connected.")

    yawn_count = 0
    fps_window = []
    last_yawn_state = False

    try:
        while True:
            start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam.")

            frame = cv2.flip(frame, 1)
            landmark_result = face_mesh_analyzer.detect(frame)
            if landmark_result is not None:
                result = detector.analyze_frame(frame, landmark_result, face_mesh_analyzer)
                status = result["status"]
                confidence = result["confidence"]
                ear = result["ear"]
                mar = result["mar"]
                alarm_on = result["alarm"]

                if result["yawn"] and not last_yawn_state:
                    yawn_count += 1
                last_yawn_state = result["yawn"]

                if alarm_on:
                    alarm.play()
                else:
                    alarm.stop()
                if args.show_landmarks:
                    frame = face_mesh_analyzer.draw_landmarks(frame, landmark_result)
            else:
                status = "ALERT"
                confidence = 0.0
                ear = 0.0
                mar = 0.0
                alarm_on = False
                alarm.stop()

            fps = 1.0 / max((time.perf_counter() - start), 1e-6)
            fps_window.append(fps)
            if len(fps_window) > 20:
                fps_window.pop(0)
            avg_fps = float(np.mean(fps_window)) if fps_window else fps

            draw_status(frame, status, confidence, ear, mar, yawn_count, alarm_on, avg_fps)
            cv2.imshow("Driver Drowsiness Protection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alarm.stop()


if __name__ == "__main__":
    main()
