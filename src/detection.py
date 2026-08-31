from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import keras
import numpy as np

from src.facial_landmarks import FaceMeshAnalyzer, extract_region


def compute_alert_score(probabilities, closed_class_index=1, yawn_class_index=2):
    return float(max(probabilities[closed_class_index], probabilities[yawn_class_index]))


@dataclass
class DrowsinessConfig:
    ear_threshold: float = 0.25
    ear_warning_threshold: float = 0.30
    mar_threshold: float = 0.58
    eye_closed_frames_required: int = 3
    yawn_frames_required: int = 2
    warning_frames_required: int = 6
    drowsy_frames_required: int = 10
    alert_reset_frames: int = 8
    model_confidence_threshold: float = 0.70
    closed_class_index: int = 0
    yawn_class_index: int = 2


@dataclass
class DrowsinessState:
    eye_closed_frames: int = 0
    yawn_frames: int = 0
    warning_frames: int = 0
    drowsy_frames: int = 0
    alert_frames: int = 0
    status: str = "ALERT"
    last_prediction: str = "ALERT"
    confidence: float = 0.0


class DrowsinessDetector:
    def __init__(self, model_path: str | Path, config: DrowsinessConfig | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.config = config or DrowsinessConfig()
        self.model = keras.models.load_model(str(self.model_path))
        self.class_names = sorted([entry.name for entry in self.model_path.parent.parent.joinpath("dataset", "train").iterdir() if entry.is_dir()]) if self.model_path.parent.parent.joinpath("dataset", "train").exists() else ["Closed_Eyes", "No_yawn", "Open_Eyes", "Yawn"]
        self.config.closed_class_index = self.class_names.index("Closed_Eyes") if "Closed_Eyes" in self.class_names else 0
        self.config.yawn_class_index = self.class_names.index("Yawn") if "Yawn" in self.class_names else 3
        self.state = DrowsinessState()

    def _predict_image(self, image: np.ndarray):
        if image is None or image.size == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        preprocessed = image.astype(np.float32) / 255.0
        if preprocessed.ndim == 3:
            preprocessed = np.expand_dims(preprocessed, axis=0)
        prediction = self.model.predict(preprocessed, verbose=0)
        return prediction[0]

    def analyze_frame(self, frame, landmarks, face_mesh_analyzer):
        if landmarks is None:
            return {
                "status": "ALERT",
                "confidence": 0.0,
                "ear": 0.0,
                "mar": 0.0,
                "eye_closed": False,
                "yawn": False,
                "alarm": False,
                "prediction": "Unknown",
            }

        ear = face_mesh_analyzer.compute_eye_aspect_ratio(landmarks)
        mar = face_mesh_analyzer.compute_mouth_aspect_ratio(landmarks)

        eye_region = face_mesh_analyzer.extract_eye_region(frame, landmarks)
        mouth_region = face_mesh_analyzer.extract_mouth_region(frame, landmarks)

        eye_prediction = self._predict_image(eye_region) if eye_region is not None else np.array([1.0, 0.0, 0.0, 0.0])
        mouth_prediction = self._predict_image(mouth_region) if mouth_region is not None else np.array([1.0, 0.0, 0.0, 0.0])

        eye_top_class = np.argmax(eye_prediction)
        mouth_top_class = np.argmax(mouth_prediction)
        eye_confidence = float(np.max(eye_prediction))
        mouth_confidence = float(np.max(mouth_prediction))

        eye_closed = ear < self.config.ear_threshold or eye_top_class == self.config.closed_class_index
        yawn = mar > self.config.mar_threshold or mouth_top_class == self.config.yawn_class_index

        if eye_closed:
            self.state.eye_closed_frames += 1
        else:
            self.state.eye_closed_frames = max(0, self.state.eye_closed_frames - 1)

        if yawn:
            self.state.yawn_frames += 1
        else:
            self.state.yawn_frames = max(0, self.state.yawn_frames - 1)

        warning_score = max(eye_confidence, mouth_confidence)
        drowsy_score = compute_alert_score(eye_prediction, self.config.closed_class_index, self.config.yawn_class_index)
        drowsy_score = max(drowsy_score, compute_alert_score(mouth_prediction, self.config.closed_class_index, self.config.yawn_class_index))

        if eye_closed or yawn:
            self.state.warning_frames += 1
        else:
            self.state.warning_frames = max(0, self.state.warning_frames - 1)

        if eye_closed or yawn:
            self.state.drowsy_frames += 1
        else:
            self.state.drowsy_frames = max(0, self.state.drowsy_frames - 1)

        if self.state.eye_closed_frames >= self.config.eye_closed_frames_required or self.state.yawn_frames >= self.config.yawn_frames_required:
            self.state.status = "WARNING"
            self.state.last_prediction = "WARNING"
        if (
            self.state.eye_closed_frames >= self.config.drowsy_frames_required
            or self.state.yawn_frames >= self.config.drowsy_frames_required
            or (self.state.warning_frames >= self.config.warning_frames_required and drowsy_score >= self.config.model_confidence_threshold)
        ):
            self.state.status = "DROWSY"
            self.state.last_prediction = "DROWSY"

        if self.state.status == "ALERT" and (eye_closed or yawn):
            self.state.alert_frames = 0

        if self.state.status != "DROWSY" and not eye_closed and not yawn:
            self.state.status = "ALERT"
            self.state.last_prediction = "ALERT"

        confidence = max(float(eye_confidence), float(mouth_confidence), float(drowsy_score))
        self.state.confidence = confidence

        alarm_on = self.state.status == "DROWSY"
        if alarm_on:
            self.state.warning_frames = max(self.state.warning_frames, self.config.warning_frames_required)

        return {
            "status": self.state.status,
            "confidence": float(confidence * 100.0),
            "ear": float(ear),
            "mar": float(mar),
            "eye_closed": bool(eye_closed),
            "yawn": bool(yawn),
            "alarm": bool(alarm_on),
            "prediction": self.class_names[np.argmax(eye_prediction)],
            "eye_prediction": eye_prediction,
            "mouth_prediction": mouth_prediction,
        }
