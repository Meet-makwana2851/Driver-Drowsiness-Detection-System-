from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp


LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 133, 173, 162, 160, 159, 158, 157, 173, 133, 33, 246, 161, 160, 159, 158]
MOUTH_INDICES = [61, 291, 0, 17, 314, 405, 311, 291, 270, 409, 267, 0]


def _distance(point_a, point_b):
    return np.linalg.norm(np.asarray(point_a) - np.asarray(point_b))


def get_landmarks(frame, face_mesh, confidence=0.5):
    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark


def compute_ear(landmarks, eye_indices=None):
    if eye_indices is None:
        eye_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES

    points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    if len(points) < 6:
        return 0.0

    vertical_1 = _distance(points[1], points[5])
    vertical_2 = _distance(points[2], points[4])
    horizontal = _distance(points[0], points[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def compute_mar(landmarks, mouth_indices=None):
    if mouth_indices is None:
        mouth_indices = MOUTH_INDICES

    points = [(landmarks[i].x, landmarks[i].y) for i in mouth_indices]
    if len(points) < 5:
        return 0.0

    upper_lower = _distance(points[0], points[6])
    left_right = _distance(points[1], points[7])
    chin = _distance(points[2], points[8])
    mouth_width = _distance(points[3], points[9])
    return (upper_lower + left_right + chin) / (3.0 * mouth_width + 1e-6)


def extract_region(frame, landmarks, indices, padding=20, target_size=(224, 224)):
    h, w = frame.shape[:2]
    coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float32)
    x_min, y_min = coords.min(axis=0).astype(int)
    x_max, y_max = coords.max(axis=0).astype(int)

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    return cv2.resize(roi, target_size)


def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2):
    h, w = frame.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame


class FaceMeshAnalyzer:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        return get_landmarks(frame, self.face_mesh)

    def compute_eye_aspect_ratio(self, landmarks, eye_indices=None):
        return compute_ear(landmarks, eye_indices)

    def compute_mouth_aspect_ratio(self, landmarks, mouth_indices=None):
        return compute_mar(landmarks, mouth_indices)

    def extract_eye_region(self, frame, landmarks, eye_indices=None, padding=15):
        if eye_indices is None:
            eye_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES
        return extract_region(frame, landmarks, eye_indices, padding=padding)

    def extract_mouth_region(self, frame, landmarks, mouth_indices=None, padding=15):
        if mouth_indices is None:
            mouth_indices = MOUTH_INDICES
        return extract_region(frame, landmarks, mouth_indices, padding=padding)

    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0), radius=2):
        return draw_landmarks(frame, landmarks, color=color, radius=radius)
