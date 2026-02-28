"""Real-time pipeline: webcam → MediaPipe → model → emoji overlay."""
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GESTURE_EMOJI_MAP
from detection.camera import get_camera, frame_to_rgb
from detection.landmarks import HandLandmarker
from data.preprocess import landmarks_to_features
from model.predict import load_model, predict_gesture


class GesturePipeline:
    def __init__(self, camera_index=0):
        self.cap = get_camera(camera_index)
        self.landmarker = HandLandmarker()
        self.model, self.encoder = load_model()
        self._current_emoji = "👋"
        self._current_label = "—"
        self._current_conf = 0.0
        self._last_landmarks = None

    def reload_model(self):
        """Reload the model and encoder from disk."""
        self.model, self.encoder = load_model()

    def read_frame(self):
        """Read one frame, run detection, return BGR frame with overlay and (label, conf, emoji)."""
        if not self.cap or not self.cap.isOpened():
            self.cap = get_camera() # Try to re-initialize
            
        if not self.cap.isOpened():
            return None, "—", 0.0, "👋"

        ret, frame = self.cap.read()
        if not ret:
            return None, "—", 0.0, "👋"
        rgb = frame_to_rgb(frame)
        hands = self.landmarker.process(rgb)
        if hands:
            # Use first hand
            features = landmarks_to_features(hands[0])
            label, conf, emoji = predict_gesture(self.model, self.encoder, features)
            self._current_label = label or "—"
            self._current_conf = conf
            self._current_emoji = emoji
            self._last_landmarks = hands[0].tolist() # Store as list for JSON
            frame = self.landmarker.draw_landmarks(frame, hands)
        else:
            self._current_label = "—"
            self._current_conf = 0.0
            self._current_emoji = "👋"
            self._last_landmarks = None

        # Overlay text
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, self._current_label, (20, h - 40), font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"{self._current_conf:.0%}", (20, h - 10), font, 0.6, (0, 255, 0), 2)
        return frame, self._current_label, self._current_conf, self._current_emoji

    def release(self):
        self.cap.release()
        self.landmarker.close()


# Global pipeline instance for Flask video feed
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = GesturePipeline()
    return _pipeline
