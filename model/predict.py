"""Load trained model and predict gesture from landmark features."""
import pickle
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_MODEL_PATH, LABEL_ENCODER_PATH, GESTURE_EMOJI_MAP, GESTURE_DISPLAY_NAMES


def load_model(model_path=None, label_encoder_path=None):
    model_path = model_path or DEFAULT_MODEL_PATH
    label_encoder_path = label_encoder_path or LABEL_ENCODER_PATH
    if not model_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    encoder = None
    if label_encoder_path.exists():
        with open(label_encoder_path, "rb") as f:
            encoder = pickle.load(f)
    return model, encoder


def predict_gesture(model, encoder, feature_vector: np.ndarray):
    """
    feature_vector: shape (63,) from preprocess.landmarks_to_features
    Returns: (label_str, confidence, emoji) or (None, 0.0, "👋") if no model.
    """
    if model is None:
        return None, 0.0, "👋"
    X = np.array([feature_vector])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        idx = np.argmax(probs)
        conf = float(probs[idx])
        if encoder is not None:
            label = encoder.inverse_transform([idx])[0]
        else:
            label = model.classes_[idx] if hasattr(model, "classes_") else str(idx)
    else:
        pred = model.predict(X)[0]
        conf = 1.0
        label = encoder.inverse_transform([pred])[0] if encoder is not None else str(pred)
    emoji = GESTURE_EMOJI_MAP.get(label, "👋")
    display = GESTURE_DISPLAY_NAMES.get(label, label)
    return display, conf, emoji
