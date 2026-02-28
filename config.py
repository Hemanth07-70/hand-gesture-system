"""Configuration for Hand Gesture Monitoring System."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Default model path (landmark-based classifier)
DEFAULT_MODEL_PATH = MODEL_DIR / "gesture_classifier.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# MediaPipe / camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Training
MIN_SAMPLES_PER_CLASS = 200
MAX_SAMPLES_PER_CLASS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Gesture → Emoji mapping (including happy, sad, crying)
GESTURE_EMOJI_MAP = {
    "thumbs_up": "👍",
    "thumbs_down": "👎",
    "peace": "✌️",
    "ok": "👌",
    "wave": "👋",
    "happy": "😊",
    "sad": "😢",
    "crying": "😭",
    "rock": "🤘",
    "fist": "✊",
    "open_palm": "🖐️",
}

# Display names for UI
GESTURE_DISPLAY_NAMES = {
    "thumbs_up": "Thumbs Up",
    "thumbs_down": "Thumbs Down",
    "peace": "Peace",
    "ok": "OK",
    "wave": "Wave",
    "happy": "Happy",
    "sad": "Sad",
    "crying": "Crying",
    "rock": "Rock",
    "fist": "Fist",
    "open_palm": "Open Palm",
}

# Auth (simple file-based for demo)
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-hand-gesture")
USERS_FILE = BASE_DIR / "data" / "users.json"
