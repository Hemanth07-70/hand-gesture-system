"""Train gesture classifier from collected landmark dataset."""
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR, DEFAULT_MODEL_PATH, LABEL_ENCODER_PATH, TEST_SIZE, RANDOM_STATE


def load_dataset():
    """Load dataset from dataset/*.npz (each file: X, y)."""
    X_list, y_list = [], []
    for npz_path in DATASET_DIR.glob("*.npz"):
        data = np.load(npz_path, allow_pickle=True)
        X_list.append(data["X"])
        y_list.append(data["y"])
    if not X_list:
        return None, None
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def train():
    """Train Random Forest and save model + label encoder."""
    try:
        X, y = load_dataset()
        if X is None or len(X) < 10:
            return False, "Not enough data. Collect samples for at least 2 gestures."
        
        classes = np.unique(y)
        if len(classes) < 2:
            return False, f"Only found one gesture ({classes[0]}). Collect samples for at least one more gesture (e.g., thumbs_down) to train a classifier."

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
        )
        
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        
        DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(le, f)
            
        return True, f"Model trained! Accuracy: {acc:.2%}. Ready to detect!"
    except Exception as e:
        return False, f"Training failed: {str(e)}"
