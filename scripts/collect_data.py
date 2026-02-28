#!/usr/bin/env python3
"""
Standalone data collection script.
Run: python scripts/collect_data.py
Then enter gesture name (e.g. thumbs_up, happy, sad) and press SPACE to capture samples.
Press 'q' to quit.
"""
import cv2
import numpy as np
from pathlib import Path

sys_path = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path))

from config import DATASET_DIR, GESTURE_EMOJI_MAP
from detection.camera import get_camera, frame_to_rgb
from detection.landmarks import HandLandmarker
from data.collect import save_samples
from data.preprocess import landmarks_to_features

def main():
    cap = get_camera()
    landmarker = HandLandmarker()
    gesture_name = input("Gesture label (e.g. thumbs_up, happy, sad, peace): ").strip().lower().replace(" ", "_")
    if not gesture_name:
        print("No label given. Exiting.")
        return
    if gesture_name not in GESTURE_EMOJI_MAP:
        print(f"Unknown gesture. Add to config or use one of: {list(GESTURE_EMOJI_MAP.keys())}")
    buffer = []
    print(f"Collecting '{gesture_name}'. SPACE = capture, Q = quit. Collect 200+ samples.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = frame_to_rgb(frame)
        hands = landmarker.process(rgb)
        if hands:
            frame = landmarker.draw_landmarks(frame, hands)
            if len(buffer) < 30:
                buffer.append(hands[0].copy())
        else:
            buffer.clear()
        cv2.putText(frame, f"{gesture_name} | Buffered: {len(buffer)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if buffer:
                n = save_samples(gesture_name, buffer)
                print(f"Saved {n} samples. Total buffer cleared.")
            buffer.clear()
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Done.")

if __name__ == "__main__":
    main()
