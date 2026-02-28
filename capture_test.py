import cv2
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection.camera import get_camera

def capture_test_image():
    print("Attempting to capture test image...")
    cap = get_camera()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    
    # Warmer cycle
    for _ in range(30):
        cap.read()
        
    ret, frame = cap.read()
    if ret:
        path = "static/test_cap.jpg"
        cv2.imwrite(path, frame)
        print(f"Success! Test image saved to {path}")
        cap.release()
        return True
    else:
        print("Error: Could not read frame.")
        cap.release()
        return False

if __name__ == "__main__":
    capture_test_image()
