"""Webcam capture."""
import cv2
import logging
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_camera(index=None):
    """Return OpenCV VideoCapture. index defaults to config CAMERA_INDEX.
    Falls back to other indices if the specified one fails.
    """
    target_index = index if index is not None else CAMERA_INDEX
    
    # Try the target index first
    cap = cv2.VideoCapture(target_index)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            logger.info(f"Camera opened successfully at index {target_index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            return cap
        cap.release()
    
    logger.warning(f"Failed to open camera at index {target_index}. Searching for available cameras...")
    
    # Fallback: Try common indices
    for i in range(5):
        if i == target_index:
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                logger.info(f"Camera found and opened at fallback index {i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                return cap
            cap.release()
    
    logger.error("No working camera found after searching indices 0-4.")
    return cv2.VideoCapture(target_index) # Return the original one even if it's closed, so caller can handle failure

def frame_to_rgb(frame_bgr):
    """Convert BGR to RGB for MediaPipe."""
    if frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
