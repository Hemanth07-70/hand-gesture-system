import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection.pipeline import GesturePipeline
import logging

logging.basicConfig(level=logging.INFO)

def test_pipeline():
    print("Initializing pipeline...")
    try:
        pipeline = GesturePipeline()
        print("Pipeline initialized.")
        ret, label, conf, emoji = pipeline.read_frame()
        if ret is not None:
            print(f"Frame read successfully. Label: {label}, Conf: {conf}")
        else:
            print("Failed to read frame.")
        pipeline.release()
    except Exception as e:
        print(f"Error during pipeline test: {e}")

if __name__ == "__main__":
    test_pipeline()
