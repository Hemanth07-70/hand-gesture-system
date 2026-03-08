import numpy as np
from detection.gesture_logic import is_distress_signal

def test_gestures():
    # Mock landmarks (21, 3)
    # Simple mock: all 0s
    lms_none = np.zeros((21, 3))
    print(f"None: {is_distress_signal(lms_none)}")

    # Mock Open Palm (Fingers up)
    lms_open = np.zeros((21, 3))
    for i in range(21):
        lms_open[i] = [0.5, 0.5 - (i*0.01), 0.0] # Y decreases for tips (upwards)
    print(f"Open Palm: {is_distress_signal(lms_open)}")

    # Mock Distress Signal (Fingers down, thumb in)
    lms_distress = np.zeros((21, 3))
    # MCPs at 0.5
    for mcp in [5, 9, 13, 17]:
        lms_distress[mcp] = [0.5 + (mcp-11)*0.02, 0.5, 0.0]
    # Tips below PIPs (Y increases downwards in MP)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lms_distress[pip] = [0.5, 0.6, 0.0]
        lms_distress[tip] = [0.5, 0.7, 0.0]
    # Thumb tucked
    lms_distress[4] = [0.55, 0.55, 0.0] # below index MCP
    
    print(f"Distress Signal: {is_distress_signal(lms_distress)}")

if __name__ == "__main__":
    test_gestures()
