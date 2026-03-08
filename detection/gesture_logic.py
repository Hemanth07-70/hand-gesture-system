import numpy as np

def is_distress_signal(landmarks):
    """
    Heuristic for the 'Signal for Help' (Distress Signal).
    1. Palm open, thumb out.
    2. Thumb tucked under fingers.
    3. Fingers closed over thumb.
    
    This function detects the 'final' state (fist with thumb tucked inside).
    landmarks: shape (21, 3) - [x, y, z] normalized.
    """
    if landmarks is None or len(landmarks) < 21:
        return False, 0.0

    # Landmark mapping:
    # 0: Wrist
    # 4: Thumb Tip, 3: Thumb IP, 2: Thumb MCP, 1: Thumb CMC
    # 8: Index Tip, 7: Index DIP, 6: Index PIP, 5: Index MCP
    # 12: Middle Tip, 11: Middle DIP, 10: Middle PIP, 9: Middle MCP
    # 16: Ring Tip, 15: Ring DIP, 14: Ring PIP, 13: Ring MCP
    # 20: Pinky Tip, 19: Pinky DIP, 18: Pinky PIP, 17: Pinky MCP

    # 1. Check if thumb is tucked (Landmark 4 is closer to palm center than other fingers)
    # Palm center approximation: midpoint of MCPs (5, 9, 13, 17)
    mcp_indices = [5, 9, 13, 17]
    palm_center_x = np.mean([landmarks[i][0] for i in mcp_indices])
    palm_center_y = np.mean([landmarks[i][1] for i in mcp_indices])
    
    thumb_tip = landmarks[4]
    
    # Simple check: thumb tip is between pinky and index MCP in X, and close in Y
    # More robust: thumb tip is 'inside' the region formed by fingers
    
    # 2. Check if fingers are closed (Tips 8, 12, 16, 20 are below PIP joints 6, 10, 14, 18)
    # In MediaPipe image coords, Y increases downwards.
    fingers_closed = True
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip][1] < landmarks[pip][1]: # Tip is above PIP (Extended)
            fingers_closed = False
            break
    
    # 3. Check if thumb is 'covered' (Thumb tip Y is greater than finger MCPs but less than finger Tips)
    # Actually, thumb tucked usually means it's pushed towards the pinky side.
    thumb_tucked = landmarks[4][0] > palm_center_x # For right hand (assuming x increases right)
    # This varies by hand side. Let's use distance to palm center.
    
    thumb_dist = np.sqrt((landmarks[4][0] - palm_center_x)**2 + (landmarks[4][1] - palm_center_y)**2)
    
    # Heuristic: Distant from palm is open, close is tucked.
    # But thumb tip distance is hard to normalize without a reference.
    
    # Better Heuristic for "Signal for Help" (Fist with thumb in):
    # - Fingers (8, 12, 16, 20) are CLOSED (Tips below PIP joints).
    # - Thumb (4) is TUCKED (4 is between Landmark 13 and 17 in X).
    
    is_fist = fingers_closed
    
    # Check if thumb is inside the fist
    # Thumb tip (4) should be closer to middle/ring MCPs (9, 13) than its own MCP (2)
    dist_thumb_to_palm = np.linalg.norm(landmarks[4] - landmarks[9])
    dist_thumb_to_origin = np.linalg.norm(landmarks[4] - landmarks[2])
    
    # If fingers are closed and thumb is tucked
    if is_fist and (landmarks[4][1] > landmarks[5][1]): # Thumb tip below Index MCP
        # Confidence score could be based on how tightly closed it is
        return True, 0.9
        
    return False, 0.0
