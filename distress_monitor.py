import cv2
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from detection.person_tracker import PersonTracker
from detection.landmarks import HandLandmarker
from detection.gesture_logic import is_distress_signal
from detection.verification import VerificationEngine
from alerts.notifier import AlertEngine
from config import ALERT_EMAIL_SENDER, ALERT_EMAIL_RECEIVER, ALERT_EMAIL_PASSWORD

def main():
    # Initialize components
    print("Initializing System...")
    tracker = PersonTracker()
    landmarker = HandLandmarker(max_num_hands=2)
    verifier = VerificationEngine(threshold_count=3, time_window=20, min_confidence=0.85)
    notifier = AlertEngine(
        sender_email=ALERT_EMAIL_SENDER,
        receiver_email=ALERT_EMAIL_RECEIVER,
        password=ALERT_EMAIL_PASSWORD
    )
    
    # Video Input
    cap = cv2.VideoCapture(0)
    
    # Window setup
    cv2.namedWindow("Distress Detection System", cv2.WINDOW_NORMAL)
    
    print("System Ready. Monitoring...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        
        # 1. Track Persons
        results, persons = tracker.track(frame)
        
        # 2. Extract Hand Landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = landmarker.process(rgb_frame)
        
        # 3. Process each hand and associate with person
        distress_triggered = False
        
        if hands:
            for hand_lms in hands:
                # Get wrist landmark in pixel coords
                wrist = hand_lms[0]
                wrist_px = (int(wrist[0] * w), int(wrist[1] * h))
                
                # Find which person this hand belongs to
                assigned_id = -1
                for person in persons:
                    bx = person["box"]
                    # Check if wrist is inside the person's bounding box
                    if bx[0] <= wrist_px[0] <= bx[2] and bx[1] <= wrist_px[1] <= bx[3]:
                        assigned_id = person["id"]
                        break
                
                # Draw hand landmarks
                frame = landmarker.draw_landmarks(frame, [hand_lms])
                
                if assigned_id != -1:
                    # 4. Classify Gesture
                    is_distress, confidence = is_distress_signal(hand_lms)
                    
                    # 5. Verify & Alert
                    alert_ready, msg = verifier.update(assigned_id, is_distress, confidence)
                    
                    if alert_ready:
                        notifier.trigger(frame, msg)
                        distress_triggered = True
                        # Draw alert on frame
                        cv2.putText(frame, "!!! DISTRESS ALERT !!!", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 6. Annotate Frame with Person Tracking
        for person in persons:
            bx = person["box"]
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Person {person['id']}", (bx[0], bx[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display output
        cv2.imshow("Distress Detection System", frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
