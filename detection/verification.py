import time

class VerificationEngine:
    def __init__(self, threshold_count=3, time_window=20, min_confidence=0.85):
        """
        threshold_count: Number of times gesture must be detected.
        time_window: Window in seconds to look for repetitions.
        min_confidence: Minimum confidence for each detection.
        """
        self.threshold_count = threshold_count
        self.time_window = time_window
        self.min_confidence = min_confidence
        
        # Structure: { track_id: [timestamp1, timestamp2, ...] }
        self.detections = {}
        # Avoid double-triggering: { track_id: last_alert_time }
        self.last_alert = {}

    def update(self, person_id, is_distress, confidence):
        """
        Update detection history for a person.
        Returns: (is_verified, message)
        """
        current_time = time.time()
        
        # Clean up old detections for this person
        if person_id in self.detections:
            self.detections[person_id] = [t for t in self.detections[person_id] if current_time - t <= self.time_window]
        else:
            self.detections[person_id] = []

        if is_distress and confidence >= self.min_confidence:
            # Check for cooldown (don't alert too frequently for the same person)
            if person_id in self.last_alert and current_time - self.last_alert[person_id] < 60:
                return False, None

            self.detections[person_id].append(current_time)
            
            count = len(self.detections[person_id])
            if count >= self.threshold_count:
                self.last_alert[person_id] = current_time
                self.detections[person_id] = [] # Reset after trigger
                return True, f"ALERT: Person ID {person_id} detected performing distress signal {count} times!"
        
        return False, None
