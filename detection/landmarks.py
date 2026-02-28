"""Extract hand landmarks using MediaPipe."""
import cv2
import mediapipe as mp
import numpy as np


class HandLandmarker:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """Return list of hand landmark arrays (21 x 3) per hand, or [] if none."""
        results = self.hands.process(frame_rgb)
        out = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                arr = []
                for lm in hand_landmarks.landmark:
                    arr.append([lm.x, lm.y, lm.z])
                out.append(np.array(arr, dtype=np.float32))
        return out

    def draw_landmarks(self, frame, hand_landmarks_list):
        """Draw landmarks on BGR frame. hand_landmarks_list from process (raw coords 0-1)."""
        if not hand_landmarks_list:
            return frame
        h, w = frame.shape[:2]
        mp_draw = mp.solutions.drawing_utils
        mp_hand = self.mp_hands.HandLandmark
        for hand_landmarks in hand_landmarks_list:
            # Convert to pixel coords for drawing
            pts = []
            for lm in hand_landmarks:
                x, y = int(lm[0] * w), int(lm[1] * h)
                pts.append((x, y))
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)
            # Draw connections (simplified: thumb, index, etc.)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # index
                (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                (5, 9), (9, 13), (13, 17),
            ]
            for i, j in connections:
                if i < len(pts) and j < len(pts):
                    cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
        return frame

    def close(self):
        self.hands.close()
