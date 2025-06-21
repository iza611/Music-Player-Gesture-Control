import numpy as np
from typing import Tuple, Optional, List
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class MediaPipeHandDetector():
    def __init__(self):
        self.hands = mp_hands.Hands(model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    def detect_keypoints(self, frame: np.array) -> Tuple[np.array, Optional[List]]:
        input_frame = frame.copy()
        input_frame.flags.writeable = False
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(input_frame)

        input_frame.flags.writeable = True
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)

        keypoints = None
        if results.multi_hand_landmarks:
            keypoints = results.multi_hand_landmarks
            for hand_landmarks in keypoints:
                mp_drawing.draw_landmarks(
                    input_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
        return input_frame, keypoints
    
    def close(self):
        self.hands.close()