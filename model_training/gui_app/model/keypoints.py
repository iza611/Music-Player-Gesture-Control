import numpy as np
from typing import Tuple
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class MediaPipeHandDetector():
    def __init__(self):
        self.hands = mp_hands.Hands(model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    def detect_keypoints(self, frame: np.array) -> Tuple[np.array, Tuple[landmark_pb2.NormalizedLandmarkList, landmark_pb2.NormalizedLandmarkList], str]:
        input_frame = frame.copy()
        input_frame.flags.writeable = False
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(input_frame)
        self.hands_lm = results.multi_hand_landmarks
        self.handedness = results.multi_handedness

        input_frame.flags.writeable = True
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)

        if self.hands_lm and self.handedness:
            keypoints = self.get_keypoints()
            for hand_landmarks in keypoints:
                mp_drawing.draw_landmarks(
                    input_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            handedness = " ".join(h.classification[0].label for h in self.handedness)
            
        else:
            keypoints = (self.create_dummy_landmark_list(), self.create_dummy_landmark_list())
            handedness = " "
            
        return input_frame, keypoints, handedness
    
    def create_dummy_landmark_list(self) -> landmark_pb2.NormalizedLandmarkList:
        dummy_landmarks = [
            landmark_pb2.NormalizedLandmark(x=0.0, y=0.0, z=0.0)
            for _ in range(21)
    ]
        return landmark_pb2.NormalizedLandmarkList(landmark=dummy_landmarks)
    
    def get_keypoints(self) -> Tuple[landmark_pb2.NormalizedLandmarkList, landmark_pb2.NormalizedLandmarkList]:
        left_hand = self.create_dummy_landmark_list()
        right_hand = self.create_dummy_landmark_list()

        # Make sure only left and right hands, only left hand or only right hand landmarks used
        # In case of eg two left hands or three right, or 4 hands in total, return dummy keypoints filled with zeros 
        label_counts = {"Left": 0, "Right": 0}

        for hand_landmarks, hand_info in zip(self.hands_lm, self.handedness):
            label = hand_info.classification[0].label 
            if label == "Left" and label_counts["Left"] == 0:
                left_hand = hand_landmarks
                label_counts["Left"] += 1
            elif label == "Right" and label_counts["Right"] == 0:
                right_hand = hand_landmarks
                label_counts["Right"] += 1
            else:
                # More than one of same type or more than 2 hands in total — invalidate all
                return (
                    self.create_dummy_landmark_list(),
                    self.create_dummy_landmark_list()
                )
        
        return left_hand, right_hand
        
    def close(self):
        self.hands.close()