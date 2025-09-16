import numpy as np
from typing import Tuple
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from dataclasses import dataclass

mp_drawing = mp.solutions.drawing_utils # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
mp_hands = mp.solutions.hands # type: ignore

@dataclass
class Keypoints:
    left_hand: landmark_pb2.NormalizedLandmarkList # type: ignore
    right_hand: landmark_pb2.NormalizedLandmarkList # type: ignore

class MediaPipeHandDetector():
    def __init__(self):
        self.hands = mp_hands.Hands(model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    def detect_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, Keypoints, str]:
        input_frame = frame.copy()
        input_frame.flags.writeable = False
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(input_frame)
        self.hands_lm = results.multi_hand_landmarks
        self.handedness = results.multi_handedness

        input_frame.flags.writeable = True
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)

        if self.hands_lm and self.handedness:
            handedness = " ".join(h.classification[0].label for h in self.handedness)
            keypoints = self._get_keypoints()
            for single_hand_keypoints in (keypoints.left_hand, keypoints.right_hand):
                mp_drawing.draw_landmarks(
                    input_frame,
                    single_hand_keypoints,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
        else:
            handedness = " "
            keypoints = Keypoints(left_hand=self._create_dummy_landmark_list(),
                                      right_hand=self._create_dummy_landmark_list())
            
        return input_frame, keypoints, handedness
    
    def _create_dummy_landmark_list(self) -> landmark_pb2.NormalizedLandmarkList: # type: ignore
        dummy_landmarks = [
            landmark_pb2.NormalizedLandmark(x=0.0, y=0.0, z=0.0) # type: ignore
            for _ in range(21)
    ]
        return landmark_pb2.NormalizedLandmarkList(landmark=dummy_landmarks) # type: ignore
    
    def _get_keypoints(self) -> Keypoints:
        left_hand = self._create_dummy_landmark_list()
        right_hand = self._create_dummy_landmark_list()

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
                return Keypoints(left_hand=self._create_dummy_landmark_list(),
                                     right_hand=self._create_dummy_landmark_list())
        
        return Keypoints(left_hand=left_hand, right_hand=right_hand)
        
    def close(self):
        self.hands.close()