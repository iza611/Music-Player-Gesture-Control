import cv2
from PIL import Image, ImageTk
import numpy as np
from typing import Tuple

class CameraCapture():

    @staticmethod
    def webcam_capture_setup() -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0)

        width, height =  1000, 600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return cap
    
    @staticmethod
    def open_camera(cap: cv2.VideoCapture) -> np.array:
        _, frame = cap.read()
        return frame
    
    @staticmethod
    def detect_keypoints(frame: np.array) -> Tuple[np.array, np.array]:
        # TBD
        # use DL model to detect keypoints and add on top of the image to display
        keypoints = np.array([])
        return frame, keypoints
    
    def convert_to_photo_image(frame_array: np.array) -> ImageTk.PhotoImage:
        opencv_image = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)

        return photo_image
