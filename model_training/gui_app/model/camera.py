import cv2
import numpy as np

class CameraCapture():
    def __init__(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    def get_frame(self) -> np.array:
        _, frame = self.cap.read()
        return frame
    
    def release(self):
        self.cap.release()
