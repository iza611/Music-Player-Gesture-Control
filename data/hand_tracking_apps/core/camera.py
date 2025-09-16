import cv2
import numpy as np

class CameraCapture():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    def get_frame(self) -> np.ndarray:
        _, frame = self.cap.read()
        return frame
    
    def release(self):
        self.cap.release()

from PIL import Image, ImageTk

def convert_to_photo_image(frame_array: np.ndarray) -> ImageTk.PhotoImage:
    opencv_image = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)

    return photo_image