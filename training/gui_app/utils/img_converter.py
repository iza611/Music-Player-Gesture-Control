import cv2
from PIL import Image, ImageTk
import numpy as np

def convert_to_photo_image(frame_array: np.array) -> ImageTk.PhotoImage:
    opencv_image = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)

    return photo_image