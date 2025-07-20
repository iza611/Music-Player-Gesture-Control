# a version for dataset prep and a version for evaluating trained models live 

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from typing import Tuple

"""View"""

class WindowContent():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GUI")
        self.root.geometry('1000x700')

        self.realtime_video = tk.Label(self.root, text="couldn't open your camera")
        self.realtime_video.grid(column=0, row=0)

        self.lbl = tk.Label(self.root, text = "")
        self.lbl.grid(column=0, row=1)

class RecordingMode(WindowContent):
    def __init__(self, record_callback):
        self.record_callback = record_callback
        super().__init__()
        self.render_recording_elements()

    def render_recording_elements(self):
        self.lbl.configure(text = "(rendering option picked)")
        btn = tk.Button(self.root, text = "Record",fg = "red", command=self.record_callback)
        btn.grid(column=0, row=2)

    def show_message(self, message: str):
        self.lbl.configure(text = message)

class PredictionMode(WindowContent):
    def __init__(self):
        super().__init__()
        self.render_prediction_elements()

    def render_prediction_elements(self):
        self.lbl.configure(text = "(predicting option picked)")

    def update_gesture_prediction(self, gesture_name: str):
        self.lbl.configure(text = gesture_name)

"""Model"""

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


class PredictGesture():
    # TBC
    # apply ML model on keypoints to detect hand gesture
    pass

class RecordGesture():
    # TBC
    # record X frames and save in desired dir (video & keypoints)
    pass

"""Presenter"""

class GUI():
    def __init__(self):
        self.cap = CameraCapture.webcam_capture_setup()
        self.display_frame()

    def display_frame(self):
        # capture, annotate and process frame
        frame = CameraCapture.open_camera(self.cap)
        frame_annotated, keypoints = CameraCapture.detect_keypoints(frame)
        photo_image = CameraCapture.convert_to_photo_image(frame_annotated)
        # display in the window
        self.window.realtime_video.photo_image = photo_image
        self.window.realtime_video.configure(image=photo_image)
        self.window.realtime_video.after(10, self.display_frame)

    def run(self):
        self.window.root.mainloop()

class RecordGesturesApp(GUI):
    def __init__(self):
        self.window = RecordingMode(self.record_clicked)
        super().__init__()

    def record_clicked(self):
        # start recording logic here with RecordGesture
        self.window.show_message("Recording started")

class PredictGestureLiveApp(GUI):
    def __init__(self):
        self.window = PredictionMode()
        super().__init__()
        # use PredictGesture and display predicted class with self.window.lbl
        # need to capture and store keypoints detected and processed somewhere - not sure yet where and how yet
        detected_gesture = "thumbs_up"
        self.window.update_gesture_prediction(detected_gesture)

"""test locally during development"""

if __name__ == "__main__":
    # app = RecordGesturesApp()
    app = PredictGestureLiveApp()
    app.run()