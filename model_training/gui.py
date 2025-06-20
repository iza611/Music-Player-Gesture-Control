# a version for dataset prep and a version for evaluating trained models live 

import tkinter as tk
import cv2
from PIL import Image, ImageTk

"""Window view logic for both dataset prep and evaluation"""

class WindowContent():
    def __init__(self):
        self.root = tk.Tk()
        self.render_window_skeleton()

    def render_window_skeleton(self):
        self.root.title("GUI")
        self.root.geometry('1000x700')

        self.realtime_video = tk.Label(self.root, text="there is a problem with your camera")
        self.realtime_video.grid(column=0, row=0)

        def clicked():
            self.lbl.configure(text = "Recording started")

        btn = tk.Button(self.root, text = "Record" ,
                    fg = "red", command=clicked)
        btn.grid(column=0, row=1)

class RecordingMode(WindowContent):
    def __init__(self):
        super().__init__()
        self.render_elements()

    def render_elements(self):
        self.lbl2 = tk.Label(self.root, text = "(rendering option picked)")
        self.lbl2.grid(column=0, row=2)

class PredictingMode(WindowContent):
    def __init__(self):
        super().__init__()
        self.render_elements()

    def render_elements(self):
        self.lbl2 = tk.Label(self.root, text = "(predicting option picked)")
        self.lbl2.grid(column=0, row=2)

"""Main GUI loop entry point"""

class GUI():
    def __init__(self, window_strategy: WindowContent):
        self.window = window_strategy
        self.webcam_capture_setup()
        self.open_camera()

    def webcam_capture_setup(self):
        self.cap = cv2.VideoCapture(0)

        width, height =  1000, 600
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def open_camera(self):
        # capture frame
        _, frame = self.cap.read()
        # convert to photo image
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        # display in the window
        self.window.realtime_video.photo_image = photo_image
        self.window.realtime_video.configure(image=photo_image)
        self.window.realtime_video.after(10, self.open_camera)

    def run(self):
        self.window.root.mainloop()

"""test locally during development"""

if __name__ == "__main__":
    app = GUI(PredictingMode())
    app.run()



# TODO:
"""
- config.yml
- MVP/MVC/MMVM
- realtime_video access change to return with normal method; change the inheritence logic i think
- add on_close()
- 
"""