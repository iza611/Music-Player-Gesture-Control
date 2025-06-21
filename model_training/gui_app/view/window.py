import tkinter as tk

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
