import tkinter as tk

class BaseWindow():
    def __init__(self):
        self.root = tk.Tk()
        # self.root.title("GUI")
        self.root.geometry('1000x700')

        self.realtime_video = tk.Label(self.root, text="couldn't open your camera")
        self.realtime_video.grid(column=0, row=0)

        self.lbl = tk.Label(self.root, text = "")
        self.lbl.grid(column=0, row=1)

    def activate_on_close_protocol(self, close_callback):
        self.root.protocol("WM_DELETE_WINDOW", close_callback)

class RecordingMode(BaseWindow):
    def __init__(self, record_callback):
        super().__init__()
        self.lbl.configure(text = "(rendering option picked)")
        btn = tk.Button(self.root, text = "Record",fg = "red", command=record_callback)
        btn.grid(column=0, row=2)

    def show_message(self, message: str):
        self.lbl.configure(text = message)

class PredictionMode(BaseWindow):
    def __init__(self):
        super().__init__()
        self.lbl.configure(text = "(predicting option picked)")

    def update_gesture_prediction(self, gesture_name: str):
        self.lbl.configure(text = gesture_name)
