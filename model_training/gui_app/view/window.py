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
        self.record_callback = record_callback
        self.lbl.configure(text = "Start recording when you're ready")
        btn = tk.Button(self.root, text = "Record",fg = "red", command=self.record_clicked)
        btn.grid(column=0, row=2)

    def record_clicked(self):
        self.countdown(3)
    
    def countdown(self, count):
        if count >= 0:
            self.show_message(f"Recording in {count}...")
            self.root.after(1000, lambda: self.countdown(count - 1))
        else:
            self.record_callback()

    def show_message(self, message: str):
        self.lbl.configure(text = message)

class PredictionMode(BaseWindow):
    def __init__(self):
        super().__init__()
        self.lbl.configure(text = "(predicting option picked)")

    def update_gesture_prediction(self, gesture_name: str):
        self.lbl.configure(text = gesture_name)
