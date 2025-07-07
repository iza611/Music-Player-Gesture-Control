import tkinter as tk
from re import match

class BaseWindow():
    def __init__(self):
        self.root = tk.Tk()
        # self.root.title("GUI")
        self.root.geometry('1000x700')

        self.realtime_video = tk.Label(self.root, 
                                       text="couldn't open your camera", 
                                       highlightthickness=6, 
                                       highlightbackground="black")
        self.realtime_video.grid(column=0, row=0)

        self.lbl = tk.Label(self.root, text = "")
        self.lbl.grid(column=0, row=1)

        self.lbl_handedness = tk.Label(self.root, text = "")
        self.lbl_handedness.grid(column=0, row=4)

    def show_detected_hands_message(self, handedness_pred: str):
        self.lbl_handedness.configure(text = handedness_pred)

    def activate_on_close_protocol(self, close_callback):
        self.root.protocol("WM_DELETE_WINDOW", close_callback)

class RecordingMode(BaseWindow):
    def __init__(self, record_callback):
        super().__init__()
        
        # Entry box for gesture name
        self.lbl.configure(text = "Gesture name (eg. thumbs_up, like, palm, crossed_fingers etc):")
        self.gesture_name = tk.StringVar()
        self.gesture_name.trace_add("write", self.check_input)
        self.entry = tk.Entry(self.root, textvariable=self.gesture_name)
        self.entry.grid(column=0, row=2, padx=10, pady=5)

        # Record button only clickable when gesture name given; countdown and callback once clicked
        self.record_callback = record_callback
        self.btn = tk.Button(self.root, 
                        text = "Record",
                        fg = "red",
                        state=tk.DISABLED,
                        command=self.record_clicked)
        self.btn.grid(column=0, row=3)

    def check_input(self, *args):
        pattern = r'^[a-z]+(_[a-z]+)*$'
        if self.gesture_name.get().strip() and match(pattern, self.gesture_name.get().strip()):
            self.btn.config(state=tk.NORMAL)
        else:
            self.btn.config(state=tk.DISABLED)

    def get_input(self):
        return self.gesture_name.get().strip()

    def record_clicked(self):
        self.countdown(0)
    
    def countdown(self, count):
        if count > 0:
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
