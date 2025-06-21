from model.camera import CameraCapture
from model.classifier import PredictGesture
from model.recorder import RecordGesture
from model.keypoints import detect_keypoints
from utils.img_converter import convert_to_photo_image
from view.window import RecordingMode, PredictionMode

class BaseApp():
    def __init__(self, window):
        self.window = window
        self.window.activate_on_close_protocol(self.on_close)
        self.cap = CameraCapture()

    def display_frame(self):
        # capture, annotate and process frame
        frame = CameraCapture.get_frame()
        frame_annotated, keypoints = detect_keypoints(frame)
        photo_image = convert_to_photo_image(frame_annotated)
        # display in the window
        self.window.realtime_video.photo_image = photo_image
        self.window.realtime_video.configure(image=photo_image)
        self.window.realtime_video.after(10, self.display_frame)

    def run(self):
        self.display_frame()
        self.window.root.mainloop()

    def on_close(self):
        print("Closing...")
        self.cap.release()
        self.window.root.destroy()

class RecordGesturesApp(BaseApp):
    def __init__(self):
        window = RecordingMode(self.record_clicked)
        super().__init__(window)

    def record_clicked(self):
        # Placeholder; start recording logic here with RecordGesture
        self.window.show_message("Recording started")

class PredictGestureLiveApp(BaseApp):
    def __init__(self):
        window = PredictionMode()
        super().__init__(window)
        # use PredictGesture and display predicted class with self.window.lbl
        # need to capture and store keypoints detected and processed somewhere - not sure yet where and how yet
        detected_gesture = "thumbs_up"
        self.window.update_gesture_prediction(detected_gesture)