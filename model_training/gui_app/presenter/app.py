from model.camera import CameraCapture
from model.classifier import PredictGesture
from model.recorder import RecordGesture
from view.window import RecordingMode, PredictionMode

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