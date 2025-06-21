from model.camera import CameraCapture
from model.classifier import PredictGesture
from model.recorder import RecordGesture
from model.keypoints import MediaPipeHandDetector
from utils.img_converter import convert_to_photo_image
from view.window import RecordingMode, PredictionMode

class BaseApp():
    """
    The skeleton for both recodring dataset and evaluating gesture prediction classifiers live.
    UI displays real-time frames from webcam. Buttons/labels below are specified by the recording/prediction versions of the app. 
    """
    def __init__(self, window):
        self.window = window
        self.window.activate_on_close_protocol(self.on_close)
        self.cap = CameraCapture()
        self.mediapipe_hands = MediaPipeHandDetector()

    def display_frame(self):
        # capture, annotate and process frame
        frame = self.cap.get_frame()
        frame_annotated, keypoints = self.mediapipe_hands.detect_keypoints(frame)
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
        self.mediapipe_hands.close()
        self.cap.release()
        self.window.root.destroy()

class RecordGesturesApp(BaseApp):
    """
    This version of BaseApp allows recording hand gestures and saving keypoints in numpy format to be used later by classifier. 
    Format: (20, 42, 3)  - 20 frames, 42 keypoints (21 per hand), 3D (x, y, z); if one hand not visible 21 zeros are placed instead.
    UI includes button to start recording. Once pressed countdown appears to indicate video length.
    """
    def __init__(self):
        window = RecordingMode(self.record_clicked)
        super().__init__(window)

    def record_clicked(self):
        # Placeholder; start recording logic here with RecordGesture
        self.window.show_message("Recording started")

class PredictGestureLiveApp(BaseApp):
    """
    This version of BaseApp displays real time predictions made by the classifier. 
    UI includes a label displaying gesture name currently predicted.
    """
    def __init__(self):
        window = PredictionMode()
        super().__init__(window)
        # use PredictGesture and display predicted class with self.window.lbl
        # need to capture and store keypoints detected and processed somewhere - not sure yet where and how yet
        detected_gesture = "thumbs_up"
        self.window.update_gesture_prediction(detected_gesture)