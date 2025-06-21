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
        frame_annotated, self.keypoints = self.mediapipe_hands.detect_keypoints(frame)
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
        window = RecordingMode(self.start_recording)
        super().__init__(window)
        self.is_recording = False
        self.recorded_keypoints = []

    def start_recording(self):
        self.window.show_message("Recording")
        self.is_recording = True

    def display_frame(self):
        super().display_frame()
        if self.is_recording:
            num_of_frames = 30
            self.recorded_keypoints.append(self.keypoints)
            self.window.show_message(f"{num_of_frames-len(self.recorded_keypoints)}")

            if len(self.recorded_keypoints) >= num_of_frames: 
                self.save_sample()
                self.window.show_message("Recording saved. Start new recording when you're ready.")
                #cleanup
                self.is_recording = False
                self.recorded_keypoints = []

    def save_sample(self):
        frame_data = []
        for hand in self.keypoints:
            hand_data = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
            frame_data.append(hand_data)
        # save in correct format in correct dir

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