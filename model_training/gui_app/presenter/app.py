from model.camera import CameraCapture
from model.classifier import PredictGesture
from model.recorder import RecordGesture
from model.keypoints import MediaPipeHandDetector
from utils.img_converter import convert_to_photo_image
from view.window import RecordingMode, PredictionMode
import numpy as np
from typing import Tuple
import os

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
        frame_annotated, self.keypoints, handedness = self.mediapipe_hands.detect_keypoints(frame)
        photo_image = convert_to_photo_image(frame_annotated)
        # display in the window
        self.window.realtime_video.photo_image = photo_image
        self.window.realtime_video.configure(image=photo_image)
        self.window.realtime_video.after(10, self.display_frame)
        self.window.show_detected_hands_message(handedness)

    def run(self):
        self.display_frame()
        self.window.root.mainloop()

    def on_close(self):
        print("Closing...")
        self.mediapipe_hands.close()
        self.cap.release()
        self.window.root.destroy()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of the current script
data_dir = os.path.join(BASE_DIR, "../../data/keypoints")
data_dir = os.path.abspath(data_dir)  # resolves ../../ properly

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
        self.window.realtime_video.configure(highlightbackground="red")
        self.is_recording = True

    def display_frame(self):
        super().display_frame()
        if self.is_recording:
            num_of_frames = 20
            self.recorded_keypoints.append(self.keypoints)
            self.window.show_message(f"{num_of_frames - len(self.recorded_keypoints)}")

            if len(self.recorded_keypoints) >= num_of_frames: 
                sample, class_name = self.convert_sample()
                self.save_sample(sample, class_name)
                self.window.show_message("Recording saved. Start new recording when you're ready.")
                self.window.realtime_video.configure(highlightbackground="black")
                #cleanup
                self.is_recording = False
                self.recorded_keypoints = []

    def convert_sample(self) -> Tuple[np.array, str]:
        gesture_name = self.window.get_input()

        all_frames_data = []
        for frame_keypoints in self.recorded_keypoints:
            frame_data = [
                np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
                for hand in frame_keypoints
            ]
            frame_data = np.stack(frame_data) # [(21,3), (21,3)] -> (2, 21, 3) list of numpy arrays to numpy array
            all_frames_data.append(frame_data)

        all_frames_data = np.stack(all_frames_data) # list of numpy arrays to numpy array
        print(all_frames_data)
        print(all_frames_data.shape)
        print(gesture_name)

        return all_frames_data, gesture_name

    def save_sample(self, sample: np.array, class_name: str):
        save_dir = data_dir + "\\" + class_name 
        os.makedirs(save_dir, exist_ok=True)

        num_of_samples = len([f for f in os.listdir(save_dir) if f.endswith('.npy')])
        new_sample_id = num_of_samples + 1

        file_path = os.path.join(save_dir, f"{new_sample_id}.npy")
        np.save(file_path, sample)
        print(f"Sample saved as {file_path}")
        print(f"Number of '{class_name}' samples = {num_of_samples+1}")
    
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