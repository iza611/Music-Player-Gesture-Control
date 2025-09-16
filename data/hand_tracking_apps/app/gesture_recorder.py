from core.keypoints import Keypoints
from ui.window import RecordingUI
from app.app_base import HandTrackingApp
from core.sample_data import Sample, save_sample

class GestureRecorder(HandTrackingApp):
    """
    App recording hand gestures and saving keypoints as numpy arrays. 
    Format: (20, 42, 3)  - 20 frames, 42 keypoints (21 per hand), 3D (x, y, z); if one hand not visible 21 zeros are placed instead.
    UI includes button to start recording. Once pressed countdown appears to indicate video length.
    """

    NUM_FRAMES = 20

    def __init__(self):
        window = RecordingUI(self.start_recording)
        super().__init__(window)
        self.is_recording = False

    def start_recording(self):
        self.window.show_message("Recording")
        self.window.realtime_video.configure(highlightbackground="red")
        self.is_recording = True
        self.sample = Sample(keypoints_sequence=[])

    def process_next_frame(self):
        keypoints: Keypoints = super().process_next_frame()

        if self.is_recording:
            if len(self.sample) >= self.NUM_FRAMES:
                self._finalise_recording()
            else:
                self.sample.append(keypoints)
                self._update_recording_ui()

    def _update_recording_ui(self):
        remaining = self.NUM_FRAMES - len(self.sample)
        self.window.show_message(f"{remaining}")

    def _finalise_recording(self):
            # save
            save_sample(sample=self.sample, class_name=self.window.get_input())
            self.window.show_message("Recording saved. Start new recording when you're ready.")

            # reset
            self.window.realtime_video.configure(highlightbackground="black")
            self.is_recording = False
            del self.sample




# class GesturePredictor(HandTrackingApp):
#     """
#     This version of HandTrackingApp allows loading a trained gesture classifier and predicting gestures live from webcam feed.
#     UI includes label to show predicted gesture name.
#     """
#     def __init__(self):
#         pass

# class DemoApp(HandTrackingApp):
#     """
#     This version of HandTrackingApp just shows real-time webcam feed with detected hand keypoints annotated.
#     UI includes label to show handedness of the detected hand(s).
#     """
#     def __init__(self):
#         pass

# class ReplayApp(HandTrackingApp):
#     """
#     This version of HandTrackingApp replays a recorded sample from the dataset folder instead of webcam feed.
#     UI includes label to show handedness of the detected hand(s).
#     """
#     def __init__(self):
#         pass

# class AppFactory():
#     @staticmethod
#     def get_app(mode: str) -> HandTrackingApp:
#         if mode == "record":
#             return GestureRecorder()
#         elif mode == "predict":
#             return GesturePredictor()
#         elif mode == "demo":
#             return DemoApp()
#         elif mode == "replay":
#             return ReplayApp()
#         else:
#             raise ValueError(f"Unknown mode: {mode}")