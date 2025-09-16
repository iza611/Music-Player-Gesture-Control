# from camera import CameraCapture, convert_to_photo_image
# from keypoints import MediaPipeHandDetector, Keypoints
# from window import RecordingUI
# import numpy as np
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = (BASE_DIR / "../keypoints_data/raw/").resolve()

# class Sample:
#     def __init__(self, keypoints_sequence: list[Keypoints]):
#         self.keypoints_sequence = keypoints_sequence

#     def __len__(self):
#         return len(self.keypoints_sequence)

#     def append(self, keypoints: Keypoints):
#         if not isinstance(keypoints, Keypoints):
#             raise TypeError("Only Keypoints instances can be appended to the sample.")
#         self.keypoints_sequence.append(keypoints)

# def sample_to_numpy(keypoints_sequence) -> np.ndarray:
#     sample_array = []
#     for keypoints in keypoints_sequence:
#         keypoints_array = [
#             np.array([[lm.x, lm.y, lm.z] for lm in single_hand_keypoints.landmark], dtype=np.float32)
#             for single_hand_keypoints in (keypoints.left_hand, keypoints.right_hand)
#         ]
#         keypoints_array = np.stack(keypoints_array) # [(21,3), (21,3)] -> (2, 21, 3) list of numpy arrays to numpy array
#         sample_array.append(keypoints_array)

#     sample_array = np.stack(sample_array) # list of numpy arrays to numpy array
#     print(sample_array)
#     print(sample_array.shape)
#     return sample_array
    
# def save_sample(sample: Sample, class_name: str): 
#         class_dir = DATA_DIR / class_name
#         class_dir.mkdir(parents=True, exist_ok=True)

#         sample_id = len(list(class_dir.glob("*.npy"))) + 1 # TODO: change to UUID instead
#         save_dir = class_dir / f"{sample_id}.npy"

#         np.save(save_dir, sample_to_numpy(sample.keypoints_sequence))
#         print(f"Sample saved as {save_dir}")

# class HandTrackingApp:
#     """
#     Base App class for hand tracking applications that connects camera, hand detector and UI window.
#     App displays real-time frames from webcam with annotated keypoints.  
#     """
#     def __init__(self, window, camera=None, detector=None):
#         self.window = window
#         self.window.activate_on_close_protocol(self._on_close)
#         self.cap = camera or CameraCapture()
#         self.detector = detector or MediaPipeHandDetector()

#     def run(self):
#         self.process_next_frame()
#         self.window.root.mainloop()

#     def process_next_frame(self):
#         frame = self.cap.get_frame()
#         frame_annotated, keypoints, handedness = self.detector.detect_keypoints(frame)
        
#         self._update_ui(frame_annotated, handedness)
#         self.window.realtime_video.after(10, self.process_next_frame)

#         return keypoints

#     def _update_ui(self, frame_annotated, handedness):
#         photo_image = convert_to_photo_image(frame_annotated)
#         self.window.realtime_video.photo_image = photo_image
#         self.window.realtime_video.configure(image=photo_image)
#         self.window.show_detected_hands_message(handedness)
    
#     def _on_close(self):
#         print("Closing...")
#         self.detector.close()
#         self.cap.release()
#         self.window.root.destroy()


# class GestureRecorder(HandTrackingApp):
#     """
#     App recording hand gestures and saving keypoints as numpy arrays. 
#     Format: (20, 42, 3)  - 20 frames, 42 keypoints (21 per hand), 3D (x, y, z); if one hand not visible 21 zeros are placed instead.
#     UI includes button to start recording. Once pressed countdown appears to indicate video length.
#     """

#     NUM_FRAMES = 20

#     def __init__(self):
#         window = RecordingUI(self.start_recording)
#         super().__init__(window)
#         self.is_recording = False

#     def start_recording(self):
#         self.window.show_message("Recording")
#         self.window.realtime_video.configure(highlightbackground="red")
#         self.is_recording = True
#         self.sample = Sample(keypoints_sequence=[])

#     def process_next_frame(self):
#         keypoints: Keypoints = super().process_next_frame()

#         if self.is_recording:
#             if len(self.sample) >= self.NUM_FRAMES:
#                 self._finalise_recording()
#             else:
#                 self.sample.append(keypoints)
#                 self._update_recording_ui()

#     def _update_recording_ui(self):
#         remaining = self.NUM_FRAMES - len(self.sample)
#         self.window.show_message(f"{remaining}")

#     def _finalise_recording(self):
#             # save
#             save_sample(sample=self.sample, class_name=self.window.get_input())
#             self.window.show_message("Recording saved. Start new recording when you're ready.")

#             # reset
#             self.window.realtime_video.configure(highlightbackground="black")
#             self.is_recording = False
#             del self.sample




# # class GesturePredictor(HandTrackingApp):
# #     """
# #     This version of HandTrackingApp allows loading a trained gesture classifier and predicting gestures live from webcam feed.
# #     UI includes label to show predicted gesture name.
# #     """
# #     def __init__(self):
# #         pass

# # class DemoApp(HandTrackingApp):
# #     """
# #     This version of HandTrackingApp just shows real-time webcam feed with detected hand keypoints annotated.
# #     UI includes label to show handedness of the detected hand(s).
# #     """
# #     def __init__(self):
# #         pass

# # class ReplayApp(HandTrackingApp):
# #     """
# #     This version of HandTrackingApp replays a recorded sample from the dataset folder instead of webcam feed.
# #     UI includes label to show handedness of the detected hand(s).
# #     """
# #     def __init__(self):
# #         pass

# # class AppFactory():
# #     @staticmethod
# #     def get_app(mode: str) -> HandTrackingApp:
# #         if mode == "record":
# #             return GestureRecorder()
# #         elif mode == "predict":
# #             return GesturePredictor()
# #         elif mode == "demo":
# #             return DemoApp()
# #         elif mode == "replay":
# #             return ReplayApp()
# #         else:
# #             raise ValueError(f"Unknown mode: {mode}")