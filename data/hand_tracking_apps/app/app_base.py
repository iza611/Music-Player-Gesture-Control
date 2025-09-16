from core.camera import CameraCapture, convert_to_photo_image
from core.keypoints import MediaPipeHandDetector

class HandTrackingApp:
    """
    Base App class for hand tracking applications that connects camera, hand detector and UI window.
    App displays real-time frames from webcam with annotated keypoints.  
    """
    def __init__(self, window, camera=None, detector=None):
        self.window = window
        self.window.activate_on_close_protocol(self._on_close)
        self.cap = camera or CameraCapture()
        self.detector = detector or MediaPipeHandDetector()

    def run(self):
        self.process_next_frame()
        self.window.root.mainloop()

    def process_next_frame(self):
        frame = self.cap.get_frame()
        frame_annotated, keypoints, handedness = self.detector.detect_keypoints(frame)
        
        self._update_ui(frame_annotated, handedness)
        self.window.realtime_video.after(10, self.process_next_frame)

        return keypoints

    def _update_ui(self, frame_annotated, handedness):
        photo_image = convert_to_photo_image(frame_annotated)
        self.window.realtime_video.photo_image = photo_image
        self.window.realtime_video.configure(image=photo_image)
        self.window.show_detected_hands_message(handedness)
    
    def _on_close(self):
        print("Closing...")
        self.detector.close()
        self.cap.release()
        self.window.root.destroy()