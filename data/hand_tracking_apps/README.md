hand_tracking_apps/
├── apps/                    # Orchestrators (App classes, connecting all parts)
│   ├──  base_app.py         # HandTrackingApp base class
│   ├──  gesture_recorder.py # App to collect data and create dataset
│
├── core/
│   ├──  camera.py           # Frame source (CameraCapture) + img utils (np -> image)
│   ├──  keypoints.py        # Keypoint detection (MediaPipeHandDetector, Keypoints dataclass)
│   ├──  sample_data.py      # Sample class + sample_to_numpy + save_sample where sample is a keypoints recording across 20 frames 
│
├── ui/                      # Pure UI/View layer with Tkinter
│   ├──  window.py           # BaseUI and RecordingUI
│
├── __main__.py              # Entrypoint running App
