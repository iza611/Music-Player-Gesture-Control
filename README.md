### Why?
I'm working on this project to improve the work environment for a tattoo artist. Music is VERY important in a tattoo shop, but controlling it while tattooing and wearing sterile gloves can be a real headache. That's why I created this project!

### How?
A modular, real-time gesture recognition system using webcam input to trigger user-defined actions — here controlling Spotify — via local HTTP APIs. Built for easy integration and full terminal control. The gesture recognition module should be reusable for other projects. 

---

### My notes
- Local Script: Captures webcam frames → posts to HTTP endpoint.
- Container A (Gesture Recognition): Receives frames → performs inference → posts predicted gesture.
- Container B (Spotify Control): Receives gesture → maps to a Spotify command → sends API call.

- But first, create a dataset & train & evaluate the model. 

🔹 Host Script 
- Initializes the webcam and launches both containers.
- View, list, and manage gesture-command setup by contacting Container 2.

🔹 Container 1: Gesture Recognition + API
- Runs real-time gesture recognition.
- Provides a local **HTTP API** at `localhost:8000` (e.g., using FastAPI or Flask).
- Exposes gesture-triggered endpoints like `/gesture/play`, `/gesture/next`, etc.
- Clean integration with any program via HTTP — "plug-and-play."

ML:
1) Keypoint detection with MediaPipe Hands (pretrained, ready to use)
2) Normalise keypoints to be camera-position and -distance independent 
3) KNN + DTW  or  ROCKET + ridge regression classifier

🔹 Container 2: Spotify Controller
- Uses Spotify's official Web API
- Listens for requests from Container 1 or other local tools.
- Manages the gesture-command setup and provides an interface for the local host to interact with the setup.

#### Example:
- User shows gesture 'pif_paf' to the webcam → each frame is sent in real-time to Container 1
- Container 1 continuously runs a trained gesture recognition model & it correctly recognises gesture 'pif_paf'
- Container 1 sends `POST /pif_paf` → triggers Spotify’s `stop()` API via Container 2.

#### Plan
1. run webcam in simplest gui with mediapipe hand keypoint detector DONE
2. record keypoints with a script DONE
3. prep dataset for a classifier, see what data it needs; decide what gestures to include, what they will correspond to DONE (only 2 gestures for now)
4. code classifier training and evaluation (IN PROGRESS)
5. train and evaluate (maybe with google colab)
6. write a script to turn on the webcam & send frames with http (local)

#### In future
add an option to add new gestures, automatically retrain & evaluate the model & update in the container if its good. 
