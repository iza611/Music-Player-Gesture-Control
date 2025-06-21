### Overview
A modular, real-time gesture recognition system using webcam input to trigger user-defined actions — here controlling Spotify — via local HTTP APIs. Built for easy integration and full terminal control. 

---

Local Script: Captures webcam frames → posts to HTTP endpoint.

Container A (Gesture Recognition): Receives frames → performs inference → posts predicted gesture.

Container B (Spotify Control): Receives gesture → maps to a Spotify command → sends API call.

### 🔹 Host Script 
- Initializes the webcam and launches both containers.
- View, list, and manage gesture-command setup by contacting Container 2.

---

### 🔹 Container 1: Gesture Recognition + API

- Runs real-time gesture recognition.
- Provides a local **HTTP API** at `localhost:8000` (e.g., using FastAPI or Flask).
- Exposes gesture-triggered endpoints like `/gesture/play`, `/gesture/next`, etc.
- Clean integration with any program via HTTP — "plug-and-play."

ML:
1) Keypoint detection with MediaPipe Hands (pretrained, ready to use)
2) Preprocessing (normalise, extract features e.g., angles, distances?)
3) Update temporal input buffer (10-30 last frames)
4) Classify with 1D CNN / lightweight TCN (needs to be trained)
OR Random Forest / XGBoost (could add Frame Deltas, mean&variance motion, Max/Min displacement, Hand centroid trajectory, Angle between fingers, Motion energy)
OR KNN with DTW
5) Post-processing (e.g., majority vote or exponential moving average?)

---

### 🔹 Container 2: Spotify Controller

- Uses Spotify's **official Web API**.
- Listens for requests from Container 1 or other local tools.
- Manages the gesture-command setup and provides an interface for the local host to interact with the setup.

---

## Training and dataset prep

- Separately in dir 'model_training'
- Training on the local machine or with Google Colab (tbd)

---

#### Example:
- Container 1 sends `POST /thumbs_up` → triggers Spotify’s `add_to_favourites()` API via Container 2.



## Step-by-Step Plan
1. run webcam in simplest gui with mediapipe hand keypoint detector
2. record keypoints with a script
3. prep dataset for a classifier, see what data it needs; decide what gestures to include, what they will correspond to
4. code classifier training and evaluation
5. train and evaluate (maybe with google colab)
6. create a setup for retraining and automatic update of the model to the container
7. write a script to turn on the webcam & send frames with http (local)

container 1
follow tutorial to setup fastapi docker ml service 
- receive frames
- use both models
- send results (detected gesture)

container 2
try to set up without help
receive gesture > transform to command > send request to spotify

## In future
add an option to add new gestures
