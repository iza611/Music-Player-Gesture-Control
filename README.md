A modular, real-time gesture recognition system using webcam input to trigger user-defined actions — such as controlling Spotify — via local HTTP APIs. Built for easy integration and full terminal control. No GUI required.

---

### Flow

1) Webcam captures video in real time.
2) Gesture recognition detects when a user performs a predefined gesture.
3) When a gesture is recognized, a mapped HTTP endpoint (e.g. `/gesture/play`) is triggered.
4) That endpoint invokes an action — like controlling Spotify — via API.

---

## Components

### 🔹 Host Script 
- Initializes the webcam and launches both containers.
- Manages webcam access for environments where Docker may have limited hardware access (e.g., Windows).

---

### 🔹 Container 1: Gesture Recognition + API

- Runs real-time gesture recognition.
- Provides a local **HTTP API** at `localhost:8000` (e.g., using FastAPI or Flask).
- Exposes gesture-triggered endpoints like `/gesture/play`, `/gesture/next`, etc.

#### Features:
- Includes some predefined gestures with mapped API endpoints.
- Allows recording and adding custom gestures via terminal interface.
- View, list, and manage gestures from command-line.
- Saves trained models and gesture embeddings to a **mounted volume** for persistence.
- Clean integration with any program via HTTP — "plug-and-play."

---

### 🔹 Container 2: Spotify Controller

- Uses Spotify's **official Web API**.
- Listens for requests from Container 1 or other local tools.
- Runs a lightweight server at `localhost:8001`, or as a callable Python script.

#### Example:
- Container 1 sends `POST /play` → triggers Spotify’s `play()` API via Container 2.

---

## Machine Learning 
(TBD)

---

## Notes

- Gesture classification from short video clips
  Trained models and embeddings saved locally
- testing?
