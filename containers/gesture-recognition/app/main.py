from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

# https://fastapi.tiangolo.com/advanced/websockets/#websockets-client
# websocket to receive frames

app = FastAPI(
    title="Gestures Recognition",
    description="Given new frame, detect and append time series of keypoints and classify hand gesture",
    version="1.0.0"
)

# model_path = os.path.join("models", "model.pkl")
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

@app.post("/test")
def test(dump: int):
    return "received frame:" + dump

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "dump_model"}
