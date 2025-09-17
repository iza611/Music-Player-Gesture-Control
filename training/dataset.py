import numpy as np
from loguru import logger
import typer
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict
import csv
from pathlib import Path
import re

from config import KEYPOINT_NORM_DATA_DIR, NUM_FRAMES, NUM_HANDS, NUM_KEYPOINTS, NUM_COORDS

app = typer.Typer()

class CustomGesturesDataset(Dataset):
    """Custom Dataset for gesture data stored in tensors."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def extract_class_name(filename: str) -> str:
    """Extract the class name from a filename of the format '<class_name>_latest.npy'."""
    name = re.match(r"(.+)_latest\.npy$", filename)
    if name:
        return name.group(1)
    else:
        raise ValueError(f"Invalid filename: {filename}. Expected format: '<class_name>_latest.npy'.")
    
def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str]]:
    """
    Load gesture dataset from the given path. 
    Each `.npy` file in the directory represents all samples for one gesture class.

    Returns:
        gesture_data (torch.Tensor): Tensor of shape 
            (num_samples, num_frames=20, num_hands=2, num_keypoints=21, num_coordinates=3).
        gesture_labels (torch.Tensor): 1D tensor of shape (num_samples,) with integer labels. 
        id_to_class_name (Dict[int, str]): Mapping from integer label class ID to class (gesture) name.
    """

    classes = [extract_class_name(c.name) for c in path.glob("*latest.npy")]
    class_name_to_id = {c: i for i, c in enumerate(classes)}

    total_samples = 0
    with open(path / "metadata.csv", mode="r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                total_samples += int(row[1])

    gesture_data = torch.empty((total_samples, NUM_FRAMES, NUM_HANDS, NUM_KEYPOINTS, NUM_COORDS), dtype=torch.float32)
    gesture_labels = torch.empty((total_samples,), dtype=torch.long)
    
    start_idx = 0
    for class_path in path.glob("*latest.npy"):
        class_name = extract_class_name(class_path.name)
        class_id = class_name_to_id[class_name]
        logger.info(f"Loading class {class_name} ({class_id+1}/{len(classes)})...")

        samples = torch.from_numpy(np.load(class_path)).type(torch.float32)
        num_samples = len(samples)
        end_idx = start_idx + num_samples

        gesture_data[start_idx: end_idx] = samples
        gesture_labels[start_idx:end_idx] = class_id # broadcasting

        start_idx = end_idx

    if len(gesture_data) != len(gesture_labels):
        raise ValueError(f"Data/labels size mismatch: gesture_data.shape={gesture_data.shape}, \n\
                         gesture_labels.shape={gesture_labels.shape}.")

    id_to_class_name = {i: c for i, c in enumerate(classes)}
    logger.info("Successfully loaded and prepared the dataset.")
    logger.info(f"gesture_data: {gesture_data.shape}")
    logger.info(f"gesture_labels: {gesture_labels.shape}")
    logger.info(f"id_to_class_name: {id_to_class_name}")
    return gesture_data, gesture_labels, id_to_class_name

@app.command()
def main():
    logger.info("Processing dataset...")
    x, y, class_to_gesture_name = load_dataset(path = KEYPOINT_NORM_DATA_DIR)
    dataset = CustomGesturesDataset(x, y)
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
