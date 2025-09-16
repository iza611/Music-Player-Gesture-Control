import numpy as np
from loguru import logger
import typer
import torch
from typing import Tuple, Dict
import csv

from config import KEYPOINT_NORM_DATA_DIR

app = typer.Typer()

def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str]]:
    """
    Load gesture dataset from KEYPOINT_NORM_DATA_DIR. 
    Each `.npy` file in the directory represents all samples for one gesture class.

    Returns:
        gesture_data (torch.Tensor): Tensor of shape 
            (num_samples, num_frames=20, num_hands=2, num_keypoints=21, num_coordinates=3).
        gesture_labels (torch.Tensor): 1D tensor of shape (num_samples,) with integer labels. 
        id_to_class_name (Dict[int, str]): Mapping from integer label class ID to class (gesture) name.
    """

    classes = [c.stem[:-7] for c in KEYPOINT_NORM_DATA_DIR.glob("*latest.npy")]
    class_name_to_id = {c: i for i, c in enumerate(classes)}

    total_samples = 0
    with open(KEYPOINT_NORM_DATA_DIR / "metadata.csv", mode="r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                total_samples += int(row[1])

    gesture_data = torch.empty((total_samples, 20, 2, 21, 3))
    gesture_labels = torch.empty((total_samples,), dtype=torch.long)
    
    start_idx = 0
    for class_path in KEYPOINT_NORM_DATA_DIR.glob("*latest.npy"):
        class_name = class_path.stem[:-7]
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
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
