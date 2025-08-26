from datetime import datetime

import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

from gesture_classification.config import KEYPOINT_DATA_DIR, KEYPOINT_NORM_DATA_DIR

app = typer.Typer()

WRIST_KEYPOINT_ID = 0
MIDDLE_MCP_KEYPOINT_ID = 9
NUM_HANDS = 2
NUM_KEYPOINTS = 21
NUM_COORDINATES = 3
EPSILON = 1e-6

def center_and_scale_keypoints(raw_keypoints: np.ndarray) -> np.ndarray:
    """
    Center and scale raw hand keypoints:

    - Centers keypoints relative to the wrist.
    - Scales keypoints to be distance-invariant using the wrist–to-middle-MCP Euclidean distance.

    Broadcasting example:
        Suppose raw_keypoints has shape (r, f, h, k, c) = (30, 20, 2, 21, 3)
        - wrist_coords = raw_keypoints[..., 0, :]              -> shape (30, 20, 2, 3)
        - wrist_coords = np.expand_dims(wrist_coords, axis=-2) -> shape (30, 20, 2, 1, 3)
        - keypoints_centered = raw_keypoints - wrist_coords    -> shape (30, 20, 2, 21, 3)
          Each keypoint is now centered relative to its wrist.

    Args:
        raw_keypoints (np.ndarray): Raw hand keypoints returned by MediaPipe model. 

    Returns:
        keypoints_normalised (np.ndarray): Location- and distance-invariant keypoints. 
    """

    # Center
    wrist_coords = raw_keypoints[..., WRIST_KEYPOINT_ID , : ]
    wrist_coords = np.expand_dims(wrist_coords, axis=-2) # expanded dims to allow broadcasting and matching shapes
    keypoints_centered = raw_keypoints - wrist_coords

    logger.info("Centered keypoints based on wrist coordinates")
    logger.info(f"Centered keypoints shape={keypoints_centered.shape}")

    # Scale
    middle_mcp_coords = keypoints_centered[..., MIDDLE_MCP_KEYPOINT_ID, :]  # shape: (r, f, h, c)
    scales = np.linalg.norm(middle_mcp_coords, axis=-1)                     # shape: (r, f, h)
    scales = np.maximum(scales, EPSILON) # to avoid division by zero
    scales = scales[..., None, None]                                        # shape: (r, f, h, 1, 1)
    keypoints_normalised = keypoints_centered / scales                      # shape: (r, f, h, k, c)

    logger.info("Normalised keypoints based on wrist-to-middle-MCP length")
    logger.info(f"Normalised keypoints shape={keypoints_normalised.shape}")

    return keypoints_normalised

def normalise_keypoints_for_class(class_name: str) -> None:
    """
    Normalise raw hand keypoints for all recorded samples of a given class. 

    Args:
        class_name (str): The name of class to normalise.

    Input Data:
        Raw keypoints are expected as .npy files in KEYPOINT_DATA_DIR/class_name.
        Each file should contain an array of shape 
        (num_recs, num_frames, num_hands=2, num_keypoints=21, num_coordinates=3)),
        in short (r, f, h, k, c).
    
    Returns:
        None
        Function saves the normalised keypoints, rather than returning them, in KEYPOINT_NORM_DATA_DIR.
        
    """

    # LOAD
    raw_keypoints_dir = KEYPOINT_DATA_DIR / class_name

    try:
        raw_keypoints = np.stack([np.load(sample_path) for sample_path in raw_keypoints_dir.glob("*.npy")])
        logger.info(f"Loaded {class_name} raw keypoints from {raw_keypoints_dir}")
        logger.info(f"Shape: {raw_keypoints.shape} (recordings, frames, hands, keypoints, coordinates)")

    except Exception as e:
        logger.error(f"Failed to load keypoints {class_name} from {raw_keypoints_dir}: {e}", exc_info=True)
        raise
    
    if len(raw_keypoints.shape) != 5 or raw_keypoints.shape[-3:] != (NUM_HANDS, NUM_KEYPOINTS, NUM_COORDINATES):
        raise ValueError(f"Expected 5D array of shape (num_recs, num_frames, {NUM_HANDS}, {NUM_KEYPOINTS}, {NUM_COORDINATES}),\n\
                         got {raw_keypoints.shape}")

    # NORMALISE
    keypoints_normalised = center_and_scale_keypoints(raw_keypoints)

    # SAVE
    save_dir = KEYPOINT_NORM_DATA_DIR / f'{class_name}_latest.npy'

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = KEYPOINT_NORM_DATA_DIR / f"{class_name}_{ts}.npy"

    save_dir.parent.mkdir(parents=True, exist_ok=True)

    np.save(save_dir, keypoints_normalised)
    np.save(archive_dir, keypoints_normalised)

    logger.info(f"Saved normalised keypoints in {save_dir} and {archive_dir}")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
