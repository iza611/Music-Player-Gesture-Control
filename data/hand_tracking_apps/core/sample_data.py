from core.keypoints import Keypoints
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../../keypoints_data/raw/").resolve()

class Sample:
    def __init__(self, keypoints_sequence: list[Keypoints]):
        self.keypoints_sequence = keypoints_sequence

    def __len__(self):
        return len(self.keypoints_sequence)

    def append(self, keypoints: Keypoints):
        if not isinstance(keypoints, Keypoints):
            raise TypeError("Only Keypoints instances can be appended to the sample.")
        self.keypoints_sequence.append(keypoints)

    
def save_sample(sample: Sample, class_name: str): 
        class_dir = DATA_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        sample_id = len(list(class_dir.glob("*.npy"))) + 1 # TODO: change to UUID instead
        save_dir = class_dir / f"{sample_id}.npy"

        np.save(save_dir, _sample_to_numpy(sample.keypoints_sequence))
        print(f"Sample saved as {save_dir}")

def _sample_to_numpy(keypoints_sequence) -> np.ndarray:
    sample_array = []
    for keypoints in keypoints_sequence:
        keypoints_array = [
            np.array([[lm.x, lm.y, lm.z] for lm in single_hand_keypoints.landmark], dtype=np.float32)
            for single_hand_keypoints in (keypoints.left_hand, keypoints.right_hand)
        ]
        keypoints_array = np.stack(keypoints_array) # [(21,3), (21,3)] -> (2, 21, 3) list of numpy arrays to numpy array
        sample_array.append(keypoints_array)

    sample_array = np.stack(sample_array) # list of numpy arrays to numpy array
    print(sample_array)
    print(sample_array.shape)
    return sample_array