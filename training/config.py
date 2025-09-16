from pathlib import Path

# from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
# load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
KEYPOINT_DATA_DIR = DATA_DIR / "keypoints"
KEYPOINT_NORM_DATA_DIR = DATA_DIR / "keypoints_normalised"

MODELS_DIR = PROJ_ROOT / "gesture_classification" / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "gesture_classification" / "figures"

N_CLASSES = 2
RANDOM_STATE = 7

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
