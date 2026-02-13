from pathlib import Path

# Paths
DB_NPZ_PATH = Path("data/db/face_db.npz")
HISTORY_DIR = Path("data/history")
ARCFACE_MODEL_PATH = "models/embedder_arcface.onnx"

# Camera
CAMERA_INDEX = 0

# Detection & Locking
HAAR_MIN_SIZE = (70, 70)
DEFAULT_DISTANCE_THRESHOLD = 0.75
LOCK_RELEASE_FRAMES = 30
LOCK_MOVEMENT_THRESHOLD_PX = 50
LOCK_ACTION_COOLDOWN_FRAMES = 15

# Mediapipe Landmarks Indices
# EAR (Eye Aspect Ratio) 6 points
LOCK_EAR_LEFT_INDICES = [33, 160, 158, 133, 153, 144]
LOCK_EAR_RIGHT_INDICES = [362, 385, 387, 263, 373, 380]
LOCK_EAR_BLINK_THRESHOLD = 0.25

LOCK_MOUTH_LEFT_INDEX = 61
LOCK_MOUTH_RIGHT_INDEX = 291
LOCK_SMILE_MOUTH_RATIO = 1.15

def ensure_dirs():
    DB_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
