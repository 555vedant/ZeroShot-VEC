import os
from pathlib import Path


def is_kaggle():
    return os.path.exists("/kaggle/input")


def is_colab():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return "google.colab" in str(type(shell))
    except Exception:
        return False


def resolve_local_wikiart_path() -> Path:
    """Prefer existing local WikiArt directory regardless of case."""
    candidates = [Path("./data/Wikiart"), Path("./data/wikiart")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class Config:

    # ENV DETECTION
    IS_KAGGLE = is_kaggle()
    IS_COLAB = is_colab()

    # PATHS
    if IS_KAGGLE:
        BASE_PATH = Path("/kaggle/input/wikiart")
        ARTEMIS_PATH = Path("/kaggle/input/artemis-dataset")
        WORK_DIR = Path("/kaggle/working")

    elif IS_COLAB:
        BASE_PATH = Path("/root/.cache/kagglehub/datasets/steubk/wikiart/versions/1")
        ARTEMIS_PATH = Path("/root/.cache/kagglehub/datasets/samamostafa03/artemis-dataset/versions/1")
        WORK_DIR = Path("/content")

    else:
        BASE_PATH = resolve_local_wikiart_path()
        ARTEMIS_PATH = Path("./data/artemis")
        WORK_DIR = Path("./data")

    DATA_FILE = WORK_DIR / "processed/pairs.json"
    CHECKPOINT_FILE = WORK_DIR / "checkpoints/clip_model.pth"

    # MODEL
    MODEL_NAME = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224
    TEXT_MAX_LENGTH = 77

    # TRAINING
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 5e-6
    DEVICE = "cuda"
    VAL_SPLIT = 0.2
    SPLIT_SEED = 42
    NEGATIVE_SEED = 123

    TEMPERATURE = 0.07

    # improved
    FREEZE_VISION = True
    FREEZE_TEXT = False

    MIXED_PRECISION = True

    # PERFORMANCE
    MULTI_GPU = True
    TF32 = True

    NUM_WORKERS = max(2, min(8, os.cpu_count() or 2))
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    NON_BLOCKING = True
    DROP_LAST_MULTI_GPU_TRAIN = True

    # Separate eval/index batch sizes to keep GPU busy during inference-heavy loops.
    EVAL_BATCH_SIZE = max(BATCH_SIZE, 64)
    INDEX_BATCH_SIZE = 128

    # SEARCH
    SEARCH_TOP_K = 5

    # ZERO-SHOT EVALUATION
    # If set, strict zero-shot split uses these as holdout labels.
    ZERO_SHOT_HOLDOUT_EMOTIONS = None
    # Auto-holdout controls (used only when ZERO_SHOT_HOLDOUT_EMOTIONS is None).
    ZERO_SHOT_HOLDOUT_RATIO = 0.2
    ZERO_SHOT_HOLDOUT_MIN_COUNT = 1
    ZERO_SHOT_MIN_HOLDOUT_EMOTIONS = 1
    ZERO_SHOT_SPLIT_SEED = SPLIT_SEED