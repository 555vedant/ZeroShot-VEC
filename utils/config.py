import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = PROJECT_ROOT / "webapp"
LOCAL_WIKIART_PATH = PROJECT_ROOT / "data" / "Wikiart"
LOCAL_ARTEMIS_PATH = PROJECT_ROOT / "data" / "artemis"
LOCAL_WORK_DIR = PROJECT_ROOT / "data"


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
    """Resolve the WikiArt directory from the repository's configured data location."""
    candidates = [
        LOCAL_WIKIART_PATH,
        PROJECT_ROOT / "data" / "wikiart",
        PROJECT_ROOT / "data" / "WikiArt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class Config:

    # ENV DETECTION
    IS_KAGGLE = is_kaggle()
    IS_COLAB = is_colab()

    # PATHS
    if IS_KAGGLE:
        BASE_PATH = Path("/kaggle/input/wikiart")
        ARTEMIS_PATH = Path("/kaggle/input/artemis-dataset")
        WORK_DIR = Path("/kaggle/input/clip")

    elif IS_COLAB:
        BASE_PATH = Path("/root/.cache/kagglehub/datasets/steubk/wikiart/versions/1")
        ARTEMIS_PATH = Path("/root/.cache/kagglehub/datasets/samamostafa03/artemis-dataset/versions/1")
        WORK_DIR = Path("/content")

    else:
        BASE_PATH = resolve_local_wikiart_path()
        ARTEMIS_PATH = LOCAL_ARTEMIS_PATH
        WORK_DIR = LOCAL_WORK_DIR

    @staticmethod
    def _prefer_webapp(default_path: Path, webapp_name: str) -> Path:
        webapp_path = WEBAPP_DIR / webapp_name
        if default_path.exists():
            return default_path
        if webapp_path.exists():
            return webapp_path
        return default_path

    DATA_FILE = _prefer_webapp(WORK_DIR / "pairs.json", "pairs.json")
    CHECKPOINT_FILE = _prefer_webapp(WORK_DIR / "clip_model.pth", "clip_model.pth")
    REJECTION_CALIBRATION_FILE = WORK_DIR / "rejection_thresholds.json"

    # MODEL
    MODEL_NAME = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224
    TEXT_MAX_LENGTH = 77

    # TRAINING
    BATCH_SIZE = 32
    EPOCHS = 10
    # CLIPFIT: Select "full" for the existing baseline or "clipfit" for the experiment.
    FINE_TUNING_STRATEGY = "clipfit"
    # CLIPFIT: Weight for frozen-teacher image representation distillation.
    CLIPFIT_KD_WEIGHT = 8.0
    LR = 5e-6
    DEVICE = "cuda"
    VAL_SPLIT = 0.2
    SPLIT_SEED = 42
    NEGATIVE_SEED = 123

    TEMPERATURE = 0.07

    # improved
    FREEZE_VISION = True
    FREEZE_TEXT = False
    UNFREEZE_VISION_TOP_LAYERS = 2
    UNFREEZE_TEXT_TOP_LAYERS = 1
    UNFREEZE_VISION_POST_LAYERNORM = True
    UNFREEZE_TEXT_FINAL_LAYERNORM = True

    MIXED_PRECISION = True
    WEIGHT_DECAY = 0.01
    BACKBONE_LR_MULTIPLIER = 0.2

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