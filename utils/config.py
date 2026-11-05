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
        WORK_DIR = Path("/kaggle/working")
        INPUT_MODEL_DIR = Path("/kaggle/input")

    elif IS_COLAB:
        BASE_PATH = Path("/root/.cache/kagglehub/datasets/steubk/wikiart/versions/1")
        ARTEMIS_PATH = Path("/root/.cache/kagglehub/datasets/samamostafa03/artemis-dataset/versions/1")
        WORK_DIR = Path("/content")
        INPUT_MODEL_DIR = WORK_DIR

    else:
        BASE_PATH = resolve_local_wikiart_path()
        ARTEMIS_PATH = LOCAL_ARTEMIS_PATH
        WORK_DIR = LOCAL_WORK_DIR
        INPUT_MODEL_DIR = WORK_DIR

    @staticmethod
    def _prefer_webapp(default_path: Path, webapp_name: str) -> Path:
        webapp_path = WEBAPP_DIR / webapp_name
        if default_path.exists():
            return default_path
        if webapp_path.exists():
            return webapp_path
        return default_path

    DATA_FILE = _prefer_webapp(WORK_DIR / "pairs.json", "pairs.json")
    OUTPUT_MODELS_DIR = WORK_DIR / "models"
    OUTPUT_LOGS_DIR = WORK_DIR / "logs"
    OUTPUT_RESULTS_DIR = WORK_DIR / "results"
    _default_checkpoint = (
        OUTPUT_MODELS_DIR / "best_model.pt"
        if IS_KAGGLE
        else _prefer_webapp(WORK_DIR / "clip_model.pth", "clip_model.pth")
    )
    CHECKPOINT_FILE = Path(
        os.getenv("CLIP_CHECKPOINT_PATH", str(_default_checkpoint))
    )
    REJECTION_CALIBRATION_FILE = WORK_DIR / "rejection_thresholds.json"
    PAINTING_GATE_THRESHOLD = 0.35

    # MODEL
    MODEL_NAME = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224
    TEXT_MAX_LENGTH = 77

    # TRAERINING
    BATCH_SIZE = 64
    EPOCHS = 4
    RESUME_ADDITIONAL_EPOCHS = True
    CHECKPOINT_INTVAL = 5
    FINE_TUNING_STRATEGY = "visual_lora"
    FREEZE_TEXT_ENCODER = True
    FREEZE_TEXT_PROJECTION = True
    FREEZE_VISUAL_BACKBONE = True
    LORA_TARGET = ["q", "v"]
    LORA_TARGET_LAST_N_BLOCKS = 4
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TRAIN_VISUAL_PROJECTION = False
    LORA_TRAIN_VISUAL_NORMS = False
    USE_KNOWLEDGE_DISTILLATION = False
    CLIPFIT_KD_WEIGHT = 0.0
    LR = 5e-4
    VAL_SPLIT = 0.2
    TRAINING_CHECKPOINT_FILE = OUTPUT_MODELS_DIR / "latest_checkpoint.pt"
    TRAINING_LOG_FILE = OUTPUT_LOGS_DIR / "training_history.csv"
    EVALUATION_RESULTS_FILE = OUTPUT_RESULTS_DIR / "evaluation_results.json"
    SPLIT_SEED = 42
    NEGATIVE_SEED = 123

    TEMPERATURE = 0.07
    LOSS_TYPE = "multi_negative_ranking"
    RANKING_LOSS_WEIGHT = 0.5
    NEGATIVE_COUNT = 3
    NEGATIVE_HARD_FRACTION = 0.25
    NEGATIVE_LOSS_WEIGHT = 0.25
    NEGATIVE_TARGET = 0.1
    CLASS_WEIGHT_POWER = 0.5
    CLASS_WEIGHT_MAX = 4.0
    WARMUP_RATIO = 0.1
    SCHEDULER = "cosine"
    EARLY_STOPPING_ENABLED = False
    EARLY_STOPPING_PATIENCE = 5

    MIXED_PRECISION = True
    GRADIENT_CLIP_NORM = 1.0
    WEIGHT_DECAY = 0.001
    BACKBONE_LR_MULTIPLIER = 0.2

    # PERFORMANCE
    MULTI_GPU = True
    GPU_IDS = None  # None means use every CUDA device visible to PyTorch.
    TF32 = True

    # Conservative Kaggle defaults avoid worker deadlocks and excessive
    # prefetched image batches on mounted datasets.
    # Notebook-safe default: worker processes can remain silent or deadlock
    # while reading mounted Kaggle files. Training logic is unchanged.
    NUM_WORKERS = 0 if IS_KAGGLE else max(2, min(8, os.cpu_count() or 2))
    PREFETCH_FACTOR = 2
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False if IS_KAGGLE else True
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
    PROMPT_TEMPLATES = (
        "a painting that evokes {emotion}",
        "an artwork expressing {emotion}",
        "a painting conveying a feeling of {emotion}",
        "an artwork associated with {emotion}",
    )
