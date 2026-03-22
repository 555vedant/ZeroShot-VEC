from pathlib import Path

class Config:
    DATA_FILE = Path("data/processed/pairs.json")
    CHECKPOINT_FILE = Path("checkpoints/clip_model.pth")

    MODEL_NAME = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224
    TEXT_MAX_LENGTH = 77

    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 5e-6
    DEVICE = "cuda"

    SEARCH_TOP_K = 5

    TEMPERATURE = 0.07

    FREEZE_VISION = True
    FREEZE_TEXT = False

    MIXED_PRECISION = True