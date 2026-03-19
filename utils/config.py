from pathlib import Path

class Config:
    DATA_FILE = Path("data/processed/pairs.json")
    CHECKPOINT_FILE = Path("checkpoints/clip_model.pth")

    MODEL_NAME = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224
    TEXT_MAX_LENGTH = 77

    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-5
    DEVICE = "cuda"

    TEMPERATURE = 0.07

    FREEZE_VISION = True
    FREEZE_TEXT = True

    MIXED_PRECISION = True