from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from pathlib import Path

from utils.helpers import load_json
from utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_image_path(raw_path):
    path_str = str(raw_path).strip().replace("\\", "/")
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path

    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.exists():
        return candidate

    # fallback for stale absolute paths from another machine
    marker = "data/"
    lower = path_str.lower()
    if marker in lower:
        rel_idx = lower.index(marker)
        fallback = (PROJECT_ROOT / path_str[rel_idx:]).resolve()  # preserve original casing
        if fallback.exists():
            return fallback

    if path.exists():
        return path.resolve()

    return None


class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(Config.DATA_FILE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = _resolve_image_path(item["image"])
        text = item["text"]

        if image_path is None:
            return None

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            return None

        return {"image": image, "text": text}


def _build_processor():
    try:
        return CLIPProcessor.from_pretrained(Config.MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"failed to load CLIP processor ({Config.MODEL_NAME}): {e}")

processor = _build_processor()


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.TEXT_MAX_LENGTH
    )
    inputs["raw_texts"] = texts
    return inputs