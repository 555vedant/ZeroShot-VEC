from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from pathlib import Path

from utils.helpers import load_json
from utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_abs(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_image_path(raw_path):
    raw_str = str(raw_path).strip()
    if not raw_str:
        return None

    # 1) Exact path as-is.
    p = Path(raw_str)
    if p.exists():
        return p.resolve()

    # 2) Normalized slashes.
    normalized = raw_str.replace("\\", "/")
    p = Path(normalized)
    if p.exists():
        return p.resolve()

    # 3) Relative to configured WikiArt base path.
    base_path = _to_abs(Config.BASE_PATH)
    candidate = (base_path / p).resolve()
    if candidate.exists():
        return candidate

    # 4) Recover stale absolute paths containing "wikiart/...".
    lower = normalized.lower()
    marker = "wikiart/"
    if marker in lower:
        suffix = normalized[lower.index(marker) + len(marker):]
        recovered = (base_path / suffix).resolve()
        if recovered.exists():
            return recovered

    # 5) If Config.BASE_PATH points to dataset root, try latest versions/<n> subfolder.
    versions_dir = base_path / "versions"
    if versions_dir.exists():
        version_dirs = [d for d in versions_dir.iterdir() if d.is_dir()]
        for version_dir in sorted(version_dirs, reverse=True):
            recovered = (version_dir / normalized).resolve()
            if recovered.exists():
                return recovered

    return None


class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(_to_abs(Config.DATA_FILE))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = resolve_image_path(item["image"])
        text = item["text"]

        if image_path is None:
            return None

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            return None

        return {"image": image, "text": text}


processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)


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

    # Keep raw prompts for label-aware evaluation.
    inputs["raw_texts"] = texts
    return inputs