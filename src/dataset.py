from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from pathlib import Path
import os

from utils.helpers import load_json
from utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_wikiart_roots():
    roots = []

    for env_name in ("WIKIART_ROOT", "WIKIART_DATA_ROOT"):
        env_val = os.environ.get(env_name)
        if env_val:
            p = Path(env_val)
            if p.exists():
                roots.append(p)

    for p in (
        Path.home() / ".cache" / "kagglehub" / "datasets" / "steubk" / "wikiart",
        Path("/content/.cache/kagglehub/datasets/steubk/wikiart"),
    ):
        if p.exists():
            versions_dir = p / "versions"
            if versions_dir.exists():
                versions = [x for x in versions_dir.iterdir() if x.is_dir()]
                roots.extend(sorted(versions, reverse=True))
            roots.append(p)

    unique_roots = []
    seen = set()
    for r in roots:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            unique_roots.append(rr)

    return unique_roots


def _resolve_image_path(raw_path, wikiart_roots):
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

    # 3) Relative to project root.
    candidate = (PROJECT_ROOT / p).resolve()
    if candidate.exists():
        return candidate

    # 4) Recover stale absolute paths containing "wikiart/...".
    lower = normalized.lower()
    marker = "wikiart/"
    if marker in lower:
        suffix = normalized[lower.index(marker) + len(marker):]
        for root in wikiart_roots:
            recovered = (root / suffix).resolve()
            if recovered.exists():
                return recovered

    return None


class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(Config.DATA_FILE)
        self.wikiart_roots = _find_wikiart_roots()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = _resolve_image_path(item["image"], self.wikiart_roots)
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
    inputs["raw_texts"] = texts
    return inputs