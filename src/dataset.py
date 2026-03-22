from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from pathlib import Path, PurePosixPath

from utils.helpers import load_json
from utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _discover_wikiart_roots():
    roots = []

    # Optional explicit override for custom environments.
    for env_name in ("WIKIART_ROOT", "WIKIART_DATA_ROOT"):
        env_val = __import__("os").environ.get(env_name)
        if env_val:
            p = Path(env_val)
            if p.exists():
                roots.append(p)

    # Common kagglehub locations (local and Colab).
    for base in (
        Path.home() / ".cache" / "kagglehub" / "datasets" / "steubk" / "wikiart",
        Path("/content/.cache/kagglehub/datasets/steubk/wikiart"),
    ):
        if base.exists():
            roots.append(base)

    expanded = []
    for root in roots:
        versions_dir = root / "versions"
        if versions_dir.exists():
            for version_path in sorted(versions_dir.iterdir(), reverse=True):
                if version_path.is_dir():
                    expanded.append(version_path)
        expanded.append(root)

    seen = set()
    uniq = []
    for root in expanded:
        r = root.resolve()
        if r not in seen and r.exists():
            seen.add(r)
            uniq.append(r)

    return uniq


def _resolve_image_path(raw_path, wikiart_roots, basename_index):
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

    # Recover stale absolute paths saved from another machine/session.
    posix_parts = PurePosixPath(path_str).parts
    if posix_parts:
        lower_parts = [p.lower() for p in posix_parts]
        suffixes = []

        if "versions" in lower_parts:
            idx = lower_parts.index("versions")
            if idx + 2 < len(posix_parts):
                suffixes.append(Path(*posix_parts[idx + 2:]))

        if "wikiart" in lower_parts:
            idx = lower_parts.index("wikiart")
            if idx + 1 < len(posix_parts):
                suffixes.append(Path(*posix_parts[idx + 1:]))

        if len(posix_parts) >= 3:
            suffixes.append(Path(*posix_parts[-3:]))

        for root in wikiart_roots:
            for suffix in suffixes:
                recovered = (root / suffix).resolve()
                if recovered.exists():
                    return recovered

    # Final fallback by basename index .
    basename = Path(path_str).name.lower()
    if basename and basename in basename_index:
        return basename_index[basename]

    return None


class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(Config.DATA_FILE)
        self.wikiart_roots = _discover_wikiart_roots()
        self.basename_index = {}
        self._basename_index_built = False

    def _build_basename_index(self):
        if self._basename_index_built:
            return

        self._basename_index_built = True
        for root in self.wikiart_roots:
            for p in root.rglob("*"):
                if p.is_file():
                    key = p.name.lower()
                    if key not in self.basename_index:
                        self.basename_index[key] = p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = _resolve_image_path(item["image"], self.wikiart_roots, self.basename_index)
        text = item["text"]

        if image_path is None and not self._basename_index_built:
            self._build_basename_index()
            image_path = _resolve_image_path(item["image"], self.wikiart_roots, self.basename_index)

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