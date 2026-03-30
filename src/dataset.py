from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from pathlib import Path
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

from utils.helpers import load_json
from utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PREFIX = "a painting that evokes "


def _to_abs(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_emotion_text(text):
    text = str(text).strip().lower()
    if text.startswith(PROMPT_PREFIX):
        return text[len(PROMPT_PREFIX):].strip()
    return text


def format_emotion_prompt(emotion):
    return f"{PROMPT_PREFIX}{str(emotion).strip().lower()}"


def _normalize_emotion_list(items):
    if not items:
        return []

    return sorted(
        {
            normalize_emotion_text(item)
            for item in items
            if str(item).strip()
        }
    )


def _emotion_counts(records):
    counts = defaultdict(int)
    for record in records:
        emotion = record.get("emotion")
        if emotion:
            counts[emotion] += 1
    return dict(counts)


def compute_zero_shot_emotion_split(records):
    """
    Build deterministic label-held-out split for strict zero-shot workflows.
    Returns a dict with holdout/seen emotions and per-emotion frequencies.
    """
    normalized_records = []
    for record in records:
        emotion = normalize_emotion_text(record.get("emotion", record.get("text", "")))
        if not emotion:
            continue
        normalized_records.append({**record, "emotion": emotion})

    if not normalized_records:
        raise RuntimeError("No emotion records found while building zero-shot split.")

    counts = _emotion_counts(normalized_records)
    all_emotions = sorted(counts.keys())

    configured = _normalize_emotion_list(getattr(Config, "ZERO_SHOT_HOLDOUT_EMOTIONS", None))
    if configured:
        missing = sorted([e for e in configured if e not in counts])
        holdout = [e for e in configured if e in counts]
        if missing:
            print(
                "Warning: Ignoring configured ZERO_SHOT_HOLDOUT_EMOTIONS not present in data: "
                f"{missing}"
            )
        if not holdout:
            raise RuntimeError(
                "ZERO_SHOT_HOLDOUT_EMOTIONS is configured, but none of those emotions exist in data."
            )
        source = "config"
    else:
        ratio = float(getattr(Config, "ZERO_SHOT_HOLDOUT_RATIO", 0.2))
        ratio = max(0.05, min(0.8, ratio))
        min_count = int(getattr(Config, "ZERO_SHOT_HOLDOUT_MIN_COUNT", 1))
        min_holdout = int(getattr(Config, "ZERO_SHOT_MIN_HOLDOUT_EMOTIONS", 1))

        eligible = [e for e in all_emotions if counts.get(e, 0) >= min_count]
        if not eligible:
            eligible = all_emotions

        target = max(min_holdout, int(round(len(eligible) * ratio)))
        target = max(1, min(target, max(len(eligible) - 1, 1)))

        seed = int(getattr(Config, "ZERO_SHOT_SPLIT_SEED", getattr(Config, "SPLIT_SEED", 42)))
        rng = random.Random(seed)
        eligible = sorted(eligible)
        rng.shuffle(eligible)
        holdout = sorted(eligible[:target])
        source = "auto"

    holdout_set = set(holdout)
    seen = sorted([e for e in all_emotions if e not in holdout_set])

    if not seen:
        raise RuntimeError(
            "Strict zero-shot split invalid: all emotions were assigned to holdout and none left for training."
        )

    return {
        "source": source,
        "holdout_emotions": holdout,
        "seen_emotions": seen,
        "all_emotions": all_emotions,
        "emotion_counts": counts,
    }


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
    def __init__(
        self,
        split="train",
        val_ratio=0.2,
        split_seed=42,
        allowed_emotions: Optional[Sequence[str]] = None,
        excluded_emotions: Optional[Sequence[str]] = None,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")

        all_data = load_json(_to_abs(Config.DATA_FILE))
        grouped = defaultdict(list)
        allowed_set: Optional[Set[str]] = None
        excluded_set: Set[str] = set()

        if allowed_emotions is not None:
            allowed_set = set(_normalize_emotion_list(allowed_emotions))
        if excluded_emotions is not None:
            excluded_set = set(_normalize_emotion_list(excluded_emotions))

        path_cache = {}

        for item in all_data:
            image_key = str(item.get("image", "")).strip()
            image_rel = str(item.get("image_rel", "")).strip()
            text = str(item.get("text", "")).strip()
            if not image_key or not text:
                continue

            resolved = path_cache.get((image_key, image_rel))
            if resolved is None:
                resolved_path = resolve_image_path(image_key)
                if resolved_path is None and image_rel:
                    resolved_path = resolve_image_path(image_rel)
                resolved = str(resolved_path) if resolved_path is not None else ""
                path_cache[(image_key, image_rel)] = resolved

            if not resolved:
                continue

            emotion = normalize_emotion_text(text)

            if allowed_set is not None and emotion not in allowed_set:
                continue
            if excluded_set and emotion in excluded_set:
                continue

            grouped[image_key].append(
                {
                    "image": image_key,
                    "image_rel": image_rel,
                    "image_resolved": resolved,
                    "text": text,
                    "emotion": emotion,
                    "image_key": image_key,
                }
            )

        image_keys = sorted(grouped.keys())
        rng = random.Random(split_seed)
        rng.shuffle(image_keys)

        val_count = int(len(image_keys) * max(0.0, min(1.0, float(val_ratio))))
        val_keys = set(image_keys[:val_count])

        if split == "train":
            selected_keys = [k for k in image_keys if k not in val_keys]
        elif split == "val":
            selected_keys = [k for k in image_keys if k in val_keys]
        else:
            selected_keys = image_keys

        self.data = []
        self.image_to_emotions = {}
        for image_key in selected_keys:
            records = grouped[image_key]
            self.data.extend(records)
            self.image_to_emotions[image_key] = {r["emotion"] for r in records if r.get("emotion")}

        self.emotions = sorted(
            {
                r["emotion"]
                for r in self.data
                if r.get("emotion")
            }
        )

        self.valid_count = len(self.data)

    def __len__(self):
        return len(self.data)

    def sample_negative_emotion(self, image_key, current_emotion=None, rng=None, candidate_pool=None):
        rng = rng or random

        positives = self.image_to_emotions.get(image_key, set())
        pool = self.emotions if candidate_pool is None else list(candidate_pool)
        candidates = [e for e in pool if e not in positives]

        if not candidates and current_emotion is not None:
            candidates = [e for e in pool if e != current_emotion]

        if not candidates:
            return None

        return rng.choice(candidates)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = Path(item["image_resolved"])
        text = item["text"]
        emotion = item.get("emotion")
        image_key = item.get("image_key", item["image"])

        if not image_path.exists():
            return None

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            return None

        return {
            "image": image,
            "text": text,
            "emotion": emotion,
            "image_key": image_key,
        }


processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=False)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    emotions = [item.get("emotion") for item in batch]
    image_keys = [item.get("image_key") for item in batch]

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
    inputs["raw_emotions"] = emotions
    inputs["image_keys"] = image_keys
    return inputs