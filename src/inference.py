import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor
from torch.utils.data import DataLoader, Dataset

from src.model import CLIPFineTuner
from src.dataset import resolve_image_path
from utils.helpers import load_json
from utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_abs(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class SearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = _to_abs(Config.CHECKPOINT_FILE)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. Run training first to create the model file."
            )

        self.model = CLIPFineTuner().to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(CLIPFineTuner.normalize_checkpoint_state_dict(state))

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.device == "cuda" and getattr(Config, "MULTI_GPU", True) and gpu_count > 1:
            self.model.enable_data_parallel()
            print(f"Search index build using DataParallel on {gpu_count} GPUs")

        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=False)

        raw_data = load_json(_to_abs(Config.DATA_FILE))
        self.data = self._build_unique_image_records(raw_data)

        if len(self.data) == 0:
            raise RuntimeError(
                "No valid images found for search index. Check data/processed/pairs.json image paths."
            )

        self.image_embs = self._build_index()

    @staticmethod
    def _build_unique_image_records(raw_data):
        unique = {}

        for item in raw_data:
            image_path = item.get("image")

            if not image_path:
                continue

            if image_path not in unique:
                unique[image_path] = {"image": image_path}

        return list(unique.values())

    def _build_index(self):
        embs = []
        kept = []

        num_workers = int(getattr(Config, "NUM_WORKERS", 2))
        pin_memory = bool(getattr(Config, "PIN_MEMORY", True)) and self.device == "cuda"
        persistent_workers = bool(getattr(Config, "PERSISTENT_WORKERS", True)) and num_workers > 0

        loader_kwargs = {
            "batch_size": int(getattr(Config, "INDEX_BATCH_SIZE", 128)),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "collate_fn": _image_collate,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(getattr(Config, "PREFETCH_FACTOR", 2))

        dataset = _ImagePathDataset(self.data)
        loader = DataLoader(dataset, **loader_kwargs)
        non_blocking = bool(getattr(Config, "NON_BLOCKING", True))

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue

                images, paths = batch
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=non_blocking) for k, v in inputs.items()}

                out = self.model.encode_images(pixel_values=inputs["pixel_values"])
                embs.append(out)
                kept.extend({"image": path} for path in paths)

        if len(embs) == 0:
            raise RuntimeError(
                "Failed to create image index. All images failed to load."
            )

        self.data = kept
        embs = torch.cat(embs)
        return F.normalize(embs, dim=-1)

    def search(self, query, top_k=Config.SEARCH_TOP_K):
        inputs = self.processor(text=[query], return_tensors="pt")
        non_blocking = bool(getattr(Config, "NON_BLOCKING", True))
        inputs = {k: v.to(self.device, non_blocking=non_blocking) for k, v in inputs.items()}

        with torch.no_grad():
            text_emb = self.model.encode_text(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        text_emb = F.normalize(text_emb, dim=-1)

        sims = (text_emb @ self.image_embs.T).squeeze(0)
        k = min(top_k, len(self.data))
        idx = sims.topk(k).indices.tolist()

        results = [self.data[i]["image"] for i in idx]
        return results


class _ImagePathDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_path = resolve_image_path(item["image"])
        if image_path is None:
            return None

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            return None

        return {"image": image, "path": str(image_path)}


def _image_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images = [x["image"] for x in batch]
    paths = [x["path"] for x in batch]
    return images, paths