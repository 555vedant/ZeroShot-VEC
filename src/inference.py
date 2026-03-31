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
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class SearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 🔹 Load model
        self.model = CLIPFineTuner().to(self.device)
        state = torch.load(_to_abs(Config.CHECKPOINT_FILE), map_location=self.device)
        self.model.load_checkpoint_state_dict(state)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=False)

        raw_data = load_json(_to_abs(Config.DATA_FILE))
        self.data = self._build_unique_image_records(raw_data)

        # 🔥 Build index (fast)
        self.image_embeddings = self._build_index()

    def _build_unique_image_records(self, raw_data):
        seen = set()
        unique = []

        for item in raw_data:
            path = item.get("image")
            if path and path not in seen:
                seen.add(path)
                unique.append({"image": path})

        return unique

    def _build_index(self):
        dataset = _ImageDataset(self.data)

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=_image_collate
        )

        all_embeddings = []

        with torch.no_grad():
            for images, paths in loader:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                emb = self.model.encode_images(pixel_values=inputs["pixel_values"])
                emb = emb.detach().cpu()  # move once

                all_embeddings.append(emb)

                # free memory
                del images, inputs, emb

        # 🔥 Single tensor (VERY IMPORTANT)
        return torch.cat(all_embeddings, dim=0)

    def search(self, query, top_k=5):
        inputs = self.processor(text=[query], return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_emb = self.model.encode_text(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        text_emb = text_emb.cpu()

        # 🔥 FAST vectorized similarity
        sims = torch.matmul(self.image_embeddings, text_emb.T).squeeze()

        top_k = min(top_k, len(sims))
        scores, indices = torch.topk(sims, k=top_k)

        return [self.data[i]["image"] for i in indices.tolist()]


class _ImageDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        path = resolve_image_path(item["image"])

        try:
            with Image.open(path) as img:
                return img.convert("RGB"), str(path)
        except:
            return None


def _image_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []

    images, paths = zip(*batch)
    return list(images), list(paths)