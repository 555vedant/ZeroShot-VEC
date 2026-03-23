import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor

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
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)

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

        with torch.no_grad():
            for item in self.data:
                image_path = resolve_image_path(item["image"])
                if image_path is None:
                    continue

                try:
                    with Image.open(image_path) as img:
                        image = img.convert("RGB")
                except Exception:
                    continue

                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                out = self.model.model.get_image_features(**inputs)
                embs.append(out)
                kept.append({"image": str(image_path)})

        if len(embs) == 0:
            raise RuntimeError(
                "Failed to create image index. All images failed to load."
            )

        self.data = kept
        embs = torch.cat(embs)
        return F.normalize(embs, dim=-1)

    def search(self, query, top_k=Config.SEARCH_TOP_K):
        inputs = self.processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_emb = self.model.model.get_text_features(**inputs)

        text_emb = F.normalize(text_emb, dim=-1)

        sims = (text_emb @ self.image_embs.T).squeeze(0)
        k = min(top_k, len(self.data))
        idx = sims.topk(k).indices.tolist()

        results = [self.data[i]["image"] for i in idx]
        return results