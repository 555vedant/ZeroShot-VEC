import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor
from torch.utils.data import DataLoader, Dataset

from src.model import CLIPFineTuner
from src.dataset import format_emotion_prompt, resolve_image_path
from utils.helpers import load_json
from utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def calibrate_rejection_thresholds(top1_scores, margins, quantile=0.05):
    """Derive rejection cutoffs from validation score distributions."""
    if not top1_scores or not margins:
        raise ValueError("Validation scores are required for rejection calibration.")

    top1 = torch.tensor(top1_scores, dtype=torch.float32)
    margin = torch.tensor(margins, dtype=torch.float32)
    q = max(0.0, min(1.0, float(quantile)))
    return {
        "top1_threshold": float(torch.quantile(top1, q).item()),
        "margin_threshold": float(torch.quantile(margin, q).item()),
        "quantile": q,
        "samples": len(top1_scores),
    }


def _to_abs(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class SearchEngine:
    def __init__(self, checkpoint_path=None, data_path=None, image_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Defaults fallback to config if not provided
        cp_path = checkpoint_path if checkpoint_path else _to_abs(Config.CHECKPOINT_FILE)
        dp_path = data_path if data_path else _to_abs(Config.DATA_FILE)

        #  Load model
        self.model = CLIPFineTuner().to(self.device)
        state = torch.load(cp_path, map_location=self.device)
        self.model.load_checkpoint_state_dict(state)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME, use_fast=False)

        self.image_dir = Path(image_dir) if image_dir else None
        if self.image_dir is not None:
            self.data = self._build_directory_records(self.image_dir)
        else:
            raw_data = load_json(dp_path)
            self.data = self._build_unique_image_records(raw_data)
#build index fast
        self.image_embeddings = self._build_index()

    @staticmethod
    def _build_directory_records(image_dir):
        directory = Path(image_dir)
        if not directory.exists():
            return []

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return [
            {"image": str(path), "image_rel": str(path)}
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]

    def refresh_index(self):
        """Rebuild the upload-only index so newly uploaded images are searchable."""
        if self.image_dir is None:
            return
        self.data = self._build_directory_records(self.image_dir)
        self.image_embeddings = self._build_index()

    def _build_unique_image_records(self, raw_data):
        seen = set()
        unique = []

        for item in raw_data:
            path = item.get("image")
            if path and path not in seen:
                seen.add(path)
                unique.append({
                    "image": path,
                    "image_rel": item.get("image_rel", ""),
                })

        return unique

    def _build_index(self):
        dataset = _ImageDataset(self.data)

        loader = DataLoader(
            dataset,
            batch_size=getattr(Config, "INDEX_BATCH_SIZE", 64),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_image_collate
        )

        all_embeddings = []
        valid_paths = []

        with torch.no_grad():
            for images, paths in loader:
                if not images:
                    continue
                
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                emb = self.model.encode_images(pixel_values=inputs["pixel_values"])
                emb = emb.detach().cpu()  # move once

                all_embeddings.append(emb)
                valid_paths.extend(paths)

                # free memory
                del images, inputs, emb

        if not all_embeddings:
            source = f"upload directory '{self.image_dir}'" if self.image_dir else f"dataset path '{Config.DATA_FILE}'"
            raise ValueError(
                f"No valid images were loaded from {source}. "
                "Add image files before building the retrieval index."
            )

        records_by_path = {}
        for record in self.data:
            resolved = self._resolve_record_path(record)
            if resolved is not None:
                records_by_path[str(resolved)] = record
        self.data = [records_by_path[path] for path in valid_paths if path in records_by_path]

        return torch.cat(all_embeddings, dim=0)

    @staticmethod
    def _resolve_record_path(record):
        for candidate in (record.get("image", ""), record.get("image_rel", "")):
            resolved = resolve_image_path(candidate)
            if resolved is not None:
                return resolved
        return None

    def _encode_text(self, text):
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=Config.TEXT_MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            return self.model.encode_text(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

    def search_with_scores(self, query, top_k=5):
        """Return the top artwork matches for an arbitrary text query."""
        if not str(query).strip():
            raise ValueError("query must not be empty")

        self.refresh_index()

        text_emb = self._encode_text(query).cpu()

        sims = torch.matmul(self.image_embeddings, text_emb.T).squeeze()

        if sims.ndim == 0:
            sims = sims.unsqueeze(0)
        top_k = min(top_k, len(sims))
        if top_k <= 0:
            return []
        scores, indices = torch.topk(sims, k=top_k)

        return [
            {
                "image": self.data[i]["image"],
                "score": float(score),
            }
            for score, i in zip(scores.tolist(), indices.tolist())
        ]

    def search(self, query, top_k=5):
        """Backward-compatible text-to-image retrieval returning image paths."""
        return [result["image"] for result in self.search_with_scores(query, top_k)]

    def score_image_against_emotions(self, image, emotions):
        """Rank an image against canonical emotion prompts using cosine similarity."""
        emotion_list = [str(emotion).strip().lower() for emotion in emotions if str(emotion).strip()]
        if not emotion_list:
            return []

        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(
            text=[format_emotion_prompt(emotion) for emotion in emotion_list],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=Config.TEXT_MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            image_emb = self.model.encode_images(pixel_values=image_inputs["pixel_values"])
            text_emb = self.model.encode_text(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )

        scores = ((image_emb @ text_emb.T) / Config.TEMPERATURE).squeeze(0).cpu().tolist()
        ranked = sorted(
            zip(emotion_list, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [{"emotion": emotion, "score": float(score)} for emotion, score in ranked]


class _ImageDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        path = SearchEngine._resolve_record_path(item)

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