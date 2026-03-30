import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import random
from collections import defaultdict
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    ArtDataset,
    collate_fn,
    processor,
    format_emotion_prompt,
    resolve_image_path,
    normalize_emotion_text,
)
from src.model import CLIPFineTuner
from utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_abs(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _move_to_device(batch, device, non_blocking):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def _make_loader(dataset, batch_size):
    num_workers = int(getattr(Config, "NUM_WORKERS", 2))
    pin_memory = bool(getattr(Config, "PIN_MEMORY", True)) and torch.cuda.is_available()
    persistent_workers = bool(getattr(Config, "PERSISTENT_WORKERS", True)) and num_workers > 0

    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(getattr(Config, "PREFETCH_FACTOR", 2))

    return DataLoader(dataset, **kwargs)


def _binary_metrics(scores, labels, threshold=0.5):
    preds = (scores >= threshold).long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = labels.numel()
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _auroc(scores, labels):
    positives = (labels == 1).sum().item()
    negatives = (labels == 0).sum().item()

    if positives == 0 or negatives == 0:
        return float("nan")

    pairs = list(zip(scores.tolist(), labels.tolist()))
    pairs.sort(key=lambda x: x[0], reverse=True)

    tpr_points = [0.0]
    fpr_points = [0.0]
    tp = 0
    fp = 0

    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr_points.append(tp / positives)
        fpr_points.append(fp / negatives)

    auc = 0.0
    for i in range(1, len(tpr_points)):
        x1, x2 = fpr_points[i - 1], fpr_points[i]
        y1, y2 = tpr_points[i - 1], tpr_points[i]
        auc += (x2 - x1) * (y1 + y2) * 0.5

    return auc


def _resolve_zero_shot_holdout_emotions(train_dataset, all_dataset):
    configured_holdout = getattr(Config, "ZERO_SHOT_HOLDOUT_EMOTIONS", None)

    if configured_holdout:
        holdout = {
            normalize_emotion_text(item)
            for item in configured_holdout
            if str(item).strip()
        }
        available = set(all_dataset.emotions)
        missing = sorted([e for e in holdout if e not in available])
        holdout = sorted([e for e in holdout if e in available])

        if missing:
            print(
                "Warning: Ignoring configured ZERO_SHOT_HOLDOUT_EMOTIONS not present in data: "
                f"{missing}"
            )

        if not holdout:
            raise RuntimeError(
                "ZERO_SHOT_HOLDOUT_EMOTIONS is configured, but none of those emotions are present in pairs.json."
            )

        return holdout, "config"

    train_emotions = set(train_dataset.emotions)
    all_emotions = set(all_dataset.emotions)
    holdout = sorted(all_emotions - train_emotions)

    if not holdout:
        raise RuntimeError(
            "No unseen emotions found for strict zero-shot evaluation. "
           
        )

    return holdout, "derived"


def _build_zero_shot_eval_dataset(split_seed, val_split):
    train_dataset = ArtDataset(split="train", val_ratio=val_split, split_seed=split_seed)
    all_dataset = ArtDataset(split="all", val_ratio=val_split, split_seed=split_seed)
    all_emotions = sorted(all_dataset.emotions)

    holdout_emotions, source = _resolve_zero_shot_holdout_emotions(train_dataset, all_dataset)
    holdout_set = set(holdout_emotions)

    filtered_records = [record for record in all_dataset.data if record.get("emotion") in holdout_set]

    if not filtered_records:
        raise RuntimeError(
            "Zero-shot holdout emotions were resolved, but no records matched them in pairs.json."
        )

    image_to_emotions = defaultdict(set)
    for record in filtered_records:
        emotion = record.get("emotion")
        image_key = record.get("image_key", record.get("image"))
        if emotion and image_key:
            image_to_emotions[image_key].add(emotion)

    all_dataset.data = filtered_records
    all_dataset.image_to_emotions = dict(image_to_emotions)
    all_dataset.emotions = holdout_emotions
    all_dataset.valid_count = len(filtered_records)

    return {
        "dataset": all_dataset,
        "candidate_emotions": holdout_emotions,
        "train_emotions": train_dataset.emotions,
        "all_emotions": all_emotions,
        "holdout_source": source,
    }


def _ranking_metrics(model, dataset, candidate_emotions, device):
    if len(dataset.image_to_emotions) == 0 or len(candidate_emotions) == 0:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0, "images": 0}

    candidate_prompts = [format_emotion_prompt(e) for e in candidate_emotions]
    text_inputs = processor(
        text=candidate_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.TEXT_MAX_LENGTH,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_emb = model.encode_text(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )

    hit1 = 0
    hit3 = 0
    mrr = 0.0
    image_count = 0

    for image_key, true_emotions in dataset.image_to_emotions.items():
        image_path = resolve_image_path(image_key)

        if image_path is None:
            continue

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_emb = model.encode_images(pixel_values=inputs["pixel_values"])

        scores = (image_emb @ text_emb.T).squeeze(0)
        ranked_idx = torch.argsort(scores, descending=True).tolist()
        ranked_emotions = [candidate_emotions[i] for i in ranked_idx]

        image_count += 1
        if ranked_emotions[0] in true_emotions:
            hit1 += 1
        if any(e in true_emotions for e in ranked_emotions[:3]):
            hit3 += 1

        reciprocal = 0.0
        for rank, emotion in enumerate(ranked_emotions, start=1):
            if emotion in true_emotions:
                reciprocal = 1.0 / rank
                break
        mrr += reciprocal

    if image_count == 0:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0, "images": 0}

    return {
        "hit_at_1": hit1 / image_count,
        "hit_at_3": hit3 / image_count,
        "mrr": mrr / image_count,
        "images": image_count,
    }


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    non_blocking = bool(getattr(Config, "NON_BLOCKING", True))

    split_seed = getattr(Config, "SPLIT_SEED", 42)
    val_split = getattr(Config, "VAL_SPLIT", 0.2)

    print("Building strict zero-shot evaluation split...")
    zero_shot = _build_zero_shot_eval_dataset(split_seed=split_seed, val_split=val_split)
    dataset = zero_shot["dataset"]
    candidate_emotions = zero_shot["candidate_emotions"]
    train_emotions = set(zero_shot["train_emotions"])
    all_emotions = set(zero_shot["all_emotions"])
    unseen_emotions = sorted(set(candidate_emotions))

    print(
        f"Zero-shot pairs: {len(dataset)} | Unique images: {len(dataset.image_to_emotions)} "
        f"| Candidate unseen emotions: {len(candidate_emotions)}"
    )
    print(
        f"Emotion coverage: train_seen={len(train_emotions)} | all={len(all_emotions)} "
        f"| unseen_eval={len(unseen_emotions)}"
    )
    print(f"Holdout source: {zero_shot['holdout_source']}")
    print(f"Unseen emotions: {unseen_emotions}")

    if len(dataset) == 0:
        raise RuntimeError("Zero-shot evaluation split is empty.")

    loader = _make_loader(dataset, batch_size=getattr(Config, "EVAL_BATCH_SIZE", Config.BATCH_SIZE))

    checkpoint_path = _to_abs(Config.CHECKPOINT_FILE)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = CLIPFineTuner().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_checkpoint_state_dict(state)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device == "cuda" and getattr(Config, "MULTI_GPU", True) and gpu_count > 1:
        model.enable_data_parallel()
        print(f"Evaluation using DataParallel on {gpu_count} GPUs")

    model.eval()

    print("Scoring positive and negative zero-shot pairs...")
    pos_scores = []
    neg_scores = []
    skipped_batches = 0
    rng = random.Random(getattr(Config, "NEGATIVE_SEED", 123) + 99)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            if batch is None:
                skipped_batches += 1
                continue

            emotions = batch.pop("raw_emotions", None)
            image_keys = batch.pop("image_keys", None)
            batch.pop("raw_texts", None)

            if not emotions or not image_keys:
                skipped_batches += 1
                continue

            batch = _move_to_device(batch, device=device, non_blocking=non_blocking)

            neg_texts = []
            for image_key, emotion in zip(image_keys, emotions):
                neg_emotion = dataset.sample_negative_emotion(
                    image_key=image_key,
                    current_emotion=emotion,
                    rng=rng,
                )
                if neg_emotion is None:
                    neg_emotion = emotion
                neg_texts.append(format_emotion_prompt(neg_emotion))

            neg_inputs = processor(
                text=neg_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=Config.TEXT_MAX_LENGTH,
            )
            neg_inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in neg_inputs.items()}

            pos_logits = model.pair_logits(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                temperature=Config.TEMPERATURE,
            )

            neg_logits = model.pair_logits(
                pixel_values=batch["pixel_values"],
                input_ids=neg_inputs["input_ids"],
                attention_mask=neg_inputs["attention_mask"],
                temperature=Config.TEMPERATURE,
            )

            pos_scores.append(torch.sigmoid(pos_logits).cpu())
            neg_scores.append(torch.sigmoid(neg_logits).cpu())

    if not pos_scores:
        raise RuntimeError(
            "No valid evaluation scores extracted. "
            "All batches were skipped, likely due to invalid image paths in pairs.json. "
            "Run preprocess.py once in the current environment and retry."
        )

    pos_scores = torch.cat(pos_scores, dim=0)
    neg_scores = torch.cat(neg_scores, dim=0)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(pos_scores, dtype=torch.long),
            torch.zeros_like(neg_scores, dtype=torch.long),
        ],
        dim=0,
    )

    metrics = _binary_metrics(scores=scores, labels=labels, threshold=0.5)
    auc = _auroc(scores=scores, labels=labels)

    ranking = _ranking_metrics(
        model=model,
        dataset=dataset,
        candidate_emotions=candidate_emotions,
        device=device,
    )

    print(f"\nEvaluation Summary: Skipped batches={skipped_batches}")
    print("Pair Matching Metrics (Zero-Shot)")
    print(f"  Pos pairs: {len(pos_scores)} | Neg pairs: {len(neg_scores)}")
    print(f"  Accuracy@0.5: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision@0.5: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall@0.5: {metrics['recall'] * 100:.2f}%")
    print(f"  F1@0.5: {metrics['f1'] * 100:.2f}%")
    print(f"  AUROC: {auc * 100:.2f}%")
    print("Ranking Metrics (Zero-Shot Images)")
    print(f"  Images evaluated: {ranking['images']}")
    print(f"  Hit@1: {ranking['hit_at_1'] * 100:.2f}%")
    print(f"  Hit@3: {ranking['hit_at_3'] * 100:.2f}%")
    print(f"  MRR: {ranking['mrr']:.4f}")


if __name__ == "__main__":
    evaluate()