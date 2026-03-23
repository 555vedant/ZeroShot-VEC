import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import re
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset, collate_fn, processor, format_emotion_prompt
from src.model import CLIPFineTuner
from utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------
# PATH HELPERS
# ---------------------------
def _to_abs(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _checkpoint_dir() -> Path:
    return _to_abs(Config.CHECKPOINT_FILE).parent / "epoch_checkpoints"


# ---------------------------
# CHECKPOINT UTILS
# ---------------------------
def _extract_epoch(path: Path):
    match = re.search(r"epoch_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else None


def _latest_epoch_checkpoint(ckpt_dir: Path):
    checkpoints = []
    for path in ckpt_dir.glob("epoch_*.pth"):
        epoch_idx = _extract_epoch(path)
        if epoch_idx is not None:
            checkpoints.append((epoch_idx, path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def _cleanup_old_checkpoints(ckpt_dir: Path, keep=3):
    files = sorted(ckpt_dir.glob("epoch_*.pth"), key=_extract_epoch)
    if len(files) > keep:
        for f in files[:-keep]:
            f.unlink()
            print(f"Deleted old checkpoint: {f}")


def _save_epoch_checkpoint(ckpt_dir, epoch, model, optimizer, scaler):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epoch_number = epoch + 1
    checkpoint_path = ckpt_dir / f"epoch_{epoch_number}.pth"

    torch.save(
        {
            "epoch": epoch_number,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_loss": getattr(model, "_best_loss", float("inf")),
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    _cleanup_old_checkpoints(ckpt_dir)


def _try_resume_training(model, optimizer, scaler, device):
    ckpt_dir = _checkpoint_dir()
    latest = _latest_epoch_checkpoint(ckpt_dir)

    if latest is None:
        print("No checkpoint found. Starting from epoch 1.")
        return 0, float("inf")

    state = torch.load(latest, map_location=device)

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    completed_epoch = int(state.get("epoch", 0))
    start_epoch = completed_epoch
    best_loss = float(state.get("best_loss", float("inf")))
    model._best_loss = best_loss

    print(f"Resumed from {latest}")
    print(f"Starting from epoch {start_epoch + 1}")

    return start_epoch, best_loss


# ---------------------------
# LOSS
# ---------------------------
def matching_bce_loss(pos_logits, neg_logits):
    pos_targets = torch.ones_like(pos_logits)
    neg_targets = torch.zeros_like(neg_logits)

    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_targets)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets)
    return 0.5 * (pos_loss + neg_loss)


def _build_negative_text_inputs(dataset, image_keys, emotions, device, rng):
    negative_texts = []

    for image_key, emotion in zip(image_keys, emotions):
        neg_emotion = dataset.sample_negative_emotion(
            image_key=image_key,
            current_emotion=emotion,
            rng=rng,
        )

        if neg_emotion is None:
            neg_emotion = emotion

        negative_texts.append(format_emotion_prompt(neg_emotion))

    neg_inputs = processor(
        text=negative_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.TEXT_MAX_LENGTH,
    )

    return {k: v.to(device) for k, v in neg_inputs.items()}


def _run_epoch(model, loader, optimizer, scaler, use_amp, device, dataset, rng, train_mode):
    total_loss = 0.0
    steps = 0
    skipped = 0

    if train_mode:
        model.train()
    else:
        model.eval()

    for step, batch in enumerate(loader):
        if batch is None:
            skipped += 1
            continue

        emotions = batch.pop("raw_emotions", None)
        image_keys = batch.pop("image_keys", None)
        batch.pop("raw_texts", None)

        if not emotions or not image_keys:
            skipped += 1
            continue

        batch = {k: v.to(device) for k, v in batch.items()}
        neg_inputs = _build_negative_text_inputs(dataset, image_keys, emotions, device, rng)

        if train_mode:
            optimizer.zero_grad()

        if train_mode:
            autocast_ctx = autocast(enabled=use_amp)
        else:
            autocast_ctx = autocast(enabled=False)

        with torch.set_grad_enabled(train_mode):
            with autocast_ctx:
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

                loss = matching_bce_loss(pos_logits, neg_logits)

        if train_mode:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        steps += 1

        if train_mode and step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    avg_loss = total_loss / max(steps, 1)
    return avg_loss, steps, skipped


# ---------------------------
# TRAIN
# ---------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_seed = getattr(Config, "SPLIT_SEED", 42)
    val_split = getattr(Config, "VAL_SPLIT", 0.2)

    train_dataset = ArtDataset(split="train", val_ratio=val_split, split_seed=split_seed)
    val_dataset = ArtDataset(split="val", val_ratio=val_split, split_seed=split_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training split is empty. Check pairs.json generation.")

    model = CLIPFineTuner().to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR
    )

    use_amp = Config.MIXED_PRECISION and device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    model.train()

    start_epoch, best_loss = _try_resume_training(model, optimizer, scaler, device)

    if start_epoch >= Config.EPOCHS:
        print("Training already completed.")
        return

    rng_train = random.Random(getattr(Config, "NEGATIVE_SEED", 123))
    rng_val = random.Random(getattr(Config, "NEGATIVE_SEED", 123) + 1)

    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

    # ---------------------------
    # TRAIN LOOP
    # ---------------------------
    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        train_loss, train_steps, train_skipped = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            dataset=train_dataset,
            rng=rng_train,
            train_mode=True,
        )

        if train_steps == 0:
            raise RuntimeError("No valid training batches were produced.")

        if len(val_dataset) > 0:
            val_loss, val_steps, val_skipped = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=False,
                device=device,
                dataset=val_dataset,
                rng=rng_val,
                train_mode=False,
            )
        else:
            val_loss, val_steps, val_skipped = train_loss, 0, 0

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} "
            f"(steps={train_steps}, skipped={train_skipped}) | "
            f"Val Loss: {val_loss:.4f} (steps={val_steps}, skipped={val_skipped})"
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = _to_abs(Config.CHECKPOINT_FILE)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST model at epoch {epoch + 1}")

        # Save checkpoint after best-loss update so resume state is accurate.
        model._best_loss = best_loss
        _save_epoch_checkpoint(_checkpoint_dir(), epoch, model, optimizer, scaler)

    print("Training complete.")


if __name__ == "__main__":
    train()
