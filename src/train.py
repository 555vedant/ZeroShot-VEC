import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re
import random
import json
import hashlib

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    ArtDataset,
    collate_fn,
    processor,
    format_emotion_prompt,
    compute_zero_shot_emotion_split,
)
from src.model import CLIPFineTuner
from src.preprocess import preprocess
from utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)

    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast_context(device: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device, enabled=enabled)

    return torch.cuda.amp.autocast(enabled=enabled)


def _setup_cuda_backend():
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(Config, "TF32", True))
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(getattr(Config, "TF32", True))


def _move_to_device(batch, device, non_blocking):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def _make_loader(dataset, shuffle, batch_size, drop_last=False):
    num_workers = int(getattr(Config, "NUM_WORKERS", 2))
    pin_memory = bool(getattr(Config, "PIN_MEMORY", True)) and torch.cuda.is_available()
    persistent_workers = bool(getattr(Config, "PERSISTENT_WORKERS", True)) and num_workers > 0

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": bool(drop_last),
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(getattr(Config, "PREFETCH_FACTOR", 2))

    return DataLoader(dataset, **kwargs)


def _safe_load_model_state(model, state_dict):
    model.load_checkpoint_state_dict(state_dict)


def _build_optimizer(model):
    base_lr = float(getattr(Config, "LR", 5e-6))
    backbone_lr_mult = float(getattr(Config, "BACKBONE_LR_MULTIPLIER", 0.2))
    weight_decay = float(getattr(Config, "WEIGHT_DECAY", 0.01))

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = (
            ".vision_model." in name
            or ".text_model." in name
            or name.startswith("model.vision_model.")
            or name.startswith("model.text_model.")
            or name.startswith("model.module.vision_model.")
            or name.startswith("model.module.text_model.")
        )

        if is_backbone:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_mult,
                "weight_decay": weight_decay,
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    print(
        "Optimizer groups | "
        f"backbone={len(backbone_params)} params @ {base_lr * backbone_lr_mult:.2e} | "
        f"head={len(head_params)} params @ {base_lr:.2e}"
    )

    return torch.optim.AdamW(param_groups)


# PATH HELPERS
def _to_abs(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _checkpoint_dir() -> Path:
    return _to_abs(Config.CHECKPOINT_FILE).parent / "epoch_checkpoints"


# CHECKPOINT UTILS
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


def _checkpoint_candidates():
    candidates = []
    full_state_candidates = []
    other_candidates = []
    resume_path = _to_abs(getattr(Config, "TRAINING_CHECKPOINT_FILE", Config.CHECKPOINT_FILE))
    if resume_path.exists():
        full_state_candidates.append(resume_path)

    input_dir = _to_abs(getattr(Config, "INPUT_MODEL_DIR", Config.CHECKPOINT_FILE.parent))
    if input_dir.exists():
        for path in input_dir.rglob("*.pth"):
            if path.is_file() and path not in candidates:
                if path.name == resume_path.name:
                    full_state_candidates.append(path)
                else:
                    other_candidates.append(path)

    output_model = _to_abs(Config.CHECKPOINT_FILE)
    if output_model.exists() and output_model not in full_state_candidates:
        other_candidates.append(output_model)

    candidates.extend(sorted(full_state_candidates, key=lambda path: path.stat().st_mtime, reverse=True))
    candidates.extend(sorted(other_candidates, key=lambda path: path.stat().st_mtime, reverse=True))
    return candidates


def _cleanup_old_checkpoints(ckpt_dir: Path, keep=3):
    files = sorted(ckpt_dir.glob("epoch_*.pth"), key=_extract_epoch)
    if len(files) > keep:
        for f in files[:-keep]:
            f.unlink()
            print(f"Deleted old checkpoint: {f}")


def _save_latest_checkpoint(ckpt_dir, epoch, model, optimizer, scaler):
    checkpoint_path = _to_abs(
        getattr(Config, "TRAINING_CHECKPOINT_FILE", Config.CHECKPOINT_FILE)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    payload = {
        "checkpoint_version": 2,
        "epoch": epoch + 1,
        "model_state_dict": model.checkpoint_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_loss": getattr(model, "_best_loss", float("inf")),
        "split_signature": getattr(model, "_split_signature", None),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    print(f"Saved resumable checkpoint after epoch {epoch + 1}: {checkpoint_path}")


def _try_resume_training(model, optimizer, scaler, device, expected_split_signature=None):
    candidates = _checkpoint_candidates()
    if not candidates:
        print("No existing checkpoint found. Starting fresh.")
        return 0, float("inf")

    for checkpoint_path in candidates:
        print(f"Trying checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            is_full_checkpoint = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            state_dict = checkpoint["model_state_dict"] if is_full_checkpoint else checkpoint
            model.load_checkpoint_state_dict(state_dict)

            checkpoint_signature = checkpoint.get("split_signature") if is_full_checkpoint else None
            if expected_split_signature and checkpoint_signature and checkpoint_signature != expected_split_signature:
                print("Skipping checkpoint: zero-shot split does not match current data split.")
                continue

            if is_full_checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scaler is not None and checkpoint.get("scaler_state_dict"):
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_epoch = int(checkpoint.get("epoch", 0))
                best_loss = float(checkpoint.get("best_loss", float("inf")))
            else:
                start_epoch = 0
                best_loss = float("inf")

            remaining_epochs = max(0, int(Config.EPOCHS) - start_epoch)
            print(
                f"Resumed after epoch {start_epoch} using {checkpoint_path}. "
                f"Remaining epochs: {remaining_epochs}"
            )
            return start_epoch, best_loss
        except Exception as error:
            print(f"Could not use {checkpoint_path}: {error}")

    print("No compatible checkpoint found. Starting fresh.")
    return 0, float("inf")

def _compute_split_signature(split_plan):
    payload = {
        "seen_emotions": sorted(split_plan.get("seen_emotions", [])),
        "holdout_emotions": sorted(split_plan.get("holdout_emotions", [])),
        "source": split_plan.get("source", "unknown"),
        "seed": int(getattr(Config, "ZERO_SHOT_SPLIT_SEED", getattr(Config, "SPLIT_SEED", 42))),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# LOSS
def matching_bce_loss(pos_logits, neg_logits):
    pos_targets = torch.ones_like(pos_logits)
    neg_targets = torch.zeros_like(neg_logits)

    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_targets)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets)  
    return 0.5 * (pos_loss + neg_loss)


# CLIPFIT: kd loss
def clipfit_kd_loss(student_embeds, teacher_embeds):
    student_embeds = F.normalize(student_embeds, dim=-1)
    teacher_embeds = F.normalize(teacher_embeds, dim=-1)
    return 1.0 - (student_embeds * teacher_embeds).sum(dim=-1).mean()


# CLIPFIT:
def _clipfit_teacher_image_embeddings(teacher, pixel_values):
    outputs = teacher.vision_model(pixel_values=pixel_values)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs[1]
    projected = teacher.visual_projection(pooled)
    return F.normalize(projected, dim=-1)


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

    non_blocking = bool(getattr(Config, "NON_BLOCKING", True))
    return {k: v.to(device, non_blocking=non_blocking) for k, v in neg_inputs.items()}


def _run_epoch(model, loader, optimizer, scaler, use_amp, device, dataset, rng, train_mode, teacher=None):
    total_loss = 0.0
    total_bce_loss = 0.0
    total_kd_loss = 0.0
    steps = 0
    skipped = 0

    if train_mode:
        model.train()
    else:
        model.eval()

    non_blocking = bool(getattr(Config, "NON_BLOCKING", True))

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

        batch = _move_to_device(batch, device=device, non_blocking=non_blocking)
        neg_inputs = _build_negative_text_inputs(dataset, image_keys, emotions, device, rng)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if train_mode:
            autocast_ctx = _autocast_context(device=device, enabled=use_amp)
        else:
            autocast_ctx = _autocast_context(device=device, enabled=False)

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

                bce_loss = matching_bce_loss(pos_logits, neg_logits)
                kd_loss = torch.zeros((), device=device)
                if teacher is not None:
                    student_image_embeds = model.encode_images(batch["pixel_values"])
                    with torch.no_grad():
                        teacher_image_embeds = _clipfit_teacher_image_embeddings(
                            teacher, batch["pixel_values"]
                        )
                    kd_loss = clipfit_kd_loss(student_image_embeds, teacher_image_embeds)

                loss = bce_loss + float(getattr(Config, "CLIPFIT_KD_WEIGHT", 8.0)) * kd_loss

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
        total_bce_loss += bce_loss.item()
        total_kd_loss += kd_loss.item()
        steps += 1

        if train_mode and step % 50 == 0:
            print(
                f"Step {step} | BCE {bce_loss.item():.4f} | "
                f"KD {kd_loss.item():.4f} | Total {loss.item():.4f}"
            )

    avg_loss = total_loss / max(steps, 1)
    return (
        avg_loss,
        total_bce_loss / max(steps, 1),
        total_kd_loss / max(steps, 1),
        steps,
        skipped,
    )


# TRAIN
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _setup_cuda_backend()

    split_seed = getattr(Config, "SPLIT_SEED", 42)
    val_split = getattr(Config, "VAL_SPLIT", 0.2)

    all_dataset = ArtDataset(split="all", val_ratio=val_split, split_seed=split_seed)
    split_plan = compute_zero_shot_emotion_split(all_dataset.data)
    seen_emotions = split_plan["seen_emotions"]
    holdout_emotions = split_plan["holdout_emotions"]
    split_signature = _compute_split_signature(split_plan)

    print(
        "Strict zero-shot plan | "
        f"source={split_plan['source']} | seen_train={len(seen_emotions)} | holdout_eval={len(holdout_emotions)}"
    )
    print(f"Holdout emotions (excluded from training): {holdout_emotions}")

    train_dataset = ArtDataset(
        split="train",
        val_ratio=val_split,
        split_seed=split_seed,
        allowed_emotions=seen_emotions,
    )
    val_dataset = ArtDataset(
        split="val",
        val_ratio=val_split,
        split_seed=split_seed,
        allowed_emotions=seen_emotions,
    )

    if len(train_dataset) == 0:
        print("No valid training pairs found. Rebuilding pairs.json via preprocess()...")
        preprocess()
        all_dataset = ArtDataset(split="all", val_ratio=val_split, split_seed=split_seed)
        split_plan = compute_zero_shot_emotion_split(all_dataset.data)
        seen_emotions = split_plan["seen_emotions"]
        holdout_emotions = split_plan["holdout_emotions"]
        split_signature = _compute_split_signature(split_plan)

        train_dataset = ArtDataset(
            split="train",
            val_ratio=val_split,
            split_seed=split_seed,
            allowed_emotions=seen_emotions,
        )
        val_dataset = ArtDataset(
            split="val",
            val_ratio=val_split,
            split_seed=split_seed,
            allowed_emotions=seen_emotions,
        )

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    train_drop_last = (
        bool(getattr(Config, "DROP_LAST_MULTI_GPU_TRAIN", True))
        and device == "cuda"
        and gpu_count > 1
        and len(train_dataset) >= int(Config.BATCH_SIZE)
    )

    train_loader = _make_loader(
        train_dataset,
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        drop_last=train_drop_last,
    )
    val_loader = _make_loader(
        val_dataset,
        shuffle=False,
        batch_size=getattr(Config, "EVAL_BATCH_SIZE", Config.BATCH_SIZE),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty after rebuild. Dataset paths are likely invalid in this runtime. "
            "Run preprocess.py and verify Config.BASE_PATH points to mounted WikiArt data."
        )

    model = CLIPFineTuner().to(device)

    # CLIPFIT: Create the teacher before any checkpoint can alter the student.
    teacher = model.create_clipfit_teacher().to(device) if getattr(
        Config, "FINE_TUNING_STRATEGY", "full"
    ) == "clipfit" else None

    if device == "cuda" and getattr(Config, "MULTI_GPU", True) and gpu_count > 1:
        model.enable_data_parallel()
        print(f"Using DataParallel on {gpu_count} GPUs")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found.")

    print(
        f"Parameters | total={total_params:,} | trainable={trainable_params:,} | "
        f"trainable%={100.0 * trainable_params / max(total_params, 1):.4f}%"
    )

    optimizer = _build_optimizer(model)

    use_amp = Config.MIXED_PRECISION and device == "cuda"
    scaler = _make_grad_scaler(use_amp=use_amp)

    model.train()

    model._split_signature = split_signature
    start_epoch, best_loss = _try_resume_training(
        model,
        optimizer,
        scaler,
        device,
        expected_split_signature=split_signature,
    )

    if start_epoch >= Config.EPOCHS:
        print("Training already completed.")
        return

    rng_train = random.Random(getattr(Config, "NEGATIVE_SEED", 123))
    rng_val = random.Random(getattr(Config, "NEGATIVE_SEED", 123) + 1)

    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

    # TRAIN LOOP
    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        train_loss, train_bce, train_kd, train_steps, train_skipped = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            dataset=train_dataset,
            rng=rng_train,
            train_mode=True,
            teacher=teacher,
        )

        if train_steps == 0:
            raise RuntimeError("No valid training batches were produced.")

        if len(val_dataset) > 0:
            val_loss, val_bce, val_kd, val_steps, val_skipped = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=False,
                device=device,
                dataset=val_dataset,
                rng=rng_val,
                train_mode=False,
                teacher=teacher,
            )
        else:
            val_loss, val_bce, val_kd, val_steps, val_skipped = train_loss, train_bce, train_kd, 0, 0

        print(
            f"Epoch {epoch + 1} | Train BCE: {train_bce:.4f} | "
            f"Train KD: {train_kd:.4f} | Train Total: {train_loss:.4f} "
            f"(steps={train_steps}, skipped={train_skipped}) | "
            f"Val BCE: {val_bce:.4f} | Val KD: {val_kd:.4f} | "
            f"Val Total: {val_loss:.4f} (steps={val_steps}, skipped={val_skipped})"
        )

        checkpoint_interval = max(1, int(getattr(Config, "CHECKPOINT_INTERVAL", 10)))
        checkpoint_due = (
            (epoch + 1) % checkpoint_interval == 0
            or epoch + 1 == Config.EPOCHS
        )

        # Persist progress only at configured checkpoint boundaries.
        if checkpoint_due and val_loss < best_loss:
            best_loss = val_loss
            best_path = _to_abs(Config.CHECKPOINT_FILE)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.checkpoint_state_dict(), best_path)
            print(f"Saved BEST model at epoch {epoch + 1} to {best_path}")

        model._best_loss = best_loss
        if checkpoint_due:
            _save_latest_checkpoint(None, epoch, model, optimizer, scaler)
        else:
            print(f"Checkpoint skipped at epoch {epoch + 1}; next save at epoch "
                  f"{min(Config.EPOCHS, ((epoch // checkpoint_interval) + 1) * checkpoint_interval)}")

    print("Training complete.")


if __name__ == "__main__":
    train()
