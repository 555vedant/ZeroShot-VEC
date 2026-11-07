import torch
from torch.utils.data import DataLoader
import re
import random
import json
import hashlib
import csv
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    ArtDataset,
    collate_fn,
    processor,
    compute_zero_shot_emotion_split,
    compute_zero_shot_split_signature,
)
from src.model import CLIPFineTuner
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
    return _to_abs(getattr(Config, "TRAINING_CHECKPOINT_FILE", Config.CHECKPOINT_FILE)).parent / "epoch_checkpoints"


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
    resume_path = _to_abs(getattr(Config, "TRAINING_CHECKPOINT_FILE", Config.CHECKPOINT_FILE))
    if resume_path.exists():
        candidates.append(resume_path)

    output_model = _to_abs(Config.CHECKPOINT_FILE)
    if output_model.exists() and output_model not in candidates:
        candidates.append(output_model)

    # Never recursively scan mounted Kaggle inputs. Users can provide one
    # additional checkpoint explicitly through CLIP_CHECKPOINT_PATH.
    explicit_path = os.getenv("CLIP_CHECKPOINT_PATH")
    if explicit_path:
        explicit_checkpoint = _to_abs(explicit_path)
        if explicit_checkpoint.exists() and explicit_checkpoint not in candidates:
            candidates.append(explicit_checkpoint)

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


def _cleanup_old_checkpoints(ckpt_dir: Path, keep=3):
    files = sorted(ckpt_dir.glob("epoch_*.pth"), key=_extract_epoch)
    if len(files) > keep:
        for f in files[:-keep]:
            f.unlink()
            print(f"Deleted old checkpoint: {f}")


def _save_latest_checkpoint(ckpt_dir, epoch, model, optimizer, scaler, scheduler=None):
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
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_loss": getattr(model, "_best_loss", float("inf")),
        "split_signature": getattr(model, "_split_signature", None),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    print(f"Saved latest checkpoint after epoch {epoch + 1}: {checkpoint_path}")


def _try_resume_training(
    model,
    optimizer,
    scaler,
    device,
    expected_split_signature=None,
    scheduler=None,
):
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
            checkpoint_signature = checkpoint.get("split_signature") if is_full_checkpoint else None
            if expected_split_signature and checkpoint_signature and checkpoint_signature != expected_split_signature:
                print("Skipping checkpoint: training strategy or zero-shot split does not match current run.")
                continue
            model.load_checkpoint_state_dict(state_dict)

            if is_full_checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler is not None and checkpoint.get("scheduler_state_dict"):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if scaler is not None and checkpoint.get("scaler_state_dict"):
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_epoch = int(checkpoint.get("epoch", 0))
                best_loss = float(checkpoint.get("best_loss", float("inf")))
            else:
                start_epoch = 0
                best_loss = float("inf")

            additional_epochs = max(0, int(getattr(Config, "EPOCHS", 0)))
            target_epoch = _target_epoch(start_epoch)
            print(
                f"Resumed after epoch {start_epoch} using {checkpoint_path}. "
                f"Additional epochs requested: {additional_epochs} | "
                f"new target epoch: {target_epoch}"
            )
            return start_epoch, best_loss
        except Exception as error:
            print(f"Could not use {checkpoint_path}: {error}")

    print("No compatible checkpoint found. Starting fresh.")
    return 0, float("inf")

def _compute_split_signature(split_plan):
    return compute_zero_shot_split_signature(split_plan)


def _target_epoch(start_epoch):
    requested_epochs = max(0, int(getattr(Config, "EPOCHS", 0)))
    if bool(getattr(Config, "RESUME_ADDITIONAL_EPOCHS", True)):
        return int(start_epoch) + requested_epochs
    return requested_epochs


# LOSS
def _multi_positive_contrastive_loss(logits, image_keys, emotions, dataset):
    batch_size = logits.shape[0]
    positive_mask = torch.zeros(
        (batch_size, batch_size),
        dtype=torch.bool,
        device=logits.device,
    )
    for row, (image_key, emotion) in enumerate(zip(image_keys, emotions)):
        image_emotions = dataset.image_to_emotions.get(image_key, {emotion})
        for column, candidate_emotion in enumerate(emotions):
            if candidate_emotion in image_emotions:
                positive_mask[row, column] = True

    if not positive_mask.any(dim=1).all():
        raise RuntimeError("Every training image must have at least one positive emotion.")

    image_loss = -(
        torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    text_loss = -(
        torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=0)
        - torch.logsumexp(logits, dim=0)
    ).mean()
    return 0.5 * (image_loss + text_loss)


def _run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    use_amp,
    device,
    dataset,
    rng,
    train_mode,
    return_details=False,
):
    total_loss = 0.0
    total_bce_loss = 0.0
    total_kd_loss = 0.0
    total_positive_loss = 0.0
    total_negative_loss = 0.0
    total_ranking_loss = 0.0
    total_positive_logit = 0.0
    total_negative_logit = 0.0
    total_valid_negatives = 0
    total_gradient_norm = 0.0
    nonfinite_batches = 0
    batch_records = []
    steps = 0
    skipped = 0

    if train_mode:
        model.train()
    else:
        model.eval()

    non_blocking = bool(getattr(Config, "NON_BLOCKING", True))

    for step, batch in enumerate(loader):
        if step == 0:
            print(
                f"{'Train' if train_mode else 'Validation'}: first batch received.",
                flush=True,
            )
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
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if train_mode:
            autocast_ctx = _autocast_context(device=device, enabled=use_amp)
        else:
            autocast_ctx = _autocast_context(device=device, enabled=False)

        with torch.set_grad_enabled(train_mode):
            with autocast_ctx:
                if step == 0:
                    print(
                        f"{'Train' if train_mode else 'Validation'}: first batch forward pass...",
                        flush=True,
                    )
                logits = model.contrastive_logits(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                contrastive_loss = _multi_positive_contrastive_loss(
                    logits, image_keys, emotions, dataset
                )
                pos_indices = torch.arange(logits.shape[0], device=device)
                pos_logits = logits[pos_indices, pos_indices]
                negative_mask = ~torch.eye(
                    logits.shape[0], dtype=torch.bool, device=device
                )
                neg_logits = logits.masked_fill(~negative_mask, float("-inf"))
                neg_logits = neg_logits.max(dim=1).values
                loss_components = {
                    "total": contrastive_loss,
                    "positive_loss": contrastive_loss,
                    "negative_loss": torch.zeros_like(contrastive_loss),
                    "ranking_loss": torch.zeros_like(contrastive_loss),
                }
                bce_loss = contrastive_loss
                kd_loss = torch.zeros((), device=device)
                loss = contrastive_loss

        if not torch.isfinite(loss).item():
            nonfinite_batches += 1
            raise FloatingPointError(
                f"Non-finite loss at {'train' if train_mode else 'validation'} batch {step}."
            )

        gradient_norm = 0.0
        if train_mode:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(getattr(Config, "GRADIENT_CLIP_NORM", 1.0)),
                    ).item()
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(getattr(Config, "GRADIENT_CLIP_NORM", 1.0)),
                    ).item()
                )
                optimizer.step()

        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_kd_loss += kd_loss.item()
        total_positive_loss += loss_components["positive_loss"].item()
        total_negative_loss += loss_components["negative_loss"].item()
        total_ranking_loss += loss_components["ranking_loss"].item()
        total_positive_logit += pos_logits.detach().float().mean().item()
        total_negative_logit += neg_logits.detach().float().mean().item()
        valid_negative_count = sum(
            1
            for image_key in image_keys
            if any(
                name not in dataset.image_to_emotions.get(image_key, set())
                for name in dataset.emotions
            )
        )
        total_valid_negatives += valid_negative_count
        total_gradient_norm += gradient_norm
        steps += 1

        if return_details:
            batch_records.append(
                {
                    "batch": step,
                    "batch_bce": bce_loss.item(),
                    "positive_pair_loss": loss_components["positive_loss"].item(),
                    "negative_pair_loss": loss_components["negative_loss"].item(),
                    "ranking_loss": loss_components["ranking_loss"].item(),
                    "learning_rate": (
                        optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
                    ),
                    "gradient_norm": gradient_norm,
                    "positive_logit_avg": pos_logits.detach().float().mean().item(),
                    "negative_logit_avg": neg_logits.detach().float().mean().item(),
                    "emotion_distribution": json.dumps(
                        {
                            emotion: emotions.count(emotion)
                            for emotion in sorted(set(emotions))
                        },
                        sort_keys=True,
                    ),
                    "valid_negatives": valid_negative_count,
                    "skipped_batches": skipped,
                    "nonfinite_batches": nonfinite_batches,
                }
            )

        if train_mode and step % 50 == 0:
            print(
                f"Step {step} | BCE {bce_loss.item():.4f} | "
                f"Pos {loss_components['positive_loss'].item():.4f} | "
                f"Neg {loss_components['negative_loss'].item():.4f} | "
                f"Rank {loss_components['ranking_loss'].item():.4f} | "
                f"Grad {gradient_norm:.4f} | "
                f"PosLogit {pos_logits.detach().float().mean().item():.4f} | "
                f"NegLogit {neg_logits.detach().float().mean().item():.4f}"
            )
        elif step > 0 and step % 25 == 0:
            print(
                f"{'Train' if train_mode else 'Validation'} progress: "
                f"batch {step}/{len(loader)}",
                flush=True,
            )

    avg_loss = total_loss / max(steps, 1)
    result = (
        avg_loss,
        total_bce_loss / max(steps, 1),
        total_kd_loss / max(steps, 1),
        steps,
        skipped,
    )
    if not return_details:
        return result
    return {
        "loss": avg_loss,
        "bce": total_bce_loss / max(steps, 1),
        "kd": total_kd_loss / max(steps, 1),
        "positive_pair_loss": total_positive_loss / max(steps, 1),
        "negative_pair_loss": total_negative_loss / max(steps, 1),
        "ranking_loss": total_ranking_loss / max(steps, 1),
        "positive_logit_avg": total_positive_logit / max(steps, 1),
        "negative_logit_avg": total_negative_logit / max(steps, 1),
        "learning_rate": (
            optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
        ),
        "gradient_norm": total_gradient_norm / max(steps, 1),
        "valid_negatives": total_valid_negatives,
        "steps": steps,
        "skipped": skipped,
        "nonfinite_batches": nonfinite_batches,
        "batch_records": batch_records,
    }


def _append_training_history(epoch, phase, metrics):
    log_path = _to_abs(getattr(Config, "TRAINING_LOG_FILE"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "phase", "batch", "batch_bce", "bce", "positive_pair_loss",
        "negative_pair_loss", "ranking_loss", "learning_rate", "gradient_norm",
        "positive_logit_avg", "negative_logit_avg", "emotion_distribution",
        "valid_negatives", "skipped_batches", "nonfinite_batches",
    ]
    records = metrics.get("batch_records", [])
    records.append(
        {
            "batch": "epoch_average",
            "bce": metrics["bce"],
            "positive_pair_loss": metrics["positive_pair_loss"],
            "negative_pair_loss": metrics["negative_pair_loss"],
            "ranking_loss": metrics["ranking_loss"],
            "learning_rate": metrics["learning_rate"],
            "gradient_norm": metrics["gradient_norm"],
            "positive_logit_avg": metrics["positive_logit_avg"],
            "negative_logit_avg": metrics["negative_logit_avg"],
            "valid_negatives": metrics["valid_negatives"],
            "skipped_batches": metrics["skipped"],
            "nonfinite_batches": metrics["nonfinite_batches"],
        }
    )
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for record in records:
            row = {field: "" for field in fields}
            row.update(record)
            row["epoch"] = epoch
            row["phase"] = phase
            writer.writerow(row)
    print(f"Training history updated: {log_path}")


# TRAIN
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _setup_cuda_backend()
    print(
        f"Training startup: device={device} | CUDA GPUs={torch.cuda.device_count()}",
        flush=True,
    )

    split_seed = getattr(Config, "SPLIT_SEED", 42)
    val_split = getattr(Config, "VAL_SPLIT", 0.2)

    print("Training startup: loading all dataset records...", flush=True)
    all_dataset = ArtDataset(split="all", val_ratio=val_split, split_seed=split_seed)
    print(
        f"Training startup: all dataset ready ({len(all_dataset)} records, "
        f"{len(all_dataset.image_to_emotions)} images).",
        flush=True,
    )
    split_plan = compute_zero_shot_emotion_split(all_dataset.data)
    seen_emotions = split_plan["seen_emotions"]
    holdout_emotions = split_plan["holdout_emotions"]
    holdout_image_keys = {
        image_key
        for image_key, emotions in all_dataset.image_to_emotions.items()
        if set(emotions).intersection(holdout_emotions)
    }
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
        excluded_image_keys=holdout_image_keys,
    )
    val_dataset = ArtDataset(
        split="val",
        val_ratio=val_split,
        split_seed=split_seed,
        allowed_emotions=seen_emotions,
        excluded_image_keys=holdout_image_keys,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty because no source images could be resolved. "
            "Configure Config.BASE_PATH or WIKIART_PATH; pairs.json was not modified."
        )
    print(
        f"Training startup: train={len(train_dataset)} records, "
        f"validation={len(val_dataset)} records.",
        flush=True,
    )

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    configured_gpu_ids = getattr(Config, "GPU_IDS", None)
    if configured_gpu_ids is None:
        active_gpu_ids = list(range(gpu_count))
    else:
        active_gpu_ids = [int(device_id) for device_id in configured_gpu_ids]
    active_gpu_count = len(active_gpu_ids) if device == "cuda" else 0
    train_drop_last = (
        bool(getattr(Config, "DROP_LAST_MULTI_GPU_TRAIN", True))
        and active_gpu_count > 1
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
    print(
        f"Training startup: loaders ready | train batches={len(train_loader)} | "
        f"validation batches={len(val_loader)} | workers={getattr(Config, 'NUM_WORKERS', 0)}",
        flush=True,
    )

    print("Training startup: loading CLIP model...", flush=True)
    model = CLIPFineTuner().to(device)
    print("Training startup: CLIP model ready.", flush=True)

    if (
        device == "cuda"
        and getattr(Config, "MULTI_GPU", True)
        and active_gpu_count > 1
    ):
        model.enable_data_parallel()
        print(
            f"Using DataParallel on GPUs {active_gpu_ids}: "
            + ", ".join(torch.cuda.get_device_name(i) for i in active_gpu_ids)
        )
        if not model.is_data_parallel():
            raise RuntimeError("Multi-GPU was requested but DataParallel is disabled.")
    elif device == "cuda":
        print(f"Using single CUDA GPU: {active_gpu_ids[0] if active_gpu_ids else 0}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found.")

    report = model.parameter_report()
    print(
        f"Parameters | total={report['total']:,} | trainable={report['trainable']:,} | "
        f"trainable%={report['trainable_percentage']:.4f}%"
    )
    print(f"Trainable modules: {report['trainable_modules']}")
    print(f"Frozen visual parameters: {report['frozen_visual']:,}")
    print(f"Frozen text parameters: {report['frozen_text']:,}")

    optimizer = _build_optimizer(model)
    requested_epochs = max(1, int(getattr(Config, "EPOCHS", 1)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=requested_epochs,
        eta_min=float(getattr(Config, "LR", 1e-6)) * 0.05,
    )

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
        scheduler=scheduler,
    )

    additional_epochs = max(0, int(getattr(Config, "EPOCHS", 0)))
    target_epoch = _target_epoch(start_epoch)
    if target_epoch <= start_epoch:
        print("No epochs requested for this run.")
        return

    rng_train = random.Random(getattr(Config, "NEGATIVE_SEED", 123))
    val_seed = getattr(Config, "NEGATIVE_SEED", 123) + 1

    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

    # TRAIN LOOP
    for epoch in range(start_epoch, target_epoch):
        print(
            f"\nEpoch {epoch + 1}/{target_epoch} "
            f"(run epoch {epoch - start_epoch + 1}/{additional_epochs})"
        )
        print("Epoch progress: loading first training batch...", flush=True)

        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            dataset=train_dataset,
            rng=rng_train,
            train_mode=True,
            return_details=True,
        )
        train_loss = train_metrics["loss"]
        train_bce = train_metrics["bce"]
        train_kd = train_metrics["kd"]
        train_steps = train_metrics["steps"]
        train_skipped = train_metrics["skipped"]

        if train_steps == 0:
            raise RuntimeError("No valid training batches were produced.")

        if len(val_dataset) > 0:
            # Reuse the same validation negatives each epoch for comparable model selection.
            rng_val = random.Random(val_seed)
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=False,
                device=device,
                dataset=val_dataset,
                rng=rng_val,
                train_mode=False,
                return_details=True,
            )
            val_loss = val_metrics["loss"]
            val_bce = val_metrics["bce"]
            val_kd = val_metrics["kd"]
            val_steps = val_metrics["steps"]
            val_skipped = val_metrics["skipped"]
        else:
            val_loss, val_bce, val_kd, val_steps, val_skipped = train_loss, train_bce, train_kd, 0, 0
            val_metrics = train_metrics

        print(
            f"Epoch {epoch + 1} | Train BCE: {train_bce:.4f} | "
            f"Train KD: {train_kd:.4f} | Train Total: {train_loss:.4f} "
            f"(steps={train_steps}, skipped={train_skipped}) | "
            f"Val BCE: {val_bce:.4f} | Val KD: {val_kd:.4f} | "
            f"Val Total: {val_loss:.4f} (steps={val_steps}, skipped={val_skipped})"
        )
        _append_training_history(epoch + 1, "train", train_metrics)
        _append_training_history(epoch + 1, "validation", val_metrics)
        scheduler.step()
        print(
            "Learning rates: "
            + ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        )

        # Evaluate and checkpoint every epoch; delayed checkpointing can miss the
        # best generalizing model on a small, noisy dataset.
        checkpoint_due = True
        # Select by supervised validation matching/ranking loss; KD is a training
        # regularizer and should not decide which checkpoint is deployed.
        selection_metric = val_bce
        improved = selection_metric < best_loss

        # Persist progress only at configured checkpoint boundaries.
        if improved:
            best_loss = selection_metric
            best_path = _to_abs(Config.CHECKPOINT_FILE)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_payload = {
                "checkpoint_version": 2,
                "epoch": epoch + 1,
                "model_state_dict": model.checkpoint_state_dict(),
                "best_validation_metric": best_loss,
                "best_loss": best_loss,
                "split_signature": split_signature,
            }
            temporary_best_path = best_path.with_suffix(".tmp")
            torch.save(best_payload, temporary_best_path)
            temporary_best_path.replace(best_path)
            print(f"Saved BEST model at epoch {epoch + 1} to {best_path}")

        model._best_loss = best_loss
        if checkpoint_due:
            _save_latest_checkpoint(
                None,
                epoch,
                model,
                optimizer,
                scaler,
                scheduler=scheduler,
            )

        patience = max(1, int(getattr(Config, "EARLY_STOPPING_PATIENCE", 5)))
        stale_epochs = int(getattr(model, "_stale_epochs", 0))
        stale_epochs = 0 if improved else stale_epochs + 1
        model._stale_epochs = stale_epochs
        if (
            bool(getattr(Config, "EARLY_STOPPING_ENABLED", False))
            and stale_epochs >= patience
        ):
            print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
            break

    print("Training complete.")


if __name__ == "__main__":
    train()
