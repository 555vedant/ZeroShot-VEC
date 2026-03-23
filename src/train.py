import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import re
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset, collate_fn
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
        return 0

    state = torch.load(latest, map_location=device)

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    completed_epoch = state["epoch"]
    start_epoch = completed_epoch

    print(f"Resumed from {latest}")
    print(f"Starting from epoch {start_epoch + 1}")

    return start_epoch


# ---------------------------
# LOSS
# ---------------------------
def contrastive_loss(img, txt):
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)

    logits = img @ txt.T / Config.TEMPERATURE
    labels = torch.arange(len(img)).to(img.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


# ---------------------------
# TRAIN
# ---------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ArtDataset()
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

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

    start_epoch = _try_resume_training(model, optimizer, scaler, device)

    if start_epoch >= Config.EPOCHS:
        print("Training already completed.")
        return

    best_loss = float("inf")

    # ---------------------------
    # TRAIN LOOP
    # ---------------------------
    for epoch in range(start_epoch, Config.EPOCHS):
        total_loss = 0.0
        steps = 0

        for step, batch in enumerate(loader):

            if batch is None:
                continue

            if "raw_texts" in batch:
                del batch["raw_texts"]

            if batch["input_ids"].size(0) < 2:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    img_emb, txt_emb = model(batch)
                    loss = contrastive_loss(img_emb, txt_emb)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                img_emb, txt_emb = model(batch)
                loss = contrastive_loss(img_emb, txt_emb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            steps += 1

            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

        avg_loss = total_loss / max(steps, 1)

        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        _save_epoch_checkpoint(_checkpoint_dir(), epoch, model, optimizer, scaler)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = _to_abs(Config.CHECKPOINT_FILE)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST model at epoch {epoch+1}")

    print("Training complete.")


if __name__ == "__main__":
    train()
