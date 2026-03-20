import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset, collate_fn
from src.model import CLIPFineTuner
from utils.config import Config


# LOSS 
def contrastive_loss(img, txt):
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)

    logits = img @ txt.T / Config.TEMPERATURE
    labels = torch.arange(len(img)).to(img.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


# TRAIN
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ArtDataset()

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Collate function:", loader.collate_fn)

    model = CLIPFineTuner().to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters found. Check FREEZE_VISION/FREEZE_TEXT settings in Config."
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR
    )

    use_amp = Config.MIXED_PRECISION and device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    model.train()

    for epoch in range(Config.EPOCHS):
        total_loss = 0.0
        update_steps = 0
        skipped_empty = 0
        skipped_too_small = 0

        for step, batch in enumerate(loader):

            if batch is None:
                skipped_empty += 1
                continue

            # Contrastive CE with a 1x1 logits matrix is always exactly 0.
            if batch["input_ids"].size(0) < 2:
                skipped_too_small += 1
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    img_emb, txt_emb = model(batch)
                    loss = contrastive_loss(img_emb, txt_emb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                img_emb, txt_emb = model(batch)
                loss = contrastive_loss(img_emb, txt_emb)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            update_steps += 1

            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

        if update_steps == 0:
            raise RuntimeError(
                "No valid training batches were produced. This usually means image paths in "
                "pairs.json are invalid (often after cache cleanup) or almost all samples fail to load."
            )

        avg_loss = total_loss / update_steps
        print(
            f"Epoch {epoch+1} | Total Loss: {total_loss:.4f} | Avg Loss: {avg_loss:.4f} "
            f"| Updates: {update_steps} | Skipped empty: {skipped_empty} "
            f"| Skipped batch<2: {skipped_too_small}"
        )

    # SAVE MODEL
    checkpoint_path = Path(__file__).resolve().parent.parent / Config.CHECKPOINT_FILE
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to: {checkpoint_path}")


if __name__ == "__main__":
    train()