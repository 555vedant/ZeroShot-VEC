import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset, collate_fn
from src.model import CLIPFineTuner
from utils.config import Config


# ---------------------------
# LOSS FUNCTION
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

    print("Collate function:", loader.collate_fn)

    model = CLIPFineTuner().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR
    )

    use_amp = Config.MIXED_PRECISION and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()

    for epoch in range(Config.EPOCHS):
        total_loss = 0

        for step, batch in enumerate(loader):

            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                img_emb, txt_emb = model(batch)
                loss = contrastive_loss(img_emb, txt_emb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

        print(f"Epoch {epoch+1} Total Loss: {total_loss:.4f}")

    # ---------------------------
    # SAVE MODEL
    # ---------------------------
    checkpoint_path = Path(__file__).resolve().parent.parent / Config.CHECKPOINT_FILE
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to: {checkpoint_path}")