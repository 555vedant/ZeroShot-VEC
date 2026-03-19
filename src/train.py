import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset
from src.model import CLIPFineTuner
from utils.config import Config


def contrastive_loss(img, txt):
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)

    logits = img @ txt.T / Config.TEMPERATURE
    labels = torch.arange(len(img)).to(img.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ArtDataset()
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = CLIPFineTuner().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR)

    scaler = GradScaler(enabled=Config.MIXED_PRECISION)

    model.train()

    for epoch in range(Config.EPOCHS):
        total_loss = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=Config.MIXED_PRECISION):
                img_emb, txt_emb = model(batch)
                loss = contrastive_loss(img_emb, txt_emb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "clip_model.pth")