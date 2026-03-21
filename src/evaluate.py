import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset, collate_fn
from src.model import CLIPFineTuner
from utils.config import Config


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = ArtDataset()
    print(f"Dataset loaded with {len(dataset)} items.")

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / Config.CHECKPOINT_FILE

    if not checkpoint_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Train first.")

    print(f"Loading model from '{checkpoint_path}'...")
    model = CLIPFineTuner().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Evaluating...")

    correct = 0
    total = 0
    skipped = 0

    with torch.no_grad():
        for batch in tqdm(loader):

            if batch is None:
                skipped += 1
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            img_emb, txt_emb = model(batch)

            # normalize embeddings
            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)

            # similarity matrix (B x B)
            logits = img_emb @ txt_emb.T

            preds = logits.argmax(dim=1)
            labels = torch.arange(len(img_emb)).to(device)

            correct += (preds == labels).sum().item()
            total += len(img_emb)

    acc = correct / total if total > 0 else 0

    print(
        f"Zero-shot Retrieval Accuracy: {acc:.4f} | "
        f"Samples: {total} | Skipped batches: {skipped}"
    )


if __name__ == "__main__":
    evaluate()