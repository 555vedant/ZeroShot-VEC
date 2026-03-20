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
        fallback = project_root / "clip_model.pth"
        checkpoint_path = fallback if fallback.exists() else checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run training first to create the model file."
        )

    print(f"Initializing model from '{checkpoint_path}'...")
    model = CLIPFineTuner().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Extracting image and text embeddings...")
    image_embs = []
    text_embs = []
    valid_samples = 0
    skipped_empty_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                skipped_empty_batches += 1
                continue

            batch_size = batch["input_ids"].size(0)
            batch = {k: v.to(device) for k, v in batch.items()}

            img, txt = model(batch)
            image_embs.append(img)
            text_embs.append(txt)
            valid_samples += batch_size

    if not image_embs:
        raise RuntimeError(
            "Evaluation produced zero valid samples. Check image paths in pairs.json and "
            "ensure the dataset cache with images still exists."
        )

    image_embs = F.normalize(torch.cat(image_embs), dim=-1).cpu()
    text_embs = F.normalize(torch.cat(text_embs), dim=-1).cpu()

    print("Calculating final accuracy...")
    n = image_embs.size(0)
    chunk_size = 1024
    correct = 0
    for start in tqdm(range(0, n, chunk_size), desc="Scoring"):
        end = min(start + chunk_size, n)
        sim_chunk = image_embs[start:end] @ text_embs.T
        preds = sim_chunk.argmax(dim=1)
        target = torch.arange(start, end)
        correct += (preds == target).sum().item()

    acc = correct / n
    print(
        f"Zero-shot Retrieval Accuracy: {acc:.4f} | "
        f"Valid samples: {valid_samples} | Skipped empty batches: {skipped_empty_batches}"
    )


if __name__ == "__main__":
    evaluate()