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
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = CLIPFineTuner().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Extracting embeddings...")
    image_embeddings = []
    text_embeddings = []
    all_texts = []
    skipped_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embed"):
            if batch is None:
                skipped_batches += 1
                continue

            texts = batch.get("raw_texts", [])
            if "raw_texts" in batch:
                del batch["raw_texts"]
            all_texts.extend(texts)

            batch = {k: v.to(device) for k, v in batch.items()}
            img_emb, txt_emb = model(batch)

            image_embeddings.append(F.normalize(img_emb, dim=-1).cpu())
            text_embeddings.append(F.normalize(txt_emb, dim=-1).cpu())

    if not image_embeddings:
        raise RuntimeError(
            "No valid embeddings extracted. "
            "All batches were skipped, likely due to invalid image paths in pairs.json. "
            "Run preprocess.py once in the current environment and retry."
        )

    print("Stacking embeddings...")
    image_embeddings = torch.cat(image_embeddings, dim=0).to(device)
    text_embeddings = torch.cat(text_embeddings, dim=0).to(device)

    total = image_embeddings.shape[0]
    use_text_match = len(all_texts) == total

    print("Scoring...")
    chunk_size = 1000
    i2t_correct = 0
    t2i_correct = 0

    for start in tqdm(range(0, total, chunk_size), desc="Retrieval"):
        end = min(start + chunk_size, total)

        img_chunk = image_embeddings[start:end]
        txt_chunk = text_embeddings[start:end]

        i2t_preds = (img_chunk @ text_embeddings.T).argmax(dim=1).cpu().tolist()
        t2i_preds = (txt_chunk @ image_embeddings.T).argmax(dim=1).cpu().tolist()

        for offset in range(end - start):
            true_idx = start + offset

            if use_text_match:
                if all_texts[i2t_preds[offset]] == all_texts[true_idx]:
                    i2t_correct += 1
                if all_texts[t2i_preds[offset]] == all_texts[true_idx]:
                    t2i_correct += 1
            else:
                if i2t_preds[offset] == true_idx:
                    i2t_correct += 1
                if t2i_preds[offset] == true_idx:
                    t2i_correct += 1

    i2t_acc = (i2t_correct / total) * 100
    t2i_acc = (t2i_correct / total) * 100

    print(f"\nZero-shot Retrieval Results: Valid samples={total} | Skipped batches={skipped_batches}")
    print(f"Image-to-Text (I2T) Accuracy: {i2t_acc:.2f}%")
    print(f"Text-to-Image (T2I) Accuracy: {t2i_acc:.2f}%")


if __name__ == "__main__":
    evaluate()