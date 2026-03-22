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

    print("Extracting embeddings...")
    
    all_img_embs = []
    all_txt_embs = []
    all_texts = []
    skipped = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting image and text embeddings"):
            if batch is None:
                skipped += 1
                continue

            # Extract raw texts for robust accuracy matching
            texts = batch.pop("raw_texts", None)
            if texts is not None:
                all_texts.extend(texts)

            batch = {k: v.to(device) for k, v in batch.items()}
            
            img_emb, txt_emb = model(batch)

            # normalize embeddings
            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)
            
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())

    print("Stacking embeddings...")
    all_img_embs = torch.cat(all_img_embs, dim=0) # [N, D]
    all_txt_embs = torch.cat(all_txt_embs, dim=0) # [N, D]
    
    N = all_img_embs.shape[0]
    
    print("Calculating final accuracy...")
    i2t_correct = 0
    t2i_correct = 0
    total = N
    
    # Process in chunks to avoid OOM
    chunk_size = 1000
    all_txt_embs = all_txt_embs.to(device)
    all_img_embs = all_img_embs.to(device)
    
    for i in tqdm(range(0, N, chunk_size), desc="Scoring"):
        end = min(i + chunk_size, N)
        img_chunk = all_img_embs[i:end]
        txt_chunk = all_txt_embs[i:end]
        
        # Image-to-Text (I2T)
        logits_i2t = img_chunk @ all_txt_embs.T
        preds_i2t = logits_i2t.argmax(dim=1).cpu().tolist()

        # Text-to-Image (T2I)
        logits_t2i = txt_chunk @ all_img_embs.T
        preds_t2i = logits_t2i.argmax(dim=1).cpu().tolist()
        
        for j in range(len(preds_i2t)):
            true_idx = i + j
            pred_txt_idx = preds_i2t[j]
            pred_img_idx = preds_t2i[j]
            
            # Robust match: if the retrieved text/image has the same ground-truth text, it's a correct match
            if len(all_texts) == N:
                if all_texts[pred_txt_idx] == all_texts[true_idx]:
                    i2t_correct += 1
                if all_texts[pred_img_idx] == all_texts[true_idx]:
                    t2i_correct += 1
            else:
                # Fallback to absolute index match
                if pred_txt_idx == true_idx:
                    i2t_correct += 1
                if pred_img_idx == true_idx:
                    t2i_correct += 1

    i2t_acc = (i2t_correct / total) * 100 if total > 0 else 0
    t2i_acc = (t2i_correct / total) * 100 if total > 0 else 0

    print(f"\nZero-shot Retrieval Results: Valid samples: {total} | Skipped: {skipped}")
    print(f"Image-to-Text (I2T) Accuracy: {i2t_acc:.2f}%")
    print(f"Text-to-Image (T2I) Accuracy: {t2i_acc:.2f}%")


if __name__ == "__main__":
    evaluate()