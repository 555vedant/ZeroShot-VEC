# import sys
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from src.dataset import ArtDataset, collate_fn
# from src.model import CLIPFineTuner
# from utils.config import Config


# def evaluate():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"device: {device}")

#     print("loading dataset...")
#     dataset = ArtDataset()
#     print(f"dataset size: {len(dataset)}")

#     loader = DataLoader(
#         dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     project_root = Path(__file__).resolve().parent.parent
#     ckpt_path = project_root / Config.CHECKPOINT_FILE

#     if not ckpt_path.exists():
#         raise FileNotFoundError(f"checkpoint not found at {ckpt_path} — run training first.")

#     print(f"loading checkpoint: {ckpt_path}")
#     model = CLIPFineTuner().to(device)
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()

#     img_emb_list = []
#     txt_emb_list = []
#     ground_truth_texts = []
#     n_skipped = 0
#     n_processed = 0
#     n_errors = 0

#     print("extracting embeddings...")
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="embed"):
#             if batch is None:
#                 n_skipped += 1
#                 continue

#             raw_texts = batch.pop("raw_texts", None)
#             if raw_texts is not None:
#                 ground_truth_texts.extend(raw_texts)

#             try:
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 img_emb, txt_emb = model(batch)
#             except Exception as e:
#                 n_errors += 1
#                 n_skipped += 1
#                 if n_errors <= 3:
#                     print(f"  batch failed: {e}")
#                 continue

#             img_emb_list.append(F.normalize(img_emb, dim=-1).cpu())
#             txt_emb_list.append(F.normalize(txt_emb, dim=-1).cpu())
#             n_processed += 1

#     if not img_emb_list:
#         raise RuntimeError(
#             f"no embeddings extracted — processed={n_processed}, skipped={n_skipped}.\n"
#             "check pairs.json for invalid paths or collate_fn for silent failures."
#         )

#     img_embs = torch.cat(img_emb_list, dim=0).to(device)   # [N, D]
#     txt_embs = torch.cat(txt_emb_list, dim=0).to(device)   # [N, D]

#     N = img_embs.shape[0]
#     has_text_labels = len(ground_truth_texts) == N

#     i2t_hits = 0
#     t2i_hits = 0
#     chunk = 1000

#     print("scoring...")
#     for start in tqdm(range(0, N, chunk), desc="retrieval"):
#         end = min(start + chunk, N)

#         img_chunk = img_embs[start:end]
#         txt_chunk = txt_embs[start:end]

#         i2t_preds = (img_chunk @ txt_embs.T).argmax(dim=1).cpu().tolist()
#         t2i_preds = (txt_chunk @ img_embs.T).argmax(dim=1).cpu().tolist()

#         for offset in range(len(i2t_preds)):
#             true_idx = start + offset

#             if has_text_labels:
#                 if ground_truth_texts[i2t_preds[offset]] == ground_truth_texts[true_idx]:
#                     i2t_hits += 1
#                 if ground_truth_texts[t2i_preds[offset]] == ground_truth_texts[true_idx]:
#                     t2i_hits += 1
#             else:
#                 if i2t_preds[offset] == true_idx:
#                     i2t_hits += 1
#                 if t2i_preds[offset] == true_idx:
#                     t2i_hits += 1

#     i2t_acc = i2t_hits / N * 100
#     t2i_acc = t2i_hits / N * 100

#     print(f"\nresults — N={N}, skipped={n_skipped}")
#     print(f"  I2T accuracy: {i2t_acc:.2f}%")
#     print(f"  T2I accuracy: {t2i_acc:.2f}%")


# if __name__ == "__main__":
#     evaluate()

# -----------------------------
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
    print(f"device: {device}")

    print("loading dataset...")
    dataset = ArtDataset()
    print(f"dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / Config.CHECKPOINT_FILE

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt_path} — run training first.")

    print(f"loading checkpoint: {ckpt_path}")
    model = CLIPFineTuner().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    img_emb_list = []
    txt_emb_list = []
    ground_truth_texts = []
    n_skipped = 0
    n_processed = 0
    n_errors = 0

    print("extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="embed"):
            if batch is None:
                n_skipped += 1
                continue

            raw_texts = batch.pop("raw_texts", None)
            if raw_texts is not None:
                ground_truth_texts.extend(raw_texts)

            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                img_emb, txt_emb = model(batch)
            except Exception as e:
                n_errors += 1
                n_skipped += 1
                if n_errors <= 3:
                    print(f"  batch failed: {e}")
                continue

            img_emb_list.append(F.normalize(img_emb, dim=-1).cpu())
            txt_emb_list.append(F.normalize(txt_emb, dim=-1).cpu())
            n_processed += 1

    if not img_emb_list:
        raise RuntimeError(
            f"no embeddings extracted — processed={n_processed}, skipped={n_skipped}.\n"
            "check pairs.json for invalid paths or collate_fn for silent failures."
        )

    img_embs = torch.cat(img_emb_list, dim=0).to(device)   # [N, D]
    txt_embs = torch.cat(txt_emb_list, dim=0).to(device)   # [N, D]

    N = img_embs.shape[0]
    has_text_labels = len(ground_truth_texts) == N

    i2t_hits = 0
    t2i_hits = 0
    chunk = 1000

    print("scoring...")
    for start in tqdm(range(0, N, chunk), desc="retrieval"):
        end = min(start + chunk, N)

        img_chunk = img_embs[start:end]
        txt_chunk = txt_embs[start:end]

        i2t_preds = (img_chunk @ txt_embs.T).argmax(dim=1).cpu().tolist()
        t2i_preds = (txt_chunk @ img_embs.T).argmax(dim=1).cpu().tolist()

        for offset in range(len(i2t_preds)):
            true_idx = start + offset

            if has_text_labels:
                if ground_truth_texts[i2t_preds[offset]] == ground_truth_texts[true_idx]:
                    i2t_hits += 1
                if ground_truth_texts[t2i_preds[offset]] == ground_truth_texts[true_idx]:
                    t2i_hits += 1
            else:
                if i2t_preds[offset] == true_idx:
                    i2t_hits += 1
                if t2i_preds[offset] == true_idx:
                    t2i_hits += 1

    i2t_acc = i2t_hits / N * 100
    t2i_acc = t2i_hits / N * 100

    print(f"\nresults — N={N}, skipped={n_skipped}")
    print(f"  I2T accuracy: {i2t_acc:.2f}%")
    print(f"  T2I accuracy: {t2i_acc:.2f}%")


if __name__ == "__main__":
    evaluate()