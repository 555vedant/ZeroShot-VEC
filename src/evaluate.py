import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import ArtDataset, collate_fn, processor
from src.model import CLIPFineTuner
from utils.config import Config


PROMPT_PREFIX = "a painting that evokes "
EMOTIONS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
    "something_else",
]
EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


def _emotion_index_from_text(text):
    text = text.strip().lower()
    if text.startswith(PROMPT_PREFIX):
        text = text[len(PROMPT_PREFIX):].strip()
    return EMOTION_TO_INDEX.get(text)


def _balanced_accuracy(preds, labels, num_classes):
    recalls = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_count = class_mask.sum().item()
        if class_count == 0:
            continue

        class_recall = (preds[class_mask] == labels[class_mask]).float().mean().item()
        recalls.append(class_recall)

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def _build_class_text_embeddings(model, device):
    prompts = [f"{PROMPT_PREFIX}{emotion}" for emotion in EMOTIONS]
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.TEXT_MAX_LENGTH,
    )

    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        class_text_emb = model.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
    return F.normalize(class_text_emb, dim=-1)


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
    label_list = []
    skipped_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embed"):
            if batch is None:
                skipped_batches += 1
                continue

            texts = batch.get("raw_texts", [])
            if "raw_texts" in batch:
                del batch["raw_texts"]

            labels = [_emotion_index_from_text(text) for text in texts]
            valid_positions = [idx for idx, label in enumerate(labels) if label is not None]
            if not valid_positions:
                skipped_batches += 1
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            img_emb, txt_emb = model(batch)

            valid_positions = torch.tensor(valid_positions, device=device)
            img_emb = F.normalize(img_emb.index_select(0, valid_positions), dim=-1)
            txt_emb = F.normalize(txt_emb.index_select(0, valid_positions), dim=-1)

            image_embeddings.append(img_emb.cpu())
            text_embeddings.append(txt_emb.cpu())
            label_list.extend([labels[idx] for idx in valid_positions.tolist()])

    if not image_embeddings:
        raise RuntimeError(
            "No valid embeddings extracted. "
            "All batches were skipped, likely due to invalid image paths in pairs.json. "
            "Run preprocess.py once in the current environment and retry."
        )

    image_embeddings = torch.cat(image_embeddings, dim=0).to(device)
    text_embeddings = torch.cat(text_embeddings, dim=0).to(device)
    labels = torch.tensor(label_list, dtype=torch.long, device=device)

    total = labels.numel()
    num_classes = len(EMOTIONS)

    print("Scoring class-wise metrics...")
    class_text_embeddings = _build_class_text_embeddings(model, device)

    # Image -> Text class prediction using class prompts.
    i2t_logits = image_embeddings @ class_text_embeddings.T
    i2t_top1 = i2t_logits.argmax(dim=1)
    i2t_top2 = i2t_logits.topk(k=2, dim=1).indices
    i2t_top1_acc = (i2t_top1 == labels).float().mean().item() * 100
    i2t_top2_acc = (i2t_top2 == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    i2t_balanced_acc = _balanced_accuracy(i2t_top1, labels, num_classes) * 100

    # Build class image prototypes, then score Text -> Image class prediction.
    class_image_prototypes = []
    class_present = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.any():
            prototype = F.normalize(image_embeddings[class_mask].mean(dim=0, keepdim=True), dim=-1)
            class_image_prototypes.append(prototype)
            class_present.append(True)
        else:
            class_image_prototypes.append(torch.zeros((1, image_embeddings.shape[1]), device=device))
            class_present.append(False)

    class_image_prototypes = torch.cat(class_image_prototypes, dim=0)
    class_present = torch.tensor(class_present, dtype=torch.bool, device=device)

    t2i_logits = text_embeddings @ class_image_prototypes.T
    t2i_logits[:, ~class_present] = -1e9
    t2i_top1 = t2i_logits.argmax(dim=1)
    t2i_top2 = t2i_logits.topk(k=2, dim=1).indices
    t2i_top1_acc = (t2i_top1 == labels).float().mean().item() * 100
    t2i_top2_acc = (t2i_top2 == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    t2i_balanced_acc = _balanced_accuracy(t2i_top1, labels, num_classes) * 100

    print(f"\nEvaluation Summary: Valid samples={total} | Skipped batches={skipped_batches}")
    print("Image -> Text (Emotion Class)")
    print(f"  Top-1 Accuracy: {i2t_top1_acc:.2f}%")
    print(f"  Top-2 Accuracy: {i2t_top2_acc:.2f}%")
    print(f"  Balanced Accuracy: {i2t_balanced_acc:.2f}%")
    print("Text -> Image (Emotion Class)")
    print(f"  Top-1 Accuracy: {t2i_top1_acc:.2f}%")
    print(f"  Top-2 Accuracy: {t2i_top2_acc:.2f}%")
    print(f"  Balanced Accuracy: {t2i_balanced_acc:.2f}%")


if __name__ == "__main__":
    evaluate()