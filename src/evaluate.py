import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset import ArtDataset
from src.model import CLIPFineTuner


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = ArtDataset()
    print(f"Dataset loaded with {len(dataset)} items.")

    print("Initializing model from 'clip_model.pth'...")
    model = CLIPFineTuner().to(device)
    model.load_state_dict(torch.load("clip_model.pth"))
    model.eval()

    print("Extracting image and text embeddings...")
    image_embs = []
    text_embs = []

    with torch.no_grad():
        for item in tqdm(dataset):
            batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
            img, txt = model(batch)

            image_embs.append(img)
            text_embs.append(txt)

    image_embs = torch.cat(image_embs)
    text_embs = torch.cat(text_embs)
    
    print("Computing similarity matrix...")
    image_embs = F.normalize(image_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    sim = image_embs @ text_embs.T

    print("Calculating final accuracy...")
    correct = 0
    for i in range(len(sim)):
        if sim[i].argmax() == i:
            correct += 1

    acc = correct / len(sim)
    print(f"Zero-shot Retrieval Accuracy: {acc:.4f}")