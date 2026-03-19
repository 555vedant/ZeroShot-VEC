import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import ArtDataset
from src.model import CLIPFineTuner


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ArtDataset()
    model = CLIPFineTuner().to(device)
    model.load_state_dict(torch.load("clip_model.pth"))
    model.eval()

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

    image_embs = F.normalize(image_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    sim = image_embs @ text_embs.T

    correct = 0
    for i in range(len(sim)):
        if sim[i].argmax() == i:
            correct += 1

    acc = correct / len(sim)
    print("Zero-shot Retrieval Accuracy:", acc)