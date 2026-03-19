from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from utils.helpers import load_json
from utils.config import Config


class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(Config.DATA_FILE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        with Image.open(item["image"]) as img:
            image = img.convert("RGB")

        text = item["text"]

        # RETURN RAW (IMPORTANT)
        return {
            "image": image,
            "text": text
        }


# ---------------------------
# GLOBAL PROCESSOR (only once)
# ---------------------------
processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)


def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,        # dynamic padding 
        truncation=True
    )

    return inputs