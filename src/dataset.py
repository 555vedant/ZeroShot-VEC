from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from utils.helpers import load_json
from utils.config import Config


class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(Config.DATA_FILE)
        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        text = item["text"]

        # return raw (no processing here)
        return {
            "image": image,
            "text": text
        }


# ---------------------------
# COLLATE FUNCTION 
# ---------------------------
def collate_fn(batch):
    processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)

    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,       
        truncation=True
    )

    return inputs