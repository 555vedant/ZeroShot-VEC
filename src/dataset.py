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

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }