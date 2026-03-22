from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from utils.helpers import load_json
from utils.config import Config


# DATASET
class ArtDataset(Dataset):
    def __init__(self):
        self.data = load_json(Config.DATA_FILE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = item["image"]
        text = item["text"]

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            return None  # skipping bad samples

        return {
            "image": image,
            "text": text
        }


# GLOBAL PROCESSOR
processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)


# COLLATE FUNCTION 
def collate_fn(batch):
    # failed samples
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.TEXT_MAX_LENGTH
    )
    
    # Attach raw texts directly into the dictionary for evaluation matching
    inputs["raw_texts"] = texts

    return inputs