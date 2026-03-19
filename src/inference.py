import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor

from src.model import CLIPFineTuner
from utils.helpers import load_json
from utils.config import Config


class SearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPFineTuner().to(self.device)
        self.model.load_state_dict(torch.load("clip_model.pth"))
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)

        self.data = load_json(Config.DATA_FILE)
        self.image_embs = self._build_index()

    def _build_index(self):
        embs = []

        with torch.no_grad():
            for item in self.data:
                image = Image.open(item["image"]).convert("RGB")

                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                out = self.model.model.get_image_features(**inputs)
                embs.append(out)

        embs = torch.cat(embs)
        return F.normalize(embs, dim=-1)

    def search(self, query, top_k=5):
        inputs = self.processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_emb = self.model.model.get_text_features(**inputs)

        text_emb = F.normalize(text_emb, dim=-1)

        sims = (text_emb @ self.image_embs.T).squeeze(0)
        idx = sims.topk(top_k).indices.tolist()

        results = [self.data[i]["image"] for i in idx]
        return results