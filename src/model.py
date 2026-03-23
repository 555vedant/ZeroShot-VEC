import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from utils.config import Config


class CLIPFineTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(Config.MODEL_NAME)

        if Config.FREEZE_VISION:
            for p in self.model.vision_model.parameters():
                p.requires_grad = False

        if Config.FREEZE_TEXT:
            for p in self.model.text_model.parameters():
                p.requires_grad = False

    def forward(self, batch):
        image_embeds = self.encode_images(batch["pixel_values"])
        text_embeds = self.encode_text(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        return image_embeds, text_embeds

    def encode_images(self, pixel_values):
        embeds = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(embeds, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        embeds = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(embeds, dim=-1)

    def pair_logits(self, pixel_values, input_ids, attention_mask, temperature=None):
        image_embeds = self.encode_images(pixel_values)
        text_embeds = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)

        if temperature is None:
            scale = self.model.logit_scale.exp().clamp(max=100)
        else:
            scale = 1.0 / float(temperature)

        logits = (image_embeds * text_embeds).sum(dim=-1) * scale
        return logits