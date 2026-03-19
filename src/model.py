import torch.nn as nn
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
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"]
        )

        return outputs.image_embeds, outputs.text_embeds