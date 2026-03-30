import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
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

    def enable_data_parallel(self):
        if isinstance(self.model, nn.DataParallel):
            return
        self.model = nn.DataParallel(self.model)

    def is_data_parallel(self):
        return isinstance(self.model, nn.DataParallel)

    def core_model(self):
        return self.model.module if self.is_data_parallel() else self.model

    def checkpoint_state_dict(self):
        state = self.state_dict()
        if not self.is_data_parallel():
            return state

        fixed = OrderedDict()
        for k, v in state.items():
            fixed[k.replace("model.module.", "model.", 1)] = v
        return fixed

    @staticmethod
    def normalize_checkpoint_state_dict(state_dict):
        normalized = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("model.module."):
                normalized[k.replace("model.module.", "model.", 1)] = v
            else:
                normalized[k] = v
        return normalized

    @staticmethod
    def _to_data_parallel_keys(state_dict):
        converted = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("model.") and not k.startswith("model.module."):
                converted[k.replace("model.", "model.module.", 1)] = v
            else:
                converted[k] = v
        return converted

    def load_checkpoint_state_dict(self, state_dict):
        normalized = self.normalize_checkpoint_state_dict(state_dict)

        if self.is_data_parallel():
            try:
                self.load_state_dict(normalized)
                return
            except RuntimeError:
                pass

            dp_ready = self._to_data_parallel_keys(normalized)
            self.load_state_dict(dp_ready)
            return

        self.load_state_dict(normalized)

    def forward(self, batch):
        image_embeds = self.encode_images(batch["pixel_values"])
        text_embeds = self.encode_text(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        return image_embeds, text_embeds

    @staticmethod
    def _to_embedding_tensor(output, modality):
        if isinstance(output, torch.Tensor):
            return output

        if modality == "image" and hasattr(output, "image_embeds"):
            return output.image_embeds

        if modality == "text" and hasattr(output, "text_embeds"):
            return output.text_embeds

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output

        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0, :]

        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(f"Unsupported {modality} output type: {type(output)!r}")

    def encode_images(self, pixel_values):
        if self.is_data_parallel():
            # CLIP full forward requires text inputs; use image feature API for image-only calls.
            embeds = self.core_model().get_image_features(pixel_values=pixel_values)
        else:
            embeds = self.model.get_image_features(pixel_values=pixel_values)
        embeds = self._to_embedding_tensor(embeds, modality="image")
        return F.normalize(embeds, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        if self.is_data_parallel():
            # CLIP full forward expects image inputs too; use text feature API for text-only calls.
            embeds = self.core_model().get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            embeds = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        embeds = self._to_embedding_tensor(embeds, modality="text")
        return F.normalize(embeds, dim=-1)

    def pair_logits(self, pixel_values, input_ids, attention_mask, temperature=None):
        if self.is_data_parallel():
            gpu_count = max(1, torch.cuda.device_count())
            batch_size = int(pixel_values.shape[0])
            balanced = batch_size >= gpu_count and (batch_size % gpu_count == 0)

            if balanced:
                # Fast path: true DataParallel execution across GPUs.
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                image_embeds = F.normalize(self._to_embedding_tensor(outputs, modality="image"), dim=-1)
                text_embeds = F.normalize(self._to_embedding_tensor(outputs, modality="text"), dim=-1)
            else:
                # Safe path for uneven final batches that can trigger DP gather shape mismatches.
                core = self.core_model()
                image_embeds = F.normalize(
                    core.get_image_features(pixel_values=pixel_values),
                    dim=-1,
                )
                text_embeds = F.normalize(
                    core.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ),
                    dim=-1,
                )
        else:
            image_embeds = self.encode_images(pixel_values)
            text_embeds = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)

        if temperature is None:
            scale = self.core_model().logit_scale.exp().clamp(max=100)
        else:
            scale = 1.0 / float(temperature)

        logits = (image_embeds * text_embeds).sum(dim=-1) * scale
        return logits