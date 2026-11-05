import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from collections import OrderedDict
from transformers import CLIPModel
from utils.config import Config


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, dropout):
        super().__init__()
        if rank <= 0:
            raise ValueError("LORA_RANK must be positive.")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRA can only wrap nn.Linear modules.")

        self.base = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        update = F.linear(self.dropout(inputs), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling
        return self.base(inputs) + update


class CLIPFineTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(Config.MODEL_NAME)
        self._configure_strategy()

    def _configure_strategy(self):
        strategy = getattr(Config, "FINE_TUNING_STRATEGY", "visual_lora")
        if strategy == "visual_lora":
            self._configure_visual_lora()
        elif strategy == "full":
            self._configure_full_fallback()
        elif strategy == "clipfit":
            self._configure_clipfit_parameters()
        else:
            raise ValueError(
                "FINE_TUNING_STRATEGY must be 'visual_lora', 'full', or 'clipfit'."
            )

    def _freeze_all_parameters(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _configure_visual_lora(self):
        if not bool(getattr(Config, "FREEZE_TEXT_ENCODER", True)):
            raise ValueError("visual_lora requires FREEZE_TEXT_ENCODER=True.")
        if not bool(getattr(Config, "FREEZE_TEXT_PROJECTION", True)):
            raise ValueError("visual_lora requires FREEZE_TEXT_PROJECTION=True.")
        if not bool(getattr(Config, "FREEZE_VISUAL_BACKBONE", True)):
            raise ValueError("visual_lora requires FREEZE_VISUAL_BACKBONE=True.")
        self._freeze_all_parameters()
        layers = self.model.vision_model.encoder.layers
        last_n = max(1, min(int(Config.LORA_TARGET_LAST_N_BLOCKS), len(layers)))
        targets = {str(value).lower() for value in Config.LORA_TARGET}
        replaced = []
        for index in range(len(layers) - last_n, len(layers)):
            attention = layers[index].self_attn
            for name in ("q_proj", "v_proj"):
                short_name = name[0]
                if short_name not in targets:
                    continue
                original = getattr(attention, name, None)
                if not isinstance(original, nn.Linear):
                    raise TypeError(
                        f"Expected vision block {index} {name} to be nn.Linear, "
                        f"got {type(original)!r}."
                    )
                setattr(
                    attention,
                    name,
                    LoRALinear(
                        original,
                        rank=int(Config.LORA_RANK),
                        alpha=float(Config.LORA_ALPHA),
                        dropout=float(Config.LORA_DROPOUT),
                    ),
                )
                replaced.append(f"vision_model.encoder.layers.{index}.self_attn.{name}")

        if not replaced:
            raise RuntimeError("Visual LoRA did not replace any attention projections.")
        if bool(getattr(Config, "LORA_TRAIN_VISUAL_PROJECTION", False)):
            for parameter in self.model.visual_projection.parameters():
                parameter.requires_grad = True
        if bool(getattr(Config, "LORA_TRAIN_VISUAL_NORMS", False)):
            for module in self.model.vision_model.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
        self._lora_modules = tuple(replaced)
        self._assert_parameter_policy()

    def _configure_full_fallback(self):
        self._freeze_all_parameters()
        for parameter in self.model.vision_model.parameters():
            parameter.requires_grad = True
        self.model.logit_scale.requires_grad = False

    def parameter_report(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        trainable_modules = sorted(
            {
                name.rsplit(".", 1)[0]
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            }
        )
        frozen_visual = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("model.vision_model.") and not parameter.requires_grad
        )
        frozen_text = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("model.text_model.") and not parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "trainable_percentage": 100.0 * trainable / max(total, 1),
            "trainable_modules": trainable_modules,
            "frozen_visual": frozen_visual,
            "frozen_text": frozen_text,
        }

    def _assert_parameter_policy(self):
        trainable_names = [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
        if not trainable_names:
            raise RuntimeError("Visual LoRA has no trainable parameters.")
        invalid = [
            name for name in trainable_names
            if "lora_A" not in name and "lora_B" not in name
        ]
        if invalid:
            raise RuntimeError(
                "Visual LoRA unexpectedly enabled non-LoRA parameters: "
                + ", ".join(invalid[:5])
            )
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        percentage = 100.0 * trainable / max(total, 1)
        if percentage >= 2.0:
            raise RuntimeError(
                f"Visual LoRA trainable percentage is {percentage:.2f}%; expected < 2%."
            )

    # CLIPFit: text FFN output biases and image-side LayerNorms only.
    def _configure_clipfit_parameters(self):
        self._freeze_all_parameters()

        for module in self.model.vision_model.modules():
            if isinstance(module, nn.LayerNorm):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        # CLIPFit updates the second (output) linear layer bias in each text FFN.
        for layer in self.model.text_model.encoder.layers:
            layer.mlp.fc2.bias.requires_grad = True

    # CLIPFIT: Preserve the pretrained image representation as a frozen teacher.
    def create_clipfit_teacher(self):
        teacher = copy.deepcopy(self.model)
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
        return teacher

    @staticmethod
    def _set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = bool(flag)

    @staticmethod
    def _unfreeze_top_encoder_layers(encoder_module, top_k):
        layers = getattr(encoder_module, "layers", None)
        if layers is None:
            return 0

        n = len(layers)
        if n == 0:
            return 0

        k = max(0, min(int(top_k), n))
        if k == 0:
            return 0

        for layer in layers[n - k:]:
            for p in layer.parameters():
                p.requires_grad = True
        return k

    def enable_data_parallel(self):
        if isinstance(self.model, nn.DataParallel):
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot enable DataParallel because CUDA is unavailable.")

        configured_ids = getattr(Config, "GPU_IDS", None)
        if configured_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = [int(device_id) for device_id in configured_ids]
            available = set(range(torch.cuda.device_count()))
            if not device_ids or not set(device_ids).issubset(available):
                raise ValueError(
                    f"GPU_IDS={device_ids} is invalid for "
                    f"{torch.cuda.device_count()} visible CUDA devices."
                )

        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self._data_parallel_device_ids = tuple(device_ids)
        else:
            self._data_parallel_device_ids = tuple(device_ids)

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
            # HF CLIP requires both modalities in its DataParallel forward.
            # Dummy text keeps image-only calls distributed across GPUs.
            batch_size = int(pixel_values.shape[0])
            dummy_ids = torch.zeros(
                (batch_size, 1),
                dtype=torch.long,
                device=pixel_values.device,
            )
            dummy_mask = torch.ones_like(dummy_ids)
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=dummy_ids,
                attention_mask=dummy_mask,
            )
            embeds = outputs.image_embeds
        else:
            embeds = self.model.get_image_features(pixel_values=pixel_values)
        embeds = self._to_embedding_tensor(embeds, modality="image")
        return F.normalize(embeds, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        if self.is_data_parallel():
            batch_size = int(input_ids.shape[0])
            dummy_pixels = torch.zeros(
                (batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                dtype=self.core_model().vision_model.embeddings.patch_embedding.weight.dtype,
                device=input_ids.device,
            )
            outputs = self.model(
                pixel_values=dummy_pixels,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeds = outputs.text_embeds
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
                # If any runtime issue appears, fall back to the safe single-device feature APIs.
                try:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    image_embeds = F.normalize(self._to_embedding_tensor(outputs, modality="image"), dim=-1)
                    text_embeds = F.normalize(self._to_embedding_tensor(outputs, modality="text"), dim=-1)
                except RuntimeError:
                    core = self.core_model()
                    image_feats = core.get_image_features(pixel_values=pixel_values)
                    text_feats = core.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    image_embeds = F.normalize(
                        self._to_embedding_tensor(image_feats, modality="image"),
                        dim=-1,
                    )
                    text_embeds = F.normalize(
                        self._to_embedding_tensor(text_feats, modality="text"),
                        dim=-1,
                    )
            else:
                # Safe path for uneven final batches that can trigger DP gather shape mismatches.
                core = self.core_model()
                image_feats = core.get_image_features(pixel_values=pixel_values)
                text_feats = core.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                image_embeds = F.normalize(
                    self._to_embedding_tensor(image_feats, modality="image"),
                    dim=-1,
                )
                text_embeds = F.normalize(
                    self._to_embedding_tensor(text_feats, modality="text"),
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
