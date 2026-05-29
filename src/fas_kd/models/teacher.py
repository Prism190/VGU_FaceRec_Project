from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import timm
import torch
import torch.nn as nn

from .magface_iresnet import iresnet18, iresnet34, iresnet50, iresnet100


class FrozenTeacher(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embedding_dim: int = 512,
        input_mode: str = "identity",
        swap_rb: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.input_mode = str(input_mode)
        self.swap_rb = bool(swap_rb)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "from_minus_one_to_zero_one":
            x = torch.clamp((x + 1.0) * 0.5, min=0.0, max=1.0)
        elif self.input_mode != "identity":
            raise ValueError(f"Unsupported teacher input_mode: {self.input_mode}")

        if self.swap_rb:
            x = x[:, [2, 1, 0], :, :]

        return x

    def _normalize_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        if emb.ndim > 2:
            emb = torch.flatten(emb, start_dim=1)
        if emb.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"Teacher output dim mismatch: expected {self.embedding_dim}, got {emb.shape[-1]}"
            )
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)

        with torch.no_grad():
            emb = self.model(x)
            return self._normalize_embedding(emb)

    def forward_with_spatial(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_input(x)

        with torch.no_grad():
            if not hasattr(self.model, "forward_features"):
                raise RuntimeError("Teacher model does not expose forward_features for spatial KD")

            spatial = self.model.forward_features(x)
            if spatial.ndim != 4:
                raise RuntimeError(
                    "Teacher forward_features must return BCHW tensor for spatial KD"
                )

            if hasattr(self.model, "forward_from_features"):
                emb = self.model.forward_from_features(spatial)
            elif hasattr(self.model, "forward_head"):
                try:
                    emb = self.model.forward_head(spatial, pre_logits=True)
                except TypeError:
                    emb = self.model.forward_head(spatial)
            else:
                raise RuntimeError(
                    "Teacher model must expose forward_from_features or forward_head for spatial KD"
                )

            return self._normalize_embedding(emb), spatial


def build_frozen_teacher(cfg: dict) -> FrozenTeacher:
    source = cfg.get("source", "torchscript")
    embedding_dim = int(cfg.get("embedding_dim", 512))
    input_mode = str(cfg.get("input_mode", "identity")).lower()
    swap_rb = bool(cfg.get("swap_rb", False))

    if source == "torchscript":
        checkpoint = Path(cfg["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")
        model = torch.jit.load(str(checkpoint), map_location="cpu")
    elif source == "timm":
        model_name = cfg.get("model_name")
        if not model_name:
            raise ValueError("teacher.model_name is required when teacher.source=timm")
        model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        checkpoint = Path(cfg["checkpoint"])
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu")
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")
    elif source == "magface":
        arch = str(cfg.get("arch", "iresnet18")).lower()
        builders = {
            "iresnet18": iresnet18,
            "iresnet34": iresnet34,
            "iresnet50": iresnet50,
            "iresnet100": iresnet100,
        }
        if arch not in builders:
            raise ValueError(f"Unsupported magface teacher arch: {arch}")

        model = builders[arch](num_classes=embedding_dim)

        checkpoint = Path(cfg["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")

        state = torch.load(checkpoint, map_location="cpu")
        raw_state = state.get("state_dict", state)

        cleaned = OrderedDict()
        model_keys = model.state_dict().keys()
        for key, value in raw_state.items():
            candidates = [
                key,
                key.replace("features.module.", ""),
                key.replace("module.features.", ""),
                key.replace("module.", ""),
            ]
            for candidate in candidates:
                if candidate in model_keys and model.state_dict()[candidate].shape == value.shape:
                    cleaned[candidate] = value
                    break

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if len(cleaned) == 0 or (len(missing) > 0 and len(cleaned) < len(model.state_dict()) * 0.6):
            raise RuntimeError(
                f"Insufficient MagFace weights loaded from {checkpoint}: "
                f"loaded={len(cleaned)}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
    else:
        raise ValueError(f"Unsupported teacher source: {source}")

    return FrozenTeacher(
        model=model,
        embedding_dim=embedding_dim,
        input_mode=input_mode,
        swap_rb=swap_rb,
    )
