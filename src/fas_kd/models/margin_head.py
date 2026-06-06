from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        normalized_weight = F.normalize(self.weight, dim=1)

        cosine = F.linear(normalized_embeddings, normalized_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        target_logits = target_logits.to(dtype=cosine.dtype)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale
        cls_loss = F.cross_entropy(logits, labels)

        return {
            "loss": cls_loss,
            "logits": logits,
            "norms": torch.linalg.norm(embeddings, dim=1),
            "cls_loss": cls_loss,
            "reg_loss": torch.zeros((), dtype=embeddings.dtype, device=embeddings.device),
        }


class MagFaceHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.35,
        magface_margin_min: float = 0.30,
        magface_margin_max: float = 0.50,
        feature_norm_lower: float = 10.0,
        feature_norm_upper: float = 110.0,
        regularizer_lambda: float = 35.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.margin_min = magface_margin_min
        self.margin_max = magface_margin_max
        self.feature_norm_lower = feature_norm_lower
        self.feature_norm_upper = feature_norm_upper
        self.regularizer_lambda = regularizer_lambda

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _adaptive_margin(self, norms: torch.Tensor) -> torch.Tensor:
        ratio = (norms - self.feature_norm_lower) / (self.feature_norm_upper - self.feature_norm_lower)
        ratio = torch.clamp(ratio, min=0.0, max=1.0)
        return self.margin_min + ratio * (self.margin_max - self.margin_min)

    def _magnitude_regularizer(self, norms: torch.Tensor) -> torch.Tensor:
        # Quality-aware regularizer encourages informative, bounded feature magnitude.
        return (norms / (self.feature_norm_upper**2)) + torch.reciprocal(norms)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        norms = torch.linalg.norm(embeddings, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=self.feature_norm_lower, max=self.feature_norm_upper)

        normalized_embeddings = embeddings / norms
        normalized_weight = F.normalize(self.weight, dim=1)

        cosine = F.linear(normalized_embeddings, normalized_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        adaptive_margin = self._adaptive_margin(norms).squeeze(1)

        idx = torch.arange(cosine.size(0), device=cosine.device)
        target_cosine = cosine[idx, labels]
        sin_theta = torch.sqrt(torch.clamp(1.0 - target_cosine**2, min=1e-7))
        cos_m = torch.cos(adaptive_margin)
        sin_m = torch.sin(adaptive_margin)
        # Normal case: cos(θ + m)
        target_margin_cosine = (target_cosine * cos_m) - (sin_theta * sin_m)
        # π-fallback (fix #13): when θ + m > π the addition formula wraps around and
        # produces a value that rewards hard negatives.  Guard with a linear approximation:
        # cos(θ) - sin(m) * m  (standard InsightFace convention).
        theta = torch.acos(target_cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target_margin_cosine = torch.where(
            (theta + adaptive_margin) > math.pi,
            target_cosine - sin_m * adaptive_margin,
            target_margin_cosine,
        )
        target_margin_cosine = target_margin_cosine.to(dtype=cosine.dtype)

        logits = cosine.clone()
        logits[idx, labels] = target_margin_cosine
        logits = logits * self.scale

        cls_loss = F.cross_entropy(logits, labels)
        reg_loss = self._magnitude_regularizer(norms.squeeze(1)).mean()
        total = cls_loss + (self.regularizer_lambda * reg_loss)

        return {
            "loss": total,
            "logits": logits,
            "norms": norms.squeeze(1),
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }


def build_margin_head(cfg: dict, in_features: int, num_classes: int) -> nn.Module:
    head_type = cfg.get("type", "magface").lower()

    if head_type == "magface":
        return MagFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            scale=float(cfg.get("scale", 64.0)),
            margin=float(cfg.get("margin", 0.35)),
            magface_margin_min=float(cfg.get("magface_margin_min", 0.30)),
            magface_margin_max=float(cfg.get("magface_margin_max", 0.50)),
            feature_norm_lower=float(cfg.get("feature_norm_lower", 10.0)),
            feature_norm_upper=float(cfg.get("feature_norm_upper", 110.0)),
            regularizer_lambda=float(cfg.get("regularizer_lambda", 35.0)),
        )

    if head_type == "arcface":
        return ArcFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            scale=float(cfg.get("scale", 64.0)),
            margin=float(cfg.get("margin", 0.5)),
        )

    raise ValueError(f"Unsupported margin head type: {head_type}")
