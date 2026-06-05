from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetV4Student(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int = 512,
        pretrained: bool = True,
        input_size: int = 112,
        projection_activation: str = "none",
        spatial_out_channels: int = 0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")

        in_features, spatial_channels = self._infer_feature_dims(input_size=input_size)
        self.spatial_channels = int(spatial_channels)
        self.spatial_out_channels = int(spatial_out_channels)
        if self.spatial_out_channels > 0:
            if self.spatial_channels <= 0:
                raise RuntimeError(
                    "Student backbone does not expose 2D spatial features, but spatial_out_channels > 0 was requested"
                )
            self.spatial_proj = nn.Conv2d(self.spatial_channels, self.spatial_out_channels, kernel_size=1, bias=False)
        else:
            self.spatial_proj = None

        self.proj = nn.Linear(in_features, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.projection_activation = projection_activation.lower()
        if self.projection_activation == "prelu":
            self.act = nn.PReLU(num_parameters=embedding_dim)
        elif self.projection_activation in {"none", "identity"}:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported projection_activation: {projection_activation}")

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        return self.backbone(x)

    def _pool_backbone_features(self, features: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_head"):
            try:
                pooled = self.backbone.forward_head(features, pre_logits=True)
            except TypeError:
                pooled = self.backbone.forward_head(features)
        else:
            pooled = features

        if pooled.ndim > 2:
            pooled = torch.flatten(F.adaptive_avg_pool2d(pooled, output_size=1), start_dim=1)
        return pooled

    def _infer_feature_dims(self, input_size: int) -> tuple[int, int]:
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self._extract_backbone_features(dummy)
            pooled = self._pool_backbone_features(features)
        if was_training:
            self.backbone.train()

        spatial_channels = int(features.shape[1]) if features.ndim == 4 else 0
        return int(pooled.shape[1]), spatial_channels

    def _project_embedding(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.proj(pooled)))

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_backbone_features(x)
        pooled = self._pool_backbone_features(features)
        emb = self._project_embedding(pooled)

        if not return_spatial:
            return emb

        if features.ndim != 4:
            raise RuntimeError(
                "Student backbone forward_features must return BCHW tensor for spatial KD"
            )

        spatial = self.spatial_proj(features) if self.spatial_proj is not None else features
        return emb, spatial

    def forward_with_spatial(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb, spatial = self.forward(x, return_spatial=True)
        return emb, spatial
