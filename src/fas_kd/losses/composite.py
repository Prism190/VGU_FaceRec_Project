from __future__ import annotations

import torch
import torch.nn as nn

from .kd import cosine_kd_loss, mse_kd_loss, rkd_angle_loss, rkd_distance_loss


class DistillationObjective(nn.Module):
    def __init__(
        self,
        lambda_cls: float,
        lambda_kd_start: float,
        lambda_kd_end: float,
        kd_ramp_epochs: int,
        kd_type: str = "cosine",
        lambda_rkd_distance: float = 0.0,
        lambda_rkd_angle: float = 0.0,
        lambda_spatial_start: float = 0.0,
        lambda_spatial_end: float = 0.0,
        spatial_ramp_epochs: int = 0,
        lambda_norm_reg: float = 0.0,
        max_norm_threshold: float = 30.0,
    ) -> None:
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_kd_start = float(lambda_kd_start)
        self.lambda_kd_end = float(lambda_kd_end)
        self.kd_ramp_epochs = int(kd_ramp_epochs)
        self.kd_type = str(kd_type).lower()
        self.lambda_rkd_distance = float(lambda_rkd_distance)
        self.lambda_rkd_angle = float(lambda_rkd_angle)
        self.lambda_spatial_start = float(lambda_spatial_start)
        self.lambda_spatial_end = float(lambda_spatial_end)
        self.spatial_ramp_epochs = int(spatial_ramp_epochs)
        self.lambda_norm_reg = float(lambda_norm_reg)
        self.max_norm_threshold = float(max_norm_threshold)

        if self.kd_type not in {"cosine", "mse"}:
            raise ValueError(f"Unsupported kd_type: {kd_type}")

    def kd_weight(self, epoch_idx: int) -> float:
        if self.kd_ramp_epochs <= 0:
            return self.lambda_kd_end
        progress = min(max(epoch_idx, 0), self.kd_ramp_epochs) / float(self.kd_ramp_epochs)
        return self.lambda_kd_start + progress * (self.lambda_kd_end - self.lambda_kd_start)

    def spatial_weight(self, epoch_idx: int) -> float:
        if self.spatial_ramp_epochs <= 0:
            return self.lambda_spatial_end
        progress = min(max(epoch_idx, 0), self.spatial_ramp_epochs) / float(self.spatial_ramp_epochs)
        return self.lambda_spatial_start + progress * (self.lambda_spatial_end - self.lambda_spatial_start)

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        class_loss: torch.Tensor,
        epoch_idx: int,
        student_spatial: torch.Tensor | None = None,
        teacher_spatial: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.kd_type == "mse":
            kd = mse_kd_loss(student_embeddings, teacher_embeddings)
        else:
            kd = cosine_kd_loss(student_embeddings, teacher_embeddings)
        kd_w = self.kd_weight(epoch_idx)
        spatial_w = self.spatial_weight(epoch_idx)

        rkd_d = torch.zeros((), device=student_embeddings.device, dtype=student_embeddings.dtype)
        rkd_a = torch.zeros((), device=student_embeddings.device, dtype=student_embeddings.dtype)
        spatial_kd = torch.zeros((), device=student_embeddings.device, dtype=student_embeddings.dtype)

        if self.lambda_rkd_distance > 0.0:
            rkd_d = rkd_distance_loss(student_embeddings, teacher_embeddings)
        if self.lambda_rkd_angle > 0.0:
            rkd_a = rkd_angle_loss(student_embeddings, teacher_embeddings)

        if spatial_w > 0.0 and student_spatial is not None and teacher_spatial is not None:
            # spatial features are None when training engine gates spatial KD off
            # (e.g. masking epochs in Phase 4 Bug #11 fix) — silently skip in that case
            if student_spatial.shape != teacher_spatial.shape:
                raise ValueError(
                    "Spatial KD requires matching BCHW feature maps, got "
                    f"student={tuple(student_spatial.shape)} teacher={tuple(teacher_spatial.shape)}"
                )
            spatial_kd = mse_kd_loss(student_spatial, teacher_spatial)

        norm_reg = torch.zeros((), device=student_embeddings.device, dtype=student_embeddings.dtype)
        if self.lambda_norm_reg > 0.0:
            # Quadratic penalty on embedding norms that exceed max_norm_threshold.
            # Counteracts MagFace's own regularizer which rewards inflating norms.
            emb_norms = student_embeddings.norm(dim=1)
            excess = torch.relu(emb_norms - self.max_norm_threshold)
            norm_reg = (excess ** 2).mean()

        total = (
            (self.lambda_cls * class_loss)
            + (kd_w * kd)
            + (self.lambda_rkd_distance * rkd_d)
            + (self.lambda_rkd_angle * rkd_a)
            + (spatial_w * spatial_kd)
            + (self.lambda_norm_reg * norm_reg)
        )

        return {
            "total_loss": total,
            "loss_cls": class_loss.detach(),
            "loss_kd": kd.detach(),
            "loss_rkd_distance": rkd_d.detach(),
            "loss_rkd_angle": rkd_a.detach(),
            "loss_spatial_kd": spatial_kd.detach(),
            "loss_norm_reg": norm_reg.detach(),
            "kd_weight": torch.tensor(kd_w, device=student_embeddings.device),
            "spatial_kd_weight": torch.tensor(spatial_w, device=student_embeddings.device),
        }
