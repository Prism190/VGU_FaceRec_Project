from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_kd_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    student_norm = F.normalize(student_embeddings, dim=1)
    teacher_norm = F.normalize(teacher_embeddings, dim=1)
    cosine = F.cosine_similarity(student_norm, teacher_norm, dim=1)
    return (1.0 - cosine).mean()


def mse_kd_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_embeddings, teacher_embeddings)


def _pairwise_distance(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + eps)
    return dist


def rkd_distance_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    eps: float = 1e-12,
    max_batch: int = 64,
) -> torch.Tensor:
    if student_embeddings.shape[0] > max_batch:
        student_embeddings = student_embeddings[:max_batch]
        teacher_embeddings = teacher_embeddings[:max_batch]

    with torch.no_grad():
        t_d = _pairwise_distance(teacher_embeddings, eps=eps)
        t_d = t_d / (t_d[t_d > 0].mean() + eps)

    s_d = _pairwise_distance(student_embeddings, eps=eps)
    s_d = s_d / (s_d[s_d > 0].mean() + eps)

    return F.smooth_l1_loss(s_d, t_d)


def rkd_angle_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    eps: float = 1e-12,
    max_batch: int = 32,
) -> torch.Tensor:
    if student_embeddings.shape[0] > max_batch:
        student_embeddings = student_embeddings[:max_batch]
        teacher_embeddings = teacher_embeddings[:max_batch]

    with torch.no_grad():
        td = teacher_embeddings.unsqueeze(0) - teacher_embeddings.unsqueeze(1)
        tn = F.normalize(td, p=2, dim=2, eps=eps)
        t_angle = torch.bmm(tn, tn.transpose(1, 2)).reshape(-1)

    sd = student_embeddings.unsqueeze(0) - student_embeddings.unsqueeze(1)
    sn = F.normalize(sd, p=2, dim=2, eps=eps)
    s_angle = torch.bmm(sn, sn.transpose(1, 2)).reshape(-1)

    return F.smooth_l1_loss(s_angle, t_angle)
