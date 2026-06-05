from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


def _far_key(far: float) -> str:
    exponent = int(round(math.log10(float(far))))
    return f"tar_far_1e{exponent}"


def _best_accuracy(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    best_acc = 0.0
    best_thr = 0.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(threshold)
    return best_acc, best_thr


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    valid = np.where(fpr <= target_far)[0]
    if valid.size == 0:
        return float(tpr[0])
    return float(tpr[valid[-1]])


@torch.no_grad()
def evaluate_pair_verification(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    target_fars: list[float],
) -> dict[str, Any]:
    model.eval()

    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for batch in dataloader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["is_same"].cpu().numpy().astype(np.int32)

        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            emb_a = model(image_a)
            emb_b = model(image_b)

        emb_a = torch.nan_to_num(emb_a, nan=0.0, posinf=0.0, neginf=0.0)
        emb_b = torch.nan_to_num(emb_b, nan=0.0, posinf=0.0, neginf=0.0)
        emb_a = F.normalize(emb_a, dim=1)
        emb_b = F.normalize(emb_b, dim=1)
        scores = torch.sum(emb_a * emb_b, dim=1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
        scores = scores.detach().cpu().numpy()

        score_batches.append(scores)
        label_batches.append(labels)

    all_scores = np.concatenate(score_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)

    non_finite_mask = ~np.isfinite(all_scores)
    num_scores_non_finite = int(non_finite_mask.sum())
    if num_scores_non_finite > 0:
        all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=1.0, neginf=-1.0)

    unique_labels = np.unique(all_labels)
    if unique_labels.size < 2:
        metrics: dict[str, Any] = {
            "accuracy": float((all_labels == all_labels[0]).mean()) if all_labels.size > 0 else 0.0,
            "roc_auc": 0.5,
            "best_threshold": 0.0,
            "num_pairs": int(all_labels.shape[0]),
            "num_scores_non_finite": num_scores_non_finite,
        }
        for far in target_fars:
            metrics[_far_key(far)] = 0.0
        return metrics

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = float(roc_auc_score(all_labels, all_scores))
    accuracy, best_thr = _best_accuracy(all_scores, all_labels, thresholds)

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "best_threshold": best_thr,
        "num_pairs": int(all_labels.shape[0]),
        "num_scores_non_finite": num_scores_non_finite,
    }

    for far in target_fars:
        metrics[_far_key(far)] = _tar_at_far(fpr, tpr, far)

    return metrics
