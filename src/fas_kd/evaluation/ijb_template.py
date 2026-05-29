from __future__ import annotations

from array import array
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image


def _far_key(far: float) -> str:
    exp = int(round(np.log10(float(far))))
    return f"tar_far_1e{exp}"


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    valid = np.where(fpr <= target_far)[0]
    if valid.size == 0:
        return float(tpr[0])
    return float(tpr[valid[-1]])


class _ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


@torch.no_grad()
def _extract_face_embeddings(
    model: torch.nn.Module,
    image_paths: list[Path],
    transform,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    num_workers: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    dataset = _ImagePathDataset(image_paths=image_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    outputs: list[np.ndarray] = []
    for images in tqdm(loader, desc="IJB face embedding", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            embeddings = model(images)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        if normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=1)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        outputs.append(embeddings.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0)


def _load_face_tid_mid(face_tid_mid_path: Path, loose_crop_root: Path) -> tuple[list[Path], np.ndarray, np.ndarray, int]:
    image_paths: list[Path] = []
    tids: list[int] = []
    mids: list[int] = []
    missing_images = 0

    with face_tid_mid_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            image_name = parts[0]
            tid = int(parts[1])
            mid = int(parts[2])
            image_path = (loose_crop_root / image_name).resolve()
            if not image_path.exists():
                missing_images += 1
                continue

            image_paths.append(image_path)
            tids.append(tid)
            mids.append(mid)

    return image_paths, np.asarray(tids, dtype=np.int32), np.asarray(mids, dtype=np.int32), missing_images


def _pool_mean(features: np.ndarray) -> np.ndarray:
    pooled = np.nan_to_num(features.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    return pooled.astype(np.float32)


def _pool_magface_weighted(features: np.ndarray) -> np.ndarray:
    feats = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    directions = feats / norms

    # MagFace quality-aware pooling: vector norm acts as confidence weight.
    weighted = np.sum(directions * norms, axis=0) / (float(np.sum(norms)) + 1e-12)
    weighted = np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)
    return weighted.astype(np.float32)


def _build_template_features(
    face_tids: np.ndarray,
    face_mids: np.ndarray,
    face_embeddings: np.ndarray,
    pooling_mode: str = "mean",
) -> dict[int, np.ndarray]:
    if pooling_mode not in {"mean", "magface_weighted"}:
        raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

    template_features: dict[int, np.ndarray] = {}
    unique_templates = np.unique(face_tids)

    for tid in tqdm(unique_templates, desc="IJB template pooling", leave=False):
        idx = np.where(face_tids == tid)[0]
        feats = np.nan_to_num(face_embeddings[idx], nan=0.0, posinf=0.0, neginf=0.0)
        mids = face_mids[idx]

        media_embeddings: list[np.ndarray] = []
        for mid in np.unique(mids):
            m_idx = np.where(mids == mid)[0]
            mid_feats = feats[m_idx]
            if pooling_mode == "magface_weighted":
                m_feat = _pool_magface_weighted(mid_feats)
            else:
                m_feat = _pool_mean(mid_feats)
                m_feat = m_feat / (np.linalg.norm(m_feat) + 1e-12)
            media_embeddings.append(m_feat.astype(np.float32))

        media_stack = np.stack(media_embeddings, axis=0)
        if pooling_mode == "magface_weighted":
            t_feat = _pool_magface_weighted(media_stack)
        else:
            t_feat = np.nan_to_num(np.mean(media_stack, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        t_feat = t_feat / (np.linalg.norm(t_feat) + 1e-12)
        t_feat = np.nan_to_num(t_feat, nan=0.0, posinf=0.0, neginf=0.0)
        template_features[int(tid)] = t_feat.astype(np.float32)

    return template_features


def _score_template_pairs(
    template_pair_label_path: Path,
    template_features: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    scores = array("f")
    labels = array("B")
    total_pairs = 0
    missing_t1 = 0
    missing_t2 = 0

    with template_pair_label_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="IJB pair scoring", leave=False):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            t1 = int(parts[0])
            t2 = int(parts[1])
            y = int(parts[2])
            total_pairs += 1

            f1 = template_features.get(t1)
            f2 = template_features.get(t2)
            if f1 is None:
                missing_t1 += 1
                continue
            if f2 is None:
                missing_t2 += 1
                continue

            score = float(np.nan_to_num(np.dot(f1, f2), nan=0.0, posinf=1.0, neginf=-1.0))
            scores.append(score)
            labels.append(1 if y > 0 else 0)

    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        total_pairs,
        missing_t1,
        missing_t2,
    )


def evaluate_ijb_template_1to1(
    model: torch.nn.Module,
    ijb_root: str | Path,
    transform,
    device: torch.device,
    use_amp: bool,
    target_fars: list[float],
    batch_size: int = 256,
    num_workers: int = 4,
    template_pooling: str = "magface_weighted",
) -> dict[str, Any]:
    ijb_root = Path(ijb_root)
    if not ijb_root.exists():
        raise FileNotFoundError(f"IJB root not found: {ijb_root}")

    prefix = ijb_root.name.lower()
    if prefix not in {"ijbb", "ijbc"}:
        raise ValueError(f"IJB root directory should be IJBB or IJBC, got: {ijb_root.name}")

    meta_root = ijb_root / "meta"
    loose_crop_root = ijb_root / "loose_crop"
    face_tid_mid_path = meta_root / f"{prefix}_face_tid_mid.txt"
    template_pair_label_path = meta_root / f"{prefix}_template_pair_label.txt"

    if not face_tid_mid_path.exists():
        raise FileNotFoundError(f"Missing face_tid_mid file: {face_tid_mid_path}")
    if not template_pair_label_path.exists():
        raise FileNotFoundError(f"Missing template_pair_label file: {template_pair_label_path}")

    image_paths, face_tids, face_mids, missing_images = _load_face_tid_mid(
        face_tid_mid_path=face_tid_mid_path,
        loose_crop_root=loose_crop_root,
    )

    if len(image_paths) == 0:
        raise RuntimeError("No valid IJB image paths found in face_tid_mid file")

    face_embeddings = _extract_face_embeddings(
        model=model,
        image_paths=image_paths,
        transform=transform,
        device=device,
        use_amp=use_amp,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_embeddings=(template_pooling == "mean"),
    )

    template_features = _build_template_features(
        face_tids=face_tids,
        face_mids=face_mids,
        face_embeddings=face_embeddings,
        pooling_mode=template_pooling,
    )

    scores, labels, total_pairs, missing_t1, missing_t2 = _score_template_pairs(
        template_pair_label_path=template_pair_label_path,
        template_features=template_features,
    )

    if scores.size == 0:
        raise RuntimeError("No valid template pairs were scored; check IJB metadata consistency")

    non_finite_mask = ~np.isfinite(scores)
    num_scores_non_finite = int(non_finite_mask.sum())
    if num_scores_non_finite > 0:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

    if np.unique(labels).size < 2:
        out: dict[str, Any] = {
            "protocol": prefix,
            "num_faces": int(len(image_paths)),
            "num_templates": int(len(template_features)),
            "num_pairs_total": int(total_pairs),
            "num_pairs_scored": int(scores.size),
            "num_pairs_missing_t1": int(missing_t1),
            "num_pairs_missing_t2": int(missing_t2),
            "missing_face_images": int(missing_images),
            "template_pooling": str(template_pooling),
            "roc_auc": 0.5,
            "num_scores_non_finite": int(num_scores_non_finite),
        }
        for far in target_fars:
            out[_far_key(float(far))] = 0.0
        return out

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(roc_auc_score(labels, scores))

    out: dict[str, Any] = {
        "protocol": prefix,
        "num_faces": int(len(image_paths)),
        "num_templates": int(len(template_features)),
        "num_pairs_total": int(total_pairs),
        "num_pairs_scored": int(scores.size),
        "num_pairs_missing_t1": int(missing_t1),
        "num_pairs_missing_t2": int(missing_t2),
        "missing_face_images": int(missing_images),
        "template_pooling": str(template_pooling),
        "roc_auc": roc_auc,
        "num_scores_non_finite": int(num_scores_non_finite),
    }

    for far in target_fars:
        out[_far_key(float(far))] = _tar_at_far(fpr=fpr, tpr=tpr, target_far=float(far))

    return out
