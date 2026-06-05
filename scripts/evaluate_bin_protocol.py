#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.transforms import build_eval_transform
from fas_kd.models.student import MobileNetV4Student
from fas_kd.models.teacher import build_frozen_teacher
from fas_kd.utils.config import load_yaml_config


def _far_key(far: float) -> str:
    return f"tar_far_1e{int(round(np.log10(float(far))))}"


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    idx = np.where(fpr <= target_far)[0]
    return float(tpr[idx[-1]]) if idx.size > 0 else float(tpr[0])


def _best_accuracy(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    best_acc, best_thr = 0.0, 0.0
    for thr in thresholds:
        pred = (scores >= thr).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_acc, best_thr


def _decode_bin_image(item) -> Image.Image:
    if isinstance(item, bytes):
        arr = np.frombuffer(item, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode bytes image")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    arr = np.asarray(item)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr.astype(np.uint8))

    flat = arr.reshape(-1)
    if flat.dtype != np.uint8:
        flat = flat.astype(np.uint8)
    img = cv2.imdecode(flat, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode ndarray image buffer shape={arr.shape}")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class BinPairDataset(Dataset):
    def __init__(self, bin_path: Path, transform) -> None:
        with bin_path.open("rb") as f:
            bins, issame = pickle.load(f, encoding="bytes")
        self.bins = bins
        self.issame = np.asarray(issame, dtype=np.int32)
        self.transform = transform

        if len(self.bins) != 2 * len(self.issame):
            raise ValueError(f"Invalid bin layout in {bin_path}: bins={len(self.bins)} pairs={len(self.issame)}")

    def __len__(self) -> int:
        return len(self.issame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_a = _decode_bin_image(self.bins[2 * index])
        image_b = _decode_bin_image(self.bins[2 * index + 1])
        return {
            "image_a": self.transform(image_a),
            "image_b": self.transform(image_b),
            "is_same": torch.tensor(int(self.issame[index]), dtype=torch.long),
        }


@torch.no_grad()
def _evaluate_bin_set(
    model: torch.nn.Module,
    bin_path: Path,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    use_flip: bool,
) -> dict[str, Any]:
    ds = BinPairDataset(bin_path=bin_path, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for batch in dl:
        a = batch["image_a"].to(device, non_blocking=True)
        b = batch["image_b"].to(device, non_blocking=True)
        y = batch["is_same"].cpu().numpy().astype(np.int32)

        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            ea = model(a)
            eb = model(b)
            if use_flip:
                ea = ea + model(torch.flip(a, dims=[3]))
                eb = eb + model(torch.flip(b, dims=[3]))

        ea = torch.nan_to_num(F.normalize(ea, dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        eb = torch.nan_to_num(F.normalize(eb, dim=1), nan=0.0, posinf=0.0, neginf=0.0)

        s = torch.sum(ea * eb, dim=1)
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0).detach().cpu().numpy().astype(np.float32)

        score_batches.append(s)
        label_batches.append(y)

    all_scores = np.concatenate(score_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)

    non_finite = int((~np.isfinite(all_scores)).sum())
    if non_finite > 0:
        all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=1.0, neginf=-1.0)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = float(roc_auc_score(all_labels, all_scores))
    accuracy, best_thr = _best_accuracy(all_scores, all_labels, thresholds)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "best_threshold": best_thr,
        "num_pairs": int(all_labels.shape[0]),
        "num_scores_non_finite": non_finite,
        _far_key(1e-3): _tar_at_far(fpr, tpr, 1e-3),
        _far_key(1e-4): _tar_at_far(fpr, tpr, 1e-4),
        _far_key(1e-5): _tar_at_far(fpr, tpr, 1e-5),
    }


def _build_student(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = MobileNetV4Student(
        backbone_name=cfg["student"]["backbone_name"],
        embedding_dim=int(cfg["student"].get("embedding_dim", 512)),
        pretrained=False,
        input_size=int(cfg["data"].get("image_size", 112)),
        projection_activation=str(cfg["student"].get("projection_activation", "none")),
        spatial_out_channels=int(cfg["student"].get("spatial_out_channels", 0)),
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["student_state"], strict=True)
    return model.to(device)


def _build_teacher(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    return build_frozen_teacher(cfg["teacher"]).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate student/teacher on InsightFace .bin verification sets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--student-checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--out", default="logs/eval_bin_protocol.json")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_eval_transform(cfg["data"])

    student = _build_student(cfg, checkpoint=Path(args.student_checkpoint), device=device)
    teacher = _build_teacher(cfg, device=device)

    bin_root = Path("data/raw/casia-webface/faces_webface_112x112")
    bin_sets = {
        "lfw": bin_root / "lfw.bin",
        "cfp_fp": bin_root / "cfp_fp.bin",
        "agedb30": bin_root / "agedb_30.bin",
    }

    out: dict[str, Any] = {"student": {}, "teacher": {}}

    for set_name, bin_path in bin_sets.items():
        out["student"][set_name] = _evaluate_bin_set(
            model=student,
            bin_path=bin_path,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=bool(cfg.get("system", {}).get("use_amp", True)),
            use_flip=not args.no_flip,
        )
        out["teacher"][set_name] = _evaluate_bin_set(
            model=teacher,
            bin_path=bin_path,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=bool(cfg.get("system", {}).get("use_amp", True)),
            use_flip=not args.no_flip,
        )

    out["student"]["aggregate"] = {
        "mean_accuracy": float(np.mean([out["student"]["lfw"]["accuracy"], out["student"]["cfp_fp"]["accuracy"], out["student"]["agedb30"]["accuracy"]])),
        "mean_roc_auc": float(np.mean([out["student"]["lfw"]["roc_auc"], out["student"]["cfp_fp"]["roc_auc"], out["student"]["agedb30"]["roc_auc"]])),
    }
    out["teacher"]["aggregate"] = {
        "mean_accuracy": float(np.mean([out["teacher"]["lfw"]["accuracy"], out["teacher"]["cfp_fp"]["accuracy"], out["teacher"]["agedb30"]["accuracy"]])),
        "mean_roc_auc": float(np.mean([out["teacher"]["lfw"]["roc_auc"], out["teacher"]["cfp_fp"]["roc_auc"], out["teacher"]["agedb30"]["roc_auc"]])),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
