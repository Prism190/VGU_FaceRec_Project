#!/usr/bin/env python3
"""Bin protocol evaluation under lower-face occlusion (surgical mask simulation).

Applies the same apply_lower_face_mask transform used during phase2/3 training
(zero-fill from y=55% down) to test images, then runs bin protocol verification.

Tests occlusion robustness: phase1 (clean-only training) vs phase3 (SWA, 30%
mask curriculum) vs phase2.

Usage:
    ./venv/bin/python scripts/evaluate_bin_occluded.py \
        --bin-root data/raw/casia-webface/faces_webface_112x112 \
        --out-dir logs/eval_occluded_$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.transforms import apply_lower_face_mask, build_eval_transform
from fas_kd.models.student import MobileNetV4Student
from fas_kd.utils.config import load_yaml_config


CHECKPOINTS = [
    ("phase1/latest", "configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml",
     "runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"),
    ("phase2/latest", "configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml",
     "runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"),
    ("phase3/swa",   "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml",
     "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/swa.pt"),
    ("phase3/latest","configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml",
     "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt"),
]


def _load_bin(bin_path: Path):
    """Read InsightFace .bin file (pickle format) → list of ((img_a, img_b), label) pairs."""
    import pickle, io
    with open(bin_path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")
    pairs, pair_labels = [], []
    for i, issame in enumerate(issame_list):
        def _to_pil(item):
            if isinstance(item, np.ndarray):
                return Image.fromarray(item).convert("RGB")
            return Image.open(io.BytesIO(item)).convert("RGB")
        img_a = _to_pil(bins[2 * i])
        img_b = _to_pil(bins[2 * i + 1])
        pairs.append((img_a, img_b))
        pair_labels.append(1 if issame else 0)
    return pairs, pair_labels


@torch.no_grad()
def _embed(model, imgs, transform, device, mask: bool, use_flip: bool = True):
    """Embed a list of PIL images, optionally applying lower-face mask."""
    embs = []
    batch_size = 256
    for start in range(0, len(imgs), batch_size):
        batch = imgs[start:start + batch_size]
        tensors = []
        for img in batch:
            t = transform(img)
            if mask:
                t = apply_lower_face_mask(t, mask_fill="zero")
            tensors.append(t)
        x = torch.stack(tensors).to(device)
        e = model(x)
        if use_flip:
            e = e + model(torch.flip(x, dims=[3]))
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def _accuracy_tar_far(scores, labels, target_fars=(0.001, 0.0001)):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    # EER threshold
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thrs = roc_curve(labels, scores, pos_label=1)
    eer_idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
    eer_thr = float(thrs[eer_idx])
    acc = float(np.mean((scores >= eer_thr) == (labels == 1)))
    tar_far = {}
    for far in target_fars:
        idxs = np.where(fpr <= far)[0]
        tar_far[f"tar_far_{far:g}"] = float(tpr[idxs[-1]]) if idxs.size else 0.0
    return {"accuracy": acc, "roc_auc": auc, "eer_threshold": eer_thr, **tar_far}


def _eval_one(model, pairs, pair_labels, transform, device, mask: bool):
    left_imgs  = [p[0] for p in pairs]
    right_imgs = [p[1] for p in pairs]
    e1 = _embed(model, left_imgs,  transform, device, mask=mask)
    e2 = _embed(model, right_imgs, transform, device, mask=mask)
    e1 = F.normalize(e1, dim=1)
    e2 = F.normalize(e2, dim=1)
    scores = (e1 * e2).sum(dim=1).numpy()
    return _accuracy_tar_far(scores, pair_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-root",
                        default="data/raw/casia-webface/faces_webface_112x112")
    parser.add_argument("--datasets", nargs="+",
                        default=["lfw", "cfp_fp", "agedb_30", "cplfw", "calfw"])
    parser.add_argument("--out-dir", default="logs/eval_occluded")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bin_root = Path(args.bin_root)
    if not bin_root.is_absolute():
        bin_root = (PROJECT_ROOT / bin_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_results = {}

    for label, cfg_path, ckpt_path in CHECKPOINTS:
        cfg_full = (PROJECT_ROOT / cfg_path).resolve()
        ckpt_full = (PROJECT_ROOT / ckpt_path).resolve()
        if not cfg_full.exists():
            print(f"[skip] {label}: config not found")
            continue
        if not ckpt_full.exists():
            print(f"[skip] {label}: checkpoint not found")
            continue

        cfg = load_yaml_config(str(cfg_full))
        sc = cfg["student"]
        model = MobileNetV4Student(
            backbone_name=sc["backbone_name"], embedding_dim=512, pretrained=False,
            input_size=112, projection_activation=str(sc.get("projection_activation","none")),
            spatial_out_channels=int(sc.get("spatial_out_channels", 0)),
        )
        ckpt = torch.load(str(ckpt_full), map_location="cpu")
        model.load_state_dict(ckpt.get("student_state", ckpt), strict=True)
        model.to(device).eval()
        transform = build_eval_transform(cfg["data"])

        print(f"\n=== {label} ===")
        results = {"clean": {}, "masked": {}}

        for ds_name in args.datasets:
            bin_path = bin_root / f"{ds_name}.bin"
            if not bin_path.exists():
                print(f"  [skip] {ds_name}: not found")
                continue

            pairs, pair_labels = _load_bin(bin_path)
            print(f"  {ds_name} ({len(pairs)} pairs) ...", end=" ", flush=True)

            clean = _eval_one(model, pairs, pair_labels, transform, device, mask=False)
            masked = _eval_one(model, pairs, pair_labels, transform, device, mask=True)

            drop = clean["accuracy"] - masked["accuracy"]
            print(f"clean={clean['accuracy']:.4f}  masked={masked['accuracy']:.4f}  drop={drop:+.4f}")

            results["clean"][ds_name] = clean
            results["masked"][ds_name] = masked

        all_results[label] = results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))

    # Summary table
    print("\n" + "=" * 90)
    print(f"OCCLUSION ROBUSTNESS — lower-face mask (zero-fill, y≥55%)")
    print(f"{'Model':<16} {'Dataset':<10} {'Clean':>8} {'Masked':>8} {'Drop':>8} {'TAR@1e-3 clean':>16} {'TAR@1e-3 mask':>15}")
    print("-" * 90)
    for label, res in all_results.items():
        for ds, cm in res["clean"].items():
            mm = res["masked"].get(ds, {})
            if not mm:
                continue
            drop = cm["accuracy"] - mm["accuracy"]
            t3c = cm.get("tar_far_0.001", 0)
            t3m = mm.get("tar_far_0.001", 0)
            print(f"{label:<16} {ds:<10} {cm['accuracy']:>8.4f} {mm['accuracy']:>8.4f} {drop:>+8.4f} {t3c:>16.4f} {t3m:>15.4f}")
    print("=" * 90)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
