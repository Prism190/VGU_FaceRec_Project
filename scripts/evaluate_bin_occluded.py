#!/usr/bin/env python3
"""Bin protocol evaluation under lower-face occlusion (surgical mask simulation).

Uses the same BinPairDataset + evaluate_pair_verification infrastructure as
evaluate_bin_protocol.py — same accuracy metric (best threshold sweep, not EER),
same DataLoader workers, same flip-TTA.

Only difference: optionally applies apply_lower_face_mask to BOTH images before
computing embeddings (clean) or WITH mask applied (masked).

Usage:
    ./venv/bin/python scripts/evaluate_bin_occluded.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.transforms import apply_lower_face_mask, build_eval_transform
from fas_kd.models.student import MobileNetV4Student
from fas_kd.utils.config import load_yaml_config

# Re-use BinPairDataset from evaluate_bin_protocol
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_bin_protocol import BinPairDataset  # type: ignore

CHECKPOINTS = [
    ("phase1/latest", "configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml",
     "runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"),
    ("phase2/latest", "configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml",
     "runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"),
    ("phase3/swa",   "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml",
     "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/swa.pt"),
]

BIN_SETS = {
    "lfw":    "lfw.bin",
    "cfp_fp": "cfp_fp.bin",
    "agedb_30": "agedb_30.bin",
    "cplfw":  "cplfw.bin",
    "calfw":  "calfw.bin",
}


def _best_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(-1.0, 1.0 + 0.001, 0.001)
    best_acc, best_thr = 0.0, 0.0
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_acc, best_thr


class _MaskedBinDataset(BinPairDataset):
    """BinPairDataset with optional lower-face mask applied to both images."""
    def __init__(self, bin_path, transform, apply_mask: bool = False):
        super().__init__(bin_path=bin_path, transform=transform)
        self.apply_mask = apply_mask

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.apply_mask:
            item["image_a"] = apply_lower_face_mask(item["image_a"], mask_fill="zero")
            item["image_b"] = apply_lower_face_mask(item["image_b"], mask_fill="zero")
        return item


@torch.no_grad()
def _run_bin(model, bin_path, transform, device, apply_mask, num_workers=4, batch_size=256):
    ds = _MaskedBinDataset(bin_path=bin_path, transform=transform, apply_mask=apply_mask)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    model.eval()
    scores_list, labels_list = [], []
    for batch in dl:
        a = batch["image_a"].to(device, non_blocking=True)
        b = batch["image_b"].to(device, non_blocking=True)
        y = batch["is_same"].cpu().numpy().astype(np.int32)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            ea = model(a) + model(torch.flip(a, dims=[3]))
            eb = model(b) + model(torch.flip(b, dims=[3]))
        ea = F.normalize(torch.nan_to_num(ea, nan=0.0), dim=1)
        eb = F.normalize(torch.nan_to_num(eb, nan=0.0), dim=1)
        s = (ea * eb).sum(dim=1).cpu().numpy()
        scores_list.append(s)
        labels_list.append(y)
    scores = np.concatenate(scores_list)
    labels = np.concatenate(labels_list)
    acc, _ = _best_accuracy(scores, labels)
    return float(acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-root", default="data/raw/casia-webface/faces_webface_112x112")
    parser.add_argument("--out-dir", default="logs/eval_occluded")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
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
        if not cfg_full.exists() or not ckpt_full.exists():
            print(f"[skip] {label}")
            continue

        cfg = load_yaml_config(str(cfg_full))
        sc = cfg["student"]
        model = MobileNetV4Student(
            backbone_name=sc["backbone_name"], embedding_dim=512, pretrained=False,
            input_size=112, projection_activation=str(sc.get("projection_activation", "none")),
            spatial_out_channels=int(sc.get("spatial_out_channels", 0)),
        )
        ckpt = torch.load(str(ckpt_full), map_location="cpu")
        model.load_state_dict(ckpt.get("student_state", ckpt), strict=True)
        model.to(device).eval()
        transform = build_eval_transform(cfg["data"])

        print(f"\n=== {label} ===", flush=True)
        results = {}
        for ds_name, fname in BIN_SETS.items():
            bin_path = bin_root / fname
            if not bin_path.exists():
                print(f"  [skip] {ds_name}", flush=True)
                continue
            clean_acc = _run_bin(model, bin_path, transform, device,
                                  apply_mask=False, num_workers=args.num_workers)
            masked_acc = _run_bin(model, bin_path, transform, device,
                                   apply_mask=True, num_workers=args.num_workers)
            drop = clean_acc - masked_acc
            print(f"  {ds_name:<10} clean={clean_acc:.4f}  masked={masked_acc:.4f}  drop={drop:+.4f}", flush=True)
            results[ds_name] = {"clean": clean_acc, "masked": masked_acc, "drop": drop}

        all_results[label] = results
        (out_dir / f"{label.replace('/','_')}.json").write_text(json.dumps(results, indent=2))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 75, flush=True)
    print(f"OCCLUSION ROBUSTNESS — lower-face mask, zero-fill y≥55% (training mask)", flush=True)
    print(f"{'Model':<14} {'Dataset':<11} {'Clean':>8} {'Masked':>8} {'Drop':>8}", flush=True)
    print("-" * 75, flush=True)
    for label, res in all_results.items():
        for ds, m in res.items():
            print(f"{label:<14} {ds:<11} {m['clean']:>8.4f} {m['masked']:>8.4f} {m['drop']:>+8.4f}", flush=True)
    print("=" * 75, flush=True)

    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
