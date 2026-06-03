#!/usr/bin/env python3
"""Evaluate teacher model with input_mode=identity on clean IJBB and IJBC.

Prints a compact comparison against the existing phase1/2/3 clean results
so you can judge which checkpoint is the best baseline for the next run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.transforms import build_eval_transform
from fas_kd.evaluation.ijb_template import evaluate_ijb_template_1to1
from fas_kd.models.teacher import build_frozen_teacher
from fas_kd.utils.config import load_yaml_config

_DEFAULT_CLEAN_ROOT = PROJECT_ROOT / "data" / "processed" / "ijb_clean_yolo11"

PHASE_EXISTING = {
    "phase1": {
        "IJBB": "logs/ijb_clean_matrix_20260601/eval_phase1_clean_ijbb_magface_weighted.json",
        "IJBC": "logs/ijb_clean_matrix_20260601/eval_phase1_clean_ijbc_magface_weighted.json",
    },
    "phase2": {
        "IJBB": "logs/ijb_clean_matrix_20260601/eval_phase2_clean_ijbb_magface_weighted.json",
        "IJBC": "logs/ijb_clean_matrix_20260601/eval_phase2_clean_ijbc_magface_weighted.json",
    },
    "phase3": {
        "IJBB": "logs/ijb_clean_matrix_20260601/eval_phase3_clean_ijbb_magface_weighted.json",
        "IJBC": "logs/ijb_clean_matrix_20260601/eval_phase3_clean_ijbc_magface_weighted.json",
    },
}


def _fmt(v) -> str:
    if v is None:
        return "  n/a  "
    return f"{float(v):.4f}"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", default=str(_DEFAULT_CLEAN_ROOT))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    CLEAN_ROOT = Path(args.clean_root)
    OUT_DIR = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "logs" / "teacher_identity_eval"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = load_yaml_config(str(PROJECT_ROOT / "configs" / "train_ms1m_magface_phase3_trueasym_swa_v1.yaml"))
    teacher_cfg = dict(cfg["teacher"])
    teacher_cfg["input_mode"] = "from_minus_one_to_zero_one"

    print("Loading teacher (input_mode=from_minus_one_to_zero_one)...")
    model = build_frozen_teacher(teacher_cfg).to(device)
    transform = build_eval_transform(cfg["data"])
    target_fars = [1e-3, 1e-4, 1e-5]

    results: dict[str, dict] = {}
    for dataset in ["IJBB", "IJBC"]:
        clean_root = CLEAN_ROOT / dataset
        if not clean_root.exists():
            print(f"[skip] clean root not found: {clean_root}")
            continue
        print(f"Evaluating teacher identity on clean {dataset}...")
        m = evaluate_ijb_template_1to1(
            model=model,
            ijb_root=clean_root,
            transform=transform,
            device=device,
            use_amp=True,
            target_fars=target_fars,
            batch_size=128,
            num_workers=4,
            use_flip=True,
            template_pooling="magface_weighted",
        )
        out_path = OUT_DIR / f"eval_teacher_identity_clean_{dataset.lower()}.json"
        out_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}")
        results[f"teacher_identity_{dataset}"] = m

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Print comparison table
    print()
    print("=" * 80)
    print(f"{'Model':<28} {'Dataset':<6} {'AUC':>8} {'TAR@1e-3':>10} {'TAR@1e-4':>10} {'TAR@1e-5':>10}")
    print("-" * 80)

    for model_name, dataset_paths in PHASE_EXISTING.items():
        for dataset, rel_path in dataset_paths.items():
            p = (PROJECT_ROOT / rel_path).resolve()
            if not p.exists():
                print(f"  [missing] {p}")
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            print(
                f"  {model_name:<26} {dataset:<6} "
                f"{_fmt(m.get('roc_auc')):>8} "
                f"{_fmt(m.get('tar_far_1e-3')):>10} "
                f"{_fmt(m.get('tar_far_1e-4')):>10} "
                f"{_fmt(m.get('tar_far_1e-5')):>10}"
            )

    for key, m in results.items():
        label, dataset = key.rsplit("_", 1)
        print(
            f"  {label:<26} {dataset:<6} "
            f"{_fmt(m.get('roc_auc')):>8} "
            f"{_fmt(m.get('tar_far_1e-3')):>10} "
            f"{_fmt(m.get('tar_far_1e-4')):>10} "
            f"{_fmt(m.get('tar_far_1e-5')):>10}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
