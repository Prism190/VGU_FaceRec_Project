#!/usr/bin/env python3
"""Run occlusion robustness eval on the teacher (iResNet-100 MagFace) only."""
from __future__ import annotations
import json, sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fas_kd.data.transforms import build_eval_transform
from fas_kd.models.teacher import build_frozen_teacher
from fas_kd.utils.config import load_yaml_config
from evaluate_bin_occluded import _run_bin, BIN_SETS  # type: ignore

CFG_PATH  = PROJECT_ROOT / "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"
BIN_ROOT  = PROJECT_ROOT / "data/raw/casia-webface/faces_webface_112x112"
OUT_FILE  = PROJECT_ROOT / "results/occlusion/teacher.json"

def main():
    cfg = load_yaml_config(str(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_frozen_teacher(cfg["teacher"]).to(device)
    transform = build_eval_transform(cfg["data"])

    print("=== teacher ===", flush=True)
    results = {}
    for ds_name, fname in BIN_SETS.items():
        bin_path = BIN_ROOT / fname
        if not bin_path.exists():
            print(f"  [skip] {ds_name}", flush=True)
            continue
        clean  = _run_bin(model, bin_path, transform, device, apply_mask=False, num_workers=4)
        masked = _run_bin(model, bin_path, transform, device, apply_mask=True,  num_workers=4)
        drop_acc = clean["accuracy"]   - masked["accuracy"]
        drop_t3  = clean["tar_far_1e-3"] - masked["tar_far_1e-3"]
        print(f"  {ds_name:<10} clean={clean['accuracy']:.4f}  masked={masked['accuracy']:.4f}"
              f"  drop={drop_acc:+.4f} | T@1e-3 clean={clean['tar_far_1e-3']:.4f}"
              f"  masked={masked['tar_far_1e-3']:.4f}  drop={drop_t3:+.4f}", flush=True)
        results[ds_name] = {
            "clean_acc":    clean["accuracy"],
            "masked_acc":   masked["accuracy"],
            "drop_acc":     drop_acc,
            "clean_tar_1e3": clean["tar_far_1e-3"],
            "masked_tar_1e3": masked["tar_far_1e-3"],
            "drop_tar_1e3": drop_t3,
            "clean_tar_1e4": clean["tar_far_1e-4"],
            "masked_tar_1e4": masked["tar_far_1e-4"],
        }

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {OUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
