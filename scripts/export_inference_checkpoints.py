#!/usr/bin/env python3
"""Extract lean inference checkpoints from full training checkpoints.

Full checkpoints include optimizer/scheduler state (~608 MB).
Inference checkpoints contain only the student weights (~37 MB).

Usage:
    ./venv/bin/python scripts/export_inference_checkpoints.py \
        --out-dir checkpoints/release
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASES = [
    {
        "name": "phase1",
        "config": "configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml",
        "checkpoint": "runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt",
        "description": "Phase 1 latest (epoch 39) — KD + MagFace. Best overall IJB: IJBB 87.98% / IJBC 90.65% TAR@1e-4.",
    },
    {
        "name": "phase1_best",
        "config": "configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml",
        "checkpoint": "runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/best.pt",
        "description": "Phase 1 best (epoch 29) — highest bin protocol mean accuracy (96.37%).",
    },
    {
        "name": "phase2",
        "config": "configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml",
        "checkpoint": "runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt",
        "description": "Phase 2 latest (epoch 32) — occlusion curriculum + spatial KD. Weaker than phase1/phase3.",
    },
    {
        "name": "phase3",
        "config": "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml",
        "checkpoint": "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt",
        "description": "Phase 3 latest (epoch 39) — true asymmetric distillation.",
    },
    {
        "name": "phase3_swa",
        "config": "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml",
        "checkpoint": "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/swa.pt",
        "description": "Phase 3 SWA (epochs 35-39 avg) — best occlusion robustness; recommended for masked/occluded deployment.",
    },
    {
        "name": "phase3_best",
        "config": "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml",
        "checkpoint": "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/best.pt",
        "description": "Phase 3 best (epoch 13) — highest LFW accuracy (99.43%) among phase3 checkpoints.",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="checkpoints/release")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for spec in PHASES:
        ckpt_path = PROJECT_ROOT / spec["checkpoint"]
        if not ckpt_path.exists():
            print(f"[skip] {spec['name']}: {ckpt_path} not found")
            continue

        print(f"Exporting {spec['name']} ...", end=" ", flush=True)
        full = torch.load(str(ckpt_path), map_location="cpu")

        lean = {
            "student_state":     full["student_state"],
            "config":            full.get("config", {}),
            "epoch":             full.get("epoch", 0),
            "best_metric":       full.get("best_metric", None),
        }

        out_path = out_dir / f"mobilenetv4_student_{spec['name']}.pt"
        torch.save(lean, out_path)
        size_mb = out_path.stat().st_size / 1e6
        sha = _sha256(out_path)
        print(f"{size_mb:.1f} MB  sha256={sha[:16]}...")

        manifest.append({
            "name":        spec["name"],
            "filename":    out_path.name,
            "description": spec["description"],
            "size_mb":     round(size_mb, 1),
            "sha256":      sha,
            "config":      spec["config"],
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")
    print("\nUpload files in", out_dir, "to GitHub Releases as release assets.")


if __name__ == "__main__":
    main()
