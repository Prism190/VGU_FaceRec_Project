#!/usr/bin/env python3
"""Chain IJB template 1:1 evaluation for teacher + phase1 + phase2 + phase3.

Supports both YOLO-cleaned and InsightFace-cleaned datasets, with flip-TTA.

Usage:
    ./venv/bin/python scripts/run_ijb_eval_chain.py                     # YOLO clean
    ./venv/bin/python scripts/run_ijb_eval_chain.py --insightface        # InsightFace clean
    ./venv/bin/python scripts/run_ijb_eval_chain.py --out-dir logs/eval_chain_custom
"""
from __future__ import annotations

import argparse
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
from fas_kd.models.student import MobileNetV4Student
from fas_kd.models.teacher import build_frozen_teacher
from fas_kd.utils.config import load_yaml_config

PHASE1_CFG = PROJECT_ROOT / "configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml"
PHASE2_CFG = PROJECT_ROOT / "configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml"
PHASE3_CFG = PROJECT_ROOT / "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"

SPECS = [
    # (label, config, checkpoint-or-"teacher")
    ("teacher", PHASE1_CFG, "teacher"),
    ("phase3",  PHASE3_CFG, PROJECT_ROOT / "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt"),
    ("phase2",  PHASE2_CFG, PROJECT_ROOT / "runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"),
    ("phase1",  PHASE1_CFG, PROJECT_ROOT / "runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"),
]


def _load_student(cfg: dict, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    sc = cfg["student"]
    model = MobileNetV4Student(
        backbone_name=sc["backbone_name"],
        embedding_dim=int(sc.get("embedding_dim", 512)),
        pretrained=False,
        input_size=int(cfg["data"].get("image_size", 112)),
        projection_activation=str(sc.get("projection_activation", "none")),
        spatial_out_channels=int(sc.get("spatial_out_channels", 0)),
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state.get("student_state", state), strict=True)
    return model.to(device).eval()


def _load_teacher(cfg: dict, device: torch.device) -> torch.nn.Module:
    return build_frozen_teacher(cfg["teacher"]).to(device).eval()


def _run(label: str, model: torch.nn.Module, cfg: dict, clean_root: Path,
         device: torch.device, batch_size: int, out_dir: Path) -> dict:
    transform = build_eval_transform(cfg["data"])
    target_fars = cfg.get("metrics", {}).get("target_fars", [1e-3, 1e-4, 1e-5])
    results = {"label": label}
    for dataset in ["IJBB", "IJBC"]:
        ds_root = clean_root / dataset
        if not ds_root.exists():
            print(f"  [{label}] {dataset} root not found: {ds_root} — skip")
            continue
        # Wait for alignment to finish if needed (IJBC may still be processing)
        loose_crop = ds_root / "loose_crop"
        if loose_crop.exists():
            import time, glob as _glob
            n = len(list(loose_crop.glob("*.jpg")))
            # IJBB ~227k images, IJBC ~469k images
            expected = 460000 if dataset == "IJBC" else 220000
            if n < expected:
                print(f"  [{label}] {dataset}: only {n} images present, waiting for alignment to finish...", flush=True)
                while len(list(loose_crop.glob("*.jpg"))) < expected:
                    time.sleep(60)
                    current = len(list(loose_crop.glob("*.jpg")))
                    print(f"    ... {current} images so far", flush=True)
                print(f"  [{label}] {dataset}: alignment complete, starting eval", flush=True)
        print(f"  [{label}] evaluating {dataset} ...", flush=True)
        m = evaluate_ijb_template_1to1(
            model=model,
            ijb_root=ds_root,
            transform=transform,
            device=device,
            use_amp=cfg.get("system", {}).get("use_amp", True),
            target_fars=target_fars,
            batch_size=batch_size,
            num_workers=4,
            template_pooling="magface_weighted",
            use_flip=True,
        )
        results[dataset] = m
        out_file = out_dir / f"{label}_{dataset.lower()}.json"
        out_file.write_text(json.dumps(m, indent=2))
        auc = m.get("roc_auc", 0)
        t3 = m.get("tar_far_1e-3", 0)
        t4 = m.get("tar_far_1e-4", 0)
        print(f"    {dataset}: AUC={auc:.4f}  TAR@1e-3={t3:.4f}  TAR@1e-4={t4:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="IJB eval chain: teacher + phase1 + phase2 + phase3")
    parser.add_argument("--insightface", action="store_true",
                        help="Use InsightFace-cleaned data instead of YOLO-cleaned")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher eval (use when teacher results already exist)")
    parser.add_argument("--clean-root", default=None,
                        help="Override clean data root directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.clean_root:
        clean_root = Path(args.clean_root)
    elif args.insightface:
        clean_root = PROJECT_ROOT / "data/processed/ijb_clean_insightface"
    else:
        clean_root = PROJECT_ROOT / "data/processed/ijb_clean_yolo11"

    if not clean_root.exists():
        print(f"[error] Clean root not found: {clean_root}")
        sys.exit(1)

    tag = "insightface" if args.insightface else "yolo"
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / f"logs/ijb_chain_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Clean root:  {clean_root}")
    print(f"Output dir:  {out_dir}")
    print(f"Device:      {device}")
    print(f"Flip-TTA:    ON")
    print()

    all_results = []
    for label, cfg_path, ckpt in SPECS:
        if args.skip_teacher and label == "teacher":
            continue
        if not Path(cfg_path).exists():
            print(f"[skip] {label}: config not found")
            continue
        if ckpt != "teacher" and not Path(str(ckpt)).exists():
            print(f"[skip] {label}: checkpoint not found at {ckpt}")
            continue

        cfg = load_yaml_config(str(cfg_path))
        print(f"\n=== {label.upper()} ===")
        if ckpt == "teacher":
            model = _load_teacher(cfg, device)
        else:
            model = _load_student(cfg, Path(str(ckpt)), device)

        res = _run(label=label, model=model, cfg=cfg, clean_root=clean_root,
                   device=device, batch_size=args.batch_size, out_dir=out_dir)
        all_results.append(res)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print()
    print("=" * 65)
    print(f"RESULTS — {tag} clean, flip-TTA=ON")
    print("=" * 65)
    print(f"{'Model':<10} {'Dataset':<7} {'AUC':>8} {'TAR@1e-3':>10} {'TAR@1e-4':>10} {'TAR@1e-5':>10}")
    print("-" * 65)
    for res in all_results:
        for ds in ["IJBB", "IJBC"]:
            m = res.get(ds, {})
            if m:
                print(f"{res['label']:<10} {ds:<7} "
                      f"{m.get('roc_auc',0):>8.4f} "
                      f"{m.get('tar_far_1e-3',0):>10.4f} "
                      f"{m.get('tar_far_1e-4',0):>10.4f} "
                      f"{m.get('tar_far_1e-5',0):>10.4f}")
    print("=" * 65)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results: {out_dir}")


if __name__ == "__main__":
    main()
