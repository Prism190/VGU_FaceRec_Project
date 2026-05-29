#!/usr/bin/env python3
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
from fas_kd.utils.config import load_yaml_config


def _resolve_ijb_root(cfg, dataset_name: str) -> Path:
    data_cfg = cfg.get("data", {})
    ijb_cfg = data_cfg.get("ijb", {})

    configured = ijb_cfg.get(f"{dataset_name.lower()}_root")
    if configured:
        return Path(configured)

    output_root = Path(cfg.get("experiment", {}).get("output_root", str(PROJECT_ROOT)))
    project_root = output_root
    if not (project_root / "src").exists() and (output_root.parent / "src").exists():
        project_root = output_root.parent
    return project_root / "data" / "raw" / "ijb" / "ijb" / dataset_name


def main() -> None:
    parser = argparse.ArgumentParser(description="IJB template-based 1:1 evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to student checkpoint")
    parser.add_argument("--dataset", choices=["IJBB", "IJBC"], default="IJBB")
    parser.add_argument(
        "--ijb-root",
        default=None,
        help="Optional dataset root override (expects IJBB/ or IJBC/ directory).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--template-pooling",
        choices=["mean", "magface_weighted"],
        default="magface_weighted",
        help="Template aggregation mode.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    student_cfg = cfg["student"]
    model = MobileNetV4Student(
        backbone_name=student_cfg["backbone_name"],
        embedding_dim=int(student_cfg.get("embedding_dim", 512)),
        pretrained=False,
        input_size=int(cfg["data"].get("image_size", 112)),
        projection_activation=str(student_cfg.get("projection_activation", "none")),
        spatial_out_channels=int(student_cfg.get("spatial_out_channels", 0)),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("student_state", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"[warn] Missing student keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected student keys: {len(unexpected)}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = build_eval_transform(cfg["data"])

    target_fars = cfg.get("metrics", {}).get("target_fars", [1e-3, 1e-4, 1e-5])
    ijb_root = Path(args.ijb_root) if args.ijb_root else _resolve_ijb_root(cfg=cfg, dataset_name=args.dataset)

    metrics = evaluate_ijb_template_1to1(
        model=model,
        ijb_root=ijb_root,
        transform=transform,
        device=device,
        use_amp=cfg.get("system", {}).get("use_amp", True),
        target_fars=target_fars,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        template_pooling=args.template_pooling,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
