#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.datasets import PairVerificationDataset
from fas_kd.data.transforms import build_eval_transform
from fas_kd.evaluation.verification import evaluate_pair_verification
from fas_kd.models.student import MobileNetV4Student
from fas_kd.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained student checkpoint on IJB 1:1 pairs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to latest.pt or best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = MobileNetV4Student(
        backbone_name=cfg["student"]["backbone_name"],
        embedding_dim=int(cfg["student"].get("embedding_dim", 512)),
        pretrained=False,
        input_size=int(cfg["data"].get("image_size", 112)),
        projection_activation=str(cfg["student"].get("projection_activation", "none")),
        spatial_out_channels=int(cfg["student"].get("spatial_out_channels", 0)),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    student.load_state_dict(checkpoint["student_state"], strict=True)
    student.eval()

    pairs_csv = cfg["data"]["ijb"]["protocol_csv"]
    dataset = PairVerificationDataset(pairs_csv=pairs_csv, transform=build_eval_transform(cfg["data"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    metrics = evaluate_pair_verification(
        model=student,
        dataloader=loader,
        device=device,
        use_amp=bool(cfg["system"].get("use_amp", True)),
        target_fars=[float(x) for x in cfg["metrics"].get("target_fars", [1e-3, 1e-4, 1e-5])],
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
