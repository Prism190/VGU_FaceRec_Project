#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.engine import run_training
from fas_kd.utils.config import apply_overrides, ensure_runtime_dirs, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV4 student with KD + margin head using torchrun/DDP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values using dotted key syntax, e.g. --override train.epochs=40",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    ensure_runtime_dirs(cfg)
    run_training(cfg)


if __name__ == "__main__":
    record(main)()
