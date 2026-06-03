#!/usr/bin/env python3
"""Post-training Stochastic Weight Averaging (SWA) for MobileNetV4 student checkpoints.

Averages the student_state_dict of all saved epoch checkpoints in [start_epoch, end_epoch]
and writes the result to checkpoints/swa.pt alongside the run's latest.pt / best.pt.

Reads SWA parameters from the config (train.swa.start_epoch) by default, or accepts
explicit CLI overrides for use with any phase.

Typical usage after Phase 3 training:

    ./venv/bin/python scripts/apply_swa.py \\
        --config configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml

This reproduces the swa.pt used in benchmarks (epochs 35-39, save_every=2 → 3 checkpoints).

Usage with explicit range (e.g. any phase):

    ./venv/bin/python scripts/apply_swa.py \\
        --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \\
        --start-epoch 30 --end-epoch 39
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.utils.config import load_yaml_config


def _find_epoch_checkpoints(ckpt_dir: Path, start: int, end: int) -> list[tuple[int, Path]]:
    """Return sorted (epoch, path) pairs for epoch_NNN.pt files in [start, end]."""
    found = []
    for f in ckpt_dir.glob("epoch_*.pt"):
        try:
            epoch = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if start <= epoch <= end:
            found.append((epoch, f))
    found.sort(key=lambda x: x[0])
    return found


def apply_swa(
    config_path: Path,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
) -> Path:
    cfg = load_yaml_config(config_path)

    output_root = Path(cfg["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    ckpt_dir = output_root / "checkpoints"

    swa_cfg = cfg.get("train", {}).get("swa", {})
    total_epochs = int(cfg.get("train", {}).get("epochs", 40))

    if start_epoch is None:
        start_epoch = int(swa_cfg.get("start_epoch", total_epochs - 5))
    if end_epoch is None:
        end_epoch = total_epochs - 1

    if out_path is None:
        out_path = ckpt_dir / "swa.pt"

    print(f"Config:       {config_path}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"SWA range:    epochs {start_epoch} – {end_epoch} (inclusive, 0-indexed)")
    print(f"Output:       {out_path}")

    candidates = _find_epoch_checkpoints(ckpt_dir, start_epoch, end_epoch)
    if not candidates:
        print(
            f"\n[ERROR] No epoch_NNN.pt files found in {ckpt_dir} "
            f"for epoch range [{start_epoch}, {end_epoch}]."
        )
        print("  Available epoch checkpoints:")
        for f in sorted(ckpt_dir.glob("epoch_*.pt")):
            print(f"    {f.name}")
        sys.exit(1)

    print(f"\nCheckpoints to average ({len(candidates)}):")
    for epoch, path in candidates:
        print(f"  epoch {epoch:03d}  {path.name}")

    if dry_run:
        print("\n[dry-run] Stopping here — no file written.")
        return out_path

    # Load and accumulate student_state
    print("\nAveraging student weights...")
    avg_state: dict[str, torch.Tensor] = {}
    ref_ckpt: dict = {}

    for i, (epoch, path) in enumerate(candidates):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("student_state") or ckpt
        if not avg_state:
            avg_state = {k: v.float().clone() for k, v in state.items()}
            ref_ckpt = ckpt
        else:
            if set(state.keys()) != set(avg_state.keys()):
                print(f"  [WARN] epoch {epoch} has different keys — skipping")
                continue
            for k in avg_state:
                avg_state[k] += state[k].float()
        print(f"  accumulated epoch {epoch:03d}")

    n = len(candidates)
    avg_state = {k: (v / n).to(ref_ckpt["student_state"][k].dtype)
                 for k, v in avg_state.items()}

    # Build output checkpoint — same structure as training checkpoints
    out_ckpt = {
        "student_state": avg_state,
        "config":        ref_ckpt.get("config", {}),
        "epoch":         candidates[-1][0],
        "best_metric":   ref_ckpt.get("best_metric"),
        "swa_epochs":    [e for e, _ in candidates],
        "swa_n":         n,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
    print(f"Averaged {n} checkpoints: epochs {[e for e, _ in candidates]}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training SWA: average epoch checkpoints into swa.pt"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to training YAML config (reads experiment.output_root and train.swa.start_epoch)"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=None,
        help="First epoch to include (0-indexed). Defaults to train.swa.start_epoch in config."
    )
    parser.add_argument(
        "--end-epoch", type=int, default=None,
        help="Last epoch to include (0-indexed, inclusive). Defaults to train.epochs - 1."
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path for swa.pt. Defaults to <output_root>/checkpoints/swa.pt."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print which checkpoints would be averaged without writing any file."
    )
    args = parser.parse_args()

    apply_swa(
        config_path=Path(args.config),
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        out_path=args.out,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
