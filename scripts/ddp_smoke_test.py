#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.utils.ddp import cleanup_distributed, init_distributed, is_main_process


def main() -> None:
    ctx = init_distributed()
    try:
        if ctx.device.type != "cuda":
            raise RuntimeError("DDP smoke test expects CUDA device")

        value = torch.tensor(float(ctx.rank + 1), device=ctx.device)
        if ctx.is_distributed:
            import torch.distributed as dist

            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            expected = (ctx.world_size * (ctx.world_size + 1)) / 2.0
            if abs(value.item() - expected) > 1e-5:
                raise RuntimeError(f"All-reduce mismatch: got {value.item()}, expected {expected}")

        if is_main_process(ctx):
            print(f"DDP smoke test passed with world_size={ctx.world_size}")
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
