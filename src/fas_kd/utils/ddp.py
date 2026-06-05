from __future__ import annotations

import datetime
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def init_distributed() -> DistributedContext:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        timeout_minutes = max(1.0, float(os.environ.get("DDP_TIMEOUT_MINUTES", "60")))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return DistributedContext(
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def seed_everything(seed: int, rank: int = 0) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def is_main_process(ctx: DistributedContext) -> bool:
    return ctx.rank == 0


def synchronize(ctx: DistributedContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def reduce_mean(value: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if not ctx.is_distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= ctx.world_size
    return reduced


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
