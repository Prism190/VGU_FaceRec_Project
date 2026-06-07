from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fas_kd.data import DataLoaderX, TrainKDDataset, create_dali_recordio_loader, load_num_classes
from fas_kd.data.transforms import build_train_transform
from fas_kd.engine.validate import validate_on_sets
from fas_kd.losses import DistillationObjective
from fas_kd.models import MobileNetV4Student, build_frozen_teacher, build_margin_head
from fas_kd.utils.ddp import (
    DistributedContext,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_mean,
    seed_everything,
    synchronize,
)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _ddp_wrap(module: torch.nn.Module, ctx: DistributedContext, find_unused_parameters: bool = False) -> torch.nn.Module:
    if ctx.is_distributed and ctx.device.type == "cuda":
        return DDP(module, device_ids=[ctx.local_rank], output_device=ctx.local_rank,
                   broadcast_buffers=False, find_unused_parameters=find_unused_parameters)
    return module


def _save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    student: torch.nn.Module,
    margin_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    best_metric: float,
) -> None:
    payload = {
        "epoch": epoch,
        "student_state": _unwrap(student).state_dict(),
        "margin_head_state": _unwrap(margin_head).state_dict(),
        "optimizer_type": type(optimizer).__name__,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_type": type(scheduler).__name__,
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config,
        "best_metric": best_metric,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically so outages cannot leave a truncated latest checkpoint.
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def _load_module_state(module: torch.nn.Module, state_dict: dict[str, Any], module_name: str) -> None:
    target = _unwrap(module)
    try:
        target.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        # Backward compatibility for checkpoints saved from wrapped modules.
        if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
            stripped_state = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
            target.load_state_dict(stripped_state, strict=True)
            return
        raise RuntimeError(f"Failed to load {module_name} weights from checkpoint") from exc


def _resolve_resume_path(train_cfg: dict[str, Any], checkpoint_dir: Path) -> Path | None:
    resume_from = train_cfg.get("resume_from", None)
    auto_resume = bool(train_cfg.get("auto_resume", False))

    if resume_from is None:
        if not auto_resume:
            return None
        candidate = checkpoint_dir / "latest.pt"
        return candidate if candidate.is_file() else None

    if not isinstance(resume_from, str):
        raise TypeError("train.resume_from must be a string path, 'auto', or null")

    resume_token = resume_from.strip()
    if resume_token == "":
        return None

    lowered = resume_token.lower()
    if lowered in {"none", "null", "false", "off"}:
        return None
    if lowered in {"auto", "latest"}:
        candidate = checkpoint_dir / "latest.pt"
        return candidate if candidate.is_file() else None

    candidate = Path(resume_token).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def _log_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _apply_lower_face_mask_batch(clear: torch.Tensor, mask_prob: float, mask_fill: str) -> torch.Tensor:
    if clear.ndim != 4:
        raise ValueError(f"Expected BCHW batch tensor, got shape {tuple(clear.shape)}")

    if mask_prob <= 0.0:
        return clear.clone()

    out = clear.clone()
    b, c, h, w = out.shape
    y_start = int(h * 0.55)

    apply_mask = torch.rand((b, 1, 1, 1), device=out.device) < float(mask_prob)
    region_mask = apply_mask.expand(b, c, h - y_start, w)

    if mask_fill == "noise":
        fill = (torch.rand_like(out[:, :, y_start:, :]) * 2.0) - 1.0
    else:
        fill = torch.zeros_like(out[:, :, y_start:, :])

    out[:, :, y_start:, :] = torch.where(region_mask, fill, out[:, :, y_start:, :])
    return out


def _to_odd_int(value: int, minimum: int = 3) -> int:
    out = max(minimum, int(value))
    if out % 2 == 0:
        out += 1
    return out


def _parse_kernel_range(raw: Any, default_low: int, default_high: int) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        low = int(raw[0])
        high = int(raw[1])
    elif raw is None:
        low = int(default_low)
        high = int(default_high)
    else:
        low = int(raw)
        high = int(raw)

    low = _to_odd_int(low)
    high = _to_odd_int(high)
    if low > high:
        low, high = high, low
    return low, high


def _linear_ramp(epoch: int, start_epoch: int, end_epoch: int, start_value: float, end_value: float) -> float:
    if end_epoch <= start_epoch:
        return float(end_value if epoch >= end_epoch else start_value)
    if epoch <= start_epoch:
        return float(start_value)
    if epoch >= end_epoch:
        return float(end_value)

    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(start_value + progress * (end_value - start_value))


def _sample_odd_kernel(low: int, high: int, device: torch.device) -> int:
    low = _to_odd_int(low)
    high = _to_odd_int(high)
    if low >= high:
        return low
    choices = ((high - low) // 2) + 1
    offset = int(torch.randint(0, choices, (1,), device=device).item())
    return low + (2 * offset)


def _build_gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel_size = _to_odd_int(kernel_size)
    sigma = max(float(sigma), 0.05)
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - ((kernel_size - 1) / 2.0)
    sq = (coords[:, None] ** 2) + (coords[None, :] ** 2)
    kernel = torch.exp(-sq / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype=dtype)


def _build_motion_kernel2d(kernel_size: int, mode: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel_size = _to_odd_int(kernel_size)
    kernel = torch.zeros((kernel_size, kernel_size), device=device, dtype=torch.float32)
    center = kernel_size // 2

    if mode == 0:
        kernel[center, :] = 1.0
    elif mode == 1:
        kernel[:, center] = 1.0
    elif mode == 2:
        idx = torch.arange(kernel_size, device=device)
        kernel[idx, idx] = 1.0
    else:
        idx = torch.arange(kernel_size, device=device)
        kernel[idx, kernel_size - 1 - idx] = 1.0

    kernel = kernel / kernel.sum()
    return kernel.to(dtype=dtype)


def _depthwise_blur_batch(images: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    channels = int(images.shape[1])
    kernel_size = int(kernel_2d.shape[0])
    depthwise_kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return F.conv2d(images, depthwise_kernel, padding=kernel_size // 2, groups=channels)


def _apply_probabilistic_mix(
    base: torch.Tensor,
    augmented: torch.Tensor,
    apply_prob: float,
) -> torch.Tensor:
    if apply_prob <= 0.0:
        return base
    if apply_prob >= 1.0:
        return augmented

    batch_size = int(base.shape[0])
    apply_mask = torch.rand((batch_size, 1, 1, 1), device=base.device) < float(apply_prob)
    return torch.where(apply_mask, augmented, base)


def _apply_training_augmentations_batch(
    clear: torch.Tensor,
    mask_prob: float,
    mask_fill: str,
    gaussian_blur_prob: float,
    gaussian_sigma: float,
    gaussian_kernel_range: tuple[int, int],
    motion_blur_prob: float,
    motion_kernel_range: tuple[int, int],
) -> torch.Tensor:
    out = _apply_lower_face_mask_batch(clear=clear, mask_prob=mask_prob, mask_fill=mask_fill)

    if gaussian_blur_prob > 0.0:
        g_kernel = _sample_odd_kernel(gaussian_kernel_range[0], gaussian_kernel_range[1], device=out.device)
        # Jitter sigma ±30 % around the scheduled value so consecutive batches at the same
        # curriculum epoch don't all share an identical blur intensity (fix #10).
        actual_sigma = float(gaussian_sigma * (0.7 + 0.6 * torch.rand(1, device=out.device).item()))
        actual_sigma = max(actual_sigma, 0.05)
        gaussian_kernel = _build_gaussian_kernel2d(
            kernel_size=g_kernel,
            sigma=actual_sigma,
            device=out.device,
            dtype=out.dtype,
        )
        blurred = _depthwise_blur_batch(out, gaussian_kernel)
        out = _apply_probabilistic_mix(base=out, augmented=blurred, apply_prob=gaussian_blur_prob)

    if motion_blur_prob > 0.0:
        b = int(out.shape[0])
        m_kernel_size = _sample_odd_kernel(motion_kernel_range[0], motion_kernel_range[1], device=out.device)
        # Build all four direction kernels once, then assign each image its own direction
        # (fix #10: previously the whole batch shared a single randomly-drawn direction).
        motion_kernels = [
            _build_motion_kernel2d(m_kernel_size, mode, out.device, out.dtype)
            for mode in range(4)
        ]
        per_sample_modes = torch.randint(0, 4, (b,), device=out.device)
        motion_blurred = out.clone()
        for mode in range(4):
            idx = (per_sample_modes == mode).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                motion_blurred[idx] = _depthwise_blur_batch(out[idx], motion_kernels[mode])
        out = _apply_probabilistic_mix(base=out, augmented=motion_blurred, apply_prob=motion_blur_prob)

    return out


def _resolve_augmentation_schedule(epoch: int, train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, Any]:
    base_mask_prob = float(data_cfg.get("mask_prob", 0.0))
    default = {
        "mask_prob": base_mask_prob,
        "gaussian_blur_prob": 0.0,
        "gaussian_sigma": 1.0,
        "gaussian_kernel_range": (7, 7),
        "motion_blur_prob": 0.0,
        "motion_kernel_range": (9, 9),
    }

    curriculum = train_cfg.get("occlusion_curriculum", {})
    if not isinstance(curriculum, dict) or not bool(curriculum.get("enabled", False)):
        mask_free_epochs = int(train_cfg.get("mask_free_epochs", 0))
        if epoch < mask_free_epochs:
            default["mask_prob"] = 0.0
        return default

    clean_epochs = int(curriculum.get("clean_epochs", 10))
    ramp_start = int(curriculum.get("ramp_start_epoch", clean_epochs))
    ramp_end = int(curriculum.get("ramp_end_epoch", 25))

    mask_start = float(curriculum.get("mask_prob_start", 0.1))
    mask_end = float(curriculum.get("mask_prob_end", 0.3))
    gauss_prob_start = float(curriculum.get("gaussian_prob_start", 0.2))
    gauss_prob_end = float(curriculum.get("gaussian_prob_end", 0.7))
    motion_prob_start = float(curriculum.get("motion_prob_start", 0.1))
    motion_prob_end = float(curriculum.get("motion_prob_end", 0.5))
    sigma_start = float(curriculum.get("gaussian_sigma_start", 1.0))
    sigma_end = float(curriculum.get("gaussian_sigma_end", 2.5))

    gauss_kernel_low, gauss_kernel_high = _parse_kernel_range(
        curriculum.get("gaussian_kernel_size", [5, 11]),
        default_low=5,
        default_high=11,
    )
    motion_kernel_low, motion_kernel_high = _parse_kernel_range(
        curriculum.get("motion_kernel_size", [7, 17]),
        default_low=7,
        default_high=17,
    )

    if epoch < clean_epochs:
        return {
            "mask_prob": 0.0,
            "gaussian_blur_prob": 0.0,
            "gaussian_sigma": sigma_start,
            "gaussian_kernel_range": (gauss_kernel_low, gauss_kernel_low),
            "motion_blur_prob": 0.0,
            "motion_kernel_range": (motion_kernel_low, motion_kernel_low),
        }

    if epoch < ramp_start:
        mask_prob = mask_start
        gauss_prob = gauss_prob_start
        motion_prob = motion_prob_start
        sigma = sigma_start
        gauss_kernel_value = float(gauss_kernel_low)
        motion_kernel_value = float(motion_kernel_low)
    else:
        mask_prob = _linear_ramp(epoch, ramp_start, ramp_end, mask_start, mask_end)
        gauss_prob = _linear_ramp(epoch, ramp_start, ramp_end, gauss_prob_start, gauss_prob_end)
        motion_prob = _linear_ramp(epoch, ramp_start, ramp_end, motion_prob_start, motion_prob_end)
        sigma = _linear_ramp(epoch, ramp_start, ramp_end, sigma_start, sigma_end)
        gauss_kernel_value = _linear_ramp(epoch, ramp_start, ramp_end, float(gauss_kernel_low), float(gauss_kernel_high))
        motion_kernel_value = _linear_ramp(
            epoch,
            ramp_start,
            ramp_end,
            float(motion_kernel_low),
            float(motion_kernel_high),
        )

    gauss_kernel = _to_odd_int(round(gauss_kernel_value), minimum=3)
    motion_kernel = _to_odd_int(round(motion_kernel_value), minimum=3)

    return {
        "mask_prob": float(max(0.0, min(1.0, mask_prob))),
        "gaussian_blur_prob": float(max(0.0, min(1.0, gauss_prob))),
        "gaussian_sigma": float(max(0.05, sigma)),
        "gaussian_kernel_range": (gauss_kernel_low, max(gauss_kernel_low, gauss_kernel)),
        "motion_blur_prob": float(max(0.0, min(1.0, motion_prob))),
        "motion_kernel_range": (motion_kernel_low, max(motion_kernel_low, motion_kernel)),
    }


def _train_one_epoch(
    epoch: int,
    total_epochs: int,
    loader: DataLoader,
    train_dataset: TrainKDDataset | None,
    student: torch.nn.Module,
    margin_head: torch.nn.Module,
    teacher: torch.nn.Module,
    objective: DistillationObjective,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    ctx: DistributedContext,
    use_amp: bool,
    grad_clip_norm: float,
    log_interval: int,
    show_progress_bar: bool,
    mask_prob: float,
    mask_fill: str,
    gaussian_blur_prob: float,
    gaussian_sigma: float,
    gaussian_kernel_range: tuple[int, int],
    motion_blur_prob: float,
    motion_kernel_range: tuple[int, int],
    use_spatial_kd: bool,
) -> dict[str, float]:
    student.train()
    margin_head.train()
    teacher.eval()
    if train_dataset is not None:
        train_dataset.set_epoch(epoch)

    sum_total = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_cls = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_kd = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_spatial_kd = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_reg = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_kd_weight = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_spatial_weight = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_steps = torch.zeros((), dtype=torch.float32, device=ctx.device)
    sum_samples = torch.zeros((), dtype=torch.float32, device=ctx.device)

    start = time.time()
    progress = None
    if show_progress_bar and is_main_process(ctx):
        progress = tqdm(
            total=len(loader),
            desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d}",
            dynamic_ncols=True,
            leave=False,
        )

    for step, batch in enumerate(loader, start=1):
        if isinstance(batch, dict):
            clear = batch["clear"].to(ctx.device, non_blocking=True)
            labels = batch["label"].to(ctx.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            clear = batch[0].to(ctx.device, non_blocking=True)
            labels = batch[1].to(ctx.device, non_blocking=True).long()
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")

        with torch.no_grad():
            masked = _apply_training_augmentations_batch(
                clear=clear,
                mask_prob=mask_prob,
                mask_fill=mask_fill,
                gaussian_blur_prob=gaussian_blur_prob,
                gaussian_sigma=gaussian_sigma,
                gaussian_kernel_range=gaussian_kernel_range,
                motion_blur_prob=motion_blur_prob,
                motion_kernel_range=motion_kernel_range,
            )

        optimizer.zero_grad(set_to_none=True)

        teacher_spatial = None
        with torch.no_grad():
            if use_spatial_kd:
                teacher_embeddings, teacher_spatial = teacher.forward_with_spatial(clear)
            else:
                teacher_embeddings = teacher(clear)

        with torch.autocast(device_type=ctx.device.type, enabled=use_amp and ctx.device.type == "cuda"):
            student_spatial = None
            if use_spatial_kd:
                student_embeddings, student_spatial = student(masked, return_spatial=True)
                if teacher_spatial is None:
                    raise RuntimeError("Spatial KD is enabled but teacher spatial features are missing")

                if student_spatial.shape[2:] != teacher_spatial.shape[2:]:
                    student_spatial = F.interpolate(
                        student_spatial,
                        size=teacher_spatial.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                if teacher_spatial.dtype != student_spatial.dtype:
                    teacher_spatial = teacher_spatial.to(dtype=student_spatial.dtype)
            else:
                student_embeddings = student(masked)

            head_out = margin_head(student_embeddings, labels)
            class_loss = head_out["loss"]
            loss_out = objective(
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_embeddings,
                class_loss=class_loss,
                epoch_idx=epoch,
                student_spatial=student_spatial,
                teacher_spatial=teacher_spatial,
            )
            total_loss = loss_out["total_loss"]

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite loss encountered at "
                f"epoch={epoch} step={step} "
                f"total={float(total_loss.detach().item())} "
                f"cls={float(loss_out['loss_cls'].detach().item())} "
                f"kd={float(loss_out['loss_kd'].detach().item())} "
                f"skd={float(loss_out['loss_spatial_kd'].detach().item())}"
            )

        scaler.scale(total_loss).backward()

        if grad_clip_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(margin_head.parameters()), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        sum_total += total_loss.detach()
        sum_cls += loss_out["loss_cls"]
        sum_kd += loss_out["loss_kd"]
        sum_spatial_kd += loss_out["loss_spatial_kd"]
        sum_reg += head_out["reg_loss"].detach()
        sum_kd_weight += loss_out["kd_weight"].detach()
        sum_spatial_weight += loss_out["spatial_kd_weight"].detach()
        sum_steps += 1.0
        sum_samples += float(labels.size(0))

        if progress is not None:
            progress.update(1)
            if step == 1 or step % max(1, log_interval // 2) == 0 or step == len(loader):
                progress.set_postfix(
                    loss=f"{float(total_loss.detach().item()):.4f}",
                    cls=f"{float(loss_out['loss_cls'].detach().item()):.4f}",
                    kd=f"{float(loss_out['loss_kd'].detach().item()):.4f}",
                    skd=f"{float(loss_out['loss_spatial_kd'].detach().item()):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )

        if step % log_interval == 0 and is_main_process(ctx):
            # Avoid per-step cross-rank sync here because shard step counts can drift
            # with DALI in distributed mode and cause deadlocks.
            msg = (
                f"[Epoch {epoch:03d} | Step {step:05d}] "
                f"total={float(total_loss.detach().item()):.4f} "
                f"cls={float(loss_out['loss_cls'].detach().item()):.4f} "
                f"kd={float(loss_out['loss_kd'].detach().item()):.4f} "
                f"skd={float(loss_out['loss_spatial_kd'].detach().item()):.4f}"
            )
            if progress is not None:
                progress.write(msg)
            else:
                print(msg)

    if progress is not None:
        progress.close()

    if hasattr(loader, "reset"):
        loader.reset()

    elapsed = max(time.time() - start, 1e-6)

    if ctx.is_distributed:
        import torch.distributed as dist

        dist.all_reduce(sum_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_cls, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_kd, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_spatial_kd, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_reg, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_kd_weight, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_spatial_weight, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_steps, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_samples, op=dist.ReduceOp.SUM)

    denom = max(sum_steps.item(), 1.0)
    stats = {
        "train_loss_total": float(sum_total.item() / denom),
        "train_loss_cls": float(sum_cls.item() / denom),
        "train_loss_kd": float(sum_kd.item() / denom),
        "train_loss_spatial_kd": float(sum_spatial_kd.item() / denom),
        "train_loss_mag_reg": float(sum_reg.item() / denom),
        "train_kd_weight": float(sum_kd_weight.item() / denom),
        "train_spatial_kd_weight": float(sum_spatial_weight.item() / denom),
        "train_samples_per_sec": float(sum_samples.item() / elapsed),
    }
    return stats


def run_training(config: dict[str, Any]) -> None:
    ctx = init_distributed()
    try:
        seed_everything(seed=int(config["experiment"]["seed"]), rank=ctx.rank)

        if config["system"].get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True

        output_root = Path(config["experiment"]["output_root"])
        checkpoint_dir = output_root / "checkpoints"
        log_path = output_root / "logs" / "train_metrics.jsonl"

        train_transform = build_train_transform(config["data"])
        train_dataset = TrainKDDataset(
            manifest_csv=config["data"]["train_manifest"],
            transform=train_transform,
            mask_prob=float(config["data"].get("mask_prob", 0.3)),
            mask_fill=str(config["data"].get("mask_fill", "zero")),
            seed=int(config["experiment"]["seed"]),
            decode_retries=int(config["system"].get("recordio_decode_retries", 16)),
        )

        requested_use_dali = bool(config["system"].get("use_dali", False))
        use_dali = requested_use_dali and ctx.device.type == "cuda" and getattr(train_dataset, "mode", "image") == "recordio"
        if requested_use_dali and not use_dali and is_main_process(ctx):
            print("[WARN] DALI requested but unavailable (requires CUDA + RecordIO manifest); falling back to DataLoader/DataLoaderX")

        train_sampler = None
        if ctx.is_distributed and not use_dali:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                drop_last=True,
            )

        configured_workers = int(config["system"].get("num_workers", 8))
        effective_workers = configured_workers
        if getattr(train_dataset, "mode", "image") == "recordio" and not use_dali:
            use_dataloaderx = bool(config["system"].get("use_dataloaderx", True)) and ctx.device.type == "cuda"
            if "recordio_num_workers" in config["system"]:
                effective_workers = int(config["system"]["recordio_num_workers"])
            elif use_dataloaderx:
                effective_workers = configured_workers
            else:
                # Conservative fallback for environments where mxnet workers are unstable.
                effective_workers = 0

            if is_main_process(ctx):
                print(
                    "[INFO] RecordIO mode: num_workers="
                    f"{effective_workers} (configured={configured_workers}, dataloaderx={use_dataloaderx})"
                )

        if use_dali:
            first_sample = train_dataset.samples[0]
            rec_file = str(first_sample["rec_path"])
            idx_file = str(first_sample["idx_path"])

            train_loader = create_dali_recordio_loader(
                batch_size=int(config["train"]["batch_size_per_gpu"]),
                rec_file=rec_file,
                idx_file=idx_file,
                num_threads=int(config["system"].get("dali_num_threads", 2)),
                local_rank=ctx.local_rank,
                initial_fill=int(config["system"].get("dali_initial_fill", 32768)),
                random_shuffle=bool(config["system"].get("dali_random_shuffle", True)),
                prefetch_queue_depth=int(config["system"].get("dali_prefetch_queue_depth", 1)),
                dali_aug=bool(config["system"].get("dali_aug", False)),
                image_size=int(config["data"].get("image_size", 112)),
                # Per-rank seed so each GPU's DALI pipeline applies different augmentations (fix #14).
                seed=int(config["experiment"]["seed"]),
            )

            if is_main_process(ctx):
                print(
                    "[INFO] Using DALI RecordIO pipeline (InsightFace-style): "
                    f"threads={int(config['system'].get('dali_num_threads', 2))}, "
                    f"prefetch={int(config['system'].get('dali_prefetch_queue_depth', 1))}, "
                    f"dali_aug={bool(config['system'].get('dali_aug', False))}"
                )
        else:
            # Per-worker seed: base + DDP rank + worker index (fix #14).
            _base_seed = int(config["experiment"]["seed"]) + ctx.rank
            def _worker_init_fn(worker_id: int, _base: int = _base_seed) -> None:
                import random as _random
                import numpy as _np
                s = (_base + worker_id) & 0xFFFF_FFFF
                _random.seed(s)
                _np.random.seed(s)
                torch.manual_seed(s)

            loader_kwargs = dict(
                dataset=train_dataset,
                batch_size=int(config["train"]["batch_size_per_gpu"]),
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=effective_workers,
                pin_memory=bool(config["system"].get("pin_memory", True)),
                drop_last=True,
                persistent_workers=effective_workers > 0,
                worker_init_fn=_worker_init_fn if effective_workers > 0 else None,
            )

            use_dataloaderx = bool(config["system"].get("use_dataloaderx", True)) and ctx.device.type == "cuda"
            if use_dataloaderx:
                train_loader = DataLoaderX(local_rank=ctx.local_rank, **loader_kwargs)
                if is_main_process(ctx):
                    print("[INFO] Using DataLoaderX (InsightFace-style GPU prefetch)")
            else:
                train_loader = DataLoader(**loader_kwargs)

        num_classes = load_num_classes(config["data"]["train_manifest"])

        student = MobileNetV4Student(
            backbone_name=config["student"]["backbone_name"],
            embedding_dim=int(config["student"].get("embedding_dim", 512)),
            pretrained=bool(config["student"].get("pretrained", True)),
            input_size=int(config["data"].get("image_size", 112)),
            projection_activation=str(config["student"].get("projection_activation", "none")),
            spatial_out_channels=int(config["student"].get("spatial_out_channels", 0)),
        ).to(ctx.device)

        margin_head = build_margin_head(
            cfg=config["margin_head"],
            in_features=int(config["student"].get("embedding_dim", 512)),
            num_classes=num_classes,
        ).to(ctx.device)

        teacher = build_frozen_teacher(config["teacher"]).to(ctx.device)

        # When spatial KD can be dynamically gated off (masked epochs), the spatial
        # projection head won't receive gradients in those steps, so DDP must allow it.
        _student_has_optional_params = int(config["student"].get("spatial_out_channels", 0)) > 0
        student = _ddp_wrap(student, ctx, find_unused_parameters=_student_has_optional_params)
        margin_head = _ddp_wrap(margin_head, ctx)

        params = list(student.parameters()) + list(margin_head.parameters())
        optim_cfg = config["optim"]
        optimizer_type = str(optim_cfg.get("type", "sgd")).lower()
        if optimizer_type == "adamw":
            betas = tuple(float(x) for x in optim_cfg.get("betas", [0.9, 0.999]))
            optimizer = torch.optim.AdamW(
                params,
                lr=float(optim_cfg["lr"]),
                betas=betas,
                weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=float(optim_cfg["lr"]),
                momentum=float(optim_cfg.get("momentum", 0.9)),
                weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
                nesterov=bool(optim_cfg.get("nesterov", False)),
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        _warmup_epochs = int(config["scheduler"].get("warmup_epochs", 1))
        # milestones in YAML are absolute epoch numbers; convert to epochs relative to warmup end
        _abs_milestones = list(config["scheduler"].get("milestones", [18, 28, 36]))
        _rel_milestones = [max(1, m - _warmup_epochs) for m in _abs_milestones]
        _warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(config["scheduler"].get("warmup_start_factor", 0.01)),
            end_factor=1.0,
            total_iters=_warmup_epochs,
        )
        _main_sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=_rel_milestones,
            gamma=float(config["scheduler"].get("gamma", 0.1)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[_warmup_sched, _main_sched],
            milestones=[_warmup_epochs],
        )

        scaler = torch.cuda.amp.GradScaler(enabled=bool(config["system"].get("use_amp", True) and ctx.device.type == "cuda"))

        loss_cfg = config["loss"]
        use_spatial_kd = bool(loss_cfg.get("use_spatial_kd", False)) or (
            float(loss_cfg.get("lambda_spatial_start", 0.0)) > 0.0
            or float(loss_cfg.get("lambda_spatial_end", 0.0)) > 0.0
        )

        if use_spatial_kd and int(config["student"].get("spatial_out_channels", 0)) <= 0:
            raise ValueError(
                "Spatial KD requires student.spatial_out_channels > 0 so the student can use a 1x1 projection"
            )

        objective = DistillationObjective(
            lambda_cls=float(loss_cfg.get("lambda_cls", 1.0)),
            lambda_kd_start=float(loss_cfg.get("lambda_kd_start", 0.2)),
            lambda_kd_end=float(loss_cfg.get("lambda_kd_end", 0.6)),
            kd_ramp_epochs=int(loss_cfg.get("kd_ramp_epochs", 8)),
            kd_type=str(loss_cfg.get("kd_type", "cosine")),
            lambda_rkd_distance=float(loss_cfg.get("lambda_rkd_distance", 0.0)),
            lambda_rkd_angle=float(loss_cfg.get("lambda_rkd_angle", 0.0)),
            lambda_spatial_start=float(loss_cfg.get("lambda_spatial_start", 0.0)),
            lambda_spatial_end=float(loss_cfg.get("lambda_spatial_end", 0.0)),
            spatial_ramp_epochs=int(loss_cfg.get("spatial_ramp_epochs", 0)),
        )

        if use_spatial_kd:
            student_probe = _unwrap(student)
            student_probe_was_training = student_probe.training
            student_probe.eval()

            with torch.no_grad():
                dummy = torch.zeros(
                    2,
                    3,
                    int(config["data"].get("image_size", 112)),
                    int(config["data"].get("image_size", 112)),
                    device=ctx.device,
                )
                _, teacher_spatial = teacher.forward_with_spatial(dummy)
                _, student_spatial = student_probe.forward_with_spatial(dummy)

            if student_probe_was_training:
                student_probe.train()

            if student_spatial.shape != teacher_spatial.shape:
                if student_spatial.shape[1] != teacher_spatial.shape[1]:
                    raise ValueError(
                        "Spatial KD channel mismatch after student projection, got "
                        f"student={tuple(student_spatial.shape)} teacher={tuple(teacher_spatial.shape)}"
                    )

            if is_main_process(ctx):
                print(
                    "[INFO] Spatial KD enabled with feature map shape "
                    f"student={tuple(student_spatial.shape)} teacher={tuple(teacher_spatial.shape)}"
                )

        best_metric = float("-inf")
        best_metric_name = str(config["train"].get("best_metric", "mean_tar_far_1e-4"))
        early_stop_patience = int(config["train"].get("early_stop_patience", 0))
        no_improve_evals = 0

        total_epochs = int(config["train"]["epochs"])
        eval_every = int(config["train"].get("eval_every", 1))
        save_every = int(config["train"].get("save_every", 0))
        start_epoch = 0

        resume_checkpoint = _resolve_resume_path(config["train"], checkpoint_dir)
        if resume_checkpoint is not None:
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

            if is_main_process(ctx):
                print("\n========================================================")
                print(f"[*] RESUMING TRAINING FROM: {resume_checkpoint}")
                print("========================================================\n")

            chkpt = torch.load(resume_checkpoint, map_location="cpu")

            _load_module_state(student, chkpt["student_state"], "student")
            _load_module_state(margin_head, chkpt["margin_head_state"], "margin_head")

            _ckpt_opt_type = chkpt.get("optimizer_type", None)
            _cur_opt_type = type(optimizer).__name__
            _optimizer_loaded = False
            # Only load optimizer state if the checkpoint explicitly records a matching type.
            # Old checkpoints without 'optimizer_type' are treated as incompatible to avoid
            # silent cross-optimizer state corruption (e.g. AdamW moments into SGD groups).
            if _ckpt_opt_type != _cur_opt_type:
                if is_main_process(ctx):
                    label = _ckpt_opt_type if _ckpt_opt_type is not None else "unknown (legacy checkpoint)"
                    print(f"[WARNING] Optimizer type mismatch: checkpoint={label}, "
                          f"current={_cur_opt_type}. Skipping optimizer state "
                          f"(warm model weights retained, training from epoch 0).")
            else:
                try:
                    optimizer.load_state_dict(chkpt["optimizer_state"])
                    _optimizer_loaded = True
                except Exception as _opt_err:
                    if is_main_process(ctx):
                        print(f"[WARNING] Optimizer state failed to load — "
                              f"starting fresh optimizer (warm model weights retained).")
                        print(f"          Reason: {_opt_err}")

            _ckpt_sched_type = chkpt.get("scheduler_type", None)
            _cur_sched_type = type(scheduler).__name__
            if "scheduler_state" in chkpt:
                if _ckpt_sched_type != _cur_sched_type:
                    if is_main_process(ctx):
                        label = _ckpt_sched_type if _ckpt_sched_type is not None else "unknown (legacy checkpoint)"
                        print(f"[WARNING] Scheduler type mismatch: checkpoint={label}, "
                              f"current={_cur_sched_type}. Skipping scheduler state.")
                else:
                    try:
                        scheduler.load_state_dict(chkpt["scheduler_state"])
                    except Exception as _sched_err:
                        if is_main_process(ctx):
                            print(f"[WARNING] Scheduler state failed to load — starting fresh scheduler.")
                            print(f"          Reason: {_sched_err}")

            if "scaler_state" in chkpt:
                try:
                    scaler.load_state_dict(chkpt["scaler_state"])
                except Exception:
                    pass  # non-critical; scaler resets safely

            # Only restore epoch position if optimizer fully loaded (same optimizer type).
            # If incompatible, train all epochs fresh with warm-started model weights.
            if _optimizer_loaded:
                start_epoch = int(chkpt["epoch"]) + 1
            best_metric = float(chkpt.get("best_metric", best_metric))

            if is_main_process(ctx):
                if _optimizer_loaded:
                    print(f"[*] Resume start epoch: {start_epoch}/{total_epochs}")
                else:
                    print(f"[*] Warm-weight start: training from epoch 0 with pretrained model weights.")
                print(f"[*] Restored best metric ({best_metric_name}): {best_metric:.6f}")

        if start_epoch >= total_epochs:
            if is_main_process(ctx):
                print(
                    f"[INFO] Checkpoint epoch already reaches target epochs "
                    f"({start_epoch} >= {total_epochs}); nothing to train."
                )
            synchronize(ctx)
            return

        for epoch in range(start_epoch, total_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            stop_training = False
            aug_schedule = _resolve_augmentation_schedule(
                epoch=epoch,
                train_cfg=config["train"],
                data_cfg=config["data"],
            )
            active_mask_prob = float(aug_schedule["mask_prob"])

            train_stats = _train_one_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                loader=train_loader,
                train_dataset=None if use_dali else train_dataset,
                student=student,
                margin_head=margin_head,
                teacher=teacher,
                objective=objective,
                optimizer=optimizer,
                scaler=scaler,
                ctx=ctx,
                use_amp=bool(config["system"].get("use_amp", True)),
                grad_clip_norm=float(config["optim"].get("grad_clip_norm", 0.0)),
                log_interval=int(config["train"].get("log_interval", 50)),
                show_progress_bar=bool(config["train"].get("show_progress_bar", True)),
                mask_prob=active_mask_prob,
                mask_fill=str(config["data"].get("mask_fill", "zero")),
                gaussian_blur_prob=float(aug_schedule["gaussian_blur_prob"]),
                gaussian_sigma=float(aug_schedule["gaussian_sigma"]),
                gaussian_kernel_range=tuple(aug_schedule["gaussian_kernel_range"]),
                motion_blur_prob=float(aug_schedule["motion_blur_prob"]),
                motion_kernel_range=tuple(aug_schedule["motion_kernel_range"]),
                # Spatial KD is contradictory when masking is active: student sees a black
                # square on the lower face but spatial MSE forces it to match teacher's clean
                # feature map in that region. Disable spatial KD for any epoch with mask_prob>0.
                use_spatial_kd=use_spatial_kd and (active_mask_prob == 0.0),
            )

            scheduler.step()

            epoch_payload: dict[str, Any] = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "mask_prob": float(active_mask_prob),
                "gaussian_blur_prob": float(aug_schedule["gaussian_blur_prob"]),
                "gaussian_sigma": float(aug_schedule["gaussian_sigma"]),
                "gaussian_kernel_range": [
                    int(aug_schedule["gaussian_kernel_range"][0]),
                    int(aug_schedule["gaussian_kernel_range"][1]),
                ],
                "motion_blur_prob": float(aug_schedule["motion_blur_prob"]),
                "motion_kernel_range": [
                    int(aug_schedule["motion_kernel_range"][0]),
                    int(aug_schedule["motion_kernel_range"][1]),
                ],
                **train_stats,
            }

            if (epoch + 1) % eval_every == 0:
                synchronize(ctx)
                if is_main_process(ctx):
                    val_out = validate_on_sets(
                        model=_unwrap(student),
                        data_cfg=config["data"],
                        val_sets=config["data"].get("val_sets", []),
                        batch_size=int(config["train"]["batch_size_per_gpu"]),
                        num_workers=int(
                            config["system"].get(
                                "val_num_workers",
                                config["system"].get("num_workers", 8),
                            )
                        ),
                        device=ctx.device,
                        use_amp=bool(config["system"].get("use_amp", True)),
                        target_fars=[float(x) for x in config["metrics"].get("target_fars", [1e-3, 1e-4])],
                        loader_timeout_s=float(config["system"].get("val_loader_timeout_s", 120.0)),
                    )
                    epoch_payload["validation"] = val_out

                    candidate_metric = val_out.get("aggregate", {}).get(best_metric_name)
                    if candidate_metric is not None and candidate_metric > best_metric:
                        best_metric = float(candidate_metric)
                        no_improve_evals = 0
                        _save_checkpoint(
                            checkpoint_path=checkpoint_dir / "best.pt",
                            epoch=epoch,
                            student=student,
                            margin_head=margin_head,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            config=config,
                            best_metric=best_metric,
                        )
                    elif candidate_metric is not None and early_stop_patience > 0:
                        no_improve_evals += 1
                        if no_improve_evals >= early_stop_patience:
                            stop_training = True
                            print(
                                f"[EARLY-STOP] No improvement in {best_metric_name} for "
                                f"{no_improve_evals} evals (patience={early_stop_patience})."
                            )
                synchronize(ctx)

            if ctx.is_distributed:
                import torch.distributed as dist

                stop_tensor = torch.tensor(
                    1 if (is_main_process(ctx) and stop_training) else 0,
                    device=ctx.device,
                    dtype=torch.int64,
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(int(stop_tensor.item()))

            if is_main_process(ctx):
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    _save_checkpoint(
                        checkpoint_path=checkpoint_dir / f"epoch_{epoch:03d}.pt",
                        epoch=epoch,
                        student=student,
                        margin_head=margin_head,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        best_metric=best_metric,
                    )
                _save_checkpoint(
                    checkpoint_path=checkpoint_dir / "latest.pt",
                    epoch=epoch,
                    student=student,
                    margin_head=margin_head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    best_metric=best_metric,
                )
                _log_jsonl(log_path, epoch_payload)
                print(f"Epoch {epoch:03d} done | lr={epoch_payload['lr']:.6f}")

            synchronize(ctx)

            if stop_training:
                if is_main_process(ctx):
                    print(f"[EARLY-STOP] Training stopped at epoch {epoch:03d}.")
                break

    finally:
        cleanup_distributed(ctx)
