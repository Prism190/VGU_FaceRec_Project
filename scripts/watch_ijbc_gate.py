#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(message, flush=True)


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_max_epoch(metrics_path: Path) -> int | None:
    if not metrics_path.exists():
        return None

    max_epoch: int | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = record.get("epoch")
            if isinstance(epoch, int):
                max_epoch = epoch if max_epoch is None else max(max_epoch, epoch)

    return max_epoch


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch train metrics and gate a run using IJBC TAR@1e-4")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs/ms1m_arcface_allin_v3_gate12")
    parser.add_argument("--config", required=True, help="Config file used by training")
    parser.add_argument("--target-epoch", type=int, default=5, help="Epoch index that must be reached before gating")
    parser.add_argument("--threshold", type=float, default=0.24, help="Minimum TAR@1e-4 to pass")
    parser.add_argument("--dataset", choices=["IJBB", "IJBC"], default="IJBC")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to evaluate; defaults to epoch_{target_epoch:03d}.pt")
    parser.add_argument("--tmux-session-to-kill", default="", help="Training tmux session to stop on gate failure")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for evaluation script")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=48.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve()
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"

    if not args.checkpoint:
        checkpoint_path = checkpoints_dir / f"epoch_{args.target_epoch:03d}.pt"
    else:
        checkpoint_path = Path(args.checkpoint).resolve()

    eval_out_path = logs_dir / f"eval_epoch{args.target_epoch:03d}_{args.dataset.lower()}_gate.json"
    decision_out_path = logs_dir / f"watchdog_decision_epoch{args.target_epoch:03d}_{args.dataset.lower()}.json"
    metrics_path = logs_dir / "train_metrics.jsonl"

    deadline = time.time() + args.max_wait_hours * 3600.0
    last_reported_epoch: int | None = None

    _log(
        "[watchdog] started | "
        f"run_dir={run_dir} target_epoch={args.target_epoch} threshold={args.threshold} "
        f"dataset={args.dataset}"
    )

    while True:
        if time.time() > deadline:
            payload = {
                "status": "timeout",
                "reason": "target epoch not reached before timeout",
                "target_epoch": args.target_epoch,
                "threshold": args.threshold,
                "dataset": args.dataset,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
            }
            _write_json(decision_out_path, payload)
            _log("[watchdog] timeout waiting for target epoch")
            return 3

        max_epoch = _read_max_epoch(metrics_path)
        if max_epoch is None:
            _log("[watchdog] waiting for train_metrics.jsonl")
            time.sleep(args.poll_seconds)
            continue

        if max_epoch != last_reported_epoch:
            _log(f"[watchdog] observed max epoch: {max_epoch}")
            last_reported_epoch = max_epoch

        if max_epoch < args.target_epoch:
            time.sleep(args.poll_seconds)
            continue

        if not checkpoint_path.exists():
            _log(f"[watchdog] target epoch reached but checkpoint missing: {checkpoint_path}")
            time.sleep(args.poll_seconds)
            continue

        _log(f"[watchdog] running {args.dataset} eval on {checkpoint_path.name}")
        cmd = [
            args.python_bin,
            "scripts/evaluate_ijb_template_1to1.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--dataset",
            args.dataset,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
        ]

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            payload = {
                "status": "eval_failed",
                "returncode": int(result.returncode),
                "stderr": result.stderr.strip(),
                "target_epoch": args.target_epoch,
                "threshold": args.threshold,
                "dataset": args.dataset,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
            }
            _write_json(decision_out_path, payload)
            _log("[watchdog] evaluation failed")
            if result.stderr.strip():
                _log(result.stderr.strip())
            return 1

        metrics = _extract_json_blob(result.stdout)
        if metrics is None:
            payload = {
                "status": "eval_parse_failed",
                "stdout_head": result.stdout[:4000],
                "target_epoch": args.target_epoch,
                "threshold": args.threshold,
                "dataset": args.dataset,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
            }
            _write_json(decision_out_path, payload)
            _log("[watchdog] could not parse evaluation json output")
            return 1

        _write_json(eval_out_path, metrics)
        tar = metrics.get("tar_far_1e-4")
        passed = isinstance(tar, (float, int)) and float(tar) >= float(args.threshold)

        decision = {
            "status": "pass" if passed else "fail",
            "tar_far_1e-4": float(tar) if isinstance(tar, (float, int)) else None,
            "threshold": float(args.threshold),
            "target_epoch": int(args.target_epoch),
            "dataset": args.dataset,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "eval_output": str(eval_out_path),
        }
        _write_json(decision_out_path, decision)

        _log(
            f"[watchdog] gate result: status={decision['status']} "
            f"tar_far_1e-4={decision['tar_far_1e-4']} threshold={args.threshold}"
        )

        if not passed and args.tmux_session_to_kill:
            _log(f"[watchdog] stopping tmux session: {args.tmux_session_to_kill}")
            subprocess.run(["tmux", "kill-session", "-t", args.tmux_session_to_kill], check=False)

        return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
