#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


def _read_last_epoch(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return -1

    last = -1
    with metrics_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = row.get("epoch")
            if isinstance(epoch, int):
                last = epoch
    return last


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _find_torchrun_pid(config_hint: str) -> Optional[int]:
    out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    config_hint = str(config_hint)

    candidates: list[int] = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, args = parts
        if "torch.distributed.run" not in args:
            continue
        if "scripts/train_ddp.py" not in args:
            continue
        if config_hint not in args and Path(config_hint).name not in args:
            continue
        try:
            candidates.append(int(pid_s))
        except ValueError:
            continue

    if not candidates:
        return None
    return sorted(candidates)[0]


def _terminate_process_tree(parent_pid: int, grace_seconds: float) -> None:
    # Ask children to stop first, then parent.
    subprocess.run(["pkill", "-TERM", "-P", str(parent_pid)], check=False)
    try:
        os.kill(parent_pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.time() + max(1.0, float(grace_seconds))
    while time.time() < deadline:
        if not _pid_alive(parent_pid):
            return
        time.sleep(1.0)

    subprocess.run(["pkill", "-KILL", "-P", str(parent_pid)], check=False)
    try:
        os.kill(parent_pid, signal.SIGKILL)
    except OSError:
        pass


def _start_tmux_resume(
    project_root: Path,
    config_path: Path,
    nproc_per_node: int,
    ddp_timeout_minutes: int,
    session_name: str,
    launch_log: Path,
) -> None:
    has_session = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if has_session.returncode == 0:
        raise RuntimeError(
            f"tmux session '{session_name}' already exists; choose another session name"
        )

    launch_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd {project_root} && "
        f"CONFIG_PATH='{config_path}' "
        f"NPROC_PER_NODE={int(nproc_per_node)} "
        f"DDP_TIMEOUT_MINUTES={int(ddp_timeout_minutes)} "
        f"bash scripts/launch_train.sh "
        f"--override train.auto_resume=true "
        f"--override train.resume_from=auto "
        f">> '{launch_log}' 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for current epoch boundary, stop foreground run, and resume in tmux"
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--ddp-timeout-minutes", type=int, default=90)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--grace-seconds", type=float, default=60.0)
    parser.add_argument("--launch-log", default="")
    parser.add_argument("--baseline-epoch", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_root = Path(cfg["experiment"]["output_root"]).resolve()
    metrics_path = output_root / "logs" / "train_metrics.jsonl"

    baseline_epoch = args.baseline_epoch
    if baseline_epoch is None:
        baseline_epoch = _read_last_epoch(metrics_path)

    pid = _find_torchrun_pid(str(config_path))
    if pid is None:
        raise RuntimeError("Could not find active torchrun process for target config")

    print(f"[watchdog] active_pid={pid}")
    print(f"[watchdog] metrics_path={metrics_path}")
    print(f"[watchdog] baseline_completed_epoch={baseline_epoch}")
    sys.stdout.flush()

    poll_seconds = max(1.0, float(args.poll_seconds))
    while True:
        if not _pid_alive(pid):
            raise RuntimeError("Training process exited before epoch-boundary migration")

        last_epoch = _read_last_epoch(metrics_path)
        print(f"[watchdog] waiting: completed_epoch={last_epoch} target>{baseline_epoch}")
        sys.stdout.flush()

        if last_epoch > baseline_epoch:
            break

        time.sleep(poll_seconds)

    print("[watchdog] epoch boundary reached; migrating to tmux")
    sys.stdout.flush()

    _terminate_process_tree(parent_pid=pid, grace_seconds=float(args.grace_seconds))

    launch_log = Path(args.launch_log).resolve() if args.launch_log else (
        project_root / "logs" / f"{args.session_name}_resume.log"
    )

    _start_tmux_resume(
        project_root=project_root,
        config_path=config_path,
        nproc_per_node=int(args.nproc_per_node),
        ddp_timeout_minutes=int(args.ddp_timeout_minutes),
        session_name=str(args.session_name),
        launch_log=launch_log,
    )

    print(f"[watchdog] started tmux session: {args.session_name}")
    print(f"[watchdog] resume_log={launch_log}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
