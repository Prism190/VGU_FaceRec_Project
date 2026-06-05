#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"


def _parse_epoch_from_name(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    if path.name == "latest.pt":
        return 10**9, path.name
    if path.name == "best.pt":
        return 10**9 - 1, path.name
    if path.name == "swa.pt":
        return 10**9 - 2, path.name
    return -1, path.name


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_from_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not locate JSON payload in command stdout")
    return json.loads(stdout[start : end + 1])


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{proc.returncode}: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _discover_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    checkpoints: list[Path] = sorted(ckpt_dir.glob("epoch_*.pt"))
    for fixed in ["best.pt", "latest.pt", "swa.pt"]:
        path = ckpt_dir / fixed
        if path.exists():
            checkpoints.append(path)

    # Remove duplicates while preserving order.
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in checkpoints:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)

    dedup.sort(key=_parse_epoch_from_name)
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical checkpoint selector: gate by bin mean accuracy, then rank by IJBC TAR@1e-4"
        )
    )
    parser.add_argument("--config", required=True, help="Training config used for evaluation")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs/ms1m_magface_phase3")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="Python executable for eval scripts")
    parser.add_argument("--bin-threshold", type=float, default=0.95, help="Minimum bin mean accuracy gate")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--template-pooling",
        choices=["mean", "magface_weighted"],
        default="magface_weighted",
        help="Template pooling for IJBC ranking",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths. If omitted, auto-discovers run checkpoints.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: <run-dir>/logs/hierarchical_gate_selection.json",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = Path(args.config).resolve()
    py_bin = str(Path(args.python_bin).resolve())

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not Path(py_bin).exists():
        raise FileNotFoundError(f"Python executable not found: {py_bin}")

    if args.checkpoints:
        checkpoints = [Path(x).resolve() for x in args.checkpoints]
    else:
        checkpoints = _discover_checkpoints(run_dir=run_dir)

    if not checkpoints:
        raise RuntimeError("No checkpoints found to evaluate")

    logs_dir = run_dir / "logs" / "hierarchical_gate"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        if not checkpoint.exists():
            continue

        row: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "status": "pending",
        }
        results.append(row)

        try:
            bin_out = logs_dir / f"{checkpoint.stem}_bin.json"
            cmd_bin = [
                py_bin,
                "scripts/evaluate_bin_protocol.py",
                "--config",
                str(cfg_path),
                "--student-checkpoint",
                str(checkpoint),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--out",
                str(bin_out),
            ]
            _run_checked(cmd_bin)

            bin_json = _load_json(bin_out)
            bin_mean_acc = float(bin_json["student"]["aggregate"]["mean_accuracy"])
            row["bin_mean_accuracy"] = bin_mean_acc
            row["bin_out"] = str(bin_out)

            if bin_mean_acc < float(args.bin_threshold):
                row["status"] = "rejected_bin_gate"
                continue

            ijbc_out = logs_dir / f"{checkpoint.stem}_ijbc.json"
            cmd_ijbc = [
                py_bin,
                "scripts/evaluate_ijb_template_1to1.py",
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                "IJBC",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--template-pooling",
                args.template_pooling,
            ]
            proc_ijbc = _run_checked(cmd_ijbc)
            ijbc_json = _extract_json_from_stdout(proc_ijbc.stdout)
            ijbc_out.write_text(json.dumps(ijbc_json, indent=2), encoding="utf-8")

            ijbc_tar_1e4 = float(ijbc_json["tar_far_1e-4"])
            row["ijbc_tar_far_1e-4"] = ijbc_tar_1e4
            row["ijbc_out"] = str(ijbc_out)
            row["status"] = "passed"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)

    passed = [r for r in results if r.get("status") == "passed"]
    passed_sorted = sorted(
        passed,
        key=lambda x: float(x.get("ijbc_tar_far_1e-4", float("-inf"))),
        reverse=True,
    )

    selected = passed_sorted[0] if passed_sorted else None

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (run_dir / "logs" / "hierarchical_gate_selection.json")
    )
    payload = {
        "config": str(cfg_path),
        "run_dir": str(run_dir),
        "bin_threshold": float(args.bin_threshold),
        "template_pooling": str(args.template_pooling),
        "selected": selected,
        "num_candidates": len(results),
        "num_passed": len(passed_sorted),
        "results": results,
        "passed_ranked": passed_sorted,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if selected is None:
        print("No checkpoint passed the bin gate.")
    else:
        print("Selected checkpoint:")
        print(json.dumps(selected, indent=2))
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
