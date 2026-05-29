#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_validation(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None

    last: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "validation" in row:
                last = row
    return last


def _fmt(x: float | int | None, digits: int = 6) -> str:
    if x is None:
        return "na"
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{digits}f}"


def _print_train_summary(run_dir: Path) -> None:
    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
    row = _latest_validation(metrics_path)
    if row is None:
        print("train: no validation rows yet")
        return

    val = row.get("validation", {})
    agg = val.get("aggregate", {})
    sets = val.get("sets", {})

    print(
        "train: epoch="
        f"{row.get('epoch', 'na')} "
        f"mean_acc={_fmt(agg.get('mean_accuracy'))} "
        f"mean_auc={_fmt(agg.get('mean_roc_auc'))} "
        f"mean_tar@1e-4={_fmt(agg.get('mean_tar_far_1e-4'))}"
    )

    for name in ["lfw", "cfp_fp", "agedb30"]:
        m = sets.get(name)
        if not m:
            continue
        print(
            f"  {name}: "
            f"acc={_fmt(m.get('accuracy'))} "
            f"auc={_fmt(m.get('roc_auc'))} "
            f"tar@1e-4={_fmt(m.get('tar_far_1e-4'))}"
        )


def _print_bin_summary(run_dir: Path, tag: str) -> None:
    path = run_dir / "logs" / f"eval_{tag}_bin_protocol.json"
    data = _read_json(path)
    if data is None:
        print(f"bin/{tag}: pending")
        return

    student = data.get("student", {})
    agg = student.get("aggregate", {})
    lfw = student.get("lfw", {})
    cfp = student.get("cfp_fp", {})
    age = student.get("agedb30", {})

    print(
        f"bin/{tag}: "
        f"mean_acc={_fmt(agg.get('mean_accuracy'))} "
        f"mean_auc={_fmt(agg.get('mean_roc_auc'))} "
        f"lfw_acc={_fmt(lfw.get('accuracy'))} "
        f"cfp_acc={_fmt(cfp.get('accuracy'))} "
        f"agedb_acc={_fmt(age.get('accuracy'))}"
    )


def _print_ijb_summary(run_dir: Path, tag: str, dataset: str) -> None:
    suffix = dataset.lower()
    path = run_dir / "logs" / f"eval_{tag}_{suffix}_template.json"
    data = _read_json(path)
    if data is None:
        print(f"ijb/{dataset}/{tag}: pending")
        return

    print(
        f"ijb/{dataset}/{tag}: "
        f"auc={_fmt(data.get('roc_auc'))} "
        f"tar@1e-4={_fmt(data.get('tar_far_1e-4'))} "
        f"tar@1e-5={_fmt(data.get('tar_far_1e-5'))}"
    )


def _is_post_eval_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "logs" / "eval_latest_bin_protocol.json",
        run_dir / "logs" / "eval_best_bin_protocol.json",
        run_dir / "logs" / "eval_latest_ijbb_template.json",
        run_dir / "logs" / "eval_latest_ijbc_template.json",
        run_dir / "logs" / "eval_best_ijbb_template.json",
        run_dir / "logs" / "eval_best_ijbc_template.json",
    ]
    return all(p.exists() for p in required)


def _print_snapshot(run_dir: Path) -> bool:
    print("=" * 72)
    print(f"snapshot_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"run_dir={run_dir}")

    _print_train_summary(run_dir)
    _print_bin_summary(run_dir, "latest")
    _print_bin_summary(run_dir, "best")
    _print_ijb_summary(run_dir, "latest", "ijbb")
    _print_ijb_summary(run_dir, "latest", "ijbc")
    _print_ijb_summary(run_dir, "best", "ijbb")
    _print_ijb_summary(run_dir, "best", "ijbc")

    complete = _is_post_eval_complete(run_dir)
    print(f"post_eval_complete={complete}")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor validation metrics during and after training")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs/cycle_v3_baseline5")
    parser.add_argument("--watch", action="store_true", help="Continuously print snapshots")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between snapshots when --watch is set")
    parser.add_argument(
        "--until-done",
        action="store_true",
        help="Exit automatically when post-train evaluation artifacts are all present",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not args.watch:
        _print_snapshot(run_dir)
        return

    while True:
        done = _print_snapshot(run_dir)
        if args.until_done and done:
            break
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
