#!/usr/bin/env python3
"""Compile all evaluation results into paper-ready tables.

Reads from:
  results/pooling_ablation/*.json   → pooling ablation table
  results/baseline/*.json           → MobileFaceNet baseline
  results/rmfrd/*.json              → RMFRD masked face eval
  docs/benchmarks/*.json            → existing IJB/bin results

Usage:
  python scripts/compile_paper_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}"


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


# ─── 1. Hardware efficiency table ───────────────────────────────────────────
print("=" * 80)
print("TABLE 1: Hardware efficiency")
print("=" * 80)
print(f"{'Model':<30} {'Params':>8} {'MACs':>10} {'Latency':>10} {'FPS':>6} {'Memory':>10}")
print("-" * 80)
print(f"{'iResNet-100 (teacher)':<30} {'65.2M':>8} {'12.15G':>10} {'18.52ms':>10} {'54':>6} {'249MB':>10}")
print(f"{'MobileNetV4-M (student)':<30} {'9.58M':>8} {'228M':>10} {'11.22ms':>10} {'89':>6} {'38.3MB':>10}")
print(f"{'MobileFaceNet (baseline)':<30} {'~1M':>8} {'~224M':>10} {'—':>10} {'—':>6} {'—':>10}")
print("Notes: latency batch=1 on Tesla P100; throughput at batch=64; memory fp32 weights")
print()

# ─── 2. Pooling ablation table ──────────────────────────────────────────────
print("=" * 80)
print("TABLE 2: Template pooling ablation (TAR@FAR, magface_weighted is default)")
print("=" * 80)
ablation_dir = PROJECT_ROOT / "results" / "pooling_ablation"

header = f"{'Model':<20} {'Pool':<20} {'IJBB AUC':>9} {'IJBB@1e-3':>10} {'IJBB@1e-4':>10} {'IJBC AUC':>9} {'IJBC@1e-3':>10} {'IJBC@1e-4':>10}"
print(header)
print("-" * len(header))

combos = [
    ("phase1_best", "mean"),
    ("phase1_best", "magface_weighted"),
    ("phase1_best", "top5"),
    ("phase1_best", "top10"),
    ("phase3_swa", "mean"),
    ("phase3_swa", "magface_weighted"),
    ("phase3_swa", "top5"),
    ("phase3_swa", "top10"),
]

# Also load existing magface_weighted from benchmarks (already known)
existing = {
    ("phase1_best", "magface_weighted", "IJBB"): _load(PROJECT_ROOT / "docs" / "benchmarks" / "phase1_ijbb.json"),
    ("phase1_best", "magface_weighted", "IJBC"): _load(PROJECT_ROOT / "docs" / "benchmarks" / "phase1_ijbc.json"),
    ("phase3_swa", "magface_weighted", "IJBB"): _load(PROJECT_ROOT / "docs" / "benchmarks" / "ijb_iff_phase3swa_ijbb.json"),
    ("phase3_swa", "magface_weighted", "IJBC"): _load(PROJECT_ROOT / "docs" / "benchmarks" / "ijb_iff_phase3swa_ijbc.json"),
}

for model, pool in combos:
    bb_key = (model, pool, "IJBB")
    bc_key = (model, pool, "IJBC")
    bb = existing.get(bb_key) or _load(ablation_dir / f"{model}_ijbb_{pool}.json")
    bc = existing.get(bc_key) or _load(ablation_dir / f"{model}_ijbc_{pool}.json")

    row = (
        f"{model:<20} {pool:<20} "
        f"{_pct(bb.get('roc_auc') if bb else None):>9} "
        f"{_pct(bb.get('tar_far_1e-3') if bb else None):>10} "
        f"{_pct(bb.get('tar_far_1e-4') if bb else None):>10} "
        f"{_pct(bc.get('roc_auc') if bc else None):>9} "
        f"{_pct(bc.get('tar_far_1e-3') if bc else None):>10} "
        f"{_pct(bc.get('tar_far_1e-4') if bc else None):>10} "
    )
    print(row)
print()

# ─── 3. Baseline comparison table ───────────────────────────────────────────
print("=" * 80)
print("TABLE 3: Lightweight model comparison on IJB-B/C (TAR@FAR=1e-4)")
print("=" * 80)
baseline_dir = PROJECT_ROOT / "results" / "baseline"

models = [
    ("iResNet-100 (teacher)",  "eval_teacher_identity_clean_ijbb.json", "eval_teacher_identity_clean_ijbc.json"),
    ("MobileFaceNet W600K",    "mobilefacenet_w600k_ijbb_magface_weighted.json", "mobilefacenet_w600k_ijbc_magface_weighted.json"),
    ("Student Phase1/best",    "phase1_ijbb.json", "phase1_ijbc.json"),
    ("Student Phase3/SWA",     "ijb_iff_phase3swa_ijbb.json", "ijb_iff_phase3swa_ijbc.json"),
]

bench_dir = PROJECT_ROOT / "docs" / "benchmarks"
print(f"{'Model':<28} {'Params':>8} {'IJBB@1e-4':>10} {'IJBC@1e-4':>10}")
print("-" * 62)
for label, bb_file, bc_file in models:
    # Try baseline dir first, then benchmarks
    bb = _load(baseline_dir / bb_file) or _load(bench_dir / bb_file)
    bc = _load(baseline_dir / bc_file) or _load(bench_dir / bc_file)
    params = {"iResNet-100 (teacher)": "65.2M", "MobileFaceNet W600K": "~1M",
              "Student Phase1/best": "9.58M", "Student Phase3/SWA": "9.58M"}.get(label, "—")
    print(f"{label:<28} {params:>8} {_pct(bb.get('tar_far_1e-4') if bb else None):>10} {_pct(bc.get('tar_far_1e-4') if bc else None):>10}")
print()

# ─── 4. RMFRD masked face evaluation ────────────────────────────────────────
print("=" * 80)
print("TABLE 4a: RMFRD — RetinaFace-aligned (primary results)")
print("=" * 80)
rmfrd_aligned_dir = PROJECT_ROOT / "results" / "rmfrd_aligned"

rmfrd_aligned_models = [
    ("MobileFaceNet W600K", "rmfrd_aligned_mobilefacenet_w600k.json"),
    ("Student Phase1/best", "rmfrd_aligned_phase1_best.json"),
    ("Student Phase3/SWA",  "rmfrd_aligned_phase3_swa.json"),
]

print(f"{'Model':<28} {'AUC':>7} {'TAR@1e-3':>9} {'TAR@1e-4':>9} {'Rank-1':>7}")
print("-" * 65)
for label, fname in rmfrd_aligned_models:
    d = _load(rmfrd_aligned_dir / fname)
    if d:
        print(f"{label:<28} {_pct(d.get('roc_auc')):>7} {_pct(d.get('tar_far_1e-3')):>9} {_pct(d.get('tar_far_1e-4')):>9} {_pct(d.get('rank1_identification')):>7}")
    else:
        print(f"{label:<28} {'[pending]':>35}")
print()

print("=" * 80)
print("TABLE 4b: RMFRD — Unaligned / simple resize (reference, not primary)")
print("=" * 80)
rmfrd_dir = PROJECT_ROOT / "results" / "rmfrd"

rmfrd_models = [
    ("MobileFaceNet W600K", "rmfrd_mobilefacenet_w600k.json"),
    ("Student Phase1/best", "rmfrd_phase1_best.json"),
    ("Student Phase3/SWA",  "rmfrd_phase3_swa.json"),
]

print(f"{'Model':<28} {'AUC':>7} {'TAR@1e-3':>9} {'TAR@1e-4':>9} {'Rank-1':>7}")
print("-" * 65)
for label, fname in rmfrd_models:
    d = _load(rmfrd_dir / fname)
    if d:
        print(f"{label:<28} {_pct(d.get('roc_auc')):>7} {_pct(d.get('tar_far_1e-3')):>9} {_pct(d.get('tar_far_1e-4')):>9} {_pct(d.get('rank1_identification')):>7}")
    else:
        print(f"{label:<28} {'[pending]':>35}")
print()

# ─── 5. Existing benchmark summary ──────────────────────────────────────────
print("=" * 80)
print("TABLE 5: Full benchmark (existing results from docs/benchmarks/)")
print("=" * 80)
summary = _load(bench_dir / "summary.json")
if summary:
    bin_models = ["phase1/latest", "phase3/swa"]
    for m in bin_models:
        d = summary.get(m, {})
        print(f"\n{m}:")
        for ds, ds_d in d.items():
            cacc = _pct(ds_d.get("clean_acc"))
            macc = _pct(ds_d.get("masked_acc"))
            print(f"  {ds:<12} clean={cacc}% masked={macc}% (drop={_pct(ds_d.get('drop_acc'))}%)")
