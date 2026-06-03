#!/usr/bin/env bash
# run_final_eval_suite.sh
# Full evaluation suite: bin protocol, IJB (missing checkpoints), occlusion robustness.
# Runs IJB in background while bin evals run sequentially.
# Gathers everything into results/ with intuitive names.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"
cd "$ROOT"

BIN_OUT="$ROOT/results/bin_protocol"
IJB_OUT="$ROOT/results/ijb"
OCC_OUT="$ROOT/results/occlusion"
mkdir -p "$BIN_OUT" "$IJB_OUT" "$OCC_OUT"

P1_CFG="$ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml"
P2_CFG="$ROOT/configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml"
P3_CFG="$ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"

P1_LATEST="$ROOT/runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"
P1_BEST="$ROOT/runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/best.pt"
P2_LATEST="$ROOT/runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"
P3_LATEST="$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt"
P3_SWA="$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/swa.pt"
P3_BEST="$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/best.pt"

IJB_CLEAN="$ROOT/data/processed/ijb_clean_insightface"

# ============================================================
# IJB: phase1/best + phase3/best — launch in background now
# so GPU runs in parallel with CPU-bound bin evals on another GPU thread
# ============================================================
echo "[suite] Launching IJB evals for phase1/best and phase3/best in background..."

run_ijb_bg() {
    local label="$1" cfg="$2" ckpt="$3"
    local log="$IJB_OUT/${label}_eval.log"
    {
        for ds in IJBB IJBC; do
            dslo="$(echo "$ds" | tr '[:upper:]' '[:lower:]')"
            echo "  [ijb] $label $ds -> $IJB_OUT/${label}_${dslo}.json"
            "$PY" "$ROOT/scripts/evaluate_ijb_template_1to1.py" \
                --config "$cfg" \
                --checkpoint "$ckpt" \
                --dataset "$ds" \
                --ijb-root "$IJB_CLEAN/$ds" \
                --batch-size 256 \
                --num-workers 8 \
                --template-pooling magface_weighted \
                > "$IJB_OUT/${label}_${dslo}.json"
        done
        echo "  [ijb] $label DONE"
    } >> "$log" 2>&1
}

run_ijb_bg "phase1_best" "$P1_CFG" "$P1_BEST" &
IJB_P1_PID=$!
run_ijb_bg "phase3_best" "$P3_CFG" "$P3_BEST" &
IJB_P3_PID=$!

# ============================================================
# BIN PROTOCOL — sequential
# ============================================================
echo ""
echo "=== BIN PROTOCOL ==="

run_bin() {
    local label="$1" cfg="$2" ckpt="$3"
    echo ""; echo "--- bin: $label ---"
    PYTHONUNBUFFERED=1 "$PY" -u "$ROOT/scripts/evaluate_bin_protocol.py" \
        --config "$cfg" \
        --student-checkpoint "$ckpt" \
        --num-workers 4 \
        --out "$BIN_OUT/$label.json" \
        2>&1 | grep --line-buffered -v "FutureWarning\|torch.load\|weights_only\|pickle\|state ="
    echo "  -> $BIN_OUT/$label.json"
}

run_bin "phase1_latest" "$P1_CFG" "$P1_LATEST"
run_bin "phase1_best"   "$P1_CFG" "$P1_BEST"
run_bin "phase3_latest" "$P3_CFG" "$P3_LATEST"
run_bin "phase3_swa"    "$P3_CFG" "$P3_SWA"
run_bin "phase3_best"   "$P3_CFG" "$P3_BEST"

echo ""
echo "=== BIN PROTOCOL DONE ==="

# ============================================================
# OCCLUSION ROBUSTNESS — all checkpoints (phase1: latest+best, phase3: latest+swa+best)
# ============================================================
echo ""
echo "=== OCCLUSION ROBUSTNESS ==="
PYTHONUNBUFFERED=1 "$PY" -u "$ROOT/scripts/evaluate_bin_occluded.py" \
    --out-dir "$OCC_OUT" \
    --num-workers 4

echo ""
echo "=== OCCLUSION ROBUSTNESS DONE ==="

# ============================================================
# Wait for IJB background jobs
# ============================================================
echo ""
echo "=== Waiting for IJB background jobs... ==="
wait $IJB_P1_PID && echo "  phase1_best IJB: DONE" || echo "  phase1_best IJB: FAILED (check $IJB_OUT/phase1_best_eval.log)"
wait $IJB_P3_PID && echo "  phase3_best IJB: DONE" || echo "  phase3_best IJB: FAILED (check $IJB_OUT/phase3_best_eval.log)"

# ============================================================
# GATHER existing IJB results
# ============================================================
echo ""
echo "=== Gathering existing IJB results into $IJB_OUT ==="

copy_ijb() {
    local src="$1" dst="$2"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        echo "  copied: $(basename $src) -> $(basename $dst)"
    else
        echo "  [missing] $src"
    fi
}

copy_ijb "$ROOT/logs/ijb_teacher_iff_eval/eval_teacher_identity_clean_ijbb.json"  "$IJB_OUT/teacher_ijbb.json"
copy_ijb "$ROOT/logs/ijb_teacher_iff_eval/eval_teacher_identity_clean_ijbc.json"  "$IJB_OUT/teacher_ijbc.json"
copy_ijb "$ROOT/docs/benchmarks/phase1_ijbb.json"                    "$IJB_OUT/phase1_latest_ijbb.json"
copy_ijb "$ROOT/docs/benchmarks/phase1_ijbc.json"                    "$IJB_OUT/phase1_latest_ijbc.json"
copy_ijb "$ROOT/docs/benchmarks/phase2_ijbb.json"                    "$IJB_OUT/phase2_latest_ijbb.json"
copy_ijb "$ROOT/docs/benchmarks/phase2_ijbc.json"                    "$IJB_OUT/phase2_latest_ijbc.json"
copy_ijb "$ROOT/docs/benchmarks/phase3_ijbb.json"                    "$IJB_OUT/phase3_latest_ijbb.json"
copy_ijb "$ROOT/docs/benchmarks/phase3_ijbc.json"                    "$IJB_OUT/phase3_latest_ijbc.json"
copy_ijb "$ROOT/logs/ijb_iff_phase3swa_ijbb.json"                    "$IJB_OUT/phase3_swa_ijbb.json"
copy_ijb "$ROOT/logs/ijb_iff_phase3swa_ijbc.json"                    "$IJB_OUT/phase3_swa_ijbc.json"

# ============================================================
# BIN SUMMARY TABLE
# ============================================================
echo ""
echo "=== BIN PROTOCOL SUMMARY ==="
"$PY" - <<'PY'
import json, glob, os
from pathlib import Path

import os; out = Path(os.environ["BIN_OUT"])
files = sorted(out.glob("*.json"))
order = ["phase1_latest", "phase1_best", "phase3_latest", "phase3_swa", "phase3_best"]
print(f"{'checkpoint':<18} {'LFW':>8} {'CFP-FP':>8} {'AgeDB':>8} {'mean':>8}")
print("-" * 46)
for name in order:
    f = out / f"{name}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    s = d.get("student") or d.get("student_best") or {}
    if not s:
        continue
    lfw = s.get("lfw", {}).get("accuracy", 0)
    cfp = s.get("cfp_fp", {}).get("accuracy", 0)
    adb = s.get("agedb30", {}).get("accuracy", 0)
    mean = s.get("aggregate", {}).get("mean_accuracy", 0)
    print(f"{name:<18} {lfw:>8.4f} {cfp:>8.4f} {adb:>8.4f} {mean:>8.4f}")
PY

# ============================================================
# IJB SUMMARY TABLE
# ============================================================
echo ""
echo "=== IJB SUMMARY (TAR@1e-4) ==="
"$PY" - <<'PY'
import json
from pathlib import Path

import os; ijb_dir = Path(os.environ["IJB_OUT"])
print(f"{'checkpoint':<22} {'IJBB TAR@1e-4':>14} {'IJBC TAR@1e-4':>14}")
print("-" * 52)
order = ["teacher", "phase1_latest", "phase1_best", "phase2_latest",
         "phase3_latest", "phase3_swa", "phase3_best"]
for stem in order:
    bb = ijb_dir / f"{stem}_ijbb.json"
    bc = ijb_dir / f"{stem}_ijbc.json"
    if not bb.exists() or not bc.exists():
        print(f"{stem:<22} {'missing':>14} {'missing':>14}")
        continue
    d_bb = json.loads(bb.read_text())
    d_bc = json.loads(bc.read_text())
    tar_bb = d_bb.get("tar_far_1e-4", 0)
    tar_bc = d_bc.get("tar_far_1e-4", 0)
    print(f"{stem:<22} {tar_bb:>14.4f} {tar_bc:>14.4f}")
PY

echo ""
echo "=== ALL DONE. Results in results/ ==="
echo "  Bin:      $BIN_OUT"
echo "  IJB:      $IJB_OUT"
echo "  Occ:      $OCC_OUT"
