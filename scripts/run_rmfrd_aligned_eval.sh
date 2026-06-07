#!/usr/bin/env bash
# Align RMFRD images with RetinaFace and re-evaluate all three models.
# Results saved to results/rmfrd_aligned/ (separate from unaligned results/rmfrd/).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJECT_ROOT/venv/bin/python"
RMFRD_RAW=/tmp/mfr2_data/self-built-masked-face-recognition-dataset
RMFRD_ALIGNED=/tmp/rmfrd_aligned
OUT="$PROJECT_ROOT/results/rmfrd_aligned"
mkdir -p "$OUT"

# ── Step 1: RetinaFace alignment ─────────────────────────────────────────────
if [ -d "$RMFRD_ALIGNED/AFDB_face_dataset" ] && [ -d "$RMFRD_ALIGNED/AFDB_masked_face_dataset" ]; then
    echo "[align] $RMFRD_ALIGNED already exists — skipping alignment"
else
    echo "[align] Running RetinaFace alignment (det_size=320, CPU)..."
    "$PY" "$PROJECT_ROOT/scripts/align_rmfrd.py" \
        --rmfrd-root "$RMFRD_RAW" \
        --out "$RMFRD_ALIGNED" \
        --det-size 320
fi

# ── Step 2: Evaluate all models ──────────────────────────────────────────────
run() {
    local label="$1"; shift
    local outfile="$OUT/${label}.json"
    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        echo "[skip] $outfile exists"
        return
    fi
    echo "[rmfrd_aligned] running $label ..."
    "$PY" "$PROJECT_ROOT/scripts/evaluate_rmfrd.py" "$@" \
        --rmfrd-root "$RMFRD_ALIGNED" \
        --out "$outfile" 2>&1 | tee "$OUT/${label}_run.log"
    echo "[rmfrd_aligned] done: $outfile"
}

run rmfrd_aligned_mobilefacenet_w600k \
    --onnx-model /home/phongtruong/.insightface/models/buffalo_sc/w600k_mbf.onnx \
    --model-name MobileFaceNet_W600K_aligned \
    --device cpu

run rmfrd_aligned_phase1_best \
    --config "$PROJECT_ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml" \
    --checkpoint "$PROJECT_ROOT/checkpoints/release/mobilenetv4_student_phase1_best.pt" \
    --model-name Student_Phase1_Best_aligned \
    --device cuda

run rmfrd_aligned_phase3_swa \
    --config "$PROJECT_ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml" \
    --checkpoint "$PROJECT_ROOT/checkpoints/release/mobilenetv4_student_phase3_swa.pt" \
    --model-name Student_Phase3_SWA_aligned \
    --device cuda

echo "=== Aligned RMFRD eval complete ==="
echo ""
echo "--- Unaligned (reference) ---"
for f in "$PROJECT_ROOT/results/rmfrd"/*.json; do
    echo "$(basename $f):"
    python3 -c "import json,sys; d=json.load(open('$f')); print(f'  AUC={d[\"roc_auc\"]*100:.2f}% Rank-1={d[\"rank1_identification\"]*100:.2f}%')" 2>/dev/null || true
done
echo ""
echo "--- Aligned ---"
for f in "$OUT"/*.json; do
    echo "$(basename $f):"
    python3 -c "import json,sys; d=json.load(open('$f')); print(f'  AUC={d[\"roc_auc\"]*100:.2f}% Rank-1={d[\"rank1_identification\"]*100:.2f}%')" 2>/dev/null || true
done
