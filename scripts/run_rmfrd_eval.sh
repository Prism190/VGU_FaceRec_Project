#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJECT_ROOT/venv/bin/python"
RMFRD=/tmp/mfr2_data/self-built-masked-face-recognition-dataset
OUT="$PROJECT_ROOT/results/rmfrd"
mkdir -p "$OUT"

run() {
  local label="$1"; shift
  local outfile="$OUT/${label}.json"
  if [ -f "$outfile" ] && [ -s "$outfile" ]; then echo "[skip] $outfile exists"; return; fi
  echo "[rmfrd] running $label ..."
  "$PY" "$PROJECT_ROOT/scripts/evaluate_rmfrd.py" "$@" --out "$outfile" 2>&1 | tee "$OUT/${label}_run.log"
  echo "[rmfrd] done: $outfile"
}

run rmfrd_mobilefacenet_w600k \
  --rmfrd-root "$RMFRD" \
  --onnx-model /home/phongtruong/.insightface/models/buffalo_sc/w600k_mbf.onnx \
  --model-name MobileFaceNet_W600K \
  --device cpu

run rmfrd_phase1_best \
  --rmfrd-root "$RMFRD" \
  --config "$PROJECT_ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml" \
  --checkpoint "$PROJECT_ROOT/checkpoints/release/mobilenetv4_student_phase1_best.pt" \
  --model-name Student_Phase1_Best \
  --device cuda

run rmfrd_phase3_swa \
  --rmfrd-root "$RMFRD" \
  --config "$PROJECT_ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml" \
  --checkpoint "$PROJECT_ROOT/checkpoints/release/mobilenetv4_student_phase3_swa.pt" \
  --model-name Student_Phase3_SWA \
  --device cuda

echo "=== RMFRD eval complete ==="
for f in "$OUT"/*.json; do
  echo "---"
  cat "$f"
done
