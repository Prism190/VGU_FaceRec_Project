#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJECT_ROOT/venv/bin/python"
ONNX="/home/phongtruong/.insightface/models/buffalo_sc/w600k_mbf.onnx"
IJB="$PROJECT_ROOT/data/processed/ijb_clean_insightface"
OUT="$PROJECT_ROOT/results/baseline"
mkdir -p "$OUT"

for ds in IJBB IJBC; do
    outfile="$OUT/mobilefacenet_w600k_${ds,,}_magface_weighted.json"
    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        echo "[baseline] already done: $outfile — skip"
    else
        echo "[baseline] MobileFaceNet $ds ..."
        "$PY" "$PROJECT_ROOT/scripts/evaluate_baseline_ijb.py" \
            --model-path "$ONNX" \
            --model-name "MobileFaceNet_W600K" \
            --dataset "$ds" \
            --ijb-root "$IJB/$ds" \
            --batch-size 64 \
            --template-pooling magface_weighted \
            --out "$outfile" 2>&1 | tee "$outfile.log"
    fi
done
echo "=== Baseline eval complete ==="
