#!/usr/bin/env bash
# Pooling ablation: mean / magface_weighted / top5 / top10
# Run on Phase1/best and Phase3/SWA across IJBB and IJBC
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJECT_ROOT/venv/bin/python"
SCRIPT="$PROJECT_ROOT/scripts/evaluate_ijb_template_1to1.py"
CFG1="$PROJECT_ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml"
CFG3="$PROJECT_ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"
CKPT1="$PROJECT_ROOT/checkpoints/release/mobilenetv4_student_phase1_best.pt"
CKPT3SWA="$PROJECT_ROOT/checkpoints/release/mobilenetv4_student_phase3_swa.pt"
IJB_ROOT="$PROJECT_ROOT/data/processed/ijb_clean_insightface"
OUT="$PROJECT_ROOT/results/pooling_ablation"
mkdir -p "$OUT"

run_eval() {
    local label="$1" cfg="$2" ckpt="$3" dataset="$4" pool="$5"
    local outfile="$OUT/${label}_${dataset,,}_${pool}.json"
    if [ -f "$outfile" ]; then
        echo "[ablation] already done: $outfile — skip"
        return
    fi
    echo "[ablation] $label $dataset $pool ..."
    "$PY" "$SCRIPT" \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --dataset "$dataset" \
        --ijb-root "$IJB_ROOT/$dataset" \
        --batch-size 256 \
        --num-workers 4 \
        --template-pooling "$pool" \
        | tee "$outfile"
    echo "[ablation] done: $outfile"
}

for pool in mean magface_weighted top5 top10; do
    for ds in IJBB IJBC; do
        run_eval "phase1_best" "$CFG1" "$CKPT1" "$ds" "$pool"
        run_eval "phase3_swa"  "$CFG3" "$CKPT3SWA" "$ds" "$pool"
    done
done

echo ""
echo "=== Pooling ablation complete ==="
echo "Results in: $OUT"

# Print summary table
"$PY" - << 'PYEOF'
import json, pathlib, os
out = pathlib.Path(os.environ.get("OUT", "/home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4/results/pooling_ablation"))
rows = []
for f in sorted(out.glob("*.json")):
    d = json.loads(f.read_text())
    rows.append({
        "file": f.stem,
        "AUC": f"{d.get('roc_auc', 0):.4f}",
        "TAR@1e-3": f"{d.get('tar_far_1e-3', 0):.4f}",
        "TAR@1e-4": f"{d.get('tar_far_1e-4', 0):.4f}",
    })
print(f"{'File':<40} {'AUC':>6} {'TAR@1e-3':>9} {'TAR@1e-4':>9}")
print("-" * 70)
for r in rows:
    print(f"{r['file']:<40} {r['AUC']:>6} {r['TAR@1e-3']:>9} {r['TAR@1e-4']:>9}")
PYEOF
