#!/usr/bin/env bash
# Chain IJB template 1:1 eval: teacher + phase1 + phase2 + phase3.
# Evaluates on YOLO-cleaned data by default; pass --insightface to use InsightFace-cleaned.
#
#   bash scripts/run_ijb_eval_chain.sh                 # YOLO-cleaned
#   bash scripts/run_ijb_eval_chain.sh --insightface   # InsightFace-cleaned
set -euo pipefail

ROOT=/home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4
PY="$ROOT/venv/bin/python"
DATE=$(date +%Y%m%d_%H%M)

CLEAN_ROOT="$ROOT/data/processed/ijb_clean_yolo11"
TAG="yolo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --insightface) CLEAN_ROOT="$ROOT/data/processed/ijb_clean_insightface"; TAG="insightface"; shift;;
        *) shift;;
    esac
done

if [ ! -d "$CLEAN_ROOT" ]; then
    echo "[error] Clean data not found: $CLEAN_ROOT"
    exit 1
fi

OUTDIR="$ROOT/logs/ijb_eval_chain_${TAG}_${DATE}"
mkdir -p "$OUTDIR"

echo "Clean root: $CLEAN_ROOT"
echo "Output:     $OUTDIR"
echo ""

run_eval() {
    local LABEL="$1"
    local CONFIG="$2"
    local CKPT="$3"
    echo "--- [$LABEL] config=$CONFIG ckpt=$CKPT ---"
    "$PY" "$ROOT/scripts/evaluate_ijb_1to1.py" \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --clean-root "$CLEAN_ROOT" \
        --device cuda \
        --batch-size 128 \
        --template-pooling magface_weighted \
        --out-dir "$OUTDIR" \
        --label "$LABEL" \
        2>&1 | tail -8
    echo ""
}

# Teacher
run_eval "teacher" \
    "$ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml" \
    "teacher"

# Phase 1
run_eval "phase1" \
    "$ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"

# Phase 2
run_eval "phase2" \
    "$ROOT/configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"

# Phase 3
run_eval "phase3" \
    "$ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt"

# Print summary
echo "========================================"
echo "SUMMARY (${TAG} clean, flip-TTA=on)"
echo "========================================"
"$PY" -c "
import json, glob, os
outdir = '$OUTDIR'
rows = []
for f in sorted(glob.glob(outdir + '/*.json')):
    d = json.load(open(f))
    label = d.get('label', os.path.basename(f))
    for ds in ['IJBB','IJBC']:
        m = d.get(ds, {})
        if m:
            rows.append((label, ds, m.get('roc_auc',0), m.get('tar_far_1e-3',0), m.get('tar_far_1e-4',0)))
print(f'{\"Model\":<12} {\"Dataset\":<7} {\"AUC\":>8} {\"TAR@1e-3\":>10} {\"TAR@1e-4\":>10}')
print('-'*50)
for r in rows:
    print(f'{r[0]:<12} {r[1]:<7} {r[2]:>8.4f} {r[3]:>10.4f} {r[4]:>10.4f}')
" 2>/dev/null

echo ""
echo "Results written to $OUTDIR"
