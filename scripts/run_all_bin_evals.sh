#!/usr/bin/env bash
# Run bin protocol (LFW/CFP-FP/AgeDB-30/CPLFW/CALFW) for all key checkpoints.
# Outputs go to results/bin_protocol/ with intuitive names.
set -euo pipefail
ROOT=/home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4
PY="$ROOT/venv/bin/python"
OUT="$ROOT/results/bin_protocol"
mkdir -p "$OUT"

run_bin() {
    local LABEL="$1"  # e.g. "phase1_latest"
    local CFG="$2"
    local CKPT="$3"
    echo ""; echo "--- $LABEL ---"
    PYTHONUNBUFFERED=1 "$PY" -u "$ROOT/scripts/evaluate_bin_protocol.py" \
        --config "$CFG" \
        --student-checkpoint "$CKPT" \
        --num-workers 4 \
        --out "$OUT/${LABEL}.json" \
        2>&1 | grep --line-buffered -v "FutureWarning\|torch.load\|weights_only\|pickle\|state ="
    echo "  -> $OUT/${LABEL}.json"
}

# Phase 1
run_bin "phase1_latest" \
    "$ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"

run_bin "phase1_best" \
    "$ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/best.pt"

# Phase 2 (latest only — best=epoch9 discarded)
run_bin "phase2_latest" \
    "$ROOT/configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"

# Phase 3
run_bin "phase3_latest" \
    "$ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt"

run_bin "phase3_swa" \
    "$ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/swa.pt"

run_bin "phase3_best" \
    "$ROOT/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml" \
    "$ROOT/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/best.pt"

echo ""
echo "All bin evals done. Results in $OUT"
# Print compact summary
"$PY" -c "
import json, glob, os
print()
print(f'{\"checkpoint\":<18} {\"LFW\":>8} {\"CFP-FP\":>8} {\"AgeDB\":>8} {\"CPLFW\":>8} {\"CALFW\":>8}')
print('-'*62)
for f in sorted(glob.glob('$OUT/*.json')):
    d = json.load(open(f))
    name = os.path.basename(f).replace('.json','')
    # Try student key first
    s = d.get('student') or d.get('student_best') or {}
    if not s: continue
    vals = [s.get(k,{}).get('accuracy',0) for k in ['lfw','cfp_fp','agedb30','cplfw','calfw']]
    print(f'{name:<18} ' + ' '.join(f'{v:>8.4f}' for v in vals))
    # Also teacher if present
    t = d.get('teacher') or d.get('teacher_iresnet100') or {}
    if t and any(t.values()):
        tvals = [t.get(k,{}).get('accuracy',0) for k in ['lfw','cfp_fp','agedb30','cplfw','calfw']]
        print(f'{\"  teacher\":<18} ' + ' '.join(f'{v:>8.4f}' for v in tvals))
" 2>/dev/null
