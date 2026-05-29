#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${PROJECT_ROOT}/logs/ijbc_clean_eval_chain_${TS}.log"
REPORT_PATH="${PROJECT_ROOT}/logs/ijbc_clean_resume_${TS}.json"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/.cache/matplotlib"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"

{
  echo "[start] $(date -Is)"
  echo "[step] clean IJBC only (resume-safe, no overwrite)"
  "${PROJECT_ROOT}/venv/bin/python" scripts/prepare_ijb_yolo_clean.py \
    --datasets IJBC \
    --output-root data/processed/ijb_clean_yolo11 \
    --report-json "${REPORT_PATH}"

  echo "[step] evaluate SWA on cleaned IJBB/IJBC"
  "${PROJECT_ROOT}/scripts/eval_ckpt_ijb.sh" \
    configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml \
    swa \
    magface_weighted \
    data/processed/ijb_clean_yolo11

  echo "[done] $(date -Is)"
  echo "[artifacts]"
  echo "- ${REPORT_PATH}"
} >>"${LOG_PATH}" 2>&1

echo "LOG_PATH=${LOG_PATH}"
