#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml}"
CHECKPOINT_PATH="${2:-${PROJECT_ROOT}/runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt}"
OUT_PATH="${3:-${PROJECT_ROOT}/logs/eval_$(basename "${CHECKPOINT_PATH%.pt}")_bin_$(date +%Y%m%d_%H%M%S).json}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "[error] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[error] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[error] Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PATH}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/evaluate_bin_protocol.py \
  --config "${CONFIG_PATH}" \
  --student-checkpoint "${CHECKPOINT_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --out "${OUT_PATH}"

echo

echo "[summary] student metrics"
"${PYTHON_BIN}" - <<'PY' "${OUT_PATH}"
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
data = json.loads(out_path.read_text(encoding="utf-8"))
student = data["student"]
agg = student["aggregate"]
print(f"file: {out_path}")
print(f"mean_accuracy: {agg['mean_accuracy']:.6f}")
print(f"mean_roc_auc: {agg['mean_roc_auc']:.6f}")
for name in ["lfw", "cfp_fp", "agedb30"]:
    m = student[name]
    print(
        f"{name}: acc={m['accuracy']:.6f} auc={m['roc_auc']:.6f} "
        f"tar@1e-4={m['tar_far_1e-4']:.6f}"
    )
PY
