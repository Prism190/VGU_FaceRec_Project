#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"

CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml}"
CHECKPOINT_ARG="${2:-latest}"
TEMPLATE_POOLING="${3:-magface_weighted}"
IJB_ROOT_OVERRIDE="${4:-${IJB_ROOT_OVERRIDE:-}}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
FREEZE_CHECKPOINT="${FREEZE_CHECKPOINT:-1}"

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "[error] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[error] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

OUTPUT_ROOT="$(${PYTHON_BIN} - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["experiment"]["output_root"])
PY
)"

case "${CHECKPOINT_ARG}" in
  latest|best|swa)
    CHECKPOINT_PATH="${OUTPUT_ROOT}/checkpoints/${CHECKPOINT_ARG}.pt"
    ;;
  *)
    CHECKPOINT_PATH="${CHECKPOINT_ARG}"
    if [[ ! "${CHECKPOINT_PATH}" = /* ]]; then
      CHECKPOINT_PATH="${PROJECT_ROOT}/${CHECKPOINT_PATH}"
    fi
    ;;
esac

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[error] Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ "${FREEZE_CHECKPOINT}" == "1" ]]; then
  SNAP_TS="$(date +%Y%m%d_%H%M%S)"
  SNAP_DIR="${OUTPUT_ROOT}/checkpoints/eval_snapshots"
  mkdir -p "${SNAP_DIR}"
  SNAP_PATH="${SNAP_DIR}/$(basename "${CHECKPOINT_PATH%.pt}")_${SNAP_TS}.pt"
  cp "${CHECKPOINT_PATH}" "${SNAP_PATH}"
  CHECKPOINT_PATH="${SNAP_PATH}"
  echo "[run] frozen checkpoint snapshot=${CHECKPOINT_PATH}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
CKPT_STEM="$(basename "${CHECKPOINT_PATH%.pt}")"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

IJBB_OUT="${LOG_DIR}/eval_${CKPT_STEM}_ijbb_template_${TEMPLATE_POOLING}_${TS}.json"
IJBC_OUT="${LOG_DIR}/eval_${CKPT_STEM}_ijbc_template_${TEMPLATE_POOLING}_${TS}.json"

echo "[run] config=${CONFIG_PATH}"
echo "[run] checkpoint=${CHECKPOINT_PATH}"
echo "[run] device=${DEVICE} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} pooling=${TEMPLATE_POOLING}"
if [[ -n "${IJB_ROOT_OVERRIDE}" ]]; then
  echo "[run] ijb_root_override=${IJB_ROOT_OVERRIDE}"
fi

IJBB_EXTRA_ARGS=()
IJBC_EXTRA_ARGS=()
if [[ -n "${IJB_ROOT_OVERRIDE}" ]]; then
  IJBB_EXTRA_ARGS+=(--ijb-root "${IJB_ROOT_OVERRIDE}/IJBB")
  IJBC_EXTRA_ARGS+=(--ijb-root "${IJB_ROOT_OVERRIDE}/IJBC")
fi

echo "[run] IJBB -> ${IJBB_OUT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ijb_template_1to1.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --dataset IJBB \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --template-pooling "${TEMPLATE_POOLING}" \
  "${IJBB_EXTRA_ARGS[@]}" \
  > "${IJBB_OUT}"

echo "[run] IJBC -> ${IJBC_OUT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_ijb_template_1to1.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --dataset IJBC \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --template-pooling "${TEMPLATE_POOLING}" \
  "${IJBC_EXTRA_ARGS[@]}" \
  > "${IJBC_OUT}"

echo ""
echo "[summary]"
"${PYTHON_BIN}" - <<'PY' "${IJBB_OUT}" "${IJBC_OUT}"
import json
import sys
from pathlib import Path

ijbb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ijbc = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

print(f"IJBB file: {sys.argv[1]}")
print(
    f"IJBB auc={ijbb['roc_auc']:.6f} tar@1e-4={ijbb['tar_far_1e-4']:.6f} "
    f"tar@1e-5={ijbb['tar_far_1e-5']:.6f}"
)
print(f"IJBC file: {sys.argv[2]}")
print(
    f"IJBC auc={ijbc['roc_auc']:.6f} tar@1e-4={ijbc['tar_far_1e-4']:.6f} "
    f"tar@1e-5={ijbc['tar_far_1e-5']:.6f}"
)
PY
