#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/train_cycle_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
AUTO_VALIDATE_ON_FINISH="${AUTO_VALIDATE_ON_FINISH:-1}"
AUTO_VALIDATE_RUN_IJB="${AUTO_VALIDATE_RUN_IJB:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-60}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

OUTPUT_ROOT="$(${PYTHON_BIN} - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["experiment"]["output_root"])
PY
)"

CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
	scripts/train_ddp.py --config "${CONFIG_PATH}" "$@"

if [[ "${AUTO_VALIDATE_ON_FINISH}" != "1" ]]; then
	echo "[post-eval] Skipped (AUTO_VALIDATE_ON_FINISH=${AUTO_VALIDATE_ON_FINISH})"
	exit 0
fi

if [[ ! -f "${CHECKPOINT_DIR}/latest.pt" ]]; then
	echo "[post-eval] Skipped: missing ${CHECKPOINT_DIR}/latest.pt" >&2
	exit 0
fi

if [[ ! -f "${CHECKPOINT_DIR}/best.pt" ]]; then
	echo "[post-eval] Warning: missing ${CHECKPOINT_DIR}/best.pt; using latest only" >&2
fi

echo "[post-eval] Running bin protocol evaluation (latest)"
if ! "${PYTHON_BIN}" scripts/evaluate_bin_protocol.py \
	--config "${CONFIG_PATH}" \
	--student-checkpoint "${CHECKPOINT_DIR}/latest.pt" \
	--batch-size "${EVAL_BATCH_SIZE}" \
	--num-workers "${EVAL_NUM_WORKERS}" \
	--out "${LOG_DIR}/eval_latest_bin_protocol.json"; then
	echo "[post-eval] Warning: bin protocol evaluation failed for latest checkpoint" >&2
fi

if [[ -f "${CHECKPOINT_DIR}/best.pt" ]]; then
	echo "[post-eval] Running bin protocol evaluation (best)"
	if ! "${PYTHON_BIN}" scripts/evaluate_bin_protocol.py \
		--config "${CONFIG_PATH}" \
		--student-checkpoint "${CHECKPOINT_DIR}/best.pt" \
		--batch-size "${EVAL_BATCH_SIZE}" \
		--num-workers "${EVAL_NUM_WORKERS}" \
		--out "${LOG_DIR}/eval_best_bin_protocol.json"; then
		echo "[post-eval] Warning: bin protocol evaluation failed for best checkpoint" >&2
	fi
fi

if [[ "${AUTO_VALIDATE_RUN_IJB}" == "1" ]]; then
	echo "[post-eval] Running IJB template evaluation (latest)"
	if ! "${PYTHON_BIN}" scripts/evaluate_ijb_template_1to1.py \
		--config "${CONFIG_PATH}" \
		--checkpoint "${CHECKPOINT_DIR}/latest.pt" \
		--dataset IJBB \
		--batch-size "${EVAL_BATCH_SIZE}" \
		--num-workers "${EVAL_NUM_WORKERS}" \
		> "${LOG_DIR}/eval_latest_ijbb_template.json"; then
		echo "[post-eval] Warning: IJBB template evaluation failed for latest checkpoint" >&2
	fi
	if ! "${PYTHON_BIN}" scripts/evaluate_ijb_template_1to1.py \
		--config "${CONFIG_PATH}" \
		--checkpoint "${CHECKPOINT_DIR}/latest.pt" \
		--dataset IJBC \
		--batch-size "${EVAL_BATCH_SIZE}" \
		--num-workers "${EVAL_NUM_WORKERS}" \
		> "${LOG_DIR}/eval_latest_ijbc_template.json"; then
		echo "[post-eval] Warning: IJBC template evaluation failed for latest checkpoint" >&2
	fi

	if [[ -f "${CHECKPOINT_DIR}/best.pt" ]]; then
		echo "[post-eval] Running IJB template evaluation (best)"
		if ! "${PYTHON_BIN}" scripts/evaluate_ijb_template_1to1.py \
			--config "${CONFIG_PATH}" \
			--checkpoint "${CHECKPOINT_DIR}/best.pt" \
			--dataset IJBB \
			--batch-size "${EVAL_BATCH_SIZE}" \
			--num-workers "${EVAL_NUM_WORKERS}" \
			> "${LOG_DIR}/eval_best_ijbb_template.json"; then
			echo "[post-eval] Warning: IJBB template evaluation failed for best checkpoint" >&2
		fi
		if ! "${PYTHON_BIN}" scripts/evaluate_ijb_template_1to1.py \
			--config "${CONFIG_PATH}" \
			--checkpoint "${CHECKPOINT_DIR}/best.pt" \
			--dataset IJBC \
			--batch-size "${EVAL_BATCH_SIZE}" \
			--num-workers "${EVAL_NUM_WORKERS}" \
			> "${LOG_DIR}/eval_best_ijbc_template.json"; then
			echo "[post-eval] Warning: IJBC template evaluation failed for best checkpoint" >&2
		fi
	fi
else
	echo "[post-eval] Skipped IJB template evaluation (AUTO_VALIDATE_RUN_IJB=${AUTO_VALIDATE_RUN_IJB})"
fi

echo "[post-eval] Done. Artifacts saved in ${LOG_DIR}"
