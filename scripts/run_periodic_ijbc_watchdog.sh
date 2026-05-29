#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=""
CONFIG=""
TRAIN_SESSION=""
EPOCHS="4,8,12,16,20,24,28,32,36,40"
THRESHOLD="0.24"
DATASET="IJBC"
POLL_SECONDS="60"
MAX_WAIT_HOURS="120"
BATCH_SIZE="256"
NUM_WORKERS="4"
DEVICE="cuda"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_periodic_ijbc_watchdog.sh \
    --run-dir <path> \
    --config <path> \
    --train-session <tmux_session_name> \
    [--epochs 4,8,12,...] \
    [--threshold 0.24] \
    [--dataset IJBC] \
    [--poll-seconds 60] \
    [--max-wait-hours 120] \
    [--batch-size 256] \
    [--num-workers 4] \
    [--device cuda] \
    [--python-bin /path/to/python] \
    [--log-file /path/to/log]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --train-session)
      TRAIN_SESSION="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --max-wait-hours)
      MAX_WAIT_HOURS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$CONFIG" || -z "$TRAIN_SESSION" ]]; then
  echo "--run-dir, --config, and --train-session are required" >&2
  usage
  exit 2
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$RUN_DIR/logs/periodic_ijbc_watchdog.log"
fi

mkdir -p "$(dirname "$LOG_FILE")"

IFS=',' read -r -a EPOCH_ARRAY <<< "$EPOCHS"

echo "[periodic-watchdog] started run_dir=$RUN_DIR train_session=$TRAIN_SESSION dataset=$DATASET threshold=$THRESHOLD epochs=$EPOCHS" | tee -a "$LOG_FILE"

for EPOCH_RAW in "${EPOCH_ARRAY[@]}"; do
  EPOCH="$(echo "$EPOCH_RAW" | tr -d '[:space:]')"
  if [[ -z "$EPOCH" ]]; then
    continue
  fi
  if [[ ! "$EPOCH" =~ ^[0-9]+$ ]]; then
    echo "[periodic-watchdog] invalid epoch value: $EPOCH" | tee -a "$LOG_FILE"
    exit 2
  fi
  if [[ "$EPOCH" -lt 1 ]]; then
    echo "[periodic-watchdog] epoch must be >= 1: $EPOCH" | tee -a "$LOG_FILE"
    exit 2
  fi

  CKPT_EPOCH=$((EPOCH - 1))
  CKPT_NAME="$(printf 'epoch_%03d.pt' "$CKPT_EPOCH")"
  CKPT_PATH="$RUN_DIR/checkpoints/$CKPT_NAME"

  echo "[periodic-watchdog] starting gate epoch=$EPOCH checkpoint=$CKPT_NAME threshold=$THRESHOLD" | tee -a "$LOG_FILE"

  if "$PYTHON_BIN" scripts/watch_ijbc_gate.py \
    --run-dir "$RUN_DIR" \
    --config "$CONFIG" \
    --target-epoch "$EPOCH" \
    --checkpoint "$CKPT_PATH" \
    --threshold "$THRESHOLD" \
    --dataset "$DATASET" \
    --tmux-session-to-kill "$TRAIN_SESSION" \
    --poll-seconds "$POLL_SECONDS" \
    --max-wait-hours "$MAX_WAIT_HOURS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" >> "$LOG_FILE" 2>&1; then
    RC=0
  else
    RC=$?
  fi

  echo "[periodic-watchdog] epoch=$EPOCH exit_code=$RC" | tee -a "$LOG_FILE"
  if [[ "$RC" -ne 0 ]]; then
    echo "[periodic-watchdog] stopping loop due to non-zero exit" | tee -a "$LOG_FILE"
    exit "$RC"
  fi
done

echo "[periodic-watchdog] completed all planned gates" | tee -a "$LOG_FILE"
