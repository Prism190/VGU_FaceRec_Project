#!/usr/bin/env bash
# Reproduces the June-2 known-good UHD demo exactly, with the tracker selectable
# via TRACKER env var (deepsort|botsort) and optional liveness via LIVENESS.
#
#   TRACKER=deepsort LIVENESS=none   bash scripts/run_demo_uhd.sh control
#   TRACKER=botsort  LIVENESS=silent_face bash scripts/run_demo_uhd.sh botsort_demo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"
TAG="${1:-demo}"
TRACKER="${TRACKER:-deepsort}"
LIVENESS="${LIVENESS:-none}"

# Phase1 checkpoint — face_db is the authoritative identity source.
CONFIG="$ROOT/configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml"
CKPT="$ROOT/runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"
SOURCE="$ROOT/data/raw/pipeline_demo/3209828-uhd_2560_1440_25fps.mp4"
DETECTOR="$ROOT/checkpoints/pretrained/yolo11n-face-age.pt"

# Tracker-specific args
if [ "$TRACKER" = "botsort" ]; then
  TRACKER_ARGS="--tracker-backend botsort --track-max-missed-frames 140 \
    --botsort-track-high-thresh 0.5 --botsort-track-low-thresh 0.1 \
    --botsort-new-track-thresh 0.6 --botsort-match-thresh 0.8"
else
  TRACKER_ARGS="--tracker-backend deepsort --track-max-missed-frames 140 \
    --track-n-init 2 --track-max-iou-distance 0.9 --track-max-cosine-distance 0.42 \
    --track-nn-budget 200"
fi

# Liveness args
if [ "$LIVENESS" = "silent_face" ]; then
  LIVENESS_ARGS="--liveness-mode silent_face \
    --liveness-silent-face-model $ROOT/checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth \
    --liveness-silent-face-device cuda --live-threshold 0.45 --liveness-every 15"
elif [ "$LIVENESS" = "litmas" ]; then
  LIVENESS_ARGS="--liveness-mode litmas \
    --liveness-litmas-model $ROOT/checkpoints/pretrained/litmas_downstream_moe.pth \
    --liveness-litmas-device cuda --live-threshold 0.40 --liveness-every 15"
else
  LIVENESS_ARGS=""
fi

# shellcheck disable=SC2086
"$PY" "$ROOT/scripts/run_face_pipeline.py" \
  --config "$CONFIG" --checkpoint "$CKPT" --source "$SOURCE" \
  --detector-model "$DETECTOR" \
  --det-conf 0.08 --det-iou 0.45 --det-imgsz 1280 \
  --det-rescue-conf 0.05 --det-rescue-imgsz 1920 --det-rescue-min-primary 2 \
  $TRACKER_ARGS \
  --quality-min 10.0 --quality-max 110.0 \
  --match-threshold 0.46 --match-topk 7 --match-min-margin 0.12 \
  --reid-min-track-frames 6 --reid-once-per-track \
  --unknown-group-threshold 0.72 --unknown-min-track-frames 6 --unknown-min-mean-magnitude 11.0 \
  $LIVENESS_ARGS \
  --max-frames 0 \
  --out-jsonl "$ROOT/logs/demo_${TAG}.jsonl" \
  --out-summary "$ROOT/logs/demo_${TAG}.summary.json" \
  --out-video "$ROOT/logs/demo_${TAG}.mp4" \
  --print-every 60
