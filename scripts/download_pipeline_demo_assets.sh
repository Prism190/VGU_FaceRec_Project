#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_OUT="${PROJECT_ROOT}/data/raw/pipeline_demo/short_hamilton_clip.mp4"
MODEL_OUT="${PROJECT_ROOT}/checkpoints/pretrained/yolo11n-face-age.pt"

VIDEO_URL="https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/short_hamilton_clip.mp4"
MODEL_URL="https://huggingface.co/AdamCodd/yolo11n-face-age/resolve/main/best.pt"

mkdir -p "$(dirname "${VIDEO_OUT}")" "$(dirname "${MODEL_OUT}")"

echo "[download] face video -> ${VIDEO_OUT}"
curl -L --fail --retry 3 -o "${VIDEO_OUT}" "${VIDEO_URL}"

echo "[download] yolo11n face model -> ${MODEL_OUT}"
curl -L --fail --retry 3 -o "${MODEL_OUT}" "${MODEL_URL}"

echo ""
echo "[done]"
ls -lh "${VIDEO_OUT}" "${MODEL_OUT}"
