# Face Labeling + IJB Clean vs Raw Command Cheatsheet

## 0) Enter Project
```bash
cd /path/to/fas-kd-mobilenetv4
```

---

## 0.1) Persistent Face Database Layout (New)
Default database root:

```bash
data/face_db
```

Layout:
- known identities: `data/face_db/known/identities/id_XXXXXX__name/`
- known identity photos: `.../photos/*.jpg`
- known identity embeddings: `.../embeddings.npz`
- stranger sessions: `data/face_db/strangers/sessions/<session_name>/`
- stranger grouped samples: `.../groups/group_XXXX/samples/*.jpg`

Notes:
- The pipeline now rebuilds known embeddings from identity photo folders by default.
- Retrieval can use pooled prototype per identity (`--known-db-retrieval-mode pooled`, default)
  or all embeddings (`--known-db-retrieval-mode all`).

---

## 0.2) Reset Face Database (Known + Stranger)
Use this now when identities are confused:

```bash
./venv/bin/python scripts/reset_face_db.py --face-db-root data/face_db --yes
```

---

## 0.3) Import Known Identities From Photos/URLs (YOLO Face Crop)
This script detects faces with YOLO11, picks the best frontal detection,
aligns/crops it, and writes into known DB folder format:

```bash
./venv/bin/python scripts/import_known_faces.py \
  --entry "Adam Driver=https://upload.wikimedia.org/wikipedia/commons/2/2b/Adam_Driver_USMC.webp" \
  --entry "Scarlett Johansson=https://upload.wikimedia.org/wikipedia/commons/b/bd/Scarlett_Johansson_2012_%28facecrop%29.jpg" \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --face-db-root data/face_db \
  --download-dir data/raw/pipeline_demo/celebs \
  --debug-dir logs/celebs_import_debug
```

Notes:
- Imported identities are added under `known/identities/id_XXXXXX__name/photos/`.
- Embeddings are rebuilt on next pipeline run when `--known-db-refresh-from-photos` is enabled.

---

## 0.4) YouTube AV1 Input Compatibility (Transcode to H.264)
Some YouTube downloads are AV1 and may decode poorly in OpenCV on this host.
Transcode once before pipeline runs:

```bash
ffmpeg -y \
  -i data/raw/pipeline_demo/videos/FDFdroN7d0w.mp4 \
  -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
  -c:a aac -movflags +faststart \
  data/raw/pipeline_demo/videos/FDFdroN7d0w_h264.mp4
```

---

## 1) One-Command Integrated Labeling Chain (No 3-Step Manual Process)
This single command now does the integrated chain:
- auto run pass1 stranger grouping + dedup gallery,
- auto launch browser UI,
- on UI Save: auto write group labels JSON and auto generate identity names JSON.

Defaults in this chain were tuned so more strangers are retained for labeling
(lower min-track/min-magnitude gates than before).

```bash
./venv/bin/python scripts/run_labeling_ui_chain.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint latest \
  --source data/raw/pipeline_demo/3209828-uhd_2560_1440_25fps.mp4 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --base-name pipeline_uhd2560_label_chain \
  --max-frames 0
```

What happens next:
1. Browser opens automatically.
2. Enter names per stranger group.
3. Click Save Labels in the UI.
4. Identity names JSON is generated automatically.

Artifacts created automatically (logs):
- pipeline_uhd2560_label_chain_group_labels.json
- pipeline_uhd2560_label_chain_identity_names.json
- pipeline_uhd2560_label_chain_gallery.npz
- pipeline_uhd2560_label_chain_gallery.manifest.json
- pipeline_uhd2560_label_chain_unknown_groups.json
- pipeline_uhd2560_label_chain_run_pass2.sh (ready-to-run pass2 command)

---

## 2) Single-Program Auto-Register + Loop-Until-Stop
Use this when you want only one running program (no pass1/pass2 split),
video loops forever, and you stop manually with Ctrl+C.

This mode auto-registers newly discovered strangers using the first/best few
embeddings per new stranger track.

> **⚠️ Production warning — `--det-conf 0.08`:** This threshold is set very low for high-recall
> labeling work. At 8% confidence, YOLO fires on reflections, partial occlusions, background
> faces in photos, and motion blur artefacts. Every false positive at this threshold passes
> through the full pipeline (alignment, embedding, liveness check, FAISS search) before the
> quality gate rejects it. On a live 30fps camera this can multiply CPU/GPU load by 3–5×
> compared to `--det-conf 0.25`. Use `0.08` for offline labeling runs only.
> For live/deployment use, set `--det-conf 0.25` (or higher) and rely on
> `--det-rescue-conf 0.08` to recover missed faces only when the primary pass finds fewer than
> `--det-rescue-min-primary` detections.

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint latest \
  --source data/raw/pipeline_demo/3209828-uhd_2560_1440_25fps.mp4 \
  --face-db-root data/face_db \
  --known-db-use \
  --known-db-refresh-from-photos \
  --known-db-retrieval-mode pooled \
  --stranger-db-use \
  --stranger-db-session-name pipeline_autoreg_loop \
  --loop-source \
  --max-frames 0 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --det-conf 0.08 \
  --det-iou 0.45 \
  --det-imgsz 1280 \
  --det-rescue-conf 0.05 \
  --det-rescue-imgsz 1920 \
  --tracker-backend deepsort \
  --track-max-missed-frames 140 \
  --track-n-init 2 \
  --track-max-iou-distance 0.9 \
  --track-max-cosine-distance 0.42 \
  --track-nn-budget 200 \
  --quality-min 10.0 \
  --quality-max 110.0 \
  --match-threshold 0.46 \
  --match-topk 7 \
  --match-min-margin 0.12 \
  --reid-min-track-frames 6 \
  --auto-register-unknowns \
  --auto-register-selection best \
  --auto-register-max-embeddings 3 \
  --auto-register-min-track-frames 6 \
  --auto-register-min-mean-magnitude 11.0 \
  --unknown-group-threshold 0.72 \
  --unknown-min-track-frames 6 \
  --unknown-min-mean-magnitude 11.0 \
  --unknown-max-samples-per-group 12 \
  --unknown-sample-min-gap 10 \
  --out-jsonl logs/pipeline_autoreg_loop.jsonl \
  --out-summary logs/pipeline_autoreg_loop.summary.json \
  --out-unknown-manifest logs/pipeline_autoreg_loop.unknown_groups.json \
  --out-unknown-review-html logs/pipeline_autoreg_loop.unknown_groups.html \
  --print-every 60
```

Notes:
- Stop manually with Ctrl+C.
- Summary now includes `runtime_seconds`, `fps_mean`, and auto-register stats.
- Stranger groups are persisted to `data/face_db/strangers/sessions/<session>/`.

---

## 2.1) Stranger GUI Label + Promote to Known DB (Separate Program)
After any pipeline run that produced stranger sessions, launch the new standalone GUI:

```bash
./venv/bin/python scripts/run_stranger_db_labeler.py \
  --face-db-root data/face_db \
  --session latest \
  --port 8770
```

Workflow:
1. Review grouped stranger photos.
2. Enter identity name for each group.
3. Click `Promote Labeled Groups`.
4. Promoted faces are copied into `known/identities/id_XXXXXX__name/photos/` and
   embeddings are appended.

---

## 2.2) Strict Re-ID Policy: Retry Unknown Tracks Faster
New behavior control:
- `--reid-stranger-retry-interval N`

Semantics:
- identified tracks remain locked (normal behavior);
- unresolved tracks can be retried every `N` accepted observations.

Examples:

Legacy one-attempt behavior when `--reid-once-per-track` is enabled:
```bash
--reid-once-per-track --reid-stranger-retry-interval 0
```

Retry unresolved strangers every 2 accepted observations:
```bash
--reid-once-per-track --reid-stranger-retry-interval 2
```

---

## 2.3) Heuristic Anti-Spoofing Mode
New liveness mode:
- `--liveness-mode hybrid`

Recommended starter settings:
```bash
--liveness-mode hybrid --live-threshold 0.45 --liveness-every 5
```

This combines texture + frequency + color cues with TTA (unless `--no-liveness-tta`).

---

## 2.4) Full YouTube Test Run (Annotated)
```bash
./venv/bin/python scripts/run_face_pipeline.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint latest \
  --source data/raw/pipeline_demo/videos/FDFdroN7d0w_h264.mp4 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --det-conf 0.08 \
  --det-iou 0.45 \
  --det-imgsz 1280 \
  --det-rescue-conf 0.05 \
  --det-rescue-imgsz 1920 \
  --det-rescue-min-primary 2 \
  --tracker-backend deepsort \
  --track-max-missed-frames 140 \
  --track-n-init 2 \
  --track-max-iou-distance 0.9 \
  --track-max-cosine-distance 0.42 \
  --track-nn-budget 200 \
  --quality-min 10.0 \
  --quality-max 110.0 \
  --liveness-mode hybrid \
  --live-threshold 0.45 \
  --liveness-every 5 \
  --match-threshold 0.46 \
  --match-topk 7 \
  --match-min-margin 0.12 \
  --reid-min-track-frames 6 \
  --reid-once-per-track \
  --reid-stranger-retry-interval 2 \
  --unknown-group-threshold 0.72 \
  --unknown-min-track-frames 6 \
  --unknown-min-mean-magnitude 11.0 \
  --max-frames 0 \
  --face-db-root data/face_db \
  --known-db-use \
  --known-db-refresh-from-photos \
  --known-db-retrieval-mode all \
  --stranger-db-use \
  --stranger-db-session-name youtube_FDFdroN7d0w_h264_test_20260602 \
  --out-jsonl logs/pipeline_youtube_FDFdroN7d0w_h264_test_20260602.jsonl \
  --out-summary logs/pipeline_youtube_FDFdroN7d0w_h264_test_20260602.summary.json \
  --out-video logs/pipeline_youtube_FDFdroN7d0w_h264_test_20260602_annotated.mp4 \
  --out-unknown-manifest logs/pipeline_youtube_FDFdroN7d0w_h264_test_20260602.unknown_groups.json \
  --out-unknown-review-html logs/pipeline_youtube_FDFdroN7d0w_h264_test_20260602.unknown_groups.review.html \
  --print-every 500
```

---

## 2.5) Real Anti-Spoofing Model (Silent-Face MiniFASNet)
Download a pretrained anti-spoof model once:

```bash
curl -L --fail \
  -o checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth \
  https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth
```

Use the model in pipeline runs:

```bash
--liveness-mode silent_face \
--liveness-silent-face-model checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth \
--liveness-silent-face-device auto \
--liveness-silent-face-live-class-index 1 \
--liveness-silent-face-input-color bgr \
--live-threshold 0.45 \
--liveness-every 5
```

Notes:
- `live_class_index=1` matches this pretrained model's real-face class convention.
- `--liveness-silent-face-device auto` uses CUDA when available, else CPU.

---

## 2.6) Open-Set Video Metrics (TPIR@FPIR + MisID/FN)
New evaluator:

```bash
./venv/bin/python scripts/evaluate_open_set_video.py \
  --pred-jsonl logs/pipeline_youtube_FDFdroN7d0w_h264_test_20260602.jsonl \
  --gt logs/youtube_open_set_gt.track_labels.json \
  --mode both \
  --decision-source retrieval \
  --probe-filter accepted \
  --fpir-targets 0.001,0.01,0.1 \
  --default-threshold 0.46 \
  --out logs/youtube_open_set_eval.json
```

Supported GT JSON formats:

Track-level map:

```json
{
  "101": 1004,
  "118": 1005,
  "132": "unknown",
  "149": "unknown"
}
```

Track-level list:

```json
{
  "track_labels": [
    {"track_id": 101, "identity_id": 1004},
    {"track_id": 118, "identity_id": 1005},
    {"track_id": 132, "is_known": false}
  ]
}
```

Observation-level list:

```json
{
  "observation_labels": [
    {"frame_idx": 245, "track_id": 101, "identity_id": 1004},
    {"frame_idx": 246, "track_id": 132, "is_known": false}
  ]
}
```

Metrics reported:
- `TPIR@FPIR` at requested FPIR targets.
- Known false reject rate (`FNIR`) and known misidentification rate (`MisIDR`).
- Unknown false positive identification rate (`FPIR`).

---

## 2.7) Anti-Spoofing Metrics (AUC/EER/APCER/BPCER/ACER)
Real PAD protocol (recommended):

```bash
./venv/bin/python scripts/evaluate_anti_spoof.py \
  --model-path checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth \
  --manifest /abs/path/to/pad_manifest.csv \
  --split test \
  --threshold 0.45 \
  --target-fars 0.01,0.001,0.0001 \
  --out logs/eval_anti_spoof_test.json
```

Quick proxy PAD (for regression sanity when no real PAD dataset is available):

```bash
./venv/bin/python scripts/build_pad_proxy_protocol.py \
  --live-source-root data/face_db/known/identities \
  --out-root data/processed/pad_proxy_protocol_v1 \
  --image-size 112 \
  --seed 20260602

./venv/bin/python scripts/evaluate_anti_spoof.py \
  --model-path checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth \
  --manifest data/processed/pad_proxy_protocol_v1/manifest.csv \
  --manifest-root data/processed/pad_proxy_protocol_v1 \
  --threshold 0.45 \
  --out logs/eval_anti_spoof_proxy_v1.json
```

---

## 3) Run Pass2 After Label Save
Use the auto-generated command script:

```bash
bash logs/pipeline_uhd2560_label_chain_run_pass2.sh
```

---

## 4) Generate Teacher + Phase1/2/3 Clean vs Raw Matrix
One command to generate the matrix report and per-model metrics:

```bash
./venv/bin/python scripts/generate_ijb_clean_matrix.py \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --template-pooling magface_weighted \
  --out-dir logs/ijb_clean_matrix_20260601
```

Outputs:
- logs/ijb_clean_matrix_20260601/matrix_clean_vs_raw.json
- logs/ijb_clean_matrix_20260601/matrix_clean_vs_raw.md
- logs/ijb_clean_matrix_20260601/eval_teacher_clean_ijbb_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_teacher_clean_ijbc_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_phase1_clean_ijbb_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_phase1_clean_ijbc_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_phase2_clean_ijbb_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_phase2_clean_ijbc_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_phase3_clean_ijbb_magface_weighted.json
- logs/ijb_clean_matrix_20260601/eval_phase3_clean_ijbc_magface_weighted.json

---

## 5) Raw FPS Benchmark (No Annotation Video)
Run one raw pass and read FPS from summary:

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint latest \
  --source data/raw/pipeline_demo/3209828-uhd_2560_1440_25fps.mp4 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --det-conf 0.08 \
  --det-iou 0.45 \
  --det-imgsz 1280 \
  --det-rescue-conf 0.05 \
  --det-rescue-imgsz 1920 \
  --tracker-backend deepsort \
  --track-max-missed-frames 140 \
  --track-n-init 2 \
  --track-max-iou-distance 0.9 \
  --track-max-cosine-distance 0.42 \
  --track-nn-budget 200 \
  --quality-min 10.0 \
  --quality-max 110.0 \
  --match-threshold 0.46 \
  --match-topk 7 \
  --match-min-margin 0.12 \
  --reid-min-track-frames 6 \
  --max-frames 0 \
  --out-jsonl logs/pipeline_raw_fps.jsonl \
  --out-summary logs/pipeline_raw_fps.summary.json \
  --print-every 60
```

Quick readout:

```bash
cat logs/pipeline_raw_fps.summary.json | jq '{frames_processed, runtime_seconds, fps_mean, stop_reason}'
```

---

## 6) Quick Checks

### Confirm clean dataset image counts
```bash
echo "raw IJBB:   $(find data/raw/ijb/ijb/IJBB/loose_crop -type f | wc -l)"
echo "clean IJBB: $(find data/processed/ijb_clean_yolo11/IJBB/loose_crop -type f | wc -l)"
echo "raw IJBC:   $(find data/raw/ijb/ijb/IJBC/loose_crop -type f | wc -l)"
echo "clean IJBC: $(find data/processed/ijb_clean_yolo11/IJBC/loose_crop -type f | wc -l)"
```

### View matrix quickly
```bash
cat logs/ijb_clean_matrix_20260601/matrix_clean_vs_raw.md
```
