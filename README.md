# AI Face Recognition Pipeline — VGU 2026

End-to-end face recognition pipeline for edge deployment.

**Core components:**
- MobileNetV4 student distilled from MagFace iResNet-100 teacher via Relational KD
- YOLO11-face (pretrained, no training needed) for detection + 5-point affine alignment
- CLAHE local contrast normalisation for uneven lighting
- MagFace magnitude-based quality gate (filters blurry / extreme-angle frames)
- BoT-SORT / DeepSORT tracking with cubic-spline tracklet interpolation
- Magnitude-weighted template pooling
- FAISS HNSW index for ANN retrieval
- Incremental DBSCAN for auto-enrollment of new identities

See [`docs/pipeline_next_stage.md`](docs/pipeline_next_stage.md) for the full runtime module map.

---

## Quick start — pipeline demo

The recommended checkpoint is **phase1** (`runs/ms1m_magface_phase1_cplus_aplus_v1`).
It out-performs phases 2 and 3 on both raw and clean IJB metrics (see clean-vs-raw matrix in `docs/`).

### 1) Environment

```bash
cd /home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4
bash scripts/bootstrap_venv.sh
source venv/bin/activate
```

Install optional runtime deps (FAISS, Ultralytics for YOLO11):

```bash
venv/bin/python -m pip install ultralytics faiss-cpu
```

### 2) Download pretrained weights

```bash
bash scripts/download_assets.sh
```

Downloads `checkpoints/pretrained/`:
- `magface_iresnet100_ms1mv2.pth` — teacher model
- `yolo11n-face-age.pt` — face detector (pretrained, no training required)
- `2.7_80x80_MiniFASNetV2.pth` — anti-spoofing model (MiniFASNetV2)

Kaggle token required for LFW / AgeDB-30:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 3) Import known identities

```bash
./venv/bin/python scripts/import_known_faces.py \
  --entry "Name=path/or/url/to/photo.jpg" \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --face-db-root data/face_db
```

### 4) Run pipeline on a video

Default tracker is **BoT-SORT** (`--tracker-backend botsort`, requires `pip install boxmot`).
Default liveness is `always_live`; switch to **LitMAS** with the weights bundled in `download_assets.sh`.

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint latest \
  --source /path/to/video.mp4 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --face-db-root data/face_db \
  --known-db-use --known-db-refresh-from-photos \
  --tracker-backend botsort \
  --track-max-missed-frames 140 \
  --liveness-mode hybrid --live-threshold 0.45 \
  --out-jsonl logs/pipeline_out.jsonl \
  --out-summary logs/pipeline_out.summary.json
```

**With LitMAS anti-spoofing** (weights downloaded by `download_assets.sh`):

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  ... \
  --liveness-mode litmas \
  --liveness-litmas-model checkpoints/pretrained/litmas_downstream_moe.pth \
  --live-threshold 0.45
```

`live_class_index` defaults to 0 (bonafide/live = class 0 in the LitMAS DeiT-MoE model).

**With BoT-SORT + ReID appearance features**:

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  ... \
  --tracker-backend botsort \
  --botsort-with-reid \
  --botsort-reid-weights checkpoints/pretrained/osnet_x0_25_msmt17.pt \
  --botsort-device cuda
```

ReID weights: download `osnet_x0_25_msmt17.pt` from the boxmot model zoo.

See [`docs/face_labeling_and_ijb_clean_eval_commands.md`](docs/face_labeling_and_ijb_clean_eval_commands.md) for annotated full-pipeline commands, labeling UI, and auto-register loop.

---

## Training (KD student from scratch)

### 1) Download datasets and pretrained weights

```bash
bash scripts/download_assets.sh
```

What it pulls: CASIA-WebFace, CFP, IJB-B/C, LFW, AgeDB-30, MagFace teacher checkpoint.

### 2) Prepare MS1M manifests

If MS1M is available as `.rec/.idx`:

```bash
venv/bin/python scripts/prepare_recordio_manifest.py \
  --rec-path /path/to/ms1m/train.rec \
  --idx-path /path/to/ms1m/train.idx \
  --output-manifest data/manifests/ms1m_train.csv \
  --output-id-map data/manifests/ms1m_id_map.csv
```

For CASIA-WebFace (JPEG tree):

```bash
python scripts/prepare_casia_manifest.py \
  --dataset-root /path/to/CASIA-WebFace \
  --output-manifest data/manifests/casia_train.csv \
  --output-id-map data/manifests/casia_id_map.csv
```

### 3) DDP smoke test

```bash
torchrun --standalone --nproc_per_node=2 scripts/ddp_smoke_test.py
```

### 4) Train — recommended phase1 config

```bash
CONFIG_PATH=configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml bash scripts/launch_train.sh
```

Key settings in phase1 config:
- backbone: `mobilenetv4_conv_medium`
- loss: MagFace classification + MSE cosine KD (ramp 5→8 over 8 epochs)
- mask-free warmup: first 20 epochs, masking enabled after
- best checkpoint selection: `mean_tar_far_1e-4` on validation pairs

Control flags:

```bash
AUTO_VALIDATE_ON_FINISH=0 bash scripts/launch_train.sh      # skip post-train eval
AUTO_VALIDATE_RUN_IJB=0 bash scripts/launch_train.sh        # bin eval only
EVAL_BATCH_SIZE=256 EVAL_NUM_WORKERS=8 bash scripts/launch_train.sh
```

With config overrides:

```bash
bash scripts/launch_train.sh \
  --override train.epochs=40 \
  --override train.batch_size_per_gpu=256
```

`configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml` uses DALI RecordIO by default (`system.use_dali: true`). Install DALI first:

```bash
venv/bin/python -m pip install nvidia-dali-cuda120
```

### 5) Post-train evaluation

Bin protocol (LFW / CFP-FP / AgeDB-30):

```bash
./venv/bin/python scripts/evaluate_bin_protocol.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --student-checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --out logs/eval_bin_protocol_latest.json
```

IJB template 1:1 (recommended: use YOLO-cleaned images):

```bash
./venv/bin/python scripts/evaluate_ijb_template_1to1.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --dataset IJBC \
  --template-pooling magface_weighted \
  --out logs/eval_ijbc_template.json
```

Teacher + all phases clean-vs-raw comparison matrix:

```bash
./venv/bin/python scripts/generate_ijb_clean_matrix.py \
  --device cuda --batch-size 128 --num-workers 4 \
  --out-dir logs/ijb_clean_matrix_$(date +%Y%m%d)
```

---

## IJB evaluation: known issues and fixes

### Issue 1: Low teacher TAR@FAR=1e-4 on raw IJB images

**Root cause A — image quality**: NIST IJB loose crops are raw bounding-box crops without face alignment. The model expects properly aligned 112×112 faces. Run the YOLO11-based clean pipeline first:

```bash
./venv/bin/python scripts/prepare_ijb_yolo_clean.py \
  --ijb-root data/raw/ijb/ijb \
  --output-root data/processed/ijb_clean_yolo11 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt
```

This applies YOLO11 face detection + InsightFace 5-point affine alignment to all loose crops. Images where YOLO11 finds no face are skipped by default (`--no-skip-fallback` to revert), which improves template quality because the evaluator excludes missing images rather than pooling garbage crops.

**Root cause B — teacher input normalisation**: `magface_iresnet100_ms1mv2.pth` was trained with InsightFace-standard preprocessing (images in `[-1, 1]`). The training configs use `teacher.input_mode: from_minus_one_to_zero_one` which converts the eval-transform output from `[-1,1]` back to `[0,1]` before the model — this is incorrect and depresses TAR at low FAR. For standalone teacher evaluation, `generate_ijb_clean_matrix.py` now defaults to `--teacher-input-mode identity`, passing `[-1,1]` directly.

To reproduce the (incorrect) training-time teacher behaviour:
```bash
./venv/bin/python scripts/generate_ijb_clean_matrix.py \
  --teacher-input-mode from_minus_one_to_zero_one ...
```

### Issue 2: Student checkpoint selection

Existing student checkpoints (phase1/2/3) were trained with the `from_minus_one_to_zero_one` teacher, so their KD targets reflect that regime. For future training, changing `teacher.input_mode: identity` in the config will give the student a better teacher signal.

---

## Training objective

$$L_{total} = \lambda_{cls} L_{MagFace} + \lambda_{kd} L_{KD} + \lambda_d L_{distance} + \lambda_a L_{angle}$$

- `L_MagFace`: margin-aware angular classification loss
- `L_KD`: cosine KD loss (ramp `lambda_kd_start` → `lambda_kd_end` over `kd_ramp_epochs`)
- `L_distance`, `L_angle`: Relational KD (off by default; set `lambda_rkd_distance > 0` to enable)

---

## Output layout

```
runs/<run_name>/
  checkpoints/
    latest.pt                         # last epoch
    best.pt                           # best mean_tar_far_1e-4
  logs/
    train_metrics.jsonl
    eval_latest_bin_protocol.json
    eval_latest_ijbb_template.json
    eval_latest_ijbc_template.json
data/face_db/
  known/identities/<id_name>/
    photos/*.jpg
    embeddings.npz
  strangers/sessions/<session_name>/
    groups/<group_id>/samples/*.jpg
```

---

## Tracker: BoT-SORT vs DeepSORT

| | BoT-SORT (default) | DeepSORT |
|--|--|--|
| Association | Two-stage (high-conf first, then low-conf) | Single-stage cosine + IoU |
| Occlusion | Kalman prediction keeps identity through gaps | Drops sooner without appearance |
| ReID | Optional (plug in `osnet_x0_25_msmt17.pt`) | Always requires appearance embedder |
| Dependency | `boxmot>=10.0` | `deep-sort-realtime` |

Switch back to DeepSORT: `--tracker-backend deepsort`.

## Anti-spoofing: LitMAS vs MiniFASNetV2

| | LitMAS (default) | MiniFASNetV2 (fallback) |
|--|--|--|
| Architecture | DeiT-tiny + MoE face expert (2025) | Depthwise MobileNet |
| Checkpoint | `litmas_downstream_moe.pth` (via `download_assets.sh`) | `2.7_80x80_MiniFASNetV2.pth` |
| Live class index | 0 (bonafide) | 1 |
| Input size | 224×224 (ImageNet normalisation) | 80×80 |
| Pipeline mode | `--liveness-mode litmas` | `--liveness-mode silent_face` |
| Requires | `pip install transformers>=4.30` | bundled |

## Docs

| File | Contents |
|------|----------|
| [`docs/pipeline_next_stage.md`](docs/pipeline_next_stage.md) | Runtime pipeline module map, MS1M migration |
| [`docs/face_labeling_and_ijb_clean_eval_commands.md`](docs/face_labeling_and_ijb_clean_eval_commands.md) | Pipeline run commands, labeling UI, IJB evaluation |
| [`docs/pipeline_metrics_and_benchmarks.md`](docs/pipeline_metrics_and_benchmarks.md) | Evaluation metric commands and interpretation |
| [`docs/ms1m_magface_full_v1_40e_postmortem_2026-05-16.md`](docs/ms1m_magface_full_v1_40e_postmortem_2026-05-16.md) | Training run analysis and pivot options |
