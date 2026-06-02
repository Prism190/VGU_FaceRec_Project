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

The recommended checkpoint is **phase3** (`runs/ms1m_magface_phase3_trueasym_swa_v1`).
It is the best student on all YOLO-cleaned IJB metrics: IJBB TAR@1e-4=0.467, IJBC=0.481.
Phase ordering: phase3 > phase1 > phase2.

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
CONFIG_PATH=configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml bash scripts/launch_train.sh
```

Key settings in phase3 config:
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

## IJB evaluation: known issues

### Why TAR@FAR=1e-4 is lower than published MagFace numbers

Published MagFace TAR@1e-4: IJBB 93.4%, IJBC 95.5%.
Our pipeline teacher on YOLO-cleaned IJB: IJBB 60.2%, IJBC 59.2%.

The gap is **alignment quality**, not a code bug:

- The official MagFace evaluation uses InsightFace RetinaFace for landmark detection and 5-point affine alignment — a larger, more precise model.
- Our pipeline uses YOLO11n (nano) for both the deployment pipeline and the IJB eval data preparation. The nano model's landmarks are less precise, introducing per-image crop noise.
- Noisy crops compress the score distributions. Ranking quality (AUC ≈ 0.97–0.98) stays high because relative ordering is preserved, but TAR collapses at strict FAR=1e-4 since the gap between genuine and impostor pairs narrows.

**Relative student-to-teacher ratios are valid** for comparing checkpoints against each other. Phase3 student achieves ~80% of the teacher ceiling on the same evaluation data.

### Image quality: raw vs YOLO-cleaned

NIST IJB loose crops are raw bounding-box crops. Always run the YOLO11 clean pipeline first:

```bash
./venv/bin/python scripts/prepare_ijb_yolo_clean.py \
  --ijb-root data/raw/ijb/ijb \
  --output-root data/processed/ijb_clean_yolo11 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt
```

Images where YOLO11 finds no face are skipped by default (use `--no-skip-fallback` to include fallback box crops). This improves TAR@1e-4 by ~5–7 points over raw crops.

### Teacher input normalisation

`magface_iresnet100_ms1mv2.pth` was trained with images divided by 255 only (`[0, 1]` range — no mean/std normalisation). Our eval transform outputs `[-1, 1]` via `Normalize(0.5, 0.5)`. The `from_minus_one_to_zero_one` input mode in all training configs correctly maps `[-1,1] → [0,1]` before the teacher. Do **not** change this to `identity` — doing so degrades AUC from 0.974 to 0.775 and TAR@1e-4 from 0.602 to 0.012.

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
