# AI Face Recognition Pipeline — VGU 2026

End-to-end face recognition pipeline for edge deployment. MobileNetV4 student distilled from MagFace iResNet-100 via knowledge distillation, with full video pipeline including detection, tracking, liveness, and FAISS-based retrieval.

**Core components:**
- MobileNetV4-Conv-Medium student (~9M params) distilled from MagFace iResNet-100 teacher
- YOLO11n-face for detection + 5-point affine alignment
- MagFace magnitude-based quality gate (filters blurry / extreme-angle frames)
- BoT-SORT / DeepSORT tracking with cubic-spline tracklet interpolation
- Magnitude-weighted template pooling
- FAISS HNSW index for ANN retrieval
- CLAHE local contrast normalisation for uneven lighting
- Incremental DBSCAN for auto-enrollment of new identities
- MiniFASNetV2 / LitMAS anti-spoofing

See [`docs/training_phases.md`](docs/training_phases.md) for the full distillation architecture.
See [`docs/pipeline_next_stage.md`](docs/pipeline_next_stage.md) for the runtime module map.

---

## Benchmarks

Evaluated with **InsightFace RetinaFace** alignment + horizontal flip-TTA, magface_weighted template pooling.
Raw JSON in [`results/`](results/).

### IJB Template 1:1 Verification (TAR@FAR)

| Model | IJBB AUC | IJBB TAR@1e-4 | IJBC AUC | IJBC TAR@1e-4 |
|---|---|---|---|---|
| Teacher (iResNet-100) | 0.9922 | 93.14% | 0.9960 | 97.64% |
| **phase1/latest** ★ clean | **0.9912** | **87.98%** | **0.9937** | **90.65%** |
| phase1/best | 0.9905 | 86.88% | 0.9929 | 89.50% |
| **phase3/swa** ★ occluded | 0.9919 | 85.27% | 0.9930 | 87.77% |
| phase3/latest | 0.9917 | 84.78% | 0.9932 | 87.23% |
| phase3/best | 0.9916 | 84.77% | 0.9929 | 87.64% |
| phase2/latest | 0.9935 | 84.52% | 0.9949 | 86.85% |

### Bin Protocol — Clean Faces (LFW / CFP-FP / AgeDB-30)

| Model | LFW | CFP-FP | AgeDB-30 | mean |
|---|---|---|---|---|
| Teacher | 99.78% | 96.27% | 98.30% | — |
| phase1/latest | 99.25% | 93.43% | 95.68% | 96.12% |
| **phase1/best** | 99.20% | **94.14%** | **95.77%** | **96.37%** |
| phase3/best | **99.43%** | 92.79% | 95.15% | 95.79% |
| phase3/latest | 99.02% | 92.06% | 93.60% | 94.89% |
| phase3/swa | 99.00% | 91.54% | 94.00% | 94.85% |

### Occlusion Robustness — TAR@1e-3 drop under lower-face mask

Drop when both verification images have their lower face masked (y ≥ 55%). Lower is more robust.

| Model | LFW | CFP-FP | AgeDB | CPLFW | CALFW |
|---|---|---|---|---|---|
| Teacher (iResNet-100) | −0.041 | −0.520 | −0.541 | **−0.724** | −0.236 |
| phase1/latest | −0.037 | −0.369 | −0.462 | −0.222 | −0.248 |
| **phase3/swa** | **−0.019** | **−0.298** | **−0.321** | **−0.148** | **−0.100** |

The teacher collapses on CPLFW (−72.4 pp) and AgeDB (−54.1 pp). Phase3/swa outperforms the teacher on every occlusion metric despite being a ~7× smaller model. Full breakdown in [`results/occlusion/`](results/occlusion/).

> Eval script: `scripts/evaluate_bin_occluded.py`

---

## Quick start — inference

### Which checkpoint?

| Deployment | Checkpoint | File |
|---|---|---|
| Clean faces (office, access control) | phase1/latest | `mobilenetv4_student_phase1.pt` |
| Masked / occluded faces | phase3/swa | `mobilenetv4_student_phase3_swa.pt` |
| Best bin accuracy (cross-pose) | phase1/best | `mobilenetv4_student_phase1_best.pt` |

### 1) Environment

```bash
cd /path/to/fas-kd-mobilenetv4
bash scripts/bootstrap_venv.sh
source venv/bin/activate
```

Optional runtime deps:

```bash
venv/bin/python -m pip install ultralytics faiss-cpu
```

### 2) Download checkpoints

Lean inference checkpoints (~37–39 MB each) are on the [Releases page](https://github.com/Prism190/AI_FaceRec_VGU_2026/releases/tag/v1.0-vgu2026):

| File | Epoch | Description |
|---|---|---|
| `mobilenetv4_student_phase1.pt` | 39 | **Recommended** — best IJB TAR@1e-4 |
| `mobilenetv4_student_phase1_best.pt` | 29 | Best bin mean accuracy (96.37%) |
| `mobilenetv4_student_phase3_swa.pt` | 35–39 avg | **Best occlusion robustness** |
| `mobilenetv4_student_phase3.pt` | 39 | Phase 3 latest |
| `mobilenetv4_student_phase3_best.pt` | 13 | Phase 3 best LFW (99.43%) |
| `mobilenetv4_student_phase2.pt` | 32 | Phase 2 reference only |

`download_assets.sh` fetches them automatically:

```bash
bash scripts/download_assets.sh
```

Also downloads to `checkpoints/pretrained/`:
- `magface_iresnet100_ms1mv2.pth` — teacher (270 MB, MagFace official)
- `yolo11n-face-age.pt` — face detector (pretrained)
- `2.7_80x80_MiniFASNetV2.pth` — anti-spoofing
- `litmas_downstream_moe.pth` — LitMAS DeiT-MoE anti-spoofing (22.8 MB)

### 3) Import known identities

```bash
./venv/bin/python scripts/import_known_faces.py \
  --entry "Name=path/to/photo.jpg" \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --face-db-root data/face_db
```

### 4) Run pipeline on video

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --source /path/to/video.mp4 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt \
  --face-db-root data/face_db \
  --known-db-use --known-db-refresh-from-photos \
  --tracker-backend deepsort \
  --liveness-mode hybrid --live-threshold 0.45 \
  --out-jsonl logs/pipeline_out.jsonl \
  --out-summary logs/pipeline_out.summary.json
```

For **occluded/masked environments** use phase3/swa:

```bash
./venv/bin/python scripts/run_face_pipeline.py \
  --config configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml \
  --checkpoint checkpoints/release/mobilenetv4_student_phase3_swa.pt \
  ...
```

With **LitMAS anti-spoofing**:

```bash
  --liveness-mode litmas \
  --liveness-litmas-model checkpoints/pretrained/litmas_downstream_moe.pth \
  --live-threshold 0.45
```

With **BoT-SORT + ReID**:

```bash
  --tracker-backend botsort \
  --botsort-with-reid \
  --botsort-reid-weights checkpoints/pretrained/osnet_x0_25_msmt17.pt
```

See [`docs/face_labeling_and_ijb_clean_eval_commands.md`](docs/face_labeling_and_ijb_clean_eval_commands.md) for annotated full-pipeline commands.

---

## Training

### Architecture and phase overview

Three phases are trained independently from scratch on MS1M. See [`docs/training_phases.md`](docs/training_phases.md) for the full design rationale.

| Phase | Key additions | Recommended ckpt |
|---|---|---|
| **1** | Base KD, light augmentation | latest (ep39) |
| **2** | + Spatial KD, occlusion curriculum | *(not recommended)* |
| **3** | + True asymmetric distillation, SWA | swa (ep35–39 avg) |

**True asymmetric distillation** (phase 3): the teacher always sees the clean image while the student sees the augmented one. This provides stable soft targets throughout the occlusion curriculum and is the primary reason phase3/swa beats the teacher at occlusion robustness.

**SWA** (phase 3, epochs 35–39): stochastic weight averaging smooths the final checkpoint over 5 epochs at a constant low LR (5×10⁻⁵), eliminating the oscillation artifacts that make `phase3/latest` less robust than `phase3/swa` at strict TAR thresholds.

### Setup

```bash
bash scripts/bootstrap_venv.sh
venv/bin/python -m pip install nvidia-dali-cuda120  # for DALI RecordIO loader
```

### Prepare MS1M manifests

```bash
# From RecordIO
venv/bin/python scripts/prepare_recordio_manifest.py \
  --rec-path /path/to/ms1m/train.rec --idx-path /path/to/ms1m/train.idx \
  --output-manifest data/manifests/ms1m_train.csv

# From JPEG tree (CASIA-WebFace)
python scripts/prepare_casia_manifest.py \
  --dataset-root /path/to/CASIA-WebFace \
  --output-manifest data/manifests/casia_train.csv
```

### Train

```bash
# Phase 1 (recommended starting point)
CONFIG_PATH=configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml bash scripts/launch_train.sh

# Phase 2
CONFIG_PATH=configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml bash scripts/launch_train.sh

# Phase 3 (asymmetric KD + SWA)
CONFIG_PATH=configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml bash scripts/launch_train.sh
```

Each phase runs for 40 epochs with AdamW, warm-up 3 epochs, LR decay at epochs 20/30/36.

### Post-train evaluation

Run all evaluations at once:

```bash
bash scripts/run_final_eval_suite.sh
```

This runs bin protocol, IJB 1:1 (InsightFace + flip-TTA), and occlusion robustness for all key checkpoints. Results go to `results/bin_protocol/`, `results/ijb/`, `results/occlusion/`.

Individual scripts:

```bash
# Bin protocol (LFW / CFP-FP / AgeDB-30)
./venv/bin/python scripts/evaluate_bin_protocol.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --student-checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --out results/bin_protocol/phase1_latest.json

# IJB template 1:1 (InsightFace clean data required first)
./venv/bin/python scripts/evaluate_ijb_template_1to1.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --dataset IJBC \
  --ijb-root data/processed/ijb_clean_insightface/IJBC \
  --template-pooling magface_weighted \
  > results/ijb/phase1_latest_ijbc.json

# Occlusion robustness
./venv/bin/python scripts/evaluate_bin_occluded.py \
  --out-dir results/occlusion/
```

---

## IJB evaluation notes

### Alignment is everything at TAR@1e-4

Raw NIST IJB crops degrade TAR@1e-4 by ~40 percentage points. Always prepare clean data first:

```bash
# InsightFace RetinaFace alignment (recommended, used for all published results)
./venv/bin/python scripts/prepare_ijb_insightface_clean.py \
  --ijb-root data/raw/ijb/ijb \
  --output-root data/processed/ijb_clean_insightface

# YOLO11n alignment (faster, lower quality at strict FAR)
./venv/bin/python scripts/prepare_ijb_yolo_clean.py \
  --ijb-root data/raw/ijb/ijb \
  --output-root data/processed/ijb_clean_yolo11 \
  --detector-model checkpoints/pretrained/yolo11n-face-age.pt
```

All benchmark numbers in this README use InsightFace RetinaFace alignment. The YOLO11n pipeline (used in the deployment video stream) achieves near-equivalent results in practice because the video frames have more controlled framing than raw NIST crops.

### Teacher input normalisation

`magface_iresnet100_ms1mv2.pth` expects `[0, 1]` input (divided by 255, no mean/std normalisation). The pipeline's eval transform outputs `[-1, 1]`. All configs set `teacher.input_mode: from_minus_one_to_zero_one` which remaps `[-1,1] → [0,1]` before the teacher forward pass. Do not change this to `identity` — it degrades teacher AUC from 0.974 to 0.775.

---

## Training loss

$$L = \lambda_{cls} \cdot L_{\text{MagFace}} + \lambda_{kd} \cdot L_{\text{KD}} + \lambda_{spatial} \cdot L_{\text{spatial}}$$

- **L_MagFace**: magnitude-aware angular margin (scale=64, margin=0.42, λ_mag=35)
- **L_KD**: MSE on cosine-normalised embeddings, ramped 5→8 over 8 epochs
- **L_spatial**: intermediate feature MSE, phase 2/3 only, ramped 1→2 over 12 epochs
- **L_distance / L_angle**: Relational KD — disabled (set `lambda_rkd_distance > 0` to enable)

---

## Output layout

```
runs/<run_name>/
  checkpoints/
    latest.pt       # last epoch weights only (stripped of optimizer)
    best.pt         # best mean_tar_far_1e-4 on validation
    swa.pt          # SWA average (phase 3 only, epochs 35–39)
results/
  bin_protocol/     # LFW / CFP-FP / AgeDB-30 per checkpoint
  ijb/              # IJBB + IJBC TAR@FAR per checkpoint
  occlusion/        # mask-robustness drop tables per checkpoint
data/face_db/
  known/identities/<name>/photos/ embeddings.npz
  strangers/sessions/<session>/groups/<id>/samples/
```

---

## Tracker comparison

| | BoT-SORT | DeepSORT (default) |
|---|---|---|
| Association | Two-stage (high-conf → low-conf) | Single-stage cosine + IoU |
| Occlusion handling | Kalman prediction through gaps | Drops sooner |
| ReID | Optional `osnet_x0_25_msmt17.pt` | Requires appearance embedder |
| Dependency | `boxmot>=19.0` | `deep-sort-realtime` |

Default is **DeepSORT** (`--tracker-backend deepsort`). Use BoT-SORT with `--tracker-backend botsort`.

**Identity source:** the pipeline loads identities exclusively from `data/face_db/known/`. Add
photos for a new person by creating `data/face_db/known/identities/id_NNNNNN__name/` with a
`meta.json` and a `photos/` folder. The pipeline re-embeds from photos on every run
(`--known-db-refresh-from-photos`, on by default). Do **not** pass `--gallery-npz` or
`--identity-names-json` alongside `--face-db-root` — the NPZ/JSON paths are from the old
label-chain pipeline and will conflict with face_db IDs and names.

## Anti-spoofing comparison

| | MiniFASNetV2 (recommended) | LitMAS |
|---|---|---|
| Architecture | Depthwise MobileNet | DeiT-tiny + MoE expert (2025) |
| Input | 80×80 | 224×224 |
| Live class index | 1 | 0 |
| AUC on proxy PAD | **0.87** | 0.47 (needs real PAD data) |
| Mode flag | `--liveness-mode silent_face` | `--liveness-mode litmas` |

MiniFASNetV2 is the recommended FAS model. LitMAS requires real PAD (print/replay) data for fine-tuning; the bundled weights gave random-chance AUC on the synthetic proxy dataset.

---

## Docs

| File | Contents |
|---|---|
| [`docs/training_phases.md`](docs/training_phases.md) | Phase 1/2/3 architecture, loss, schedules, results |
| [`docs/pipeline_fixes_2026-06-05.md`](docs/pipeline_fixes_2026-06-05.md) | Runtime pipeline bug fixes (OOM leaks, duplicate tracking, landmark fallback, greedy override) |
| [`docs/pipeline_next_stage.md`](docs/pipeline_next_stage.md) | Runtime pipeline module map |
| [`docs/face_labeling_and_ijb_clean_eval_commands.md`](docs/face_labeling_and_ijb_clean_eval_commands.md) | Full pipeline commands, labeling UI, IJB eval |
| [`docs/pipeline_metrics_and_benchmarks.md`](docs/pipeline_metrics_and_benchmarks.md) | Metric interpretation and eval commands |
| [`docs/ms1m_magface_full_v1_40e_postmortem_2026-05-16.md`](docs/ms1m_magface_full_v1_40e_postmortem_2026-05-16.md) | Phase 1 training run analysis |
