# MobileNetV4 KD + Margin-Aware Face Recognition Training

This project trains a MobileNetV4 student with:
- Masked-input distillation from a frozen MagFace/ResNet-100 teacher
- Margin-aware identity supervision (MagFace by default, ArcFace optional)
- Native PyTorch DDP with torchrun on 2 GPUs

All runtime artifacts stay under this project root to avoid interfering with other users.

## 1) Setup isolated environment

```bash
cd /home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4
bash scripts/bootstrap_venv.sh
source venv/bin/activate
```

Default bootstrap installs GPU wheels:
- `torch==2.5.1`
- `torchvision==0.20.1`
- index `https://download.pytorch.org/whl/cu118`

Optional DALI install during bootstrap:

```bash
INSTALL_DALI=1 DALI_PKG=nvidia-dali-cuda120 bash scripts/bootstrap_venv.sh
```

Override if needed:

```bash
TORCH_CUDA_TAG=cu121 TORCH_VERSION=2.5.1 TORCHVISION_VERSION=0.20.1 bash scripts/bootstrap_venv.sh
```

## 2) Prepare manifests

## 2) Download datasets and pretrained weights

Run automated download:

```bash
bash scripts/download_assets.sh
```

What it pulls:
- CASIA-WebFace (Google Drive)
- CFP (direct zip)
- IJB (Google Drive)
- LFW (Kaggle, requires token)
- AgeDB-30 bundle (Kaggle, requires token)
- MagFace teacher checkpoints (Google Drive)

Kaggle token requirement:
- Place `kaggle.json` at `~/.kaggle/kaggle.json`
- Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Downloaded layout:
- archives: `data/archives`
- extracted datasets: `data/raw`
- pretrained weights: `checkpoints/pretrained`

## 3) Prepare manifests

### 2.1 CASIA-WebFace train manifest

```bash
python scripts/prepare_casia_manifest.py \
  --dataset-root /path/to/CASIA-WebFace \
  --output-manifest data/manifests/casia_train.csv \
  --output-id-map data/manifests/casia_id_map.csv
```

### 2.2 LFW pairs manifest (from pairs.txt)

```bash
python scripts/prepare_pairs_manifest.py \
  --format lfw \
  --protocol /path/to/pairs.txt \
  --images-root /path/to/lfw-aligned \
  --output-csv data/manifests/lfw_pairs.csv
```

### 2.3 CFP / AgeDB / IJB protocol manifests

Use triplet format protocol lines:

```
relative/or/abs/path_a.jpg relative/or/abs/path_b.jpg 1
relative/or/abs/path_c.jpg relative/or/abs/path_d.jpg 0
```

Then convert:

```bash
python scripts/prepare_pairs_manifest.py \
  --format triplet \
  --protocol /path/to/protocol.txt \
  --images-root /path/to/images-root \
  --output-csv data/manifests/cfp_fp_pairs.csv
```

Repeat for AgeDB and IJB 1:1 protocol CSV.

## 4) Configure training

Edit config:
- `configs/train_base.yaml`

Important fields:
- `data.train_manifest`
- `data.val_sets[*].pairs_csv`
- `data.ijb.protocol_csv`
- `data.ijb.ijbb_root`
- `data.ijb.ijbc_root`
- `teacher.checkpoint`
- `student.backbone_name`

## 5) DDP smoke test

```bash
torchrun --standalone --nproc_per_node=2 scripts/ddp_smoke_test.py
```

## 6) Train on 2 GPUs

```bash
bash scripts/launch_train.sh
```

By default, rank 0 shows a live tqdm progress bar in terminal with current loss and LR.
After a successful training run, `scripts/launch_train.sh` now auto-runs post-train validation and writes artifacts to the run's `logs/` directory:
- InsightFace `.bin` protocol on `latest.pt` and `best.pt`
- IJB template 1:1 for IJBB and IJBC on `latest.pt` and `best.pt`

Control flags:

```bash
AUTO_VALIDATE_ON_FINISH=0 bash scripts/launch_train.sh      # disable all post-train eval
AUTO_VALIDATE_RUN_IJB=0 bash scripts/launch_train.sh        # run bin eval only
EVAL_BATCH_SIZE=256 EVAL_NUM_WORKERS=8 bash scripts/launch_train.sh
```

With overrides:

```bash
bash scripts/launch_train.sh \
  --override train.epochs=40 \
  --override train.batch_size_per_gpu=160 \
  --override loss.lambda_kd_end=0.8
```

Recommended 5-epoch clean baseline retry (no synthetic mask, stronger KD, PReLU projection):

```bash
CONFIG_PATH=configs/train_cycle_v3_baseline5.yaml bash scripts/launch_train.sh
```

Recommended production retry (40 epochs, safe masking 0.15, RKD enabled):

```bash
CONFIG_PATH=configs/train_cycle_v4_long40_mask015.yaml bash scripts/launch_train.sh
```

MS1M profile (direct RecordIO manifest path, no image conversion):

```bash
CONFIG_PATH=configs/train_ms1m_cycle_v1.yaml bash scripts/launch_train.sh
```

`train_ms1m_cycle_v1.yaml` uses InsightFace-style RecordIO with DALI by default (`system.use_dali: true`).

Disable the bar if you prefer plain logs:

```bash
bash scripts/launch_train.sh --override train.show_progress_bar=false
```

Monitor validation during training and automatically stop once post-train eval artifacts are ready:

```bash
venv/bin/python scripts/monitor_validation.py \
  --run-dir runs/cycle_v3_baseline5 \
  --watch \
  --interval 20 \
  --until-done
```

See [docs/pipeline_next_stage.md](docs/pipeline_next_stage.md) for the expanded runtime pipeline modules and MS1M migration commands.

## 7) Evaluate IJB 1:1

Legacy image-pair evaluation (placeholder CSV based):

```bash
python scripts/evaluate_ijb_1to1.py \
  --config configs/train_base.yaml \
  --checkpoint checkpoints/best.pt
```

Protocol-correct template-based evaluation (recommended):

```bash
python scripts/evaluate_ijb_template_1to1.py \
  --config configs/train_base.yaml \
  --checkpoint checkpoints/best.pt \
  --dataset IJBB \
  --batch-size 256 \
  --num-workers 4
```

## Core loss used

The composite objective follows:

\[
L_{total} = \lambda_{cls} L_{MagFace} + \lambda_{kd} L_{KD} + \lambda_{d} L_{distance} + \lambda_{a} L_{angle}
\]

Default behavior:
- `L_MagFace` always on
- `L_KD` cosine loss with KD ramp from `lambda_kd_start` to `lambda_kd_end`
- RKD distance/angle off by default (set >0 to enable)

## Outputs

- Training logs: `logs/train_metrics.jsonl`
- Latest checkpoint: `checkpoints/latest.pt`
- Best checkpoint: `checkpoints/best.pt`
- Post-train bin eval: `logs/eval_latest_bin_protocol.json`, `logs/eval_best_bin_protocol.json`
- Post-train IJB template eval: `logs/eval_latest_ijbb_template.json`, `logs/eval_latest_ijbc_template.json`, `logs/eval_best_ijbb_template.json`, `logs/eval_best_ijbc_template.json`

## Phase 2 note

Face anti-spoofing test-time domain generalization is intentionally separated as Phase 2 integration.
