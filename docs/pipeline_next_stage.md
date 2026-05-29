# Next-Stage Face Pipeline Plan

This document maps the implemented scaffolding to the target production pipeline.

## 1) KD Training Roadmap

- Baseline recovery (already done): `configs/train_cycle_v3_baseline5.yaml`
- Long run with safe masking + RKD: `configs/train_cycle_v4_long40_mask015.yaml`
- MS1M upgrade profile: `configs/train_ms1m_cycle_v1.yaml`

Run 40-epoch long cycle:

```bash
cd /home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4
CONFIG_PATH=configs/train_cycle_v4_long40_mask015.yaml bash scripts/launch_train.sh
```

Monitor until post-eval is complete:

```bash
venv/bin/python scripts/monitor_validation.py \
  --run-dir runs/cycle_v4_long40_mask015 \
  --watch \
  --interval 20 \
  --until-done
```

## 2) MS1M RecordIO Manifest

If MS1M is available as `.rec/.idx`, build manifest directly without image conversion:

```bash
venv/bin/python scripts/prepare_recordio_manifest.py \
  --rec-path /path/to/ms1m/train.rec \
  --idx-path /path/to/ms1m/train.idx \
  --output-manifest data/manifests/ms1m_train.csv \
  --output-id-map data/manifests/ms1m_id_map.csv
```

Then launch:

```bash
CONFIG_PATH=configs/train_ms1m_cycle_v1.yaml bash scripts/launch_train.sh
```

Recommended staged execution for MS1M:

```bash
# Stage 1: sanity run (5 epochs)
CONFIG_PATH=configs/train_ms1m_cycle_v1_warmup5.yaml bash scripts/launch_train.sh

# Stage 2: full run (40 epochs) after warmup metrics look healthy
CONFIG_PATH=configs/train_ms1m_cycle_v1.yaml bash scripts/launch_train.sh
```

`configs/train_ms1m_cycle_v1.yaml` is configured for InsightFace-style DALI RecordIO training:
- `system.use_dali: true`
- `system.dali_aug: true`
- `system.dali_num_threads: 2`

Install DALI first (inside the project venv):

```bash
venv/bin/python -m pip install nvidia-dali-cuda120
```

## 3) Runtime Pipeline Modules (Scaffolded)

- Detection + landmarks adapter: `src/fas_kd/pipeline/detection.py`
- CLAHE + 5-point affine alignment: `src/fas_kd/pipeline/preprocess.py`
- Liveness gate + test-time domain augmentation: `src/fas_kd/pipeline/liveness.py`
- MagFace magnitude quality filter: `src/fas_kd/pipeline/quality_gate.py`
- Tracking + cubic-spline interpolation fallback: `src/fas_kd/pipeline/tracking.py`
- Magnitude-weighted pooling: `src/fas_kd/pipeline/aggregation.py`
- ANN retrieval (FAISS HNSW optional, numpy fallback): `src/fas_kd/pipeline/retrieval.py`
- Incremental clustering (DBSCAN buffer): `src/fas_kd/pipeline/clustering.py`
- End-to-end orchestrator: `src/fas_kd/pipeline/runtime.py`

Smoke check:

```bash
venv/bin/python scripts/pipeline_smoke.py
```

## 4) Optional Runtime Dependencies

For full detector/retrieval stack:

```bash
venv/bin/python -m pip install ultralytics faiss-cpu
```

`faiss-cpu` is optional because retrieval has a numpy fallback.
