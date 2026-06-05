# Pipeline Metrics And Benchmarks

This guide focuses on quantitative evaluation (not only functional demos).

## 1) Anti-Spoofing Metrics (PAD)

### 1.1 Preferred: real PAD dataset
Use a labeled PAD protocol (for example OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof) and create a CSV manifest:

```csv
path,label,split
path/to/live_001.jpg,1,test
path/to/spoof_001.jpg,0,test
```

Run:

```bash
./venv/bin/python scripts/evaluate_anti_spoof.py \
  --model-path checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth \
  --manifest /abs/path/to/pad_manifest.csv \
  --split test \
  --threshold 0.45 \
  --target-fars 0.01,0.001,0.0001 \
  --out logs/eval_anti_spoof_test.json
```

Reported metrics include:
- ROC AUC
- EER and EER threshold
- APCER, BPCER, ACER at your threshold
- best-ACER operating point
- TAR/TPR at FAR targets
- BPCER at APCER targets (1%, 5%, 10%)

### 1.2 Immediate local proxy benchmark (quick sanity only)
When no real PAD dataset is available, create a synthetic print/replay proxy set from known live faces:

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

Important: proxy numbers are useful for regression checks, not publication-grade PAD claims.

## 2) Open-Set Video Identification Metrics

Run pipeline first, then evaluate against GT labels:

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

Core metrics:
- TPIR@FPIR targets
- FNIR (known false reject)
- MisIDR (known misidentification)
- Unknown FPIR (false identification of unknown)

## 3) Verification Metrics (LFW/CFP-FP/AgeDB-30)

```bash
./venv/bin/python scripts/evaluate_bin_protocol.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --student-checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --out logs/eval_bin_protocol_latest.json
```

Use mean accuracy/AUC and TAR@FAR (1e-3/1e-4/1e-5) as regression gates.

## 4) IJB Template Metrics (IJBB/IJBC)

```bash
./venv/bin/python scripts/evaluate_ijb_template_1to1.py \
  --config configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml \
  --checkpoint runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt \
  --dataset IJBC \
  --template-pooling magface_weighted \
  --out logs/eval_ijbc_template.json
```

Track TAR@1e-4 and TAR@1e-5 as key low-FAR indicators.

## 5) Runtime Throughput Metrics

`run_face_pipeline.py` summaries already report:
- frames_processed
- runtime_seconds
- fps_mean
- accepted_observations
- recognized_observations
- match_reject_reasons

Use these for A/B tests of detector/tracker/re-id/liveness policy.
