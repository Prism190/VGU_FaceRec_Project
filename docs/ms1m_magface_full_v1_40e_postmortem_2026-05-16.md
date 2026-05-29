# MS1M MagFace KD Full v1 40e: Postmortem and Pivot Plan

Date: 2026-05-16
Run: runs/ms1m_magface_full_v1_40e
Config: configs/train_ms1m_magface_full_v1_40e.yaml

## 1) Process So Far (What We Did)

1. Route selection phase:
   - ArcFace path was attempted first and failed gate criteria.
   - MagFace path passed rescue gate and was promoted to full training.

2. Reliability and detached operations:
   - Training was run in tmux for disconnect-safe operation.
   - Added milestone watchdogs for IJBC gating (epochs 8,12,16,20,24,28,32,36,40).
   - Added auto-resume behavior and periodic checkpoint checks.

3. Debug/fix cycle during run:
   - Resolved optimizer mismatch regression in training loop.
   - Fixed watchdog launch and quoting failures in wrapper script.
   - Diagnosed premature stop: this was early-stop restoration behavior, not watchdog kill.
   - Set early_stop_patience to 0 in full config to avoid accidental early stop on resume.

4. End state of this route:
   - Training reached epoch index 39 (40th epoch complete).
   - Post-evaluation artifacts were generated.
   - Main quality signal plateaued after a temporary peak.

## 2) Quick Latest Performance Check (Today)

### A) Fresh rerun check on latest checkpoint (IJBC)

Command family executed:
- scripts/evaluate_ijb_template_1to1.py with checkpoint runs/ms1m_magface_full_v1_40e/checkpoints/latest.pt

Rerun output file:
- runs/ms1m_magface_full_v1_40e/logs/eval_latest_ijbc_template_rerun_20260516.json

Observed IJBC metrics:
- roc_auc: 0.954989496736339
- tar_far_1e-3: 0.4659201308994222
- tar_far_1e-4: 0.28353019379250394
- tar_far_1e-5: 0.1599938640895843

Result: matches existing saved latest IJBC evaluation exactly.

### B) Latest saved IJBB snapshot

Source file:
- runs/ms1m_magface_full_v1_40e/logs/eval_latest_ijbb_template.json

Observed IJBB metrics:
- roc_auc: 0.9489032260528238
- tar_far_1e-3: 0.44410905550146057
- tar_far_1e-4: 0.259396299902629
- tar_far_1e-5: 0.13846153846153847

### C) Milestone IJBC gate trajectory

| Epoch | TAR@FAR=1e-4 | ROC AUC |
|---|---:|---:|
| 4 | 0.276934 | 0.966455 |
| 8 | 0.281792 | 0.964974 |
| 12 | 0.274173 | 0.963512 |
| 16 | 0.275502 | 0.962739 |
| 20 | 0.264049 | 0.959557 |
| 24 | 0.255356 | 0.956372 |
| 28 | 0.257964 | 0.955415 |
| 32 | 0.284604 | 0.956819 |
| 36 | 0.283326 | 0.956096 |

Key facts:
- Best gate value occurred at epoch 32 (0.284604).
- Epoch 36 stayed near the same level (0.283326).
- Net gain from epoch 20 to 36: +0.019277, but late-stage trend is effectively flat.

## 3) Current Pipeline (As It Stands)

1. Data ingest:
   - RecordIO-based training data pipeline with retry and mask augmentation support.
   - Optional DALI RecordIO loader path.

2. Model stack:
   - Student: MobileNetV4 (timm) + embedding projection.
   - Teacher: frozen MagFace iresnet100.
   - Margin head: MagFace angular margin classifier.

3. Training objective:
   - Composite loss: classification + cosine KD (relational KD terms currently disabled).
   - AdamW optimizer with multi-step LR schedule.

4. Validation and selection:
   - In-loop validation: pair CSV protocol (LFW/CFP-FP/AgeDB) aggregate mean_roc_auc.
   - Checkpoint best_metric currently tied to in-loop mean_roc_auc.

5. External gate and final scoring:
   - Watchdog periodically runs IJBC template evaluation and applies threshold gate.
   - End-of-run scripts evaluate latest checkpoint on bin protocol and IJB templates.

## 4) Core Problems in the Current Pipeline

1. Objective mismatch (major):
   - Optimization and best-checkpoint selection use pair-set mean_roc_auc.
   - Product decision metric is IJBC TAR at very low FAR (1e-4/1e-5).
   - This mismatch can produce checkpoints that look good in-loop but stall on target metric.

2. Late-stage diminishing returns:
   - After around epoch 32, IJBC improvements are negligible.
   - Additional epochs mainly consume compute without moving target quality.

3. Student-teacher ceiling gap remains high:
   - Latest bin protocol aggregate (student): mean_roc_auc 0.9712.
   - Latest bin protocol aggregate (teacher): mean_roc_auc 0.9887.
   - Distillation setup is not closing enough of the teacher gap.

4. Watchdog end-epoch edge case:
   - Final milestone gate can wait indefinitely due epoch index convention mismatch
     (human epoch target vs zero-based epoch in run metrics/checkpoint naming).

5. Distillation signal may be too weak for this target:
   - Current KD mostly cosine-level alignment.
   - No feature-level or relation-level pressure to preserve teacher geometry at low FAR.

6. Schedule/regularization interaction:
   - LR reaches very low values late in run; updates become tiny.
   - Constant mask probability may undercut clean-template discrimination if not scheduled.

## 5) Significant Pivot Options (Non-Tame Changes)

These are intentionally large moves, not micro-tuning.

### Pivot A: Make the training target match deployment target

What changes:
- Promote IJBC TAR@1e-4 to first-class checkpoint selector.
- Add periodic in-train IJB template eval and pick best by target metric, not pair mean_roc_auc.
- Introduce FAR-focused hard-negative mining (batch memory/queue) to directly pressure low-FAR behavior.

Why this is significant:
- It changes what the model is actually optimized and selected for.
- Removes the current metric proxy mismatch.

### Pivot B: Two-stage distillation with stronger geometric transfer

What changes:
- Stage 1 (clean alignment): train without mask augmentation to match teacher manifold.
- Stage 2 (occlusion specialization): re-enable masking with curriculum schedule.
- Enable relational KD (distance/angle) + intermediate feature distillation at selected blocks.

Why this is significant:
- Replaces single-stage weak KD with explicit geometry transfer and domain specialization.

### Pivot C: Capacity jump instead of squeezing current tiny student

What changes:
- Move from current MobileNetV4 small profile to a higher-capacity student
  (for example MobileNetV4 medium/large or lightweight ConvNeXt variant).
- Keep latency budget by pruning/quantizing after convergence rather than constraining too early.

Why this is significant:
- Raises representational ceiling before compression, avoiding current under-capacity plateau.

### Pivot D: Data regime overhaul for low-FAR objective

What changes:
- Build a quality-filtered and class-balanced training subset.
- Add template-style sampling (multi-image identity groups), not only pair-centric validation.
- Add hard identity confusion mining from previous checkpoint embeddings.

Why this is significant:
- Low-FAR performance is heavily data-distribution dependent; this attacks the bottleneck directly.

## 6) Recommended Next Wave (Concrete, High-Impact)

Run three experiments, stop any run that misses interim gates early.

1. Exp-1 (Metric-aligned training):
   - Best checkpoint metric = IJBC TAR@1e-4.
   - Early gate at epoch 12 and 20.
   - Abort if no clear gain over current baseline band.

2. Exp-2 (Two-stage KD + relational terms):
   - Stage 1 clean alignment, Stage 2 mask curriculum.
   - Enable RKD distance/angle and one intermediate feature loss.

3. Exp-3 (Capacity bump + later compression):
   - Larger student backbone under same objective.
   - Compare against baseline at equal epoch budget.

Success criterion suggestion:
- Must exceed TAR@1e-4 = 0.295 on IJBC by mid-run gate to continue.
- If not met, terminate early and switch track.

## 7) Operational Note

At time of writing, watchdog processes for this run are still active and waiting on final gate logic. If this route is fully retired, shut down watchdog session and related processes to avoid resource waste.
