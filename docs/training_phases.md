# Training Phases — Architecture & Design

This document explains the three training phases for the MobileNetV4 student, what each one adds, and when to use each resulting checkpoint.

---

## Overview

All three phases share the same base architecture: **MobileNetV4-Conv-Medium** distilled from a frozen **MagFace iResNet-100** teacher on MS1M. The phases are not sequential fine-tuning — each is trained from scratch with a different objective and augmentation regime. Phase 2 and 3 build on insights from phase 1 rather than its weights.

```
MS1M dataset
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Teacher: MagFace iResNet-100 (frozen)                      │
│  • 65.7M params  • 512-d embeddings                         │
│  • IJBB 93.14% / IJBC 97.64% TAR@1e-4                      │
│  • Input: [0,1] (from_minus_one_to_zero_one mode applied)   │
└─────────────────────────────────────────────────────────────┘
    │ soft targets (embeddings)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Student: MobileNetV4-Conv-Medium                           │
│  • ~9M params  • 512-d embeddings                           │
│  • Input: [-1,1] (ImageNet-style mean=0.5 std=0.5)          │
│  • Phase 1: no spatial head                                 │
│  • Phase 2/3: + spatial projection head (512-ch)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Loss function

All phases use the same base loss:

```
L_total = λ_cls · L_MagFace  +  λ_kd · L_KD  +  λ_spatial · L_spatial (phase2/3 only)
```

- **L_MagFace**: magnitude-aware angular margin classification (scale=64, margin=0.42). The margin enforces inter-class separation; the magnitude regulariser (λ=35) pushes low-quality embeddings toward the origin.
- **L_KD**: MSE loss between student and teacher embeddings (cosine-normalised). Ramped from `λ_kd_start=5.0` to `λ_kd_end=8.0` over the first 8 epochs.
- **L_spatial**: intermediate feature matching between student spatial map and teacher spatial features, phase 2 and 3 only (λ ramped 1.0 → 2.0 over 12 epochs).

---

## Phase 1 — Base KD (`train_ms1m_magface_phase1_cplus_aplus_v1.yaml`)

**Goal:** establish the best possible clean-face identity representation.

**Key design choices:**
- No spatial KD head — student is a plain backbone + linear projection.
- Mild augmentation: `mask_prob=0.10` (random lower-face mask on 10% of training images), introduced only after 20 mask-free warmup epochs so the backbone learns clean features first.
- LR schedule: AdamW, warm-up 3 epochs (start factor 0.1), decay at epochs 20/30/36 (×0.1 each). No SWA.

**Training schedule:**
```
Epochs 0–2:    LR warm-up (1e-5 → 1e-4)
Epochs 3–19:   Clean training (no mask augmentation)
Epochs 20–39:  mask_prob=0.10 introduced
               LR drops at ep20, ep30, ep36
```

**Result:** best clean-face benchmark across all phases.

| Checkpoint | IJBB TAR@1e-4 | IJBC TAR@1e-4 | LFW | CFP-FP | AgeDB |
|---|---|---|---|---|---|
| phase1/latest (ep39) | **87.98%** | **90.65%** | 99.25% | 93.43% | 95.68% |
| phase1/best (ep29) | 86.88% | 89.50% | 99.20% | **94.14%** | **95.77%** |

> **Use phase1/latest** for clean-face deployment. **Use phase1/best** if CFP-FP (cross-pose) or mean bin accuracy matters more than IJB.

---

## Phase 2 — Occlusion Curriculum + Spatial KD (`train_ms1m_magface_phase2_occlusion_spatial_v1.yaml`)

**Goal:** add structured occlusion robustness and intermediate feature alignment.

**What's new vs phase 1:**
- **Spatial KD head**: a 512-channel projection on the student's last feature map is matched against the teacher's spatial output. This forces the student to build spatially-aware intermediate representations, not just the final embedding.
- **Occlusion curriculum**: augmentation intensity ramps up over epochs 10–25:

  | Epoch range | Mask prob | Gaussian blur prob | Motion blur prob |
  |---|---|---|---|
  | 0–9 | 0 | 0 | 0 |
  | 10–25 (ramp) | 0.10 → 0.30 | 0.25 → 0.75 | 0.10 → 0.55 |
  | 26–40 | 0.30 | 0.75 | 0.55 |

  Gaussian blur σ ramps 1.0 → 2.8 (larger kernels than phase3). Motion blur kernel ramps 7 → 17px.

**Why phase 2 is weaker than expected:** the occlusion curriculum was introduced without SWA, and the LR schedule is inherited unchanged from phase 1. The model overtrained on the mask augmentation in later epochs. Phase 2/best (epoch 9) emerged extremely early, which is a sign the model peaked before the occlusion curriculum had a chance to stabilise. IJBB 84.52% is the weakest of all three phases.

| Checkpoint | IJBB TAR@1e-4 | IJBC TAR@1e-4 | LFW T@1e-3 drop | CFP-FP T@1e-3 drop |
|---|---|---|---|---|
| phase2/latest (ep32) | 84.52% | 86.85% | −0.017 | −0.385 |

> **Phase 2 is not recommended for deployment.** It is outperformed by phase1 on clean faces and by phase3/swa on occluded faces.

---

## Phase 3 — True Asymmetric Distillation + SWA (`train_ms1m_magface_phase3_trueasym_swa_v1.yaml`)

**Goal:** combine occlusion robustness and clean-face accuracy via asymmetric training and SWA stabilisation.

**What's new vs phase 2:**

### True asymmetric distillation (`dali_true_asymmetry: true`)
In phase 1 and 2, both the student and teacher see the same augmented image. In phase 3, the teacher always sees the **clean (unaugmented)** image while the student sees the **augmented** one. This is the key change:

```
Phase 1/2:   student(aug_img) ─── KD loss ──► teacher(aug_img)
Phase 3:     student(aug_img) ─── KD loss ──► teacher(clean_img)
```

The teacher now provides clean, unbiased soft targets even when the student input is occluded or blurred. This prevents the student from distilling noise from a confused teacher and gives a stable learning signal throughout the entire curriculum.

### SWA (Stochastic Weight Averaging)
Starting at epoch 35, a running average of model weights is maintained with a constant low LR (5×10⁻⁵). The SWA checkpoint averages epochs 35–39, smoothing over the oscillations that occur at the end of cosine/step annealing. This is what makes `phase3/swa` significantly more robust than `phase3/latest` at strict TAR thresholds.

### Tighter occlusion augmentation
Phase 3 uses slightly softer motion blur (kernel 5–9px vs phase 2's 7–17px) and Gaussian σ up to 1.5 (vs 2.8 in phase 2). The goal is a less aggressive curriculum that pairs better with asymmetric distillation.

**Training schedule:**
```
Epochs 0–9:   Clean training
Epochs 10–25: Occlusion ramp (mask, Gaussian blur, motion blur)
Epochs 26–34: Full occlusion augmentation at peak intensity
Epochs 35–39: SWA accumulation (constant LR 5e-5)
              → swa.pt = averaged weights of epochs 35–39
```

**Result:**

| Checkpoint | IJBB TAR@1e-4 | IJBC TAR@1e-4 | LFW T@1e-3 drop | CFP-FP T@1e-3 drop | CPLFW T@1e-3 drop |
|---|---|---|---|---|---|
| phase3/latest (ep39) | 84.78% | 87.23% | −0.026 | −0.221 | −0.266 |
| **phase3/swa** (ep35–39) | 85.27% | 87.77% | **−0.019** | **−0.298** | **−0.148** |
| phase3/best (ep13) | 84.77% | 87.64% | — | — | — |

> **Use phase3/swa** for masked or partially-occluded deployment. It beats the teacher on every occlusion metric despite being a student model.

---

## Phase comparison

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Spatial KD | ✗ | ✓ | ✓ |
| Occlusion curriculum | Light (10%) | Full | Full |
| Asymmetric distillation | ✗ | ✗ | ✓ |
| SWA | ✗ | ✗ | ✓ (ep35–39) |
| Best IJB clean | ✓ | ✗ | — |
| Best occlusion robustness | ✗ | ✗ | ✓ (swa) |
| Recommended checkpoint | latest (ep39) | — | swa |

---

## How to reproduce

```bash
# Phase 1
CONFIG_PATH=configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml bash scripts/launch_train.sh

# Phase 2
CONFIG_PATH=configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml bash scripts/launch_train.sh

# Phase 3
CONFIG_PATH=configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml bash scripts/launch_train.sh
```

Each phase trains independently from scratch (no weight transfer between phases). The SWA checkpoint is written automatically at the end of phase 3 training.

Full eval suite after training:

```bash
bash scripts/run_final_eval_suite.sh
```

This runs bin protocol (LFW/CFP-FP/AgeDB), IJB template 1:1 (InsightFace + flip-TTA), and occlusion robustness for all key checkpoints, and writes everything to `results/`.
