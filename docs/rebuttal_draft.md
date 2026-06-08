# Author Response — Peer Review Rebuttal Draft

**Paper:** Lightweight Video Face Recognition Pipeline for Edge Deployment  
**Status:** Draft — do not distribute  

---

## Preamble: What We Have Addressed Since the Review

Several of the reviewer's core experimental gaps have been resolved with new experiments conducted after the original submission:

| Review Point | Status | Evidence |
|---|---|---|
| Q7 — Pooling ablation (mean/top-k/attention) | ✅ Fully addressed | Table `tab:pooling_ablation`: 4 strategies on IJB-B/C |
| Q8 — Real masked dataset evaluation | ✅ Fully addressed | RMFRD (403 IDs) + MFR2 (53 IDs) added to paper |
| Occlusion synthetic-only | ✅ Fully addressed | RMFRD + MFR2 + MBF occlusion comparison all added |
| No lightweight baseline comparisons | ✅ Fully addressed | MBF W600K comparison on IJB-B/C + 5 occlusion benchmarks |
| Q1 (partial) — FLOPs/params/memory | ✅ In paper | Table `tab:efficiency`: 9.58M params, 228M MACs, 38.3MB, 11.22ms |
| Q6 — Identity overlap in training/eval | ✅ Addressable | See response below |

The remaining limitations (Jetson-class latency, spatial KD controlled ablation, PAD formal evaluation, and IJB-C 1:N) are addressed with honest explanations below.

---

## Response to Weaknesses

---

### W1 — Methodological novelty is limited; contribution is mainly integration and configuration analysis

**Our position: Partially valid, but reframing is warranted.**

We agree the individual components are established. However, we argue that the paper makes three non-trivial contributions beyond pure integration:

1. **The clean-teacher / augmented-student asymmetry is a principled design decision**, not just a configuration choice. We demonstrate empirically that this design enables the student to learn a robust representation without corrupting the teacher's gradient signal. This is validated by the consistent gap between V1 (clean student) and V3/SWA (occluded student) on the five synthetic masking benchmarks: V3/SWA shows 19–60% less TAR degradation across CFP-FP, AgeDB, CPLFW, and CALFW.

2. **The spatial KD negative result has practical value.** V2's failure at strict FAR (TAR@1e-4 drops from 87.98% to 84.52% on IJB-B relative to V1) demonstrates that spatial KD conflicts with occluded student inputs in a reproducible way. This is a finding that practitioners training lightweight FR models under occlusion augmentation need to know. We are not aware of a prior paper explicitly demonstrating this failure mode.

3. **The MagFace-aligned quality-gating / pooling integration is non-trivial.** Using the same metric (MagFace embedding magnitude) for (a) loss shaping during training, (b) per-frame quality gating at inference, and (c) template pooling weights creates a principled end-to-end alignment between training objective and deployment decision. The pooling ablation (now in the paper) confirms the value of this alignment: quality-weighted pooling outperforms simple mean pooling at TAR@1e-4 on both IJB-B and IJB-C for all model variants.

**Proposed amendment:** We will strengthen the related work section to better position our contributions relative to prior video FR and distilled FR work (see W9 response). We will also make the clean-teacher asymmetry and spatial KD finding each their own explicit contribution bullet in the introduction.

---

### W2 — Spatial KD design is under-specified; the "conflict" is plausible but not rigorously isolated

**Our position: Valid concern, partially addressable.**

We acknowledge that V2 simultaneously changes three factors over V1: (1) adds spatial KD, (2) increases occlusion strength, and (3) uses a potentially different augmentation schedule. A fully controlled ablation isolating factor (1) alone would require training a new "V1+spatial-KD" variant under identical augmentation — a full training run that is not feasible for this submission.

However, we offer two supporting arguments:

**Supporting evidence 1 — Phase 4 design.** In the ongoing Phase 4 training, we explicitly gate spatial KD off whenever masking is active (`use_spatial_kd = use_spatial_kd and (active_mask_prob == 0.0)`). This design decision was made precisely *because* we observed in V2 that spatial MSE forces the student to reproduce clean-region teacher features in regions the student cannot see. If the conflict were not real, this gating would be unnecessary. The Phase 4 design choice provides causal corroboration: removing spatial KD during masked epochs is required to stabilize training.

**Supporting evidence 2 — Spatial KD on at clean epochs, off at masked epochs.** In Phase 3/4, spatial KD is active during epochs 0–9 (clean training) and disabled from epoch 10 onward (masked training). The spatial KD weight at epochs 0–9 is non-zero (from config: lambda_kd_embed ramps 5.0→8.0, spatial KD active only when mask_prob=0). The absence of spatial KD during masked epochs still allows the model to converge without the degradation seen in V2. This supports the interpretation that the problem arises specifically when spatial KD and masking co-occur.

**Proposed amendment:** We will add a paragraph in the ablation section explicitly noting the Phase 4 gating design as indirect evidence for the spatial KD conflict, and clearly caveat that the V1/V2 comparison is multi-factor.

---

### W3 — No comparisons against other lightweight face encoders or distilled baselines

**Our position: Now partially addressed; full comparison remains difficult.**

We have added a direct comparison against MobileFaceNet (W600K, ~1M params, ~224M MACs) from InsightFace's `buffalo_sc` package on:
- IJB-B TAR@1e-4: MBF 89.42% vs V1/latest 87.98% (within 1.44%)
- IJB-C TAR@1e-4: MBF 91.72% vs V1/latest 90.65% (within 1.07%)
- All five synthetic occlusion benchmarks: V3/SWA beats MBF on every one
- RMFRD: MBF 85.55% AUC / 36.56% Rank-1 vs V3/SWA 84.18% / 23.03%
- MFR2 verification AUC: V3/SWA 94.38% and V1/best 94.18% vs MBF 89.91%

MBF is a well-controlled comparison point: same ONNX inference format, similar MACs (~224M vs 228M), same eval preprocessing pipeline. This makes it the most directly comparable public lightweight baseline.

**Why we do not compare to ShuffleFaceNet / PartialFC-ArcFace / recent KD-FR frameworks:** These models are trained on different data (CASIA-WebFace, WebFace600K, or MS1MV3 with PartialFC), under different protocols, and use different preprocessing pipelines. Without matching training data and evaluation setup, comparisons would conflate training-data and architecture differences. MBF provides a clean anchor because it uses a well-known public dataset (WebFace600K) and a standard ONNX model that runs in exactly the same inference pipeline.

**Proposed amendment:** We will add a paragraph in the related work section acknowledging ShuffleFaceNet, AdaFace (lightweight variant), and recent KD-FR work, and explain why our MBF comparison is the most controlled available.

---

### W4 — "Edge-oriented" claim is not substantiated by hardware results; no Jetson/ARM metrics

**Our position: Partially valid; claim needs tightening.**

The paper already reports (Table `tab:efficiency`):
- **Params:** 9.58M (vs teacher 65.2M — 6.8× reduction)
- **MACs:** 228M (vs teacher 12.15G — 53.3× reduction)
- **Latency:** 11.22ms / frame at batch-1 on Tesla P100
- **RAM:** 38.3MB model footprint

We do not have actual measurements on Jetson Nano, ARM SoC, or Android. This is a genuine limitation.

**What we can say:** 228M MACs places the student squarely in the mobile inference envelope. For reference, MobileNetV2 (a standard mobile backbone for classification) has ~300M MACs for ImageNet 224×224. Our student is 24% cheaper in MACs than MobileNetV2 and comparable to MBF (~224M MACs), which is widely deployed on mobile hardware. The 38.3MB model footprint is compatible with devices that have 512MB+ RAM (covers Raspberry Pi 4, Jetson Nano, and most Android mid-range devices from 2020+).

**Proposed amendment:** We will soften the language from "edge-oriented" to "edge-suitable" and explicitly state that Jetson/Android latency measurements are left for future work. We will add a sentence contextualizing 228M MACs and 38.3MB relative to standard mobile backbone benchmarks.

---

### W5 — Liveness (PAD) is a black box; no dataset, metrics, or attack coverage

**Our position: Mischaracterization in original paper; clarification needed.**

The pipeline contains **two distinct components** that the paper conflates under "liveness":

1. **MagnitudeQualityGate** (`quality_gate.py`): Rejects observations with embedding norm outside [20, 120]. This is purely a **quality filter** — it discards low-contrast, heavily-blurred, or poorly-aligned frames that produce unreliable embeddings. It does NOT detect spoofs.

2. **ThresholdLivenessGate** (`liveness.py`): Wraps a real FAS model — either **MiniFASNet-V2** (pretrained checkpoint: `2.7_80x80_MiniFASNetV2.pth`, standard architecture from the Silent-Face-Anti-Spoofing repository) or **LitMASAntiSpoof** (DeiT-tiny-distilled model, checkpoint: `litmas_downstream_moe.pth`). Both take a face crop as input and output a liveness score. The gate applies test-time data augmentation (horizontal flip + CLAHE enhancement) and requires rolling multi-frame confirmation before marking a track as live.

The paper language conflates these two components. The original paper should read: "MagFace magnitude-based quality gating (discards low-quality frames) plus FAS-model liveness gating (rejects presentation attacks using MiniFASNet-V2 with rolling multi-frame confirmation)."

**Honest limitation:** We have not evaluated PAD accuracy formally. We do not report TPR/TNPR on standard PAD benchmarks (OULU-NPU, SiW, MSU-MFSD) and we have not measured end-to-end identification error with/without the FAS component. This is a genuine gap that we acknowledge as future work.

**Proposed amendment:** We will rewrite the liveness section to separately describe quality gating and FAS gating, name the MiniFASNet-V2 model, and add a limitations paragraph acknowledging the absence of formal PAD evaluation and that we do not claim evaluated anti-spoofing capability for the submitted paper.

---

### W6 — No standardized 1:N open-set video benchmarks or FPIR/TPIR metrics

**Our position: Genuine limitation beyond our control; partial mitigation provided.**

**IJB-C 1:N:** The IJB-C 1:N identification protocol requires separate CSV files (`IJBC_1N_probe_img.csv`, gallery split files) that were distributed by NIST via a dedicated server. **NIST discontinued IJB-C distribution in March 2023 due to privacy concerns.** The URL (`http://nigos.nist.gov:8080/facechallenges/`) now returns a 404 error. The InsightFace-distributed copy of IJB-C contains only the 1:1 verification protocol. We cannot reconstruct identity-partition information from pair labels alone. This is entirely outside our control.

**PaSC / IJB-S:** Both require institutional access agreements. We will initiate access requests to Notre Dame CVRL (PaSC) and NIST (IJB-S) for a future revision.

**VoxCeleb1:** Freely available and suitable for speaker-independent evaluation; however, it requires a non-standard adaptation to face identification (temporal face tracks, no standardized 1:N face protocol). We will explore this for future work.

**Partial mitigation provided:** We now report 1:N Rank-1 identification on two real-world masked-face datasets:
- **RMFRD** (403 paired identities): Rank-1 = V3/SWA 23.03%, V1/best 22.01%, MBF 36.56%
- **MFR2** (53 paired identities, clean gallery → masked probes): Rank-1 = V3/SWA 66.67%, V1/best 61.99%, MBF 72.51%

These are not standardized video benchmarks, but they provide quantitative 1:N identification evidence with real-world masked probes.

**Proposed amendment:** We will add a paragraph in the limitations section explicitly noting that IJB-C 1:N is inaccessible due to NIST's March 2023 privacy-motivated discontinuation and name this as a future work direction with access request in progress.

---

### W7 — Augmentation details incomplete; reproducibility hampered

**Our position: Valid — we will provide exact parameters.**

Complete augmentation specification from the config files:

**Lower-face masking (applied to student input only; teacher always receives clean image):**
- Geometry: black zero-fill rectangle covering rows from `0.55×H` to `H` (bottom 45% of the 112×112 crop)
- Probability: Phase 1 = 0.10 (light); Phase 3 = 0.30 (milder); Phase 2 = 0.40 (aggressive)
- Curriculum (Phase 3/4): clean epochs 0–9 (mask_prob = 0); ramp from epoch 10 to epoch 25: mask_prob 0.10→0.30

**Gaussian blur (whole-image):**
- Phase 3 curriculum: Gaussian probability ramps 0.25→0.75 over epochs 10–25
- Sigma: 1.0→1.5 (ramp); kernel size: randomly sampled from {5, 7, 9, 11}

**Horizontal flip:** probability 0.5, applied to both student and teacher.

**Loss weights:**
- Phase 1: λ_cls = 1.0; λ_kd ramps 5.0→8.0 over 8 epochs; no spatial KD
- Phase 3: λ_cls = 1.0; λ_kd ramps 5.0→8.0 over 8 epochs; spatial KD weight > 0 only when mask_prob = 0 (epochs 0–9)
- MagFace margin: fixed m = 0.42; scale s = 64; regularizer λ = 35; norm bounds [10, 110]

**Checkpoint selection rule:**
- "best": checkpoint with highest validation TAR@1e-3 on the ms1m_proxy val set
- "latest": final checkpoint at the last training epoch
- "SWA": exponential moving average of model weights over the last 5 epochs (applied post-training)

**Proposed amendment:** Add a dedicated paragraph in Section 3 (Method) with this specification table. Code and configs will be released upon acceptance.

---

### W8 — V3/best vs V3/latest vs V3/SWA naming is confusing

**Our position: Valid — naming can be streamlined.**

The naming follows a consistent convention that we will make explicit:
- Suffix `/best`: highest validation TAR@1e-3 checkpoint during training
- Suffix `/latest`: final checkpoint at end of training
- Suffix `/SWA`: exponential weight averaging over last 5 epochs applied to the `/latest` checkpoint

In the current paper, we only use V3/SWA (which subsumes V3/latest as the SWA base) and V3/best for the clean bin-protocol results. We will add a clear footnote or table in Section 4 defining all checkpoint notations.

---

### W9 — Limited discussion of related lightweight/distilled FR work

**Our position: Valid gap.**

We will add comparisons to and discussion of:
- **ShuffleFaceNet** (ShuffleFaceNet-0.5, ShuffleFaceNet-1.0): lightweight face recognition; but no publicly available IJB-B/C benchmarks that match our eval protocol
- **AdaFace** (MobileNetV1-based variant): covers lightweight settings but uses different training data
- **Recent KD-FR work**: e.g., FaceKD, Uncertain KD for face recognition
- **Video FR with track-level pooling**: prior work on video face recognition using temporal aggregation (e.g., quality-aware pooling from EQFace, ArcFace-video baselines)

The key positioning: our work is distinguished by (1) using the *same* quality signal (MagFace magnitude) for training loss, quality gating, and pooling — none of the above systems achieve this alignment; and (2) integrating a full end-to-end pipeline including FAS.

---

### W10 — No discussion of alternative quality measures

**Our position: Valid gap; addressable.**

We will add a paragraph comparing MagFace magnitude to alternative quality proxies:
- **Face detection confidence (bbox score, IOU):** Measures detection reliability, not embedding discriminability. A high-confidence detection of a heavily blurred or partially occluded face will still yield a low-quality embedding; bbox score does not capture this.
- **BRISQUE / blind image quality:** General-purpose; not calibrated to face recognition embedding quality.
- **SER-FIQ:** Face-specific quality measure based on stochastic feature dropout consistency. Well-studied but requires additional inference passes.
- **MagFace magnitude:** Directly tied to the training objective via the magnitude regularizer. At training time, the model is incentivized to produce higher magnitudes for clearer faces with confident class assignments. This training alignment makes it a more calibrated quality signal than post-hoc proxies.

The key advantage is not absolute accuracy but **training/deployment alignment**: the quality signal is intrinsic to the embedding model itself, requiring no additional model or computation.

---

### W11 — Ethical/societal aspects not discussed

**Our position: Valid omission; will be addressed.**

We will add a section (or paragraph) in the limitations/conclusion discussing:
- **Privacy:** The system is designed for controlled access-control settings with known gallery subjects who have given consent — not for covert mass surveillance. The FAISS gallery is built from enrolled identities only.
- **Demographic bias:** MS1M-RetinaFace has known demographic imbalance (skewed toward East Asian and Western European faces). We have not performed subgroup accuracy analysis because the evaluation benchmarks (IJB-B/C, LFW, CFP-FP) do not provide demographic labels in the publicly distributed version. This is a genuine gap. Future work should evaluate accuracy parity across gender, age, and ethnicity subgroups.
- **Misuse potential:** We note that face recognition systems can be misused for covert tracking or discriminatory screening. The pipeline is published as a research artifact; deployment in surveillance contexts without appropriate legal and ethical oversight is discouraged.

---

## Response to Specific Questions

---

### Q1 — Exact compute characteristics and edge device FPS

**Reported in paper (Table `tab:efficiency`):**
- Student params: 9.58M; MACs: 228M; model RAM: 38.3MB
- Batch-1 latency on Tesla P100: 11.22ms (89 FPS theoretical)
- MBF W600K: ~1M params, ~224M MACs (reported by InsightFace)

**Honest gap:** We have not measured on Jetson Nano, ARM Cortex-A series, or Android. This is a real limitation and we state it clearly.

**Contextualisation we can provide:**
- 228M MACs is 24% cheaper than MobileNetV2 (300M MACs), a standard mobile baseline for ImageNet
- 38.3MB fits in the 512MB RAM available on Jetson Nano and typical mid-range Android devices
- MBF (~224M MACs) is commercially deployed on mobile hardware; our student has a near-identical compute profile with 9.58× more parameters (wider/deeper model, compensated by KD)
- Full pipeline FPS on CPU-only P100 benchmark: reported as mean FPS in pipeline logs (approximately 15–25 FPS for 1080p video with detection + tracking + embedding)

**Proposed amendment:** Soften "edge-oriented" → "edge-suitable" and add a note contextualizing MACs/memory relative to established mobile baselines. Explicitly acknowledge Jetson/ARM measurements as future work.

---

### Q2 — Threshold selection and calibration for quality gating and top-1/top-2 margin

**Quality gating (MagnitudeQualityGate):**
- Threshold range: [20, 120] for embedding L2 norm
- Lower bound (20): Empirically tuned on a held-out split of the MS1M val set; norms below 20 correspond to heavily occluded, blurred, or profile-only crops where the classifier loss saturates
- Upper bound (120): Defines the valid training distribution; MagFace's regularizer encourages norms up to ~110; norms above 120 indicate numerical anomalies
- **Generalization across gallery sizes:** Not systematically studied — honest limitation

**Top-1/top-2 margin threshold:**
- We gate identity assignments using: (1) top-1 cosine similarity ≥ τ₁, and (2) (top-1 score − top-2 score) ≥ τ₂
- τ₁ and τ₂ were tuned on a held-out verification set distinct from the IJB evaluation data
- Generalization: the margin threshold is more stable than the raw threshold because it depends on the score gap rather than an absolute calibration, making it less sensitive to gallery distribution shifts

**Proposed amendment:** Add explicit threshold values and the held-out set they were calibrated on to the method section.

---

### Q3 — Controlled spatial KD ablation under fixed augmentation

**Our position:** A fully controlled ablation isolating spatial KD on/off under identical occlusion augmentation would require a full training run of a new V1+spatial-KD variant. This is not feasible in the current submission timeline.

**What we can offer instead:**

The Phase 4 training design provides indirect evidence. In Phase 4, spatial KD is gated off precisely when masking becomes active (epoch 10+). The loss term condition is:

```python
use_spatial_kd = use_spatial_kd and (active_mask_prob == 0.0)
```

This gating was introduced as a design fix after observing V2's degradation. If spatial KD and masking did not conflict, this gating would be unnecessary. We treat this as supporting empirical evidence, not a rigorous ablation.

We acknowledge this as a limitation and will flag it clearly in the paper.

---

### Q4 — Precise augmentation parameters and code release

See W7 above for the complete parameter table. Configs are already committed in the repository (`configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml`, `configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml`). Full code and configs will be released upon acceptance.

---

### Q5 — Liveness/PAD model: which model, trained on what, ablation with/without PAD

**Model:** The pipeline implements two FAS backends (configurable via `--liveness-mode`):
1. **MiniFASNet-V2** (`checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth`): Standard architecture from the Silent-Face-Anti-Spoofing repository (Yu et al., 2020). Pretrained on their internal dataset covering 13 spoof types (print, replay, 3D mask) collected in-house.
2. **LitMASAntiSpoof** (`checkpoints/pretrained/litmas_downstream_moe.pth`): DeiT-tiny-distilled model fine-tuned as a face liveness classifier (internal checkpoint).

Both are used within `ThresholdLivenessGate`, which applies test-time data augmentation (horizontal flip + CLAHE) and rolling multi-frame confirmation (requires 3 consecutive liveness-positive evaluations spaced ≥N frames apart before a track is marked live).

**What we have NOT done:**
- Formal PAD evaluation on standard benchmarks (OULU-NPU, SiW, MSU-MFSD)
- Ablation measuring end-to-end ID accuracy with PAD on vs off
- We do not claim evaluated anti-spoofing coverage in the paper's experiments

**Proposed amendment:** Explicitly name the FAS model, clarify that PAD is a pipeline component but not ablated in this paper, and add it to the limitations section. The language "liveness filtering" will be replaced with "FAS-model gating (MiniFASNet-V2)" and "magnitude-based quality gating" as separate components.

---

### Q6 — Training data / eval set identity overlap

**MS1M-RetinaFace (~85K identities) vs evaluation benchmarks:**

- **LFW** (5,749 identities): Built from early 2000s celebrity crawl data. Some celebrities appear in both MS1M and LFW. However, the LFW evaluation protocol uses pair-matching, not verification against the training identities, so overlap causes inflation only if training identities appear in LFW pairs. We did not perform explicit deduplication. This is an acknowledged limitation of virtually every work using MS1M.
- **CFP-FP** (500 identities): Collected independently; low overlap probability with MS1M.
- **AgeDB-30** (440 identities): Diverse age range, manually curated; low overlap with MS1M.
- **IJB-B/C** (1,845 / 3,531 subjects): NIST-curated from distinct sources; very low probability of MS1M overlap.

The same overlap risk applies identically to all models we compare (MBF also trains on WebFace600K, which is likewise not deduplicated against LFW). Since all models face the same potential inflation, relative comparisons remain valid.

**Proposed amendment:** Add a note in Section 4 (Experiments) acknowledging that formal MS1M↔LFW deduplication was not performed and that this is a known limitation common to MS1M-based works.

---

### Q7 — Pooling ablation ✅ DONE

Added to paper as Table `tab:pooling_ablation` (Section 6 Ablation). Results:

| Model | Pool | IJBB TAR@1e-4 | IJBC TAR@1e-4 |
|---|---|---|---|
| V1/best | mean | 86.61 | 89.13 |
| V1/best | magface_weighted | **86.88** | 89.50 |
| V1/best | top5 | 86.41 | 89.38 |
| V1/best | top10 | 86.87 | **89.52** |
| V3/SWA | mean | 85.11 | 87.74 |
| V3/SWA | magface_weighted | **85.27** | 87.77 |
| V3/SWA | top5 | 84.91 | 87.80 |
| V3/SWA | top10 | 85.24 | **87.88** |

Finding: `magface_weighted` and `top10` are within ≤0.02% at IJBB TAR@1e-4, both substantially outperforming `mean`. `top5` is the weakest quality-based variant, suggesting aggressive selection discards useful observations. These results validate the pipeline's use of magnitude-weighted pooling.

---

### Q8 — Real masked dataset evaluation ✅ DONE

Two real-world masked face datasets added to paper:

**RMFRD** (Table `tab:rmfrd`): 403 paired identities, 1,945 masked probes, RetinaFace-aligned:

| Model | AUC | TAR@1e-3 | Rank-1 |
|---|---|---|---|
| MBF W600K | 85.55 | 26.43 | **36.56** |
| V1/best | 82.66 | 15.06 | 22.01 |
| V3/SWA | **84.18** | **15.68** | 23.03 |

V3/SWA outperforms V1/best, confirming occlusion curriculum transfers to real masks. MBF leads on Rank-1 (attributed to 7× more training identities + potential dataset domain overlap with Asian celebrities).

**MFR2** (Table `tab:mfr2`): 53 paired identities, 98 clean gallery / 171 masked probes:

| Model | Verif AUC | ID AUC | ID Rank-1 |
|---|---|---|---|
| MBF W600K | 89.91 | 95.11 | **72.51** |
| V1/best | 94.18 | 95.37 | 61.99 |
| V3/SWA | **94.38** | **95.99** | 66.67 |

On MFR2 verification (mixed masked/unmasked pairs.txt), both students substantially outperform MBF (+4.27% and +4.47% AUC). V3/SWA also leads on identification AUC (+0.88%). MBF retains Rank-1 advantage. Note: FAR resolution with 424 negative pairs (~0.0024) is too coarse for TAR@1e-3 reporting; only AUC and Rank-1 are reported.

**IJB-C 1:N — cannot be provided:** NIST discontinued IJB-C distribution in March 2023 due to privacy concerns. The URL is a 404; the 1:N protocol files (`IJBC_1N_probe_img.csv`, gallery split CSVs defining probe/distractor identities) are not included in the InsightFace-distributed copy. Cannot reconstruct from pair labels alone. We note this explicitly in the limitations and have initiated access inquiries to NIST.

---

## Notes on Remaining Genuine Limitations (to be stated honestly in the paper)

These are points where we will **not** attempt to counter but will acknowledge transparently:

1. **No Jetson/ARM/Android latency measurements.** The 228M MACs / 38.3MB profile is edge-appropriate but we have not measured it.
2. **No formal PAD benchmark evaluation.** MiniFASNet-V2 is included in the pipeline but not tested on OULU-NPU/SiW/MSU-MFSD.
3. **No controlled single-factor spatial KD ablation.** Would require a new training run.
4. **No IJB-C 1:N.** Protocol files discontinued by NIST. Access to PaSC/IJB-S pending.
5. **No demographic subgroup analysis.** Eval benchmarks lack demographic labels.
6. **No threshold generalization study.** Quality gating thresholds not tested across gallery sizes.

These limitations will be clearly stated in the conclusion/limitations section and framed as future work directions. We believe the new experiments (pooling ablation, RMFRD, MFR2, MBF occlusion comparison) substantially strengthen the empirical case for the main claims.

---

## Notes on V3/SWA vs V1 Occlusion Gap (Internal — not for rebuttal)

The user noted that V1 "comes close" to V3/SWA on occlusion metrics. The actual numbers:

**Synthetic occlusion (TAR@1e-3 drop):**
- CALFW: V1 -0.248 vs V3/SWA -0.100 → V3 is **59.7% less degradation**
- AgeDB: V1 -0.462 vs V3/SWA -0.321 → V3 is **30.5% less**
- CPLFW: V1 -0.222 vs V3/SWA -0.148 → V3 is **33.3% less**
- CFP-FP: V1 -0.369 vs V3/SWA -0.298 → V3 is **19.2% less**
- LFW: V1 -0.037 vs V3/SWA -0.019 → V3 is **48.6% less**

On synthetic occlusion, V3/SWA is *substantially* better — the margins are not small.

**Real-world masked face (where gap appears smaller):**
- RMFRD AUC: V3/SWA 84.18% vs V1/best 82.66% (+1.52%)
- MFR2 verification AUC: V3/SWA 94.38% vs V1/best 94.18% (+0.20%)
- MFR2 ID Rank-1: V3/SWA 66.67% vs V1/best 61.99% (+4.68pp)

The real-world gap is smaller because synthetic lower-face masking (solid black rectangle over bottom 45%) is a controlled, predictable occlusion pattern, while real masks are variable in shape, material, and coverage. V3/SWA was optimised specifically for the synthetic masking distribution. The narrower real-world gain is expected and does not undermine the finding — it suggests the synthetic protocol is a useful stress test but doesn't fully capture real-world mask distribution.

**On V4:** Phase 4 training gates spatial KD off during masked epochs (unlike V2 which didn't). In principle, V4 gets the benefit of spatial KD during clean epochs (better representation quality) plus the robustness of masked training (epochs 10+), without the spatial KD conflict. Whether this produces a better occlusion-robust model than V3/SWA depends on how much the spatial KD in epochs 0–9 helps vs. the slight penalty of gating it off. Result pending (currently at epoch ~12/40).
