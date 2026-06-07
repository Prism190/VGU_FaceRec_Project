# Peer Review Report: Lightweight Video Face Recognition Pipeline for Edge Deployment

## Summary
This paper presents a lightweight video face recognition pipeline for edge deployment built around a MobileNetV4-Conv-Medium student distilled from a frozen MagFace iResNet-100 teacher. The authors study three independently trained variants (clean KD, spatial KD with aggressive occlusion, and an SWA-stabilized robust variant) and integrate the distilled encoder into a full video pipeline with YOLO detection, 5-point alignment, tracking, liveness gating, MagFace-magnitude quality filtering, magnitude-weighted track pooling, and FAISS-based retrieval. Experiments on standard image and template verification benchmarks (LFW/CFP-FP/AgeDB-30 and IJB-B/C) show a trade-off: the clean KD variant excels at low-FAR verification, while the SWA variant is more robust to synthetic lower-face occlusion.

## Strengths

### Technical novelty and innovation
* Uses MagFace as both a teacher and a principled cue for quality-aware gating and pooling, aligning training and deployment signals.
* Clean-teacher/augmented-student distillation is a sound design to promote robustness without degrading the teacher input.
* Identifies and analyzes a practical failure mode of spatial KD when combined with strong occlusion augmentation, providing a helpful negative result.

### Experimental rigor and validation
* Evaluates on both bin-level (LFW/CFP-FP/AgeDB-30) and template-based (IJB-B/C) protocols, with an emphasis on low-FAR TAR—appropriate for deployment contexts.
* Provides an occlusion robustness analysis via synthetic masking with clear degradation statistics across multiple benchmarks.

### Clarity of presentation
* The paper is clearly written and well organized; pipeline stages, training variants, and evaluation metrics are described coherently.
* Limitations and design trade-offs are candidly discussed, including the configuration-level nature of ablations.

### Significance of contributions
* Addresses a practically important problem: building a deployable, quality-aware, track-level video FR pipeline on resource-constrained devices.
* The robustness vs. clean-accuracy trade-off and recommendations per deployment scenario can be valuable for practitioners.

## Weaknesses

### Technical limitations or concerns
* Methodological novelty is limited; core elements (embedding-level KD, MagFace loss/quality, YOLO detection, DeepSORT/BoT-SORT, FAISS, SWA) are established. The contribution is mainly integration and configuration analysis.
* Spatial KD design remains under-specified and not isolated; the “conflict” is plausible but not rigorously teased apart from other factors.

### Experimental gaps or methodological issues
* No comparisons against other lightweight face encoders or distilled baselines (e.g., MobileFaceNet/ShuffleFaceNet/PartialFC-based lightweight ArcFace, or recent KD frameworks for FR), limiting context on how competitive the student is for a given compute budget.
* “Edge-oriented” claim is not substantiated by hardware results: no FLOPs/params, latency/FPS, energy on mobile/embedded platforms, or memory footprint.
* Liveness (PAD) is included in the pipeline but not specified or evaluated; claims about increased deployment reliability are thus weakly supported.
* Runtime pipeline evaluation lacks standardized open-set video benchmarks and quantitative 1:N identification metrics (e.g., IJB-C 1:N, FPIR/TPIR) beyond descriptive statistics.
* Occlusion robustness is assessed only via synthetic lower-face masks; no evaluation on real masked/occluded datasets (e.g., RMFRD, MaskedFace, MFR2) or surveillance-like settings.

### Clarity or presentation issues
* Augmentation details (mask geometry, blur kernel ranges, frequencies, schedule) and exact loss weights/enablement chronology are incomplete, hampering reproducibility.
* The relationship between “V3/best,” “V3/latest,” and “V3/SWA” checkpoints could be streamlined; it is not always clear which selection rule applies for “best.”

### Missing related work or comparisons
* Limited discussion and empirical comparison to prior lightweight/distilled FR models and video FR pipelines that include track-level aggregation or quality-aware selection.
* No discussion of alternative quality measures (e.g., face detection confidence, predicted quality scores) and how MagFace magnitude compares.

## Detailed Comments

### Technical soundness evaluation
* The core objective—MagFace classification plus embedding-level MSE KD—is technically sound and widely used in FR. Clean-teacher vs. augmented-student supervision is a well-motivated asymmetry.
* The spatial KD analysis is insightful but not isolated: V2 simultaneously changes occlusion strength, adds spatial KD, and may have different schedules. The paper acknowledges this, yet a more controlled ablation (e.g., toggling only spatial KD with fixed augmentations) would strengthen the causal claim.
* The magnitude-weighted track pooling is consistent with MagFace’s quality-magnitude linkage; however, an ablation against simple average pooling, top-k pooling, or attention would clarify its added value.
* Open-set thresholds (top-1 score, top-1/top-2 margin) are sensible, but threshold selection, calibration, and stability across galleries are not detailed.

### Experimental evaluation assessment
* Benchmarking on LFW/CFP-FP/AgeDB-30 and IJB-B/C is expected, and emphasizing TAR@1e-4 is appropriate for access control. Results show plausible trends: V1 best at strict FAR on IJB; V3/SWA best under synthetic occlusion.
* Absent: standardized 1:N evaluation (e.g., IJB-C identification protocols with TPIR/FPIR) and any large-scale open-set retrieval analysis, which is critical for deployment claims.
* Missing comparisons to lightweight baselines and SOTAs with similar compute budgets; without FLOPs/params and wall-clock latency, the “edge” efficiency claim is not quantifiably supported.
* The PAD/liveness component is treated as a black box with no dataset, metrics, or attack coverage, undermining the claim that the full pipeline improves deployment reliability.
* Synthetic-occlusion evaluation is a useful stress test, but complementing with real masked-face datasets or in-the-wild occlusions would significantly increase confidence in the robustness claims.

### Comparison with related work
* The provided related work summaries are largely orthogonal to face recognition; however, within FR, more comprehensive positioning relative to lightweight FR and FR-specific KD literature is needed. The paper cites key margin losses (ArcFace/CosFace) and MagFace, but does not compare to other compact/edge architectures or recent distillation strategies tuned for FR embeddings.
* Video FR pipelines with track-level aggregation have been explored; the paper could better contextualize how magnitude-based pooling compares to established aggregation or quality-aware selection schemes in prior video FR works.

### Discussion of broader impact and significance
* The paper responsibly lists limitations and practical takeaways (choose V1 for clean settings; V3/SWA for occluded contexts).
* Ethical and societal aspects (privacy, demographic fairness/bias, consent, and misuse of surveillance technology) are not discussed. Given deployment emphasis, at least a brief discussion and, ideally, demographic subgroup analyses would be appropriate.
* The integration emphasis is valuable for practitioners, but for a top-tier venue, stronger methodological novelty or more comprehensive quantitative evidence (including edge device metrics and standardized open-set video identification) would be expected to justify publication.

## Questions for Authors
1. What are the exact student model compute characteristics (parameter count, FLOPs/MACs, and memory footprint), and what end-to-end FPS/latency do you achieve on representative edge devices (e.g., Jetson, ARM SoCs, Android/iOS) for full pipeline operation?
2. How were thresholds for magnitude-based quality gating and the top-1/top-2 margin chosen and calibrated? Do they generalize across gallery sizes and domains?
3. Could you provide a controlled ablation isolating spatial KD (on/off) under fixed augmentation strength to validate the hypothesized conflict with masked inputs?
4. What are the precise augmentation parameters and schedules (mask placement/shape/ratio, blur kernel sizes, probabilities, curriculum timings)? Can you release code or configs for reproducibility?
5. Which liveness/PAD model was used, on what datasets was it trained/evaluated, and how does PAD performance affect end-to-end identification error (e.g., ablation with/without PAD)?
6. Do your training data and evaluation sets have identity overlap (e.g., MS1M vs. LFW/CFP-FP/AgeDB)? If so, how is this addressed to avoid biased estimates?
7. Can you evaluate magnitude-weighted pooling against other pooling strategies (mean, median, top-k, attention) and report their impact on IJB and occlusion robustness?
8. Will you include standardized open-set 1:N evaluations (e.g., IJB-C identification TPIR/FPIR) and, if possible, a real masked/occluded dataset to corroborate the synthetic-occlusion results?

## Overall Assessment
This paper tackles a relevant and impactful problem—deployable, lightweight video face recognition with quality-aware aggregation—and presents a well-written, pragmatic system study. The main empirical finding, a trade-off between clean low-FAR accuracy (V1) and occlusion robustness (V3/SWA), is clearly demonstrated and practically useful. 

However, the methodological novelty is limited, and several critical pieces of evidence are missing to substantiate the “edge-oriented” and “deployment-ready” claims: no hardware performance metrics or compute comparisons, no standardized open-set video identification evaluation, no PAD evaluation, and limited comparisons to lightweight/distilled face recognition baselines. The spatial KD conflict is a good observation but is not rigorously isolated.

Given ACCV’s standards, the work currently reads more as a solid engineering study and system integration report than a research contribution with strong novelty and comprehensive validation. Strengthening the evaluation with compute/latency metrics on edge hardware, standardized 1:N open-set results, real occlusion datasets, PAD evaluation, and controlled ablations would significantly improve the paper’s suitability. As it stands, I recommend rejection, with encouragement to resubmit after addressing these gaps.
