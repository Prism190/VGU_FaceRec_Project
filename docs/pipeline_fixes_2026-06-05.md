# Pipeline Bug Fixes — 2026-06-05

This document records all bugs diagnosed and fixed in the runtime face recognition pipeline
(`src/fas_kd/pipeline/` and `scripts/run_face_pipeline.py`). None of these affected training
or model weights; they are all deployment-layer issues.

---

## Critical — Duplicate Detection (Root Cause of Track Fragmentation)

**Bug:** `yolo11n-face-age.pt` classifies faces by age bracket (young / adult / old). YOLO applies
NMS **per class**, so a face detected as both "young" and "adult" produces two nearly-identical
bounding boxes (IoU ≈ 0.97–0.99) that survive YOLO's internal NMS. Both boxes are forwarded to
the tracker as separate `FaceDetection` objects.

**Impact:** DeepSORT created a separate track for each duplicate detection of the same face.
Result: 27 tracks for 4 people, every identity label flickering as the two tracks alternated
getting matched. The `merge_iou_thres` config field existed in `DetectionConfig` for exactly this
case but the deduplication code was never written.

**Fix:** Added `YOLO11FaceDetector._merge_detections()` — a greedy cross-class NMS pass that
suppresses any detection whose IoU with a higher-confidence detection exceeds `merge_iou_thres`
(default 0.55). Called at the end of `detect()`. Prefers detections with real (non-synthetic)
landmarks when scores are equal.

```
src/fas_kd/pipeline/detection.py — _merge_detections() static method
```

**Result:** 27 tracks → 11 tracks for the same video; no more duplicate identity labels.

---

## High — TrackHistory Memory Leak (H1)

**Bug:** In `TrackManager`, dead tracks were removed from `self.tracks` in all three backends
(`_update_hungarian`, `_update_deepsort`, `_update_botsort`) but never from `self.history`.
`_update_history()` only appended for live tracks. In a 24/7 deployment the history dict grows
without bound.

**Fix:** Added `self.history.pop(tid, None)` alongside every `self.tracks.pop(tid, None)` in
all three dead-track removal sites.

```
src/fas_kd/pipeline/tracking.py — all three _update_* methods
```

---

## High — TrackEmbeddingBuffer Memory Leak (H2)

**Bug:** In `RuntimePipeline.process_frame`, `self._liveness_cache` was pruned for dead tracks
but `self.track_buffers` (up to 64 embeddings each) was never pruned. Embedding buffers for
dead tracks accumulated indefinitely.

**Fix:** Added a matching prune line after the existing liveness cache cleanup:
```python
self.track_buffers = {tid: buf for tid, buf in self.track_buffers.items() if tid in live_track_ids}
```

```
src/fas_kd/pipeline/runtime.py — process_frame()
```

---

## High — YOLO Landmark Fallback Poisons Embeddings (H3)

**Bug:** When YOLO detects a face but produces no keypoints, `_landmarks_from_bbox()` generates
synthetic 5-point landmarks at fixed bounding-box proportions. These fake points were fed into
`FacePreprocessor.align()` which runs `cv2.estimateAffinePartial2D` against the InsightFace
canonical template, producing a completely mangled 112×112 crop. The `MagnitudeQualityGate`
only checks vector norm, not alignment quality, so garbage embeddings passed silently into track
buffers and FAISS.

**Fix (three-part):**
1. Added `landmarks_synthetic: bool = False` field to `FaceDetection` in `types.py`.
2. Set `landmarks_synthetic=True` in the fallback path in `detection.py`.
3. Added `FacePreprocessor.crop_center()` — plain bbox crop + resize, no affine warp.
4. `RuntimePipeline.process_frame()` branches on `det.landmarks_synthetic`: uses `crop_center`
   instead of the affine `align` path.

```
src/fas_kd/pipeline/types.py       — FaceDetection.landmarks_synthetic field
src/fas_kd/pipeline/detection.py   — sets landmarks_synthetic=True in fallback path
src/fas_kd/pipeline/preprocess.py  — FacePreprocessor.crop_center()
src/fas_kd/pipeline/runtime.py     — branches on landmarks_synthetic
```

---

## Medium — Double IoU Greedy Override After Tracker (M3)

**Bug:** After `track_manager.update()` returned tracker-assigned faces (DeepSORT/BoT-SORT with
Kalman + Hungarian), a second **greedy IoU loop** in `process_frame()` reassigned `det_idx →
track_idx` with a 0.3 threshold. This silently overrode the tracker's optimal global assignment
with a suboptimal greedy one, breaking identity continuity on multi-person frames.

**Fix:** Removed the 15-line secondary greedy loop entirely. Instead, each tracker backend now
stamps `track.matched_det_idx` (the detection index it was assigned) on every `TrackedFace`
object. `process_frame()` reads this directly to build the det→track mapping.

Changes:
- Added `matched_det_idx: int | None = None` field to `TrackedFace` in `types.py`.
- All three backends (`_update_hungarian`, `_update_deepsort`, `_update_botsort`) set
  `matched_det_idx` on every track update.
- `process_frame()` builds `det_to_track_id` directly from `track.matched_det_idx`.
- Removed unused `iou_xyxy` import from `runtime.py`.

```
src/fas_kd/pipeline/types.py    — TrackedFace.matched_det_idx field
src/fas_kd/pipeline/tracking.py — matched_det_idx propagation in all backends
src/fas_kd/pipeline/runtime.py  — replaced greedy loop with direct matched_det_idx lookup
```

---

## Pipeline Config — Stale Gallery NPZ and Identity Names

**Bug:** The UHD demo script loaded both `face_db` (current identity source) **and** an old
`pipeline_uhd2560_label_chain_gallery.npz` + `identity_names.json` from a previous labelling
pass. The old files:
- Used conflicting ID numbering → duplicated Sarah (IDs 1000 and 1006) and triple-loaded John
- Overrode correct `face_db` names with wrong ones (ID 1001: Paul → overridden to John)
- Added 3 phantom identity IDs (1006, 1007, 1008) that existed only in the gallery NPZ

**Fix:** Removed `--gallery-npz` and `--identity-names-json` from `run_demo_uhd.sh`. The
`face_db` at `data/face_db/known/` is now the sole authoritative identity source; it holds all
photos, embeddings, and names. Any number of identities can be added to `face_db` — the
pipeline will handle them without configuration changes.

```
scripts/run_demo_uhd.sh — removed --gallery-npz and --identity-names-json flags
```

---

## Annotation — Ghost Boxes for Kalman-Predicted Tracks

**Feature:** When a tracked face is momentarily undetectable (brief occlusion, YOLO miss at
0.08 confidence), DeepSORT keeps the track alive via Kalman prediction but no `FaceObservation`
is generated, so the annotation box vanished from the video.

**Fix:** Added a ghost-box draw pass in the main video loop. After drawing all current
observations, the loop iterates over `pipeline.track_manager.tracks` for any track with a
known identity that has no observation this frame. If `missed_frames ≤ 10`, it draws a
corner-segment style box (not a solid rectangle) with the last known identity name. The box
fades as `missed_frames` increases.

The 10-frame cap is deliberate: Kalman prediction drifts significantly after more than ~10
frames. Drawing a ghost box beyond that wandered off-screen and caused re-appearing faces to
be assigned a new identity instead of the existing track.

Added function: `_draw_ghost_track()` in `scripts/run_face_pipeline.py`.

```
scripts/run_face_pipeline.py — _draw_ghost_track(), ghost-box draw loop in main annotation
```

---

## Summary Table

| ID | Severity | Component | Description | Fix |
|---|---|---|---|---|
| — | Critical | detection.py | YOLO face-age model emits duplicate bboxes per age class; cross-class NMS not applied | Added `_merge_detections()` post-NMS dedup |
| H1 | High | tracking.py | `TrackHistory` never cleaned on track death → OOM leak | `history.pop(tid)` in all 3 backends |
| H2 | High | runtime.py | `TrackEmbeddingBuffer` never pruned for dead tracks → OOM leak | Dict comprehension prune alongside liveness_cache |
| H3 | High | detection.py / preprocess.py | Synthetic bbox-derived landmarks fed to affine aligner → mangled crops pass quality gate | `landmarks_synthetic` flag, `crop_center()` fallback |
| M3 | Medium | runtime.py | Secondary greedy IoU loop overrides tracker's optimal Hungarian/DeepSORT assignment | Removed loop; use `matched_det_idx` from tracker |
| — | Config | run_demo_uhd.sh | Stale gallery NPZ + names JSON conflict with face_db → duplicate/wrong names | Removed both; face_db is sole identity source |
| — | UX | run_face_pipeline.py | Label vanishes when face briefly occluded (DeepSORT coasting, no observation) | Ghost-box draw pass, capped at 10 missed frames |
