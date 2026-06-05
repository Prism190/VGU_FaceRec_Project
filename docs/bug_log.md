# Bug Log

Full list of known issues. Status column reflects current state of the codebase.

Legend: ✅ Fixed | ⬜ Open | 📝 Doc-only

---

## Pipeline — Detection / Embedding Quality

| # | Status | Severity | Issue | File |
|---|---|---|---|---|
| — | ✅ | Critical | **YOLO cross-class duplicate detections** — `yolo11n-face-age.pt` runs per-age-class NMS; a face detected as both "young" and "adult" produces two boxes (IoU ≈ 0.97–0.99) that survive YOLO's internal NMS. Caused 27 tracks for 4 people. `merge_iou_thres` config field existed but dedup code was never written. Fixed by `_merge_detections()` in `detect()`. | `pipeline/detection.py` |
| 1 | ✅ | High | **YOLO landmark fallback poisons embeddings** — fake bbox-proportion landmarks fed to affine aligner → mangled 112×112 crop → garbage embeddings bypass quality gate silently. Fixed with `landmarks_synthetic` flag + `crop_center()` fallback path. | `pipeline/detection.py`, `pipeline/preprocess.py`, `pipeline/runtime.py` |
| 2 | ✅ | Medium | **Double IoU greedy loop overrides tracker** — after DeepSORT/BoT-SORT returns Kalman+Hungarian-optimised tracks, a second greedy pass re-assigns by arbitrary detection array order, orphaning faces. Fixed by removing the loop; tracker now stamps `matched_det_idx` on each `TrackedFace`. | `pipeline/runtime.py` |
| 3 | ⬜ | Medium | **MSE KD forces student to fake quality** — `mse_kd_loss` compares raw (unnormalized) student/teacher embeddings; student gets masked input (low quality) but is penalised for not matching teacher's high-magnitude vector; degrades quality-magnitude correlation in the trained student. | `losses/kd.py` |

---

## Pipeline — Liveness / Tracking

| # | Status | Severity | Issue | File |
|---|---|---|---|---|
| 4 | ✅ | Medium | **Liveness cache free-pass exploit** — single `is_live=True` locked result for N frames with no confidence averaging; a spoof that passed once got unlimited free passes during the cache window. Fixed with rolling score buffer (`liveness_confirm_frames=3`). | `pipeline/runtime.py` |
| 20 | ⬜ | Medium | **DBSCAN label volatility** — full DBSCAN rerun from scratch every 16 frames; cluster IDs reshuffled arbitrarily each time; `latest_label()` cannot reliably track a stranger's identity across frames. | `pipeline/clustering.py` |

---

## Memory Leaks (OOM in Deployment)

| # | Status | Severity | Issue | File |
|---|---|---|---|---|
| 6 | ✅ | High | **TrackHistory never cleaned** — dead tracks popped from `self.tracks` but `self.history` accumulates indefinitely; OOM on 24/7 stream. Fixed with `history.pop(tid, None)` in all three tracker backends. | `pipeline/tracking.py` |
| 7 | ✅ | High | **TrackEmbeddingBuffer never cleaned** — `self._liveness_cache` pruned for dead tracks, `self.track_buffers` not; up to 64 embeddings per dead track leak forever. Fixed with dict comprehension prune alongside liveness cache cleanup. | `pipeline/runtime.py` |
| 8 | ✅ | Medium | **RTSP frame buffer grows without bound** — synchronous `cv2.VideoCapture` at 30fps stream with ~10fps pipeline throughput accumulates stale frames; Kalman filters assume consistent dt and break on dropped frames. Fixed with background grab thread + 2-slot bounded queue for live sources. | `scripts/run_face_pipeline.py` |

---

## Training Quality

| # | Status | Severity | Issue | File |
|---|---|---|---|---|
| 9 | ✅ | Medium | **Autograd graph built through augmentations** — `_apply_training_augmentations_batch()` called outside `torch.no_grad()`; PyTorch traces through `F.conv2d`, `torch.exp`, `torch.where` for every step; wastes VRAM and backward time. Fixed by wrapping call in `with torch.no_grad()`. Measured speedup: 79ms → 38ms per batch (2.08×). | `engine/train.py` |
| 11 | ✅ | Medium | **Spatial KD + masking contradiction** (root cause of Phase 2 failure) — spatial MSE enforced between student and teacher intermediate maps while student's lower 45% is a black square; student cannot reconstruct occluded jaw/chin features; contradictory gradients; explains `best.pt` at epoch 9. Fixed by gating `use_spatial_kd` on `active_mask_prob == 0.0` at the call site. | `engine/train.py` |
| 5 | ⬜ | Medium | **Scheduler resume bug** — if optimizer fails to load (e.g. cross-phase resume), `start_epoch` resets to 0 but scheduler still loads its end-of-training state → epoch 0 training at 1e-6 LR. | `engine/train.py ~L850` |
| 10 | ⬜ | Medium | **Batch-uniform augmentation** — motion blur angle sampled once per batch (same kernel for all 256 images); Gaussian sigma constant for the entire epoch; destroys intra-batch variance and skews BatchNorm stats at epoch boundaries. | `engine/train.py` |
| 12 | ⬜ | Low | **LR decay / curriculum collision** — Phase 1 `mask_free_epochs: 20` and LR milestone at epoch 20 fire simultaneously; hardest data arrives exactly when the network has the lowest plasticity. | Phase 1 config |
| 13 | ⬜ | Low | **MagFace π-fallback missing** — `MagFaceHead` has no guard for θ + m > π; when fired, loss rewards hard negatives; only affects early training when cosine ≈ −0.9; trained models unaffected. | `models/margin_head.py` |
| 14 | ⬜ | Low | **DDP seed duplication** — global `seed: 3407` not offset by `local_rank`; if DALI initialises all ranks with identical seed, all GPUs apply identical augmentations; effective variance = 1 GPU. (Unverified — needs DALI seed path check.) | All phase configs |

---

## Infrastructure / Performance

| # | Status | Severity | Issue | File |
|---|---|---|---|---|
| 15 | ⬜ | Medium | **DDP + DALI potential deadlock** — per-step cross-rank sync removed to avoid DALI step-count drift, but end-of-epoch `dist.all_reduce` kept; if step counts actually drift, all ranks hard-deadlock. | `engine/train.py ~L522, ~L547` |
| 16 | ⬜ | Low | **DALI RAM-hog path extraction** — full 5.8M-row `TrainKDDataset` instantiated (minutes, ~10GB RAM) just to read `samples[0]["rec_path"]` for DALI; dataset immediately discarded. | `engine/train.py ~L584–629` |
| 17 | ⬜ | Low | **DALI multiplexer uses tensor arithmetic** — `_multiplexing()` uses `condition * true_case + neg_condition * false_case` instead of `fn.multiplex`; forces full-batch multiply+add on every augmentation branch regardless of whether it's active. | `data/dali_loader.py` |
| 18 | ⬜ | Low | **RecordIO concurrency workaround** — `recordio_decode_retries: 64` is a brute-force retry loop masking a race condition from shared `MXIndexedRecordIO` file pointers across forked workers; burns CPU. | All phase configs |
| 19 | ⬜ | Low | **run_final_eval_suite.sh CUDA OOM trap** — no `CUDA_VISIBLE_DEVICES` set; two background IJB jobs + foreground bin eval compete for `cuda:0`; single-GPU machines OOM immediately. | `scripts/run_final_eval_suite.sh` |

---

## Documentation Fixes

| # | Status | Issue | File |
|---|---|---|---|
| 21 | ✅ | **Asymmetric distillation was always active** — `dali_true_asymmetry: true` is dead config (flag never read anywhere in source); `teacher(clear)` / `student(masked)` is hardcoded in `train.py` for all three phases. Phase 3's real differentiators are softer blur curriculum + SWA. | `docs/training_phases.md` |
| 22 | ✅ | **Phase 2 failure explanation incomplete** — doc cited "overtraining on masks"; root cause is spatial KD vs masked input contradiction (bug #11). | `docs/training_phases.md` |
| 23 | ✅ | **`--det-conf 0.08` needs production warning** — every false positive at 8% confidence is fully embedded, tracked, and liveness-checked before quality gate; destroys FPS in live deployment. | `docs/face_labeling_and_ijb_clean_eval_commands.md` |
| 24 | ✅ | **Phase comparison table wrong** — Asymmetric distillation row shows ✗/✗/✓; should be ✓/✓/✓. | `docs/training_phases.md` |

---

## Confirmed Not Bugs

| Claim | Verdict |
|---|---|
| MagFace retrieval paradox (raw embeddings in FAISS) | False — L2-normalised before both insert and search |
| Best checkpoint amnesia on resume | False — `best_metric` explicitly loaded from checkpoint |
