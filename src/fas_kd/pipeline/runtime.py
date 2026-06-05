from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .aggregation import TrackEmbeddingBuffer
from .liveness import ThresholdLivenessGate
from .preprocess import FacePreprocessor
from .quality_gate import MagnitudeQualityGate
from .tracking import TrackManager
from .types import FaceDetection, FaceObservation


@dataclass
class RuntimePipeline:
    preprocess: FacePreprocessor
    liveness_gate: ThresholdLivenessGate
    quality_gate: MagnitudeQualityGate
    embed_fn: Callable[[np.ndarray], np.ndarray]
    liveness_interval_frames: int = 0
    # Require this many liveness evaluations (spaced by liveness_interval_frames) to agree
    # before marking a track as live. Prevents a single spoofed frame from caching is_live=True
    # for the entire window. Set to 1 to restore the original single-evaluation behaviour.
    liveness_confirm_frames: int = 3
    track_manager: TrackManager = field(default_factory=TrackManager)
    track_buffers: dict[int, TrackEmbeddingBuffer] = field(default_factory=dict)
    # Cache stores (last_eval_frame_idx, [recent_liveness_scores]) for rolling confirmation.
    _liveness_cache: dict[int, tuple[int, list[float]]] = field(default_factory=dict)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        detections: list[FaceDetection],
        frame_idx: int,
    ) -> list[FaceObservation]:
        face_rgbs: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        for det in detections:
            if det.landmarks_synthetic:
                face_rgb = self.preprocess.crop_center(frame_bgr, det.bbox_xyxy)
            else:
                face_rgb = self.preprocess(frame_bgr, det.landmarks5)
            emb = self.embed_fn(face_rgb)
            emb = np.asarray(emb, dtype=np.float32)
            face_rgbs.append(face_rgb)
            embeddings.append(emb)

        tracks = self.track_manager.update(
            detections=detections,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            detection_embeddings=embeddings,
        )
        live_track_ids = {int(t.track_id) for t in tracks}

        # Liveness cache and embedding buffers are only meaningful for currently active tracks.
        self._liveness_cache = {tid: st for tid, st in self._liveness_cache.items() if tid in live_track_ids}
        self.track_buffers = {tid: buf for tid, buf in self.track_buffers.items() if tid in live_track_ids}

        # Use the tracker's own det→track assignment; no secondary greedy override.
        det_to_track_id: dict[int, int] = {}
        for track in tracks:
            if track.matched_det_idx is not None:
                det_to_track_id[track.matched_det_idx] = track.track_id
        tracks_by_id = {t.track_id: t for t in tracks}

        observations: list[FaceObservation] = []
        for det_idx, det in enumerate(detections):
            track_id = det_to_track_id.get(det_idx)
            if track_id is None:
                continue
            track = tracks_by_id.get(track_id)
            if track is None:
                continue

            face_rgb = face_rgbs[det_idx]
            interval = max(0, int(self.liveness_interval_frames))
            confirm = max(1, int(self.liveness_confirm_frames))
            cached = self._liveness_cache.get(track.track_id)
            if interval > 0 and cached is not None and (frame_idx - int(cached[0])) < interval:
                # Reuse cached rolling scores without re-evaluating.
                scores = cached[1]
            else:
                # Time to re-evaluate: get raw score and append to rolling window.
                _, raw_score = self.liveness_gate.is_live(face_rgb)
                prev_scores = cached[1] if cached is not None else []
                scores = (prev_scores + [raw_score])[-confirm:]
                self._liveness_cache[track.track_id] = (int(frame_idx), scores)
            mean_score = sum(scores) / len(scores) if scores else 0.0
            is_live = (len(scores) >= confirm) and (mean_score >= self.liveness_gate.live_threshold)
            liveness_score = mean_score

            emb = embeddings[det_idx]
            quality_pass, magnitude = self.quality_gate.evaluate(emb)

            obs = FaceObservation(
                track_id=track.track_id,
                frame_idx=frame_idx,
                bbox_xyxy=tuple(float(v) for v in det.bbox_xyxy),
                embedding=emb,
                magnitude=magnitude,
                liveness_score=liveness_score,
                is_live=is_live,
                quality_pass=quality_pass,
            )
            observations.append(obs)

            if is_live and quality_pass:
                buf = self.track_buffers.setdefault(track.track_id, TrackEmbeddingBuffer(max_size=64))
                buf.push(embedding=emb, magnitude=magnitude)

        return observations

    def pooled_track_embedding(self, track_id: int) -> np.ndarray | None:
        buf = self.track_buffers.get(track_id)
        if buf is None:
            return None
        return buf.pooled()
