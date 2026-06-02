from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .aggregation import TrackEmbeddingBuffer
from .liveness import ThresholdLivenessGate
from .preprocess import FacePreprocessor
from .quality_gate import MagnitudeQualityGate
from .tracking import TrackManager, iou_xyxy
from .types import FaceDetection, FaceObservation


@dataclass
class RuntimePipeline:
    preprocess: FacePreprocessor
    liveness_gate: ThresholdLivenessGate
    quality_gate: MagnitudeQualityGate
    embed_fn: Callable[[np.ndarray], np.ndarray]
    liveness_interval_frames: int = 0
    track_manager: TrackManager = field(default_factory=TrackManager)
    track_buffers: dict[int, TrackEmbeddingBuffer] = field(default_factory=dict)
    _liveness_cache: dict[int, tuple[int, bool, float]] = field(default_factory=dict)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        detections: list[FaceDetection],
        frame_idx: int,
    ) -> list[FaceObservation]:
        face_rgbs: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        for det in detections:
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

        # Liveness cache is only meaningful for currently active tracks.
        self._liveness_cache = {tid: st for tid, st in self._liveness_cache.items() if tid in live_track_ids}

        observations: list[FaceObservation] = []
        det_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for det_idx, det in enumerate(detections):
            best_track_idx = -1
            best_iou = 0.3
            for track_idx, track in enumerate(tracks):
                if track_idx in used_tracks:
                    continue
                score = iou_xyxy(track.bbox_xyxy, det.bbox_xyxy)
                if score > best_iou:
                    best_iou = score
                    best_track_idx = track_idx
            if best_track_idx >= 0:
                used_tracks.add(best_track_idx)
                det_to_track[det_idx] = best_track_idx

        for det_idx, det in enumerate(detections):
            track_idx = det_to_track.get(det_idx)
            if track_idx is None:
                continue
            track = tracks[track_idx]

            face_rgb = face_rgbs[det_idx]
            interval = max(0, int(self.liveness_interval_frames))
            cached = self._liveness_cache.get(track.track_id)
            if interval > 0 and cached is not None and (frame_idx - int(cached[0])) < interval:
                is_live = bool(cached[1])
                liveness_score = float(cached[2])
            else:
                is_live, liveness_score = self.liveness_gate.is_live(face_rgb)
                self._liveness_cache[track.track_id] = (int(frame_idx), bool(is_live), float(liveness_score))

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
