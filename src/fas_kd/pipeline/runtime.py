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
    track_manager: TrackManager = field(default_factory=TrackManager)
    track_buffers: dict[int, TrackEmbeddingBuffer] = field(default_factory=dict)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        detections: list[FaceDetection],
        frame_idx: int,
    ) -> list[FaceObservation]:
        tracks = self.track_manager.update(detections=detections, frame_idx=frame_idx)

        observations: list[FaceObservation] = []
        track_by_bbox = {tuple(t.bbox_xyxy): t for t in tracks}

        for det in detections:
            # Associate detection to most recent track by bbox exact match from this update step.
            track = track_by_bbox.get(tuple(det.bbox_xyxy))
            if track is None:
                continue

            face_rgb = self.preprocess(frame_bgr, det.landmarks5)
            is_live, liveness_score = self.liveness_gate.is_live(face_rgb)

            emb = self.embed_fn(face_rgb)
            emb = np.asarray(emb, dtype=np.float32)
            quality_pass, magnitude = self.quality_gate.evaluate(emb)

            obs = FaceObservation(
                track_id=track.track_id,
                frame_idx=frame_idx,
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
