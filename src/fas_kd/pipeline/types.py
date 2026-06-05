from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceDetection:
    bbox_xyxy: tuple[float, float, float, float]
    landmarks5: np.ndarray
    score: float
    landmarks_synthetic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedFace:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    landmarks5: np.ndarray
    last_frame_idx: int
    missed_frames: int = 0
    matched_det_idx: int | None = None


@dataclass
class FaceObservation:
    track_id: int
    frame_idx: int
    bbox_xyxy: tuple[float, float, float, float]
    embedding: np.ndarray
    magnitude: float
    liveness_score: float
    is_live: bool
    quality_pass: bool
