from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceDetection:
    bbox_xyxy: tuple[float, float, float, float]
    landmarks5: np.ndarray
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedFace:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    landmarks5: np.ndarray
    last_frame_idx: int
    missed_frames: int = 0


@dataclass
class FaceObservation:
    track_id: int
    frame_idx: int
    embedding: np.ndarray
    magnitude: float
    liveness_score: float
    is_live: bool
    quality_pass: bool
