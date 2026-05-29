from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MagnitudeQualityGate:
    min_magnitude: float = 20.0
    max_magnitude: float = 120.0

    def evaluate(self, embedding: np.ndarray) -> tuple[bool, float]:
        magnitude = float(np.linalg.norm(embedding, ord=2))
        ok = self.min_magnitude <= magnitude <= self.max_magnitude
        return ok, magnitude
