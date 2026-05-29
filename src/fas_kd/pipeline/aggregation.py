from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def magnitude_weighted_pool(embeddings: np.ndarray, magnitudes: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    if magnitudes.ndim != 1 or magnitudes.shape[0] != embeddings.shape[0]:
        raise ValueError("Magnitude vector shape mismatch")

    weights = np.maximum(magnitudes, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= eps:
        pooled = np.mean(embeddings, axis=0)
    else:
        pooled = np.sum(embeddings * weights[:, None], axis=0) / (weight_sum + eps)

    norm = np.linalg.norm(pooled, ord=2)
    if norm <= eps:
        return pooled
    return pooled / norm


@dataclass
class TrackEmbeddingBuffer:
    max_size: int = 64
    embeddings: deque[np.ndarray] = field(default_factory=deque)
    magnitudes: deque[float] = field(default_factory=deque)

    def push(self, embedding: np.ndarray, magnitude: float) -> None:
        if len(self.embeddings) >= self.max_size:
            self.embeddings.popleft()
            self.magnitudes.popleft()
        self.embeddings.append(np.asarray(embedding, dtype=np.float32))
        self.magnitudes.append(float(magnitude))

    def pooled(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        embs = np.stack(list(self.embeddings), axis=0)
        mags = np.asarray(list(self.magnitudes), dtype=np.float32)
        return magnitude_weighted_pool(embs, mags)
