from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class IncrementalUnknownClusterer:
    eps: float = 0.6
    min_samples: int = 5
    max_buffer_size: int = 10000
    buffer: list[np.ndarray] = field(default_factory=list)
    _latest_label: int | None = None

    def add(self, embedding: np.ndarray) -> None:
        self.buffer.append(np.asarray(embedding, dtype=np.float32))
        if len(self.buffer) > self.max_buffer_size:
            overflow = len(self.buffer) - self.max_buffer_size
            self.buffer = self.buffer[overflow:]

        # Keep this lightweight for streaming use; final labels come from cluster().
        self._latest_label = None

    def latest_label(self) -> int | None:
        return self._latest_label

    def __len__(self) -> int:
        return len(self.buffer)

    def cluster(self) -> np.ndarray:
        if not self.buffer:
            return np.asarray([], dtype=np.int32)
        X = np.stack(self.buffer, axis=0)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit_predict(X)
        return labels.astype(np.int32)
