from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class IncrementalUnknownClusterer:
    eps: float = 0.6
    min_samples: int = 5
    max_buffer_size: int = 10000
    update_every: int = 16
    buffer: list[np.ndarray] = field(default_factory=list)
    _latest_label: int | None = None
    _added_since_update: int = 0
    _cached_labels: np.ndarray | None = None

    def add(self, embedding: np.ndarray) -> None:
        self.buffer.append(np.asarray(embedding, dtype=np.float32))
        overflow = 0
        if len(self.buffer) > self.max_buffer_size:
            overflow = len(self.buffer) - self.max_buffer_size
            self.buffer = self.buffer[overflow:]
            self._cached_labels = None

        if len(self.buffer) < max(1, int(self.min_samples)):
            self._latest_label = None
            self._added_since_update = 0
            return

        self._added_since_update += 1
        need_update = (
            self._cached_labels is None
            or overflow > 0
            or self._added_since_update >= max(1, int(self.update_every))
        )
        if need_update:
            self._cached_labels = self.cluster()
            self._added_since_update = 0

        labels = self._cached_labels
        if labels is None or labels.size != len(self.buffer):
            self._latest_label = None
            return

        latest = int(labels[-1])
        self._latest_label = latest if latest >= 0 else None

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
