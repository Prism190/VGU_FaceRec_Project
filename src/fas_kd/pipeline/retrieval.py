from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ANNResult:
    identity_id: int
    score: float


@dataclass
class IdentityIndex:
    dim: int
    use_faiss: bool = True
    hnsw_m: int = 32
    vectors: list[np.ndarray] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._faiss = None
        self._index = None
        if self.use_faiss:
            try:
                import faiss

                self._faiss = faiss
                self._index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
                self._index.hnsw.efConstruction = 200
                self._index.hnsw.efSearch = 64
            except Exception:
                self._faiss = None
                self._index = None

    def _l2_normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        return x / np.maximum(n, eps)

    def add(self, identity_id: int, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        vec = self._l2_normalize(vec)
        self.ids.append(int(identity_id))
        self.vectors.append(vec[0])
        if self._index is not None:
            self._index.add(vec)

    def search(self, query_embedding: np.ndarray, k: int = 1) -> list[ANNResult]:
        if not self.ids:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = self._l2_normalize(query)

        if self._index is not None:
            dists, idxs = self._index.search(query, k)
            out: list[ANNResult] = []
            for dist, idx in zip(dists[0], idxs[0]):
                if idx < 0:
                    continue
                # HNSWFlat with L2 distance on normalized vectors: sim = 1 - d^2 / 2.
                score = float(max(-1.0, min(1.0, 1.0 - (float(dist) / 2.0))))
                out.append(ANNResult(identity_id=self.ids[int(idx)], score=score))
            return out

        # Numpy fallback.
        mat = np.stack(self.vectors, axis=0)
        sims = mat @ query[0]
        order = np.argsort(-sims)[:k]
        return [ANNResult(identity_id=self.ids[int(i)], score=float(sims[int(i)])) for i in order]
