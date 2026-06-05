#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.pipeline import (
    FaceDetection,
    FacePreprocessor,
    IdentityIndex,
    IncrementalUnknownClusterer,
    MagnitudeQualityGate,
    PreprocessConfig,
    RuntimePipeline,
    ThresholdLivenessGate,
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _test_preprocess() -> None:
    h, w = 240, 320
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 60
    frame[:, :, 1] = 100
    frame[:, :, 2] = 140

    landmarks = np.array(
        [[130, 90], [190, 90], [160, 120], [138, 150], [182, 150]],
        dtype=np.float32,
    )

    pre = FacePreprocessor(PreprocessConfig(image_size=112, use_clahe=True))
    out = pre(frame, landmarks)

    _assert(out.shape == (112, 112, 3), f"unexpected preprocess output shape: {out.shape}")
    _assert(out.dtype == np.uint8, f"unexpected preprocess dtype: {out.dtype}")


def _test_runtime_and_pooling() -> None:
    def embed(face_rgb: np.ndarray) -> np.ndarray:
        base = float(np.mean(face_rgb)) / 255.0
        rng = np.random.default_rng(int(base * 1_000_000) + 13)
        emb = rng.standard_normal(512).astype(np.float32)
        emb /= max(np.linalg.norm(emb), 1e-8)
        emb *= 35.0
        return emb

    def liveness(_: np.ndarray) -> float:
        return 0.95

    pipeline = RuntimePipeline(
        preprocess=FacePreprocessor(PreprocessConfig(image_size=112, use_clahe=False)),
        liveness_gate=ThresholdLivenessGate(infer_fn=liveness, live_threshold=0.5, use_ttda=False),
        quality_gate=MagnitudeQualityGate(min_magnitude=10.0, max_magnitude=120.0),
        embed_fn=embed,
        liveness_interval_frames=10,
    )

    for frame_idx in range(4):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, :] = 120
        cx = 150 + frame_idx * 2
        detection = FaceDetection(
            bbox_xyxy=(float(cx - 40), 70.0, float(cx + 40), 180.0),
            landmarks5=np.array(
                [
                    [cx - 16.0, 102.0],
                    [cx + 16.0, 102.0],
                    [cx + 0.0, 124.0],
                    [cx - 14.0, 148.0],
                    [cx + 14.0, 148.0],
                ],
                dtype=np.float32,
            ),
            score=0.99,
        )
        observations = pipeline.process_frame(frame_bgr=frame, detections=[detection], frame_idx=frame_idx)
        _assert(len(observations) == 1, "runtime should produce exactly one observation")
        _assert(observations[0].quality_pass, "quality gate should pass synthetic embedding")
        _assert(observations[0].is_live, "liveness gate should pass synthetic frame")

    pooled = pipeline.pooled_track_embedding(track_id=1)
    _assert(pooled is not None, "pooled embedding should exist for track 1")
    _assert(pooled.shape == (512,), f"unexpected pooled shape: {pooled.shape}")


def _test_retrieval_and_clustering() -> None:
    dim = 512
    index = IdentityIndex(dim=dim, use_faiss=False)

    rng = np.random.default_rng(3407)
    e1 = rng.standard_normal(dim).astype(np.float32)
    e1 /= max(np.linalg.norm(e1), 1e-8)
    e2 = rng.standard_normal(dim).astype(np.float32)
    e2 /= max(np.linalg.norm(e2), 1e-8)

    index.add(identity_id=101, embedding=e1)
    index.add(identity_id=202, embedding=e2)

    q = e1 + 0.01 * rng.standard_normal(dim).astype(np.float32)
    results = index.search(q, k=1)
    _assert(len(results) == 1, "identity index should return one hit")
    _assert(results[0].identity_id == 101, f"expected id 101, got {results[0].identity_id}")

    clusterer = IncrementalUnknownClusterer(eps=0.2, min_samples=3, max_buffer_size=200)
    c1 = e1.copy()
    c2 = e2.copy()
    for _ in range(20):
        p1 = c1 + 0.01 * rng.standard_normal(dim).astype(np.float32)
        p2 = c2 + 0.01 * rng.standard_normal(dim).astype(np.float32)
        p1 /= max(np.linalg.norm(p1), 1e-8)
        p2 /= max(np.linalg.norm(p2), 1e-8)
        clusterer.add(p1)
        clusterer.add(p2)

    labels = clusterer.cluster()
    _assert(labels.shape[0] == 40, f"unexpected label count: {labels.shape[0]}")
    non_noise = labels[labels >= 0]
    _assert(non_noise.size > 0, "clusterer should produce at least one non-noise assignment")


def main() -> None:
    _test_preprocess()
    print("[ok] preprocess")
    _test_runtime_and_pooling()
    print("[ok] runtime + pooling")
    _test_retrieval_and_clustering()
    print("[ok] retrieval + clustering")
    print("PIPELINE SELFCHECK PASSED")


if __name__ == "__main__":
    main()
