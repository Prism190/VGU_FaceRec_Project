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
    MagnitudeQualityGate,
    PreprocessConfig,
    RuntimePipeline,
    ThresholdLivenessGate,
)


def mock_embed(face_rgb: np.ndarray) -> np.ndarray:
    flat = face_rgb.astype(np.float32).reshape(-1)
    rng = np.random.default_rng(int(float(np.mean(flat)) * 1000.0) % 1_000_000)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= max(np.linalg.norm(emb), 1e-8)
    emb *= 40.0
    return emb


def mock_liveness(face_rgb: np.ndarray) -> float:
    del face_rgb
    return 0.9


def main() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dummy landmarks near center.
    landmarks = np.array(
        [
            [290.0, 220.0],
            [350.0, 220.0],
            [320.0, 260.0],
            [295.0, 300.0],
            [345.0, 300.0],
        ],
        dtype=np.float32,
    )

    detection = FaceDetection(
        bbox_xyxy=(260.0, 180.0, 380.0, 340.0),
        landmarks5=landmarks,
        score=0.99,
    )

    pipeline = RuntimePipeline(
        preprocess=FacePreprocessor(PreprocessConfig(image_size=112, use_clahe=True)),
        liveness_gate=ThresholdLivenessGate(infer_fn=mock_liveness, live_threshold=0.5, use_ttda=True),
        quality_gate=MagnitudeQualityGate(min_magnitude=20.0, max_magnitude=120.0),
        embed_fn=mock_embed,
    )

    observations = pipeline.process_frame(frame_bgr=frame, detections=[detection], frame_idx=0)
    print(f"observations={len(observations)}")
    if observations:
        obs = observations[0]
        print(
            f"track_id={obs.track_id} live={obs.is_live} quality={obs.quality_pass} "
            f"mag={obs.magnitude:.3f} liveness={obs.liveness_score:.3f}"
        )
        pooled = pipeline.pooled_track_embedding(obs.track_id)
        print(f"pooled_embedding_shape={None if pooled is None else pooled.shape}")


if __name__ == "__main__":
    main()
