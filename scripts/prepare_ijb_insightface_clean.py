#!/usr/bin/env python3
"""Re-align IJB-B and IJB-C using InsightFace RetinaFace + norm_crop.

This produces better-aligned faces than YOLO11n because RetinaFace was
trained specifically for face landmark detection and gives more precise
5-point landmarks, matching the official MagFace evaluation pipeline.

Usage:
    ./venv/bin/python scripts/prepare_ijb_insightface_clean.py \
        --ijb-root data/raw/ijb/ijb \
        --output-root data/processed/ijb_clean_insightface \
        --device 0          # GPU id; -1 for CPU
        --overwrite         # rebuild even if exists
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_app(device_id: int):
    from insightface.app import FaceAnalysis
    provider = "CUDAExecutionProvider" if device_id >= 0 else "CPUExecutionProvider"
    app = FaceAnalysis(name="buffalo_sc", providers=[provider])
    app.prepare(ctx_id=device_id, det_size=(640, 640))
    return app


def _load_meta_landmarks(name_5pts_path: Path) -> dict[str, np.ndarray]:
    """Parse ijbX_name_5pts_score.txt → {image_name: float32 (5,2)}."""
    lms: dict[str, np.ndarray] = {}
    with name_5pts_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            name = parts[0]
            coords = [float(x) for x in parts[1:11]]
            pts = np.array(coords, dtype=np.float32).reshape(5, 2)
            lms[name] = pts
    return lms


def _norm_crop(img_bgr: np.ndarray, kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """InsightFace standard 5-point affine alignment."""
    from insightface.utils.face_align import norm_crop
    return norm_crop(img_bgr, landmark=kps, image_size=image_size)


def _process_dataset(
    src_dataset: Path,
    dst_dataset: Path,
    app,
    overwrite: bool,
    image_size: int = 112,
) -> dict:
    prefix = src_dataset.name.lower()
    src_loose = src_dataset / "loose_crop"
    src_meta = src_dataset / "meta"
    face_tid_mid_path = src_meta / f"{prefix}_face_tid_mid.txt"
    name_5pts_path = src_meta / f"{prefix}_name_5pts_score.txt"

    if not face_tid_mid_path.exists():
        raise FileNotFoundError(face_tid_mid_path)

    dst_loose = dst_dataset / "loose_crop"
    dst_meta = dst_dataset / "meta"
    dst_loose.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)

    # Copy metadata files unchanged
    for p in src_meta.glob("*"):
        dst = dst_meta / p.name
        if not dst.exists() or overwrite:
            shutil.copy2(p, dst)

    # Load meta landmarks as fallback
    meta_lms = _load_meta_landmarks(name_5pts_path) if name_5pts_path.exists() else {}

    # Load image list from face_tid_mid
    image_names: list[str] = []
    with face_tid_mid_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                image_names.append(parts[0])

    stats = dict(total=len(image_names), retinaface=0, meta_fallback=0, skipped=0)

    for name in tqdm(image_names, desc=f"{prefix.upper()} InsightFace align", unit="img"):
        dst_path = dst_loose / name
        if dst_path.exists() and not overwrite:
            continue

        src_path = src_loose / name
        if not src_path.exists():
            stats["skipped"] += 1
            continue

        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            stats["skipped"] += 1
            continue

        # Try RetinaFace detection
        faces = app.get(img_bgr)
        kps = None
        if faces:
            # Use highest-confidence detection
            face = max(faces, key=lambda f: f.det_score)
            kps = face.kps.astype(np.float32)
            stats["retinaface"] += 1
        elif name in meta_lms:
            kps = meta_lms[name]
            stats["meta_fallback"] += 1
        else:
            stats["skipped"] += 1
            continue

        aligned = _norm_crop(img_bgr, kps, image_size=image_size)
        cv2.imwrite(str(dst_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IJB clean dataset using InsightFace RetinaFace")
    parser.add_argument("--ijb-root", default="data/raw/ijb/ijb")
    parser.add_argument("--output-root", default="data/processed/ijb_clean_insightface")
    parser.add_argument("--device", type=int, default=0, help="GPU id (-1=CPU)")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ijb_root = Path(args.ijb_root)
    if not ijb_root.is_absolute():
        ijb_root = (PROJECT_ROOT / ijb_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    print("Loading InsightFace buffalo_sc model (downloads if not cached)...")
    app = _get_app(args.device)
    print("Model ready.")

    for dataset_name in ["IJBB", "IJBC"]:
        src = ijb_root / dataset_name
        if not src.exists():
            print(f"[skip] {dataset_name} not found at {src}")
            continue
        dst = output_root / dataset_name
        print(f"\n=== {dataset_name} ===")
        stats = _process_dataset(src_dataset=src, dst_dataset=dst, app=app,
                                 overwrite=args.overwrite, image_size=args.image_size)
        total = stats["total"]
        print(f"  total={total}  retinaface={stats['retinaface']} ({100*stats['retinaface']/total:.1f}%)"
              f"  meta_fallback={stats['meta_fallback']} ({100*stats['meta_fallback']/total:.1f}%)"
              f"  skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
