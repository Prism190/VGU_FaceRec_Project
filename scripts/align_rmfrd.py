#!/usr/bin/env python3
"""Pre-process RMFRD/AFDB images using RetinaFace face alignment.

For each image in both the clean (AFDB_face_dataset) and masked
(AFDB_masked_face_dataset) directories:
  1. Run RetinaFace (det_500m.onnx via insightface buffalo_sc) to get 5-point landmarks
  2. Apply InsightFace ArcFace alignment -> 112×112 BGR crop
  3. Save as PNG to the output directory (same identity/filename structure)

For images where detection fails (too small, low quality, heavy occlusion),
fall back to simple BILINEAR resize to 112×112.

Usage:
  python scripts/align_rmfrd.py \
      --rmfrd-root /tmp/mfr2_data/self-built-masked-face-recognition-dataset \
      --out /tmp/rmfrd_aligned \
      [--insightface-root ~/.insightface] \
      [--det-size 320]   # 320 or 640; 320 is faster
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_detector(insightface_root: str, det_size: int):
    """Return FaceAnalysis detector (detection-only, CPU)."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_sc",
        root=insightface_root,
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _align_one(
    img_path: Path,
    out_path: Path,
    detector,
    fallback_count: list[int],
) -> None:
    """Detect, align, and save. Falls back to resize if detection fails."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        img_pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    faces = detector.get(img_bgr)

    if faces:
        # Pick the face with highest detection score
        face = max(faces, key=lambda f: f.det_score)
        from insightface.utils.face_align import norm_crop
        aligned_bgr = norm_crop(img_bgr, face.kps, image_size=112, mode="arcface")
        # Save as RGB PNG so PIL reads correctly (standard convention for downstream models)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(aligned_rgb).save(str(out_path))
    else:
        # Fallback: simple resize — already RGB from PIL round-trip
        fallback_count[0] += 1
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil.resize((112, 112), Image.BILINEAR).save(str(out_path))


def _process_dir(
    src: Path, dst: Path, detector, desc: str
) -> tuple[int, int]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    fallback_count = [0]
    for img_path in tqdm(all_images, desc=desc):
        rel = img_path.relative_to(src)
        out_path = dst / rel.with_suffix(".png")
        _align_one(img_path, out_path, detector, fallback_count)
    return len(all_images), fallback_count[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaFace-align RMFRD/AFDB images")
    parser.add_argument("--rmfrd-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--insightface-root", default="/home/phongtruong/.insightface")
    parser.add_argument("--det-size", type=int, default=320, choices=[160, 320, 640])
    args = parser.parse_args()

    rmfrd_root = Path(args.rmfrd_root)
    out_root = Path(args.out)

    print(f"[align] Loading RetinaFace detector (det_size={args.det_size})…")
    detector = _load_detector(args.insightface_root, args.det_size)

    for subdir in ["AFDB_face_dataset", "AFDB_masked_face_dataset"]:
        src = rmfrd_root / subdir
        dst = out_root / subdir
        if not src.exists():
            print(f"[align] {src} not found — skipping")
            continue
        n, fb = _process_dir(src, dst, detector, desc=subdir)
        pct = 100 * fb / n if n else 0
        print(f"[align] {subdir}: {n} images, {fb} fallback-resized ({pct:.1f}%)")

    print(f"[align] Done. Aligned images saved to {out_root}")


if __name__ == "__main__":
    main()
