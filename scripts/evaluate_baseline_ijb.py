#!/usr/bin/env python3
"""Evaluate a pretrained baseline model (ONNX) on IJB-B/C template verification.

Supports the insightface buffalo_sc MobileFaceNet (w600k_mbf.onnx) and any
ArcFace-family ONNX model that expects 112x112 BGR [-1, 1] input.

Usage:
  python scripts/evaluate_baseline_ijb.py \
      --model-path ~/.insightface/models/buffalo_sc/w600k_mbf.onnx \
      --dataset IJBB \
      --ijb-root data/processed/ijb_clean_insightface/IJBB \
      --template-pooling magface_weighted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.evaluation.ijb_template import (
    _build_template_features,
    _far_key,
    _load_face_tid_mid,
    _score_template_pairs,
    _tar_at_far,
)
from sklearn.metrics import roc_auc_score, roc_curve


class _OnnxExtractor:
    """Wraps an ArcFace-family ONNX model for batch embedding extraction."""

    def __init__(self, model_path: str, use_flip: bool = True) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.use_flip = use_flip

    def _preprocess(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize((112, 112), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR (insightface convention)
        arr = arr / 127.5 - 1.0
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def extract(self, image_paths: list[Path], batch_size: int = 64) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Baseline embedding", leave=False):
            batch_paths = image_paths[start : start + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p)
                imgs.append(self._preprocess(img))

            batch = np.stack(imgs, axis=0).astype(np.float32)
            emb = self.sess.run([self.output_name], {self.input_name: batch})[0]

            if self.use_flip:
                flipped = batch[:, :, :, ::-1].copy()
                emb_flip = self.sess.run([self.output_name], {self.input_name: flipped})[0]
                emb = emb + emb_flip

            # L2 normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / (norms + 1e-12)
            all_embeddings.append(emb.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="IJB baseline eval using ONNX model")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model file")
    parser.add_argument("--model-name", default="baseline", help="Label for output")
    parser.add_argument("--dataset", choices=["IJBB", "IJBC"], default="IJBB")
    parser.add_argument("--ijb-root", required=True, help="Path to IJBB or IJBC directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--template-pooling",
        choices=["mean", "magface_weighted", "top5", "top10"],
        default="magface_weighted",
    )
    parser.add_argument("--no-flip", action="store_true", help="Disable flip augmentation")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    ijb_root = Path(args.ijb_root)
    prefix = ijb_root.name.lower()
    meta_root = ijb_root / "meta"
    face_tid_mid_path = meta_root / f"{prefix}_face_tid_mid.txt"
    template_pair_label_path = meta_root / f"{prefix}_template_pair_label.txt"
    loose_crop_root = ijb_root / "loose_crop"

    image_paths, face_tids, face_mids, missing_images = _load_face_tid_mid(
        face_tid_mid_path=face_tid_mid_path,
        loose_crop_root=loose_crop_root,
    )
    print(f"[baseline] {len(image_paths)} images ({missing_images} missing)")

    extractor = _OnnxExtractor(model_path=args.model_path, use_flip=not args.no_flip)
    face_embeddings = extractor.extract(image_paths, batch_size=args.batch_size)
    print(f"[baseline] embeddings: {face_embeddings.shape}")

    template_features = _build_template_features(
        face_tids=face_tids,
        face_mids=face_mids,
        face_embeddings=face_embeddings,
        pooling_mode=args.template_pooling,
    )

    scores, labels, total_pairs, missing_t1, missing_t2 = _score_template_pairs(
        template_pair_label_path=template_pair_label_path,
        template_features=template_features,
    )

    non_finite_mask = ~np.isfinite(scores)
    num_non_finite = int(non_finite_mask.sum())
    if num_non_finite:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

    target_fars = [1e-3, 1e-4, 1e-5]
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(roc_auc_score(labels, scores))

    out: dict = {
        "model": args.model_name,
        "protocol": prefix,
        "template_pooling": args.template_pooling,
        "num_faces": len(image_paths),
        "num_templates": len(template_features),
        "num_pairs_total": total_pairs,
        "num_pairs_scored": int(scores.size),
        "missing_face_images": missing_images,
        "roc_auc": roc_auc,
        "num_scores_non_finite": num_non_finite,
    }
    for far in target_fars:
        out[_far_key(float(far))] = _tar_at_far(fpr=fpr, tpr=tpr, target_far=float(far))

    print(json.dumps(out, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[baseline] saved to {args.out}")


if __name__ == "__main__":
    main()
