#!/usr/bin/env python3
"""Evaluate MBF ONNX on bin protocol datasets under synthetic lower-face occlusion.

Computes both clean and masked accuracy/TAR@FAR for MobileFaceNet (W600K) so we
can directly compare occlusion robustness against our V3/SWA student.

Masking: zeros out pixels below y=0.55*H (same rule as evaluate_bin_occluded.py).

Usage:
    python scripts/evaluate_bin_mbf_occluded.py \
        --onnx ~/.insightface/models/buffalo_sc/w600k_mbf.onnx \
        --bin-root data/raw/casia-webface/faces_webface_112x112 \
        --out results/occlusion/mobilefacenet_w600k.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import roc_curve
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BIN_SETS = {
    "lfw":      "lfw.bin",
    "cfp_fp":   "cfp_fp.bin",
    "agedb_30": "agedb_30.bin",
    "cplfw":    "cplfw.bin",
    "calfw":    "calfw.bin",
}


def _decode_bin_image(item) -> Image.Image:
    if isinstance(item, bytes):
        arr = np.frombuffer(item, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode bytes image")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    arr = np.asarray(item)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr.astype(np.uint8))
    flat = arr.reshape(-1).astype(np.uint8)
    img = cv2.imdecode(flat, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _prep_onnx(pil_img: Image.Image, apply_mask: bool = False) -> np.ndarray:
    """PIL RGB → CHW float32 in [-1,1] BGR, with optional lower-face zeroing."""
    img = pil_img.convert("RGB").resize((112, 112), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)          # H W 3 RGB
    if apply_mask:
        y_start = int(arr.shape[0] * 0.55)
        arr[y_start:, :, :] = 0.0
        # Convert to pixel range for proper normalisation before normalise step
        # (mask is in [0,255] domain at this point — set to 0 and normalise below)
    arr = arr[:, :, ::-1]                          # RGB → BGR (InsightFace convention)
    arr = arr / 127.5 - 1.0                        # → [-1, 1]
    return arr.transpose(2, 0, 1)                  # HWC → CHW


def _best_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.arange(-1.0, 1.001, 0.001)
    best = 0.0
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        acc = float((preds == labels).mean())
        if acc > best:
            best = acc
    return best


def _tar_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> float:
    mask = np.isfinite(scores)
    s, l = scores[mask], labels[mask]
    if len(np.unique(l)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(l, s, pos_label=1)
    idx = np.where(fpr <= target_far)[0]
    return float(tpr[idx[-1]]) if idx.size > 0 else 0.0


def _run_bin(sess, input_name, output_name, bin_path: Path, apply_mask: bool,
             batch_size: int = 256) -> dict:
    with bin_path.open("rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    labels = np.asarray(issame, dtype=np.int32)
    n = len(labels)

    # Build flat list: image_0a, image_0b, image_1a, image_1b, ...
    all_arrs = []
    for i in tqdm(range(n), desc=f"{'masked' if apply_mask else 'clean':6} {bin_path.stem}", leave=False):
        pil_a = _decode_bin_image(bins[2 * i])
        pil_b = _decode_bin_image(bins[2 * i + 1])
        all_arrs.append(_prep_onnx(pil_a, apply_mask))
        all_arrs.append(_prep_onnx(pil_b, apply_mask))

    all_arrs_np = np.stack(all_arrs, axis=0).astype(np.float32)  # (2N, 3, 112, 112)
    total = len(all_arrs_np)

    embeddings = []
    for start in range(0, total, batch_size):
        batch = all_arrs_np[start:start + batch_size]
        emb = sess.run([output_name], {input_name: batch})[0]
        # Flip TTA
        flipped = batch[:, :, :, ::-1].copy()
        emb_flip = sess.run([output_name], {input_name: flipped})[0]
        emb = emb + emb_flip
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)
        embeddings.append(emb.astype(np.float32))

    embeddings = np.concatenate(embeddings, axis=0)  # (2N, 512)
    ea = embeddings[0::2]   # (N, 512)
    eb = embeddings[1::2]   # (N, 512)
    scores = (ea * eb).sum(axis=1)

    return {
        "accuracy":      _best_accuracy(scores, labels),
        "tar_far_1e-3":  _tar_at_far(scores, labels, 1e-3),
        "tar_far_1e-4":  _tar_at_far(scores, labels, 1e-4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="/home/phongtruong/.insightface/models/buffalo_sc/w600k_mbf.onnx")
    parser.add_argument("--bin-root", default="data/raw/casia-webface/faces_webface_112x112")
    parser.add_argument("--out", default="results/occlusion/mobilefacenet_w600k.json")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    bin_root = Path(args.bin_root)
    if not bin_root.is_absolute():
        bin_root = (PROJECT_ROOT / bin_root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    inp  = sess.get_inputs()[0].name
    outp = sess.get_outputs()[0].name
    print(f"[mbf-occ] Provider: {sess.get_providers()[0]}")

    results: dict[str, dict] = {}
    for ds_name, fname in BIN_SETS.items():
        bin_path = bin_root / fname
        if not bin_path.exists():
            print(f"  [skip] {ds_name}")
            continue
        print(f"\n{ds_name}")
        clean  = _run_bin(sess, inp, outp, bin_path, apply_mask=False,  batch_size=args.batch_size)
        masked = _run_bin(sess, inp, outp, bin_path, apply_mask=True,   batch_size=args.batch_size)
        drop_acc = clean["accuracy"]     - masked["accuracy"]
        drop_t3  = clean["tar_far_1e-3"] - masked["tar_far_1e-3"]
        drop_t4  = clean["tar_far_1e-4"] - masked["tar_far_1e-4"]
        print(f"  clean acc={clean['accuracy']:.4f}  masked acc={masked['accuracy']:.4f}  drop={drop_acc:+.4f}")
        print(f"  clean TAR@1e-3={clean['tar_far_1e-3']:.4f}  masked={masked['tar_far_1e-3']:.4f}  drop={drop_t3:+.4f}")
        results[ds_name] = {
            "clean_acc":    clean["accuracy"],
            "masked_acc":   masked["accuracy"],
            "drop_acc":     drop_acc,
            "clean_tar_1e3":  clean["tar_far_1e-3"],
            "masked_tar_1e3": masked["tar_far_1e-3"],
            "drop_tar_1e3":   drop_t3,
            "clean_tar_1e4":  clean["tar_far_1e-4"],
            "masked_tar_1e4": masked["tar_far_1e-4"],
            "drop_tar_1e4":   drop_t4,
        }

    out_path.write_text(json.dumps({"MobileFaceNet_W600K": results}, indent=2))
    print(f"\n[mbf-occ] Saved to {out_path}")

    # Print summary table
    print("\n=== Occlusion TAR@1e-3 drop summary ===")
    print(f"{'dataset':<12}  {'clean':>7}  {'masked':>7}  {'drop':>7}")
    for ds, v in results.items():
        print(f"{ds:<12}  {v['clean_tar_1e3']:7.4f}  {v['masked_tar_1e3']:7.4f}  {v['drop_tar_1e3']:+7.4f}")


if __name__ == "__main__":
    main()
