#!/usr/bin/env python3
"""Evaluate on Real-World Masked Face Dataset (RMFRD / AFDB).

Protocol:
  - Gallery: clean face crops from AFDB_face_dataset/ (up to --gallery-per-id images per identity)
  - Probes: all masked face crops from AFDB_masked_face_dataset/
  - Pairs: positive = same identity in both; negative = cross-identity random sample
  - Reports TAR@FAR (1:1 verification) and Rank-1 (1:N identification)

Usage:
  python scripts/evaluate_rmfrd.py \
      --rmfrd-root /tmp/mfr2_data/self-built-masked-face-recognition-dataset \
      --config configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml \
      --checkpoint checkpoints/release/mobilenetv4_student_phase3_swa.pt \
      --out results/rmfrd_phase3_swa.json

  # For baseline ONNX model:
  python scripts/evaluate_rmfrd.py \
      --rmfrd-root /tmp/mfr2_data/self-built-masked-face-recognition-dataset \
      --onnx-model ~/.insightface/models/buffalo_sc/w600k_mbf.onnx \
      --out results/rmfrd_mbf_baseline.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_rmfrd(rmfrd_root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Return (clean_by_id, masked_by_id) for paired identities only."""
    clean_dir = rmfrd_root / "AFDB_face_dataset"
    masked_dir = rmfrd_root / "AFDB_masked_face_dataset"

    clean_ids = {p.name for p in clean_dir.iterdir() if p.is_dir()}
    masked_ids = {p.name for p in masked_dir.iterdir() if p.is_dir()}
    paired_ids = sorted(clean_ids & masked_ids)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    clean_by_id: dict[str, list[Path]] = {}
    masked_by_id: dict[str, list[Path]] = {}
    for uid in paired_ids:
        c_imgs = [p for p in (clean_dir / uid).iterdir() if p.suffix.lower() in exts]
        m_imgs = [p for p in (masked_dir / uid).iterdir() if p.suffix.lower() in exts]
        if c_imgs and m_imgs:
            clean_by_id[uid] = c_imgs
            masked_by_id[uid] = m_imgs

    return clean_by_id, masked_by_id


class _PyTorchExtractor:
    """Wraps the student model for embedding extraction."""

    def __init__(self, cfg, checkpoint: str, device: torch.device, use_amp: bool, use_flip: bool) -> None:
        from fas_kd.data.transforms import build_eval_transform
        from fas_kd.models.student import MobileNetV4Student
        from fas_kd.utils.config import load_yaml_config

        self.transform = build_eval_transform(cfg["data"])
        self.device = device
        self.use_amp = use_amp
        self.use_flip = use_flip

        s_cfg = cfg["student"]
        model = MobileNetV4Student(
            backbone_name=s_cfg["backbone_name"],
            embedding_dim=int(s_cfg.get("embedding_dim", 512)),
            pretrained=False,
            input_size=int(cfg["data"].get("image_size", 112)),
            projection_activation=str(s_cfg.get("projection_activation", "none")),
            spatial_out_channels=int(s_cfg.get("spatial_out_channels", 0)),
        )
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("student_state", ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        self.model = model

    @torch.no_grad()
    def embed(self, img_paths: list[Path]) -> np.ndarray:
        out = []
        for path in tqdm(img_paths, desc="Embedding", leave=False):
            img = Image.open(path).convert("RGB").resize((112, 112), Image.BILINEAR)
            t = self.transform(img).unsqueeze(0).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp and self.device.type == "cuda"):
                e = self.model(t)
                if self.use_flip:
                    e = e + self.model(torch.flip(t, dims=[3]))
            e = F.normalize(e, dim=1)
            e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(e.squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(out, axis=0)


class _OnnxExtractor:
    """Wraps an ArcFace ONNX model."""

    def __init__(self, model_path: str, use_flip: bool = True) -> None:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.use_flip = use_flip

    def _prep(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize((112, 112), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)[:, :, ::-1] / 127.5 - 1.0
        return arr.transpose(2, 0, 1)

    def embed(self, img_paths: list[Path]) -> np.ndarray:
        out = []
        for path in tqdm(img_paths, desc="Embedding", leave=False):
            img = Image.open(path)
            x = self._prep(img)[None].astype(np.float32)
            e = self.sess.run([self.output_name], {self.input_name: x})[0]
            if self.use_flip:
                xf = x[:, :, :, ::-1].copy()
                e = e + self.sess.run([self.output_name], {self.input_name: xf})[0]
            norm = np.linalg.norm(e, axis=1, keepdims=True)
            e = e / (norm + 1e-12)
            out.append(e.squeeze(0).astype(np.float32))
        return np.stack(out, axis=0)


def _rank1(gallery_emb: dict[str, np.ndarray], probe_emb: np.ndarray, probe_ids: list[str]) -> float:
    ids = list(gallery_emb.keys())
    gallery_matrix = np.stack([gallery_emb[i] for i in ids], axis=0)
    scores = probe_emb @ gallery_matrix.T  # (P, G)
    top1_ids = [ids[int(np.argmax(scores[i]))] for i in range(len(probe_ids))]
    correct = sum(p == g for p, g in zip(probe_ids, top1_ids))
    return correct / len(probe_ids)


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    valid = np.where(fpr <= target_far)[0]
    if valid.size == 0:
        return float(tpr[0])
    return float(tpr[valid[-1]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on RMFRD masked face dataset")
    parser.add_argument("--rmfrd-root", required=True, help="Path containing AFDB_face_dataset/ and AFDB_masked_face_dataset/")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--onnx-model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gallery-per-id", type=int, default=10, help="Max clean images per gallery identity")
    parser.add_argument("--neg-ratio", type=int, default=5, help="Negative pairs per positive pair")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rmfrd_root = Path(args.rmfrd_root)
    clean_by_id, masked_by_id = _load_rmfrd(rmfrd_root)
    paired_ids = sorted(clean_by_id.keys())
    print(f"[rmfrd] {len(paired_ids)} paired identities; gallery cap={args.gallery_per_id}")

    # Build extractor
    use_flip = not args.no_flip
    if args.onnx_model:
        extractor = _OnnxExtractor(args.onnx_model, use_flip=use_flip)
        model_label = args.model_name or Path(args.onnx_model).stem
    else:
        if args.config is None or args.checkpoint is None:
            raise ValueError("Either --onnx-model or both --config and --checkpoint required")
        from fas_kd.utils.config import load_yaml_config
        cfg = load_yaml_config(args.config)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        extractor = _PyTorchExtractor(
            cfg=cfg,
            checkpoint=args.checkpoint,
            device=device,
            use_amp=cfg.get("system", {}).get("use_amp", True),
            use_flip=use_flip,
        )
        model_label = args.model_name or Path(args.checkpoint).stem

    # Extract gallery embeddings (mean of clean images, up to gallery_per_id)
    print("[rmfrd] Building gallery...")
    gallery_emb: dict[str, np.ndarray] = {}
    all_gallery_paths: list[Path] = []
    gallery_id_list: list[str] = []
    for uid in paired_ids:
        imgs = clean_by_id[uid][:args.gallery_per_id]
        all_gallery_paths.extend(imgs)
        gallery_id_list.extend([uid] * len(imgs))

    raw_gallery = extractor.embed(all_gallery_paths)
    for uid in paired_ids:
        mask = [i for i, gid in enumerate(gallery_id_list) if gid == uid]
        g_emb = raw_gallery[mask].mean(axis=0)
        g_emb = g_emb / (np.linalg.norm(g_emb) + 1e-12)
        gallery_emb[uid] = g_emb.astype(np.float32)

    # Extract probe (masked) embeddings
    print("[rmfrd] Extracting masked probes...")
    all_probe_paths: list[Path] = []
    probe_ids: list[str] = []
    for uid in paired_ids:
        imgs = masked_by_id[uid]
        all_probe_paths.extend(imgs)
        probe_ids.extend([uid] * len(imgs))

    probe_emb = extractor.embed(all_probe_paths)

    # L2 normalize probe embeddings
    norms = np.linalg.norm(probe_emb, axis=1, keepdims=True)
    probe_emb = probe_emb / (norms + 1e-12)

    # Build verification pairs (positive + negative)
    from sklearn.metrics import roc_auc_score, roc_curve

    scores_list: list[float] = []
    labels_list: list[int] = []
    other_ids = paired_ids.copy()

    for i, (uid, p_emb) in enumerate(zip(probe_ids, probe_emb)):
        # positive pair
        g_emb = gallery_emb[uid]
        scores_list.append(float(np.dot(p_emb, g_emb)))
        labels_list.append(1)
        # negative pairs
        neg_candidates = [x for x in other_ids if x != uid]
        neg_sample = random.sample(neg_candidates, min(args.neg_ratio, len(neg_candidates)))
        for neg_id in neg_sample:
            scores_list.append(float(np.dot(p_emb, gallery_emb[neg_id])))
            labels_list.append(0)

    scores = np.array(scores_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(roc_auc_score(labels, scores))

    # Rank-1 identification on all probes
    rank1 = _rank1(gallery_emb, probe_emb, probe_ids)

    target_fars = [1e-3, 1e-4, 1e-5]
    result: dict[str, Any] = {
        "model": model_label,
        "dataset": "RMFRD",
        "num_paired_identities": len(paired_ids),
        "num_gallery_images": len(all_gallery_paths),
        "num_probe_images": len(all_probe_paths),
        "num_positive_pairs": n_pos,
        "num_negative_pairs": n_neg,
        "gallery_per_id": args.gallery_per_id,
        "roc_auc": roc_auc,
        "rank1_identification": rank1,
    }
    for far in target_fars:
        exp = int(round(np.log10(float(far))))
        result[f"tar_far_1e{exp}"] = _tar_at_far(fpr, tpr, far)

    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[rmfrd] saved to {args.out}")


if __name__ == "__main__":
    main()
