#!/usr/bin/env python3
"""Evaluate on MFR2 (Masked Face Recognition in Real-world).

MFR2 contains 55 celebrity identities with both real masked and clean (no-mask) face images.

Two protocols:
  1. Verification  — pairs.txt (LFW-style same/different pairs), reports AUC + TAR@FAR
  2. 1:N Identification — no-mask images as gallery; masked images as probes; reports Rank-1

Usage:
  # Student checkpoint
  python scripts/evaluate_mfr2.py \
      --mfr2-root data/raw/mfr2 \
      --config configs/train_ms1m_magface_phase1_v1.yaml \
      --checkpoint checkpoints/release/mobilenetv4_student_phase1_best.pt \
      --out results/mfr2/phase1_best.json

  # MBF ONNX baseline
  python scripts/evaluate_mfr2.py \
      --mfr2-root data/raw/mfr2 \
      --onnx-model ~/.insightface/models/buffalo_sc/w600k_mbf.onnx \
      --out results/mfr2/mobilefacenet_w600k.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
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


def _load_mfr2(mfr2_root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]], list[tuple]]:
    """Parse mfr2_labels.txt; return (clean_by_id, masked_by_id, pairs).

    pairs is a list of (path_a, path_b, label) where label=1 for same-identity.
    """
    labels_path = mfr2_root / "mfr2_labels.txt"
    pairs_path = mfr2_root / "pairs.txt"

    # Map (identity, img_num) -> mask_type
    img_type: dict[tuple[str, int], str] = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            identity, num, mask_type = parts[0], int(parts[1]), parts[2]
            img_type[(identity, num)] = mask_type

    # Build clean / masked lists
    clean_by_id: dict[str, list[Path]] = defaultdict(list)
    masked_by_id: dict[str, list[Path]] = defaultdict(list)
    for (identity, num), mask_type in img_type.items():
        img_path = mfr2_root / identity / f"{identity}_{num:04d}.png"
        if not img_path.exists():
            continue
        if mask_type == "no-mask":
            clean_by_id[identity].append(img_path)
        else:
            masked_by_id[identity].append(img_path)

    # Keep only identities that have both clean and masked images
    paired_ids = sorted(set(clean_by_id.keys()) & set(masked_by_id.keys()))
    clean_by_id = {k: sorted(clean_by_id[k]) for k in paired_ids}
    masked_by_id = {k: sorted(masked_by_id[k]) for k in paired_ids}

    # Parse pairs.txt (LFW-style)
    # Same-identity: "Identity n1 n2"  (3 tokens)
    # Diff-identity: "Identity1 n1 Identity2 n2"  (4 tokens)
    pairs: list[tuple[Path, Path, int]] = []
    with open(pairs_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 3:
                identity, n1, n2 = tokens[0], int(tokens[1]), int(tokens[2])
                pa = mfr2_root / identity / f"{identity}_{n1:04d}.png"
                pb = mfr2_root / identity / f"{identity}_{n2:04d}.png"
                if pa.exists() and pb.exists():
                    pairs.append((pa, pb, 1))
            elif len(tokens) == 4:
                id1, n1, id2, n2 = tokens[0], int(tokens[1]), tokens[2], int(tokens[3])
                pa = mfr2_root / id1 / f"{id1}_{n1:04d}.png"
                pb = mfr2_root / id2 / f"{id2}_{n2:04d}.png"
                if pa.exists() and pb.exists():
                    pairs.append((pa, pb, 0))

    return dict(clean_by_id), dict(masked_by_id), pairs


class _PyTorchExtractor:
    def __init__(self, cfg, checkpoint: str, device: torch.device, use_amp: bool, use_flip: bool) -> None:
        from fas_kd.data.transforms import build_eval_transform
        from fas_kd.models.student import MobileNetV4Student

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
    scores = probe_emb @ gallery_matrix.T
    top1_ids = [ids[int(np.argmax(scores[i]))] for i in range(len(probe_ids))]
    correct = sum(p == g for p, g in zip(probe_ids, top1_ids))
    return correct / len(probe_ids)


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    valid = np.where(fpr <= target_far)[0]
    if valid.size == 0:
        return float(tpr[0])
    return float(tpr[valid[-1]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on MFR2 masked face dataset")
    parser.add_argument("--mfr2-root", required=True, help="Path to MFR2 root (contains mfr2_labels.txt, pairs.txt)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--onnx-model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    mfr2_root = Path(args.mfr2_root)
    clean_by_id, masked_by_id, pairs = _load_mfr2(mfr2_root)
    paired_ids = sorted(clean_by_id.keys())

    n_clean = sum(len(v) for v in clean_by_id.values())
    n_masked = sum(len(v) for v in masked_by_id.values())
    n_same = sum(1 for _, _, lbl in pairs if lbl == 1)
    n_diff = sum(1 for _, _, lbl in pairs if lbl == 0)
    print(f"[mfr2] {len(paired_ids)} paired identities | gallery: {n_clean} clean | probes: {n_masked} masked")
    print(f"[mfr2] pairs.txt: {n_same} same-identity / {n_diff} different-identity")

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
            cfg=cfg, checkpoint=args.checkpoint, device=device,
            use_amp=cfg.get("system", {}).get("use_amp", True), use_flip=use_flip,
        )
        model_label = args.model_name or Path(args.checkpoint).stem

    # --- Verification using pairs.txt ---
    print("[mfr2] Running verification (pairs.txt)...")
    unique_paths = list({pa for pa, _, _ in pairs} | {pb for _, pb, _ in pairs})
    all_embs_raw = extractor.embed(unique_paths)
    path_to_emb = {str(p): e for p, e in zip(unique_paths, all_embs_raw)}

    verif_scores: list[float] = []
    verif_labels: list[int] = []
    for pa, pb, lbl in pairs:
        ea = path_to_emb[str(pa)]
        eb = path_to_emb[str(pb)]
        verif_scores.append(float(np.dot(ea, eb)))
        verif_labels.append(lbl)

    from sklearn.metrics import roc_auc_score, roc_curve
    scores_arr = np.array(verif_scores, dtype=np.float32)
    labels_arr = np.array(verif_labels, dtype=np.int32)
    fpr_v, tpr_v, _ = roc_curve(labels_arr, scores_arr)
    verif_auc = float(roc_auc_score(labels_arr, scores_arr))

    # --- 1:N Identification: gallery=clean, probes=masked ---
    print("[mfr2] Building gallery (no-mask images)...")
    all_gallery_paths: list[Path] = []
    gallery_id_list: list[str] = []
    for uid in paired_ids:
        all_gallery_paths.extend(clean_by_id[uid])
        gallery_id_list.extend([uid] * len(clean_by_id[uid]))

    raw_gallery = extractor.embed(all_gallery_paths)
    gallery_emb: dict[str, np.ndarray] = {}
    for uid in paired_ids:
        indices = [i for i, gid in enumerate(gallery_id_list) if gid == uid]
        g = raw_gallery[indices].mean(axis=0)
        g = (g / (np.linalg.norm(g) + 1e-12)).astype(np.float32)
        gallery_emb[uid] = g

    print("[mfr2] Extracting masked probes...")
    all_probe_paths: list[Path] = []
    probe_ids: list[str] = []
    for uid in paired_ids:
        all_probe_paths.extend(masked_by_id[uid])
        probe_ids.extend([uid] * len(masked_by_id[uid]))

    probe_emb_raw = extractor.embed(all_probe_paths)
    norms = np.linalg.norm(probe_emb_raw, axis=1, keepdims=True)
    probe_emb = (probe_emb_raw / (norms + 1e-12)).astype(np.float32)

    # Rank-1 identification
    rank1 = _rank1(gallery_emb, probe_emb, probe_ids)

    # Verification scores from gallery-probe pairs for TAR@FAR
    id_scores: list[float] = []
    id_labels: list[int] = []
    for uid, p_emb in zip(probe_ids, probe_emb):
        id_scores.append(float(np.dot(p_emb, gallery_emb[uid])))
        id_labels.append(1)
        neg_ids = [x for x in paired_ids if x != uid]
        for neg_id in random.sample(neg_ids, min(5, len(neg_ids))):
            id_scores.append(float(np.dot(p_emb, gallery_emb[neg_id])))
            id_labels.append(0)

    id_scores_arr = np.array(id_scores, dtype=np.float32)
    id_labels_arr = np.array(id_labels, dtype=np.int32)
    fpr_id, tpr_id, _ = roc_curve(id_labels_arr, id_scores_arr)
    id_auc = float(roc_auc_score(id_labels_arr, id_scores_arr))

    result: dict[str, Any] = {
        "model": model_label,
        "dataset": "MFR2",
        "num_paired_identities": len(paired_ids),
        "num_gallery_images": len(all_gallery_paths),
        "num_probe_images": len(all_probe_paths),
        # Verification (pairs.txt)
        "verification_auc": verif_auc,
        "verification_num_same_pairs": n_same,
        "verification_num_diff_pairs": n_diff,
        # 1:N Identification (clean gallery → masked probes)
        "identification_auc": id_auc,
        "identification_rank1": rank1,
    }
    for far in [1e-3, 1e-4]:
        exp = int(round(np.log10(float(far))))
        result[f"verification_tar_far_1e{exp}"] = _tar_at_far(fpr_v, tpr_v, far)
        result[f"identification_tar_far_1e{exp}"] = _tar_at_far(fpr_id, tpr_id, far)

    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[mfr2] saved to {args.out}")


if __name__ == "__main__":
    main()
