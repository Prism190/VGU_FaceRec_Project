#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MPL_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from fas_kd.data.transforms import build_eval_transform
from fas_kd.models.student import MobileNetV4Student
from fas_kd.pipeline import (
    DetectionConfig,
    FaceDetection,
    FacePreprocessor,
    IdentityIndex,
    MagnitudeQualityGate,
    PreprocessConfig,
    RuntimePipeline,
    SilentFaceAntiSpoof,
    TrackManager,
    ThresholdLivenessGate,
    YOLO11FaceDetector,
    ensure_face_db_layout,
    load_known_face_gallery,
    persist_stranger_session,
)
from fas_kd.utils.config import load_yaml_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_ms1m_magface_phase3_trueasym_swa_v1.yaml"
DEFAULT_IDENTITY_NAMES = [
    "Sarah",
    "John",
    "Emma",
    "Michael",
    "Olivia",
    "Daniel",
    "Sophia",
    "David",
    "Ava",
    "James",
    "Mia",
    "Noah",
    "Isabella",
    "Liam",
    "Grace",
    "Lucas",
    "Chloe",
    "Ethan",
    "Amelia",
    "Henry",
]


def _epoch_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def _resolve_checkpoint(cfg: dict, checkpoint_arg: str) -> Path:
    output_root = Path(cfg["experiment"]["output_root"]).resolve()
    ckpt_dir = output_root / "checkpoints"

    alias = checkpoint_arg.strip().lower()
    if alias in {"auto", "current"}:
        epoch_paths = sorted(ckpt_dir.glob("epoch_*.pt"), key=_epoch_key)
        if epoch_paths:
            return epoch_paths[-1]
        for fixed in ["latest.pt", "best.pt", "swa.pt"]:
            path = ckpt_dir / fixed
            if path.exists():
                return path
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if alias in {"latest", "best", "swa"}:
        path = ckpt_dir / f"{alias}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint alias not found: {path}")
        return path

    candidate = Path(checkpoint_arg)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def _build_student(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    student_cfg = cfg["student"]
    model = MobileNetV4Student(
        backbone_name=student_cfg["backbone_name"],
        embedding_dim=int(student_cfg.get("embedding_dim", 512)),
        pretrained=False,
        input_size=int(cfg["data"].get("image_size", 112)),
        projection_activation=str(student_cfg.get("projection_activation", "none")),
        spatial_out_channels=int(student_cfg.get("spatial_out_channels", 0)),
    )

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("student_state", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing student keys while loading {checkpoint_path}: {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"Unexpected student keys while loading {checkpoint_path}: {unexpected[:5]}")

    model.eval().to(device)
    return model


def _build_embed_fn(model: torch.nn.Module, transform, device: torch.device, use_amp: bool):
    @torch.no_grad()
    def _embed(face_rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(face_rgb.astype(np.uint8), mode="RGB")
        x = transform(pil).unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            emb = model(x)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        return emb[0].detach().cpu().numpy().astype(np.float32)

    return _embed


def _infer_liveness_always_live(_: np.ndarray) -> float:
    return 1.0


def _infer_liveness_texture(face_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    var_lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # Heuristic score in [0,1]. This is a placeholder, not a production anti-spoof model.
    score = (var_lap - 5.0) / 60.0
    return float(np.clip(score, 0.0, 1.0))


def _infer_liveness_hybrid(face_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Texture sharpness cue: print/replay attacks are often flatter in high frequencies.
    var_lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    lap_score = float(np.clip((var_lap - 8.0) / 90.0, 0.0, 1.0))

    # Frequency cue: compute high-frequency power ratio from the face crop spectrum.
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.square(np.abs(fft))
    h, w = gray.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = max(6.0, 0.10 * float(min(h, w)))
    high_power = float(np.sum(power[rr >= cutoff]))
    total_power = float(np.sum(power) + 1e-6)
    hf_ratio = high_power / total_power
    hf_score = float(np.clip((hf_ratio - 0.56) / 0.22, 0.0, 1.0))

    # Color/illumination cue: spoof media tends to show weaker natural variation.
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[..., 1])) / 255.0
    val_std = float(np.std(hsv[..., 2])) / 64.0
    color_score = float(np.clip(0.6 * sat_mean + 0.4 * np.clip(val_std, 0.0, 1.0), 0.0, 1.0))

    # Penalty for large clipped bright regions with low saturation (screen glare-like artifact).
    spec_mask = (hsv[..., 2] >= 245) & (hsv[..., 1] <= 40)
    spec_ratio = float(np.mean(spec_mask.astype(np.float32)))
    spec_penalty = float(np.clip((spec_ratio - 0.08) / 0.12, 0.0, 1.0))

    score = 0.45 * lap_score + 0.35 * hf_score + 0.20 * color_score - 0.25 * spec_penalty
    return float(np.clip(score, 0.0, 1.0))


def _load_gallery(index: IdentityIndex, gallery_npz: Path | None) -> list[int]:
    if gallery_npz is None:
        return []
    data = np.load(gallery_npz)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    ids = np.asarray(data["ids"], dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError(f"gallery embeddings must be 2D, got {embeddings.shape}")
    if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
        raise ValueError("gallery ids shape mismatch")

    loaded_ids: list[int] = []
    for emb, identity in zip(embeddings, ids):
        identity_i = int(identity)
        index.add(identity_id=identity_i, embedding=emb)
        loaded_ids.append(identity_i)
    return loaded_ids


def _load_identity_names(
    identity_names_json: Path | None,
    known_ids: list[int],
    known_name_map: dict[int, str] | None = None,
) -> dict[int, str]:
    out: dict[int, str] = {}

    # Start from persisted known-db names so on-disk identity metadata is authoritative.
    if known_name_map:
        for raw_id, raw_name in known_name_map.items():
            try:
                identity_id = int(raw_id)
            except Exception:
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            out[identity_id] = name

    # Optional identity-names JSON is an explicit override layer.
    if identity_names_json is not None:
        payload = json.loads(identity_names_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            iterable = payload.items()
        elif isinstance(payload, list):
            iterable = []
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                if "id" not in entry or "name" not in entry:
                    continue
                iterable.append((entry["id"], entry["name"]))
        else:
            raise ValueError("--identity-names-json must be a dict or list of {id,name}")

        for raw_id, raw_name in iterable:
            try:
                identity_id = int(raw_id)
            except Exception:
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            out[identity_id] = name

    used = {name.lower() for name in out.values()}
    next_default = 0
    for identity_id in sorted({int(v) for v in known_ids}):
        if identity_id in out:
            continue
        while next_default < len(DEFAULT_IDENTITY_NAMES) and DEFAULT_IDENTITY_NAMES[next_default].lower() in used:
            next_default += 1
        if next_default < len(DEFAULT_IDENTITY_NAMES):
            candidate = DEFAULT_IDENTITY_NAMES[next_default]
            next_default += 1
        else:
            candidate = f"Person-{identity_id}"
        out[identity_id] = candidate
        used.add(candidate.lower())

    return out


def _refresh_known_db_embeddings_from_photos(
    *,
    face_db_root: Path,
    embed_fn,
    emb_dim: int,
) -> dict[str, int]:
    layout = ensure_face_db_layout(face_db_root)
    identities_dir = layout["known_identities"]

    stats = {
        "identities_scanned": 0,
        "identities_with_photos": 0,
        "identities_updated": 0,
        "photos_total": 0,
        "photos_embedded": 0,
        "photos_failed": 0,
    }

    identity_dir_re = re.compile(r"^id_(\d+)(?:__.*)?$")
    for identity_dir in sorted(p for p in identities_dir.iterdir() if p.is_dir()):
        stats["identities_scanned"] += 1
        photos_dir = identity_dir / "photos"
        if not photos_dir.exists():
            continue

        photo_paths = sorted(
            p
            for p in photos_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not photo_paths:
            continue

        stats["identities_with_photos"] += 1
        stats["photos_total"] += int(len(photo_paths))

        embeddings: list[np.ndarray] = []
        for photo_path in photo_paths:
            image_bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
            if image_bgr is None or image_bgr.size == 0:
                stats["photos_failed"] += 1
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            emb = np.asarray(embed_fn(image_rgb), dtype=np.float32).reshape(-1)
            if emb.size != int(emb_dim):
                stats["photos_failed"] += 1
                continue
            emb = _normalize_embedding(emb)
            if not np.isfinite(emb).all() or float(np.linalg.norm(emb)) < 1e-6:
                stats["photos_failed"] += 1
                continue
            embeddings.append(emb)

        if not embeddings:
            continue

        emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
        np.savez(identity_dir / "embeddings.npz", embeddings=emb_arr)
        stats["identities_updated"] += 1
        stats["photos_embedded"] += int(emb_arr.shape[0])

        meta_path = identity_dir / "meta.json"
        meta: dict[str, object] = {}
        if meta_path.exists():
            try:
                loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded_meta, dict):
                    meta = dict(loaded_meta)
            except Exception:
                meta = {}

        raw_id = meta.get("identity_id")
        identity_id = None
        try:
            if raw_id is not None:
                identity_id = int(raw_id)
        except Exception:
            identity_id = None
        if identity_id is None:
            m = identity_dir_re.match(identity_dir.name)
            if m is not None:
                identity_id = int(m.group(1))

        if identity_id is not None:
            meta["identity_id"] = int(identity_id)
        if not str(meta.get("name", "")).strip():
            meta["name"] = identity_dir.name
        meta["embedding_count"] = int(emb_arr.shape[0])
        meta["photo_count"] = int(len(photo_paths))

        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    return {k: int(v) for k, v in stats.items()}


def _draw_observation(frame_bgr: np.ndarray, obs: dict, *, show_track_id: bool = False) -> None:
    h, w = frame_bgr.shape[:2]
    base = float(min(h, w))
    font_scale = max(0.9, base / 1200.0)
    box_thickness = max(2, int(round(base / 500.0)))
    text_thickness = max(2, int(round(font_scale * 2.0)))

    x1, y1, x2, y2 = [int(v) for v in obs["bbox_xyxy"]]
    is_live = bool(obs["is_live"])
    quality = bool(obs["quality_pass"])
    identity_id = obs.get("identity_id")
    identity_name = obs.get("identity_name")
    cluster_label = obs.get("cluster_label")

    if identity_id is not None:
        color = (0, 220, 0)
    elif is_live and quality:
        color = (0, 180, 255)
    else:
        color = (0, 80, 255)

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, box_thickness)
    if identity_name:
        display_name = str(identity_name)
    elif identity_id is not None:
        display_name = f"Person-{int(identity_id)}"
    elif cluster_label is not None:
        display_name = f"Stranger-{int(cluster_label)}"
    else:
        display_name = "Unknown"

    label_parts = [display_name]
    if show_track_id:
        label_parts.append(f"T{obs['track_id']}")
    label_parts.append(f"live={int(is_live)}")
    label_parts.append(f"q={int(quality)}")
    if identity_id is not None and obs.get("match_score") is not None:
        label_parts.append(f"sim={float(obs['match_score']):.2f}")
    label = " | ".join(label_parts)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    pad = max(4, box_thickness + 1)
    tx1 = max(0, x1)
    ty2 = max(th + baseline + 2 * pad, y1 - 2)
    ty1 = max(0, ty2 - (th + baseline + 2 * pad))
    tx2 = min(w - 1, tx1 + tw + 2 * pad)

    # Filled background for readability on bright/complex scenes.
    cv2.rectangle(frame_bgr, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        label,
        (tx1 + pad, ty2 - baseline - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        text_thickness,
        cv2.LINE_AA,
    )


def _draw_ghost_track(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    identity_name: str,
    missed_frames: int,
    *,
    show_track_id: bool = False,
    track_id: int | None = None,
) -> None:
    """Draw a dimmed corner-segment box for a track coasting on Kalman prediction."""
    h, w = frame_bgr.shape[:2]
    base = float(min(h, w))
    font_scale = max(0.75, base / 1400.0)
    text_thickness = max(1, int(round(font_scale * 1.5)))

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    fade = max(0.25, 1.0 - float(missed_frames) / 20.0)
    color = (0, int(160 * fade), 0)

    # Corner-segment style to distinguish from solid observation boxes.
    seg = max(12, (x2 - x1) // 5)
    for sx1, sy1, sx2, sy2 in [
        (x1, y1, x1 + seg, y1), (x2 - seg, y1, x2, y1),
        (x1, y2, x1 + seg, y2), (x2 - seg, y2, x2, y2),
        (x1, y1, x1, y1 + seg), (x1, y2 - seg, x1, y2),
        (x2, y1, x2, y1 + seg), (x2, y2 - seg, x2, y2),
    ]:
        cv2.line(frame_bgr, (sx1, sy1), (sx2, sy2), color, 2)

    label_parts = [identity_name]
    if show_track_id and track_id is not None:
        label_parts.append(f"T{track_id}")
    label = " | ".join(label_parts)
    cv2.putText(
        frame_bgr, label,
        (max(0, x1), max(15, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA,
    )


def _normalize_embedding(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if not np.isfinite(norm) or norm < eps:
                return np.zeros_like(arr, dtype=np.float32)
        return (arr / norm).astype(np.float32)


def _collapse_identity_hits(hits: list) -> list[tuple[int, float]]:
        best_by_identity: dict[int, float] = {}
        for hit in hits:
                identity_id = int(hit.identity_id)
                score = float(hit.score)
                prev = best_by_identity.get(identity_id)
                if prev is None or score > prev:
                        best_by_identity[identity_id] = score
        return sorted(best_by_identity.items(), key=lambda kv: kv[1], reverse=True)


def _strict_identity_decision(
        index: IdentityIndex,
        embedding: np.ndarray,
        *,
        top_k: int,
        min_score: float,
        min_margin: float,
) -> tuple[int | None, float | None, float | None, float | None, str]:
        hits = index.search(embedding, k=max(1, int(top_k)))
        ranked = _collapse_identity_hits(hits)
        if not ranked:
                return None, None, None, None, "empty_gallery"

        top_id, top_score = ranked[0]
        second_score = float(ranked[1][1]) if len(ranked) > 1 else -1.0
        margin = float(top_score - second_score)

        if float(top_score) < float(min_score):
                return None, float(top_score), second_score, margin, "low_score"
        if float(margin) < float(min_margin):
                return None, float(top_score), second_score, margin, "low_margin"
        return int(top_id), float(top_score), second_score, margin, "accepted"


def _crop_face_patch(
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float] | list[float],
        *,
        pad_ratio: float = 0.1,
) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        px = pad_ratio * bw
        py = pad_ratio * bh

        ix1 = max(0, int(round(x1 - px)))
        iy1 = max(0, int(round(y1 - py)))
        ix2 = min(w, int(round(x2 + px)))
        iy2 = min(h, int(round(y2 + py)))
        if ix2 <= ix1 or iy2 <= iy1:
                return None

        crop = frame_bgr[iy1:iy2, ix1:ix2]
        if crop.size == 0:
                return None
        return crop


def _collect_registration_candidate(
    *,
    track_candidates: dict[int, list[dict[str, object]]],
    track_id: int,
    frame_idx: int,
    magnitude: float,
    embedding: np.ndarray,
    selection: str,
    max_keep: int,
) -> None:
    vec = _normalize_embedding(embedding)
    if vec.size == 0 or not np.isfinite(vec).all():
        return
    if float(np.linalg.norm(vec)) < 1e-6:
        return

    bucket = track_candidates.setdefault(int(track_id), [])
    rec = {
        "frame_idx": int(frame_idx),
        "magnitude": float(magnitude),
        "embedding": vec,
    }

    keep = max(1, int(max_keep))
    if str(selection) == "first":
        if len(bucket) < keep:
            bucket.append(rec)
        return

    bucket.append(rec)
    bucket.sort(key=lambda item: (-float(item["magnitude"]), int(item["frame_idx"])))
    if len(bucket) > keep:
        del bucket[keep:]


def _select_registration_candidates(
    *,
    candidates: list[dict[str, object]],
    selection: str,
    max_count: int,
) -> list[dict[str, object]]:
    if not candidates:
        return []

    keep = max(1, int(max_count))
    if str(selection) == "first":
        ranked = sorted(candidates, key=lambda item: int(item["frame_idx"]))
    else:
        ranked = sorted(candidates, key=lambda item: (-float(item["magnitude"]), int(item["frame_idx"])))
    return ranked[:keep]


def _write_unknown_review_html(
        *,
        manifest: dict,
        out_html: Path,
        unknown_name_prefix: str,
) -> None:
        payload = json.dumps(manifest, ensure_ascii=True)
        html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Unknown Face Group Review</title>
    <style>
        :root {{
            --bg: #f2efe8;
            --card: #ffffff;
            --ink: #1f2937;
            --muted: #6b7280;
            --accent: #0f766e;
            --border: #d6d3d1;
        }}
        body {{
            margin: 0;
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(1200px 500px at 0% 0%, #e6f4ea 0%, transparent 60%),
                radial-gradient(900px 450px at 100% 0%, #fde68a 0%, transparent 55%),
                var(--bg);
        }}
        .wrap {{
            max-width: 1300px;
            margin: 0 auto;
            padding: 24px;
        }}
        h1 {{
            margin: 0 0 6px;
            font-size: 28px;
            letter-spacing: 0.2px;
        }}
        p {{
            margin: 0 0 14px;
            color: var(--muted);
        }}
        .toolbar {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 18px;
        }}
        button {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--card);
            color: var(--ink);
            padding: 10px 14px;
            cursor: pointer;
            font-weight: 600;
        }}
        button:hover {{ border-color: var(--accent); color: var(--accent); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 14px;
        }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        }}
        .title {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 10px;
            margin-bottom: 6px;
        }}
        .title strong {{ font-size: 17px; }}
        .stats {{
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        .samples {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 6px;
            margin-bottom: 8px;
        }}
        .samples img {{
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #f8fafc;
        }}
        label {{
            display: block;
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 4px;
        }}
        input[type=\"text\"] {{
            width: 100%;
            box-sizing: border-box;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .empty {{
            padding: 18px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--muted);
        }}
    </style>
</head>
<body>
    <div class=\"wrap\">
        <h1>Unknown Face Group Review</h1>
        <p>Assign a human label per group and export the map as JSON.</p>
        <div class=\"toolbar\">
            <button id=\"btnExport\">Export Labels JSON</button>
            <button id=\"btnClear\">Clear Saved Labels</button>
        </div>
        <div id=\"container\"></div>
    </div>
    <script>
        const data = {payload};
        const storageKey = "unknown_face_group_labels_v1";
        const namePrefix = {json.dumps(str(unknown_name_prefix), ensure_ascii=True)};
        const labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}") || {{}};

        const container = document.getElementById("container");
        const clusters = Array.isArray(data.clusters) ? data.clusters : [];

        function save() {{
            localStorage.setItem(storageKey, JSON.stringify(labels));
        }}

        function render() {{
            if (!clusters.length) {{
                container.innerHTML = '<div class="empty">No grouped unknowns found in this run.</div>';
                return;
            }}

            const grid = document.createElement("div");
            grid.className = "grid";
            clusters.forEach((cluster) => {{
                const gid = Number(cluster.group_id);
                const key = String(gid);
                const card = document.createElement("div");
                card.className = "card";

                const title = document.createElement("div");
                title.className = "title";
                const left = document.createElement("strong");
                left.textContent = `${{namePrefix}}-${{gid}}`;
                const right = document.createElement("span");
                right.textContent = `tracks=${{Number(cluster.num_tracks || 0)}}`;
                right.style.color = "#6b7280";
                right.style.fontSize = "12px";
                title.appendChild(left);
                title.appendChild(right);

                const stats = document.createElement("div");
                stats.className = "stats";
                const avgMag = Number(cluster.avg_track_magnitude || 0).toFixed(2);
                const sim = cluster.max_similarity_to_group;
                const simText = sim == null ? "n/a" : Number(sim).toFixed(3);
                stats.textContent = `frames=${{cluster.first_frame}}..${{cluster.last_frame}} | avgMag=${{avgMag}} | maxGroupSim=${{simText}}`;

                const sampleBox = document.createElement("div");
                sampleBox.className = "samples";
                const samples = Array.isArray(cluster.samples) ? cluster.samples : [];
                samples.forEach((src) => {{
                    const img = document.createElement("img");
                    img.loading = "lazy";
                    img.src = src;
                    img.alt = `${{namePrefix}}-${{gid}} sample`;
                    sampleBox.appendChild(img);
                }});

                const label = document.createElement("label");
                label.textContent = "Manual label";
                const input = document.createElement("input");
                input.type = "text";
                input.placeholder = "Example: Sarah";
                input.value = String(labels[key] || "");
                input.addEventListener("change", () => {{
                    const v = input.value.trim();
                    if (v) {{ labels[key] = v; }} else {{ delete labels[key]; }}
                    save();
                }});

                card.appendChild(title);
                card.appendChild(stats);
                card.appendChild(sampleBox);
                card.appendChild(label);
                card.appendChild(input);
                grid.appendChild(card);
            }});
            container.innerHTML = "";
            container.appendChild(grid);
        }}

        document.getElementById("btnExport").addEventListener("click", () => {{
            const payload = {{
                created_from: data.manifest_path || null,
                unknown_name_prefix: namePrefix,
                labels,
            }};
            const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "unknown_group_labels.json";
            a.click();
            URL.revokeObjectURL(a.href);
        }});

        document.getElementById("btnClear").addEventListener("click", () => {{
            localStorage.removeItem(storageKey);
            Object.keys(labels).forEach((k) => delete labels[k]);
            render();
        }});

        render();
    </script>
</body>
</html>
"""
        out_html.write_text(html_text, encoding="utf-8")


def _open_annotated_writer(
    out_video_path: Path,
    width: int,
    height: int,
    fps: float,
) -> tuple[cv2.VideoWriter, Path, str]:
    candidates: list[tuple[str, Path]] = [
        ("avc1", out_video_path),
        ("H264", out_video_path),
        ("mp4v", out_video_path.with_name(f"{out_video_path.stem}.tmp_mp4v{out_video_path.suffix}")),
    ]

    for fourcc_tag, path in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_tag),
            float(fps),
            (int(width), int(height)),
        )
        if writer.isOpened():
            return writer, path, fourcc_tag
        writer.release()

    raise RuntimeError(
        "Could not initialize video writer. Tried codecs: avc1, H264, mp4v. "
        "Install FFmpeg-backed OpenCV or use a build with H.264 support."
    )


def _transcode_mp4_to_h264(src_path: Path, dst_path: Path) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 0


def _synthetic_stream(max_frames: int) -> Iterable[tuple[int, np.ndarray, list[FaceDetection]]]:
    count = max_frames if max_frames > 0 else 180
    width = 640
    height = 480

    for frame_idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cx = 320 + int(90.0 * np.sin(frame_idx / 12.0))
        cy = 240

        cv2.circle(frame, (cx, cy), 78, (185, 170, 140), -1)
        cv2.circle(frame, (cx - 24, cy - 16), 8, (30, 30, 30), -1)
        cv2.circle(frame, (cx + 24, cy - 16), 8, (30, 30, 30), -1)
        cv2.ellipse(frame, (cx, cy + 24), (26, 12), 0, 0, 180, (40, 40, 40), 2)

        bbox = (float(cx - 80), float(cy - 96), float(cx + 80), float(cy + 96))
        landmarks = np.array(
            [
                [cx - 25.0, cy - 16.0],
                [cx + 25.0, cy - 16.0],
                [cx + 0.0, cy + 4.0],
                [cx - 20.0, cy + 36.0],
                [cx + 20.0, cy + 36.0],
            ],
            dtype=np.float32,
        )
        detections = [FaceDetection(bbox_xyxy=bbox, landmarks5=landmarks, score=0.99)]
        yield frame_idx, frame, detections


def _video_stream(
    source: str,
    detector: YOLO11FaceDetector,
    max_frames: int,
    *,
    loop_source: bool,
):
    source_is_camera = source.isdigit()
    if source_is_camera:
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if bool(loop_source) and not source_is_camera and (max_frames <= 0 or frame_idx < max_frames):
                    seek_ok = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
                    if not seek_ok:
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        if not cap.isOpened():
                            break
                    continue
                break
            detections = detector.detect(frame)
            yield frame_idx, frame, detections
            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the face pipeline with current student checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML training config")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="Checkpoint path or alias: auto/current/latest/best/swa",
    )
    parser.add_argument("--source", default="", help="Video path or camera index. Ignore when --demo-synthetic")
    parser.add_argument("--demo-synthetic", action="store_true", help="Run with generated synthetic frames and fake landmarks")
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to process. <=0 means full stream")
    parser.add_argument(
        "--loop-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Loop file source back to frame 0 on EOF. Useful with --max-frames 0 for manual stop runs.",
    )

    parser.add_argument("--detector-model", default="", help="YOLO face model path for real video mode")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--det-max", type=int, default=50)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--det-disable-rescue-pass", action="store_true")
    parser.add_argument("--det-rescue-conf", type=float, default=0.08)
    parser.add_argument("--det-rescue-iou", type=float, default=0.45)
    parser.add_argument("--det-rescue-imgsz", type=int, default=1280)
    parser.add_argument("--det-rescue-min-primary", type=int, default=2)
    parser.add_argument("--det-merge-iou", type=float, default=0.55)

    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--quality-min", type=float, default=None, help="Minimum embedding norm. Default from config margin_head.feature_norm_lower")
    parser.add_argument("--quality-max", type=float, default=None, help="Maximum embedding norm. Default from config margin_head.feature_norm_upper")
    parser.add_argument(
        "--liveness-mode",
        choices=["always_live", "texture", "hybrid", "silent_face", "litmas"],
        default="always_live",
    )
    parser.add_argument("--live-threshold", type=float, default=0.5)
    parser.add_argument("--no-liveness-tta", action="store_true")
    parser.add_argument("--liveness-every", type=int, default=15, help="Run liveness every N frames per track")
    parser.add_argument(
        "--liveness-silent-face-model",
        default="checkpoints/pretrained/2.7_80x80_MiniFASNetV2.pth",
        help="Path to Silent-Face MiniFASNet .pth checkpoint used when --liveness-mode silent_face",
    )
    parser.add_argument(
        "--liveness-silent-face-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device for Silent-Face model",
    )
    parser.add_argument(
        "--liveness-silent-face-live-class-index",
        type=int,
        default=1,
        help="Softmax class index interpreted as live for Silent-Face model",
    )
    parser.add_argument(
        "--liveness-silent-face-input-color",
        choices=["bgr", "rgb"],
        default="bgr",
        help="Expected color order for Silent-Face model input",
    )
    # LitMAS anti-spoofing options
    parser.add_argument(
        "--liveness-litmas-model",
        default="checkpoints/pretrained/litmas_downstream_moe.pth",
        help="Path to LitMAS DeiT+MoE checkpoint (used when --liveness-mode litmas)",
    )
    parser.add_argument(
        "--liveness-litmas-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device for LitMAS model",
    )
    parser.add_argument(
        "--liveness-litmas-live-class-index",
        type=int,
        default=0,
        help="Softmax class index for 'live' in LitMAS output (0=bonafide, 1=spoof)",
    )
    parser.add_argument("--tracker-backend", choices=["deepsort", "botsort", "hungarian"], default="deepsort")
    parser.add_argument(
        "--track-max-missed-frames",
        type=int,
        default=20,
        help="Keep a track alive for this many missed frames before dropping it",
    )
    parser.add_argument("--track-n-init", type=int, default=2, help="DeepSORT confirmation frames")
    parser.add_argument("--track-max-iou-distance", type=float, default=0.75, help="DeepSORT IoU gating threshold")
    parser.add_argument("--track-max-cosine-distance", type=float, default=0.25, help="DeepSORT appearance cosine distance threshold")
    parser.add_argument(
        "--track-nn-budget",
        type=int,
        default=100,
        help="DeepSORT appearance memory budget per track (0 disables limit)",
    )
    parser.add_argument(
        "--track-nms-max-overlap",
        type=float,
        default=1.0,
        help="DeepSORT NMS overlap threshold",
    )
    parser.add_argument(
        "--track-gating-only-position",
        action="store_true",
        help="DeepSORT Kalman gating uses only center position (less strict shape gating)",
    )
    # BoT-SORT options (used when --tracker-backend botsort)
    parser.add_argument(
        "--botsort-device",
        default="cpu",
        help="Device for BoT-SORT Kalman tracker (cpu recommended; set to cuda when using ReID)",
    )
    parser.add_argument(
        "--botsort-reid-weights",
        default=None,
        help="Optional path to ReID weights for BoT-SORT appearance features (e.g. osnet_x0_25_msmt17.pt)",
    )
    parser.add_argument("--botsort-with-reid", action="store_true", help="Enable ReID features in BoT-SORT")
    parser.add_argument("--botsort-track-high-thresh", type=float, default=0.10,
                        help="High-conf detection threshold for first-pass association (lower for face pipelines)")
    parser.add_argument("--botsort-track-low-thresh", type=float, default=0.03)
    parser.add_argument("--botsort-new-track-thresh", type=float, default=0.10)
    parser.add_argument("--botsort-match-thresh", type=float, default=0.8)

    parser.add_argument(
        "--face-db-root",
        default="data/face_db",
        help="Root folder for persistent face database (known + strangers)",
    )
    parser.add_argument(
        "--known-db-use",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load known identities from folder database at <face-db-root>/known",
    )
    parser.add_argument(
        "--known-db-retrieval-mode",
        choices=["pooled", "all"],
        default="pooled",
        help="Use pooled prototype per identity or all stored embeddings for retrieval",
    )
    parser.add_argument(
        "--known-db-refresh-from-photos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild known identity embeddings from photo folders before loading FAISS gallery",
    )
    parser.add_argument(
        "--stranger-db-use",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist grouped strangers into folder database at <face-db-root>/strangers",
    )
    parser.add_argument(
        "--stranger-db-session-name",
        default="",
        help="Optional custom stranger session name; defaults to output JSONL stem",
    )

    parser.add_argument("--gallery-npz", default="", help="Optional gallery npz with arrays: embeddings, ids")
    parser.add_argument("--match-threshold", type=float, default=0.35)
    parser.add_argument("--match-topk", type=int, default=5, help="Retrieve top-k neighbors before deciding identity")
    parser.add_argument(
        "--match-min-margin",
        type=float,
        default=0.08,
        help="Minimum top1-top2 score margin required to accept a match",
    )
    parser.add_argument("--identity-names-json", default="", help="Optional JSON map/list to convert identity ids to names")
    parser.add_argument("--unknown-name-prefix", default="Stranger", help="Display prefix used when identity is unknown")
    parser.add_argument(
        "--reid-min-track-frames",
        type=int,
        default=6,
        help="Minimum accepted observations before first recognition attempt per track",
    )
    parser.add_argument(
        "--reid-once-per-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recognize each track once and lock identity until track is lost",
    )
    parser.add_argument(
        "--reid-stranger-retry-interval",
        type=int,
        default=0,
        help=(
            "Accepted-frame interval before retrying unresolved tracks. "
            "With --reid-once-per-track, 0 keeps legacy one-attempt behavior; "
            "set >0 to periodically retry strangers without relabeling already identified tracks."
        ),
    )
    parser.add_argument("--show-track-id", action="store_true", help="Show internal tracker id in overlay")
    parser.add_argument(
        "--out-gallery-npz",
        default="",
        help="Optional output gallery npz generated from pooled track embeddings",
    )
    parser.add_argument(
        "--gallery-min-track-frames",
        type=int,
        default=8,
        help="Minimum accepted observations per track for enrollment when writing --out-gallery-npz",
    )
    parser.add_argument(
        "--gallery-id-offset",
        type=int,
        default=1000,
        help="Offset added to enrolled identity ids when writing --out-gallery-npz",
    )
    parser.add_argument(
        "--gallery-dedupe-threshold",
        type=float,
        default=0.68,
        help="Similarity threshold to merge two track prototypes into one enrolled identity",
    )
    parser.add_argument(
        "--gallery-min-mean-magnitude",
        type=float,
        default=None,
        help="Minimum per-track mean embedding norm for gallery enrollment (default uses quality-min)",
    )
    parser.add_argument(
        "--out-gallery-manifest",
        default="",
        help="Optional JSON report for track-to-identity enrollment and dedupe decisions",
    )

    parser.add_argument("--dbscan-eps", type=float, default=0.35, help="Cosine-distance threshold for online unknown clustering")
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--unknown-max-buffer", type=int, default=2000)
    parser.add_argument(
        "--unknown-group-threshold",
        type=float,
        default=0.62,
        help="Similarity threshold to merge an unknown track into an existing unknown group",
    )
    parser.add_argument(
        "--unknown-min-track-frames",
        type=int,
        default=10,
        help="Minimum accepted observations before assigning an unknown group",
    )
    parser.add_argument(
        "--unknown-min-mean-magnitude",
        type=float,
        default=None,
        help="Minimum per-track mean norm before unknown-group assignment (default uses quality-min)",
    )
    parser.add_argument(
        "--unknown-max-samples-per-group",
        type=int,
        default=8,
        help="Maximum saved face crops per unknown group",
    )
    parser.add_argument(
        "--unknown-sample-min-gap",
        type=int,
        default=10,
        help="Minimum frame gap between saved samples for the same unknown group",
    )
    parser.add_argument(
        "--auto-register-unknowns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically register unknown groups into the live identity index during the run.",
    )
    parser.add_argument(
        "--auto-register-selection",
        choices=["first", "best"],
        default="best",
        help="Use first-N or best-N embeddings when registering a newly discovered stranger.",
    )
    parser.add_argument(
        "--auto-register-max-embeddings",
        type=int,
        default=3,
        help="How many embeddings to add per auto-registered stranger track.",
    )
    parser.add_argument(
        "--auto-register-candidate-pool-size",
        type=int,
        default=24,
        help="Internal per-track candidate buffer used before first/best selection.",
    )
    parser.add_argument(
        "--auto-register-min-track-frames",
        type=int,
        default=None,
        help="Minimum accepted frames before auto-register (default: unknown-min-track-frames).",
    )
    parser.add_argument(
        "--auto-register-min-mean-magnitude",
        type=float,
        default=None,
        help="Minimum mean magnitude before auto-register (default: unknown-min-mean-magnitude).",
    )
    parser.add_argument(
        "--auto-register-id-offset",
        type=int,
        default=10000,
        help="Identity id offset for auto-registered strangers.",
    )
    parser.add_argument(
        "--auto-register-name-prefix",
        default="Stranger",
        help="Display prefix used for auto-registered stranger identities.",
    )
    parser.add_argument("--out-unknown-manifest", default="", help="Optional JSON output for grouped unknown tracks")
    parser.add_argument(
        "--out-unknown-review-html",
        default="",
        help="Optional HTML output for manual unknown-group labeling",
    )
    parser.add_argument("--cluster-every", type=int, default=64, help="Deprecated (ignored): kept for backward compatibility")
    parser.add_argument(
        "--cluster-interval-sec",
        type=float,
        default=2.0,
        help="Deprecated (ignored): kept for backward compatibility",
    )
    parser.add_argument(
        "--cluster-sync",
        action="store_true",
        help="Deprecated (ignored): online clustering is always incremental",
    )

    parser.add_argument("--out-jsonl", default="logs/pipeline_run.jsonl")
    parser.add_argument("--out-summary", default="", help="Optional summary JSON path")
    parser.add_argument("--out-video", default="", help="Optional annotated output video path")
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    face_db_root = Path(args.face_db_root)
    if not face_db_root.is_absolute():
        face_db_root = (PROJECT_ROOT / face_db_root).resolve()
    face_db_layout = ensure_face_db_layout(face_db_root)
    known_db_root = face_db_layout["known_root"]
    stranger_db_root = face_db_layout["strangers_root"]

    cfg = load_yaml_config(args.config)
    ckpt_path = _resolve_checkpoint(cfg=cfg, checkpoint_arg=args.checkpoint)
    quality_min = (
        float(args.quality_min)
        if args.quality_min is not None
        else float(cfg.get("margin_head", {}).get("feature_norm_lower", 10.0))
    )
    quality_max = (
        float(args.quality_max)
        if args.quality_max is not None
        else float(cfg.get("margin_head", {}).get("feature_norm_upper", 120.0))
    )
    unknown_min_mean_magnitude = (
        float(args.unknown_min_mean_magnitude)
        if args.unknown_min_mean_magnitude is not None
        else float(quality_min)
    )
    gallery_min_mean_magnitude = (
        float(args.gallery_min_mean_magnitude)
        if args.gallery_min_mean_magnitude is not None
        else float(quality_min)
    )
    auto_register_min_track_frames = (
        max(1, int(args.auto_register_min_track_frames))
        if args.auto_register_min_track_frames is not None
        else max(1, int(args.unknown_min_track_frames))
    )
    auto_register_min_mean_magnitude = (
        float(args.auto_register_min_mean_magnitude)
        if args.auto_register_min_mean_magnitude is not None
        else float(unknown_min_mean_magnitude)
    )

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    use_amp = bool(cfg.get("system", {}).get("use_amp", True))
    model = _build_student(cfg=cfg, checkpoint_path=ckpt_path, device=device)
    transform = build_eval_transform(cfg["data"])
    embed_fn = _build_embed_fn(model=model, transform=transform, device=device, use_amp=use_amp)

    pre_cfg = PreprocessConfig(
        image_size=int(cfg["data"].get("image_size", 112)),
        use_clahe=bool(cfg["data"].get("use_clahe", False)),
    )

    silent_face_model_path: Path | None = None
    silent_face_device_used: str | None = None

    if args.liveness_mode == "always_live":
        liveness_infer = _infer_liveness_always_live
    elif args.liveness_mode == "texture":
        liveness_infer = _infer_liveness_texture
    elif args.liveness_mode == "hybrid":
        liveness_infer = _infer_liveness_hybrid
    elif args.liveness_mode == "silent_face":
        silent_face_model_path = Path(args.liveness_silent_face_model)
        if not silent_face_model_path.is_absolute():
            silent_face_model_path = (PROJECT_ROOT / silent_face_model_path).resolve()

        requested_liveness_device = str(args.liveness_silent_face_device).strip().lower()
        if requested_liveness_device == "auto":
            silent_face_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested_liveness_device == "cuda":
            if not torch.cuda.is_available():
                print("[warn] Silent-Face liveness requested cuda but CUDA is unavailable; falling back to cpu")
                silent_face_device = torch.device("cpu")
            else:
                silent_face_device = torch.device("cuda")
        else:
            silent_face_device = torch.device("cpu")

        silent_face = SilentFaceAntiSpoof(
            model_path=silent_face_model_path,
            device=silent_face_device,
            live_class_index=int(args.liveness_silent_face_live_class_index),
            expect_bgr_input=str(args.liveness_silent_face_input_color).strip().lower() == "bgr",
        )
        liveness_infer = silent_face.score
        silent_face_device_used = str(silent_face_device)
    else:  # litmas
        from fas_kd.pipeline.anti_spoof import LitMASAntiSpoof

        litmas_model_path = Path(args.liveness_litmas_model)
        if not litmas_model_path.is_absolute():
            litmas_model_path = (PROJECT_ROOT / litmas_model_path).resolve()

        requested_litmas_device = str(args.liveness_litmas_device).strip().lower()
        if requested_litmas_device == "auto":
            litmas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested_litmas_device == "cuda":
            litmas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            litmas_device = torch.device("cpu")

        litmas = LitMASAntiSpoof(
            model_path=litmas_model_path,
            device=litmas_device,
            live_class_index=int(args.liveness_litmas_live_class_index),
        )
        liveness_infer = litmas.score

    tracker_backend = str(args.tracker_backend).strip().lower()

    pipeline = RuntimePipeline(
        preprocess=FacePreprocessor(pre_cfg),
        liveness_gate=ThresholdLivenessGate(
            infer_fn=liveness_infer,
            live_threshold=float(args.live_threshold),
            use_ttda=not args.no_liveness_tta,
        ),
        quality_gate=MagnitudeQualityGate(
            min_magnitude=quality_min,
            max_magnitude=quality_max,
        ),
        embed_fn=embed_fn,
        liveness_interval_frames=max(0, int(args.liveness_every)),
        track_manager=TrackManager(
            backend=tracker_backend,
            max_missed_frames=max(1, int(args.track_max_missed_frames)),
            deepsort_n_init=max(1, int(args.track_n_init)),
            deepsort_max_iou_distance=float(args.track_max_iou_distance),
            deepsort_max_cosine_distance=float(args.track_max_cosine_distance),
            deepsort_nn_budget=None if int(args.track_nn_budget) <= 0 else int(args.track_nn_budget),
            deepsort_nms_max_overlap=float(args.track_nms_max_overlap),
            deepsort_gating_only_position=bool(args.track_gating_only_position),
            botsort_device=str(args.botsort_device),
            botsort_model_weights=args.botsort_reid_weights,
            botsort_with_reid=bool(args.botsort_with_reid),
            botsort_track_high_thresh=float(args.botsort_track_high_thresh),
            botsort_track_low_thresh=float(args.botsort_track_low_thresh),
            botsort_new_track_thresh=float(args.botsort_new_track_thresh),
            botsort_match_thresh=float(args.botsort_match_thresh),
        ),
    )

    emb_dim = int(cfg["student"].get("embedding_dim", 512))
    index = IdentityIndex(dim=emb_dim, use_faiss=True)
    gallery_ids_loaded: list[int] = []
    known_db_name_map: dict[int, str] = {}
    known_db_stats = {
        "identities_total": 0,
        "vectors_loaded": 0,
        "identities_with_embeddings": 0,
        "photos_total": 0,
    }
    known_db_refresh_stats = {
        "identities_scanned": 0,
        "identities_with_photos": 0,
        "identities_updated": 0,
        "photos_total": 0,
        "photos_embedded": 0,
        "photos_failed": 0,
    }

    if bool(args.known_db_use):
        if bool(args.known_db_refresh_from_photos):
            known_db_refresh_stats = _refresh_known_db_embeddings_from_photos(
                face_db_root=face_db_root,
                embed_fn=embed_fn,
                emb_dim=emb_dim,
            )
        known_embeddings, known_ids, known_name_map, known_db_stats = load_known_face_gallery(
            db_root=face_db_root,
            expected_dim=emb_dim,
            retrieval_mode=str(args.known_db_retrieval_mode),
        )
        for emb, identity in zip(known_embeddings, known_ids):
            identity_i = int(identity)
            index.add(identity_id=identity_i, embedding=emb)
            gallery_ids_loaded.append(identity_i)
        known_db_name_map = {int(k): str(v) for k, v in known_name_map.items()}

    gallery_path = Path(args.gallery_npz).resolve() if args.gallery_npz else None
    if gallery_path is not None:
        gallery_ids_loaded.extend(_load_gallery(index=index, gallery_npz=gallery_path))
    num_gallery = len(index.ids)

    identity_names_path = Path(args.identity_names_json).resolve() if args.identity_names_json else None
    if identity_names_path is not None and not identity_names_path.exists():
        raise FileNotFoundError(f"identity names json not found: {identity_names_path}")
    identity_name_map = _load_identity_names(
        identity_names_json=identity_names_path,
        known_ids=gallery_ids_loaded,
        known_name_map=known_db_name_map,
    )
    reid_min_track_frames = max(1, int(args.reid_min_track_frames))

    detector = None
    if not args.demo_synthetic:
        if not args.source:
            raise ValueError("--source is required unless --demo-synthetic is set")
        if not args.detector_model:
            raise ValueError("--detector-model is required unless --demo-synthetic is set")
        if args.source.startswith("/path/to/") or args.detector_model.startswith("/path/to/"):
            raise ValueError(
                "Replace placeholder paths. Example: --source data/raw/pipeline_demo/short_hamilton_clip.mp4 "
                "--detector-model checkpoints/pretrained/yolo11n-face-age.pt"
            )
        detector = YOLO11FaceDetector(
            model_path=args.detector_model,
            cfg=DetectionConfig(
                conf_thres=float(args.det_conf),
                iou_thres=float(args.det_iou),
                max_det=int(args.det_max),
                imgsz=int(args.det_imgsz),
                enable_rescue_pass=not bool(args.det_disable_rescue_pass),
                rescue_conf_thres=float(args.det_rescue_conf),
                rescue_iou_thres=float(args.det_rescue_iou),
                rescue_imgsz=int(args.det_rescue_imgsz),
                rescue_min_primary_detections=int(args.det_rescue_min_primary),
                merge_iou_thres=float(args.det_merge_iou),
            ),
        )

    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (PROJECT_ROOT / out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_video_path: Path | None = None
    if args.out_video:
        out_video_path = Path(args.out_video)
        if not out_video_path.is_absolute():
            out_video_path = (PROJECT_ROOT / out_video_path).resolve()
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

    if args.out_summary:
        out_summary = Path(args.out_summary)
        if not out_summary.is_absolute():
            out_summary = (PROJECT_ROOT / out_summary).resolve()
    else:
        out_summary = out_jsonl.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    if args.out_gallery_npz:
        out_gallery = Path(args.out_gallery_npz)
        if not out_gallery.is_absolute():
            out_gallery = (PROJECT_ROOT / out_gallery).resolve()
        out_gallery.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_gallery = None

    if args.out_gallery_manifest:
        out_gallery_manifest = Path(args.out_gallery_manifest)
        if not out_gallery_manifest.is_absolute():
            out_gallery_manifest = (PROJECT_ROOT / out_gallery_manifest).resolve()
    elif out_gallery is not None:
        out_gallery_manifest = out_gallery.with_suffix(".manifest.json")
    else:
        out_gallery_manifest = None
    if out_gallery_manifest is not None:
        out_gallery_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.out_unknown_manifest:
        out_unknown_manifest = Path(args.out_unknown_manifest)
        if not out_unknown_manifest.is_absolute():
            out_unknown_manifest = (PROJECT_ROOT / out_unknown_manifest).resolve()
    else:
        out_unknown_manifest = out_jsonl.with_suffix(".unknown_groups.json")
    out_unknown_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.out_unknown_review_html:
        out_unknown_review_html = Path(args.out_unknown_review_html)
        if not out_unknown_review_html.is_absolute():
            out_unknown_review_html = (PROJECT_ROOT / out_unknown_review_html).resolve()
    else:
        out_unknown_review_html = out_jsonl.with_suffix(".unknown_groups.review.html")
    out_unknown_review_html.parent.mkdir(parents=True, exist_ok=True)

    unknown_samples_dir = out_unknown_manifest.parent / f"{out_jsonl.stem}_unknown_faces"

    if args.demo_synthetic:
        stream = _synthetic_stream(max_frames=int(args.max_frames))
    else:
        assert detector is not None
        stream = _video_stream(
            source=args.source,
            detector=detector,
            max_frames=int(args.max_frames),
            loop_source=bool(args.loop_source),
        )

    writer = None
    writer_path: Path | None = None
    writer_codec: str | None = None

    frames_processed = 0
    total_detections = 0
    total_observations = 0
    accepted_observations = 0
    recognized_observations = 0
    match_attempts = 0
    match_reject_reasons: dict[str, int] = defaultdict(int)
    reid_stranger_retry_interval = int(args.reid_stranger_retry_interval)

    track_accept_counts: dict[int, int] = {}
    track_magnitude_sum: dict[int, float] = {}
    track_magnitude_count: dict[int, int] = {}
    track_identity_cache: dict[int, dict[str, object]] = {}
    reid_last_attempt_accept_count: dict[int, int] = {}

    unknown_index = IdentityIndex(dim=emb_dim, use_faiss=True)
    unknown_next_group_id = 0
    unknown_groups: dict[int, dict[str, object]] = {}
    unknown_group_sample_count: dict[int, int] = {}
    unknown_group_last_sample_frame: dict[int, int] = {}
    unknown_group_registered_tracks: dict[int, set[int]] = {}
    auto_register_candidates: dict[int, list[dict[str, object]]] = {}
    auto_registered_identity_ids: set[int] = set()
    auto_registered_tracks = 0
    auto_registered_embeddings = 0
    stranger_session_dir: Path | None = None

    if index.ids:
        next_auto_identity_id = max(int(args.auto_register_id_offset), max(int(v) for v in index.ids) + 1)
    else:
        next_auto_identity_id = int(args.auto_register_id_offset)

    run_start_time = time.perf_counter()
    stop_reason = "stream_end_or_max_frames"

    print(f"[pipeline] config={args.config}")
    print(f"[pipeline] checkpoint={ckpt_path}")
    print(f"[pipeline] device={device.type} amp={use_amp}")
    print(f"[pipeline] quality_gate=[{quality_min:.3f}, {quality_max:.3f}]")
    print(
        "[pipeline] liveness="
        f"mode={args.liveness_mode} threshold={float(args.live_threshold):.3f} "
        f"tta={int(not args.no_liveness_tta)} every={max(0, int(args.liveness_every))}"
    )
    if args.liveness_mode == "silent_face":
        print(
            "[pipeline] liveness_silent_face="
            f"model={silent_face_model_path} device={silent_face_device_used} "
            f"live_class_index={int(args.liveness_silent_face_live_class_index)} "
            f"input_color={str(args.liveness_silent_face_input_color).lower()}"
        )
    print(
        "[pipeline] detector="
        f"conf={args.det_conf:.3f} iou={args.det_iou:.3f} imgsz={args.det_imgsz} "
        f"rescue={int(not args.det_disable_rescue_pass)} rescue_conf={args.det_rescue_conf:.3f} "
        f"rescue_imgsz={args.det_rescue_imgsz}"
    )
    print(f"[pipeline] tracker={pipeline.track_manager.backend}")
    print(f"[pipeline] tracker_max_missed_frames={int(pipeline.track_manager.max_missed_frames)}")
    if str(pipeline.track_manager.backend).lower() == "deepsort":
        print(
            "[pipeline] tracker_deepsort="
            f"n_init={int(pipeline.track_manager.deepsort_n_init)} "
            f"max_iou={float(pipeline.track_manager.deepsort_max_iou_distance):.3f} "
            f"max_cos={float(pipeline.track_manager.deepsort_max_cosine_distance):.3f} "
            f"nn_budget={pipeline.track_manager.deepsort_nn_budget} "
            f"nms_max_overlap={float(pipeline.track_manager.deepsort_nms_max_overlap):.3f} "
            f"gating_only_position={int(bool(pipeline.track_manager.deepsort_gating_only_position))}"
        )
    print("[pipeline] unknown_grouping=faiss_online")
    if args.cluster_sync or int(args.cluster_every) != 64 or float(args.cluster_interval_sec) != 2.0:
        print("[warn] --cluster-sync/--cluster-every/--cluster-interval-sec are deprecated and ignored")
    print(f"[pipeline] gallery_size={num_gallery}")
    print(
        "[pipeline] face_db="
        f"root={face_db_root} known_enabled={int(bool(args.known_db_use))} "
        f"known_mode={args.known_db_retrieval_mode} stranger_enabled={int(bool(args.stranger_db_use))}"
    )
    print(
        "[pipeline] known_db_stats="
        f"identities_total={int(known_db_stats.get('identities_total', 0))} "
        f"identities_with_embeddings={int(known_db_stats.get('identities_with_embeddings', 0))} "
        f"photos_total={int(known_db_stats.get('photos_total', 0))} "
        f"vectors_loaded={int(known_db_stats.get('vectors_loaded', 0))}"
    )
    print(
        "[pipeline] known_db_refresh="
        f"enabled={int(bool(args.known_db_refresh_from_photos))} "
        f"identities_scanned={int(known_db_refresh_stats.get('identities_scanned', 0))} "
        f"identities_with_photos={int(known_db_refresh_stats.get('identities_with_photos', 0))} "
        f"identities_updated={int(known_db_refresh_stats.get('identities_updated', 0))} "
        f"photos_total={int(known_db_refresh_stats.get('photos_total', 0))} "
        f"photos_embedded={int(known_db_refresh_stats.get('photos_embedded', 0))} "
        f"photos_failed={int(known_db_refresh_stats.get('photos_failed', 0))}"
    )
    print(
        "[pipeline] strict_match="
        f"threshold={float(args.match_threshold):.3f} "
        f"topk={max(1, int(args.match_topk))} "
        f"min_margin={float(args.match_min_margin):.3f}"
    )
    print(
        "[pipeline] unknown_gate="
        f"group_threshold={float(args.unknown_group_threshold):.3f} "
        f"min_track_frames={max(1, int(args.unknown_min_track_frames))} "
        f"min_mean_magnitude={unknown_min_mean_magnitude:.3f}"
    )
    print(
        "[pipeline] gallery_enroll_gate="
        f"min_track_frames={max(1, int(args.gallery_min_track_frames))} "
        f"dedupe_threshold={float(args.gallery_dedupe_threshold):.3f} "
        f"min_mean_magnitude={gallery_min_mean_magnitude:.3f}"
    )
    if bool(args.auto_register_unknowns):
        print(
            "[pipeline] auto_register_unknowns="
            f"enabled=1 selection={args.auto_register_selection} "
            f"max_embeddings={max(1, int(args.auto_register_max_embeddings))} "
            f"min_track_frames={int(auto_register_min_track_frames)} "
            f"min_mean_magnitude={float(auto_register_min_mean_magnitude):.3f} "
            f"id_offset={int(args.auto_register_id_offset)}"
        )
    if bool(args.loop_source):
        print("[pipeline] source_looping=enabled")
    if identity_name_map:
        preview_items = list(sorted(identity_name_map.items()))[:6]
        preview_text = ", ".join(f"{k}:{v}" for k, v in preview_items)
        suffix = "..." if len(identity_name_map) > len(preview_items) else ""
        print(f"[pipeline] identity_names={len(identity_name_map)} {preview_text}{suffix}")
    print(
        "[pipeline] reid="
        f"once_per_track={int(bool(args.reid_once_per_track))} "
        f"min_track_frames={reid_min_track_frames} "
        f"stranger_retry_interval={int(reid_stranger_retry_interval)}"
    )
    print(f"[pipeline] output_jsonl={out_jsonl}")
    print(f"[pipeline] output_unknown_manifest={out_unknown_manifest}")

    with out_jsonl.open("w", encoding="utf-8") as fp:
        try:
            for frame_idx, frame_bgr, detections in stream:
                frames_processed += 1
                total_detections += len(detections)

                observations = pipeline.process_frame(frame_bgr=frame_bgr, detections=detections, frame_idx=frame_idx)
                total_observations += len(observations)

                active_track_ids = {int(tid) for tid in pipeline.track_manager.tracks.keys()}
                if track_identity_cache:
                    track_identity_cache = {tid: st for tid, st in track_identity_cache.items() if tid in active_track_ids}
                if reid_last_attempt_accept_count:
                    reid_last_attempt_accept_count = {
                        tid: cnt
                        for tid, cnt in reid_last_attempt_accept_count.items()
                        if tid in active_track_ids
                    }
                if auto_register_candidates:
                    auto_register_candidates = {
                        tid: samples
                        for tid, samples in auto_register_candidates.items()
                        if tid in active_track_ids
                    }

                rows_for_draw: list[dict] = []
                for obs in observations:
                    track_id = int(obs.track_id)
                    cached_identity = track_identity_cache.get(track_id)
                    identity_id = None if cached_identity is None else cached_identity.get("identity_id")
                    identity_name = None if cached_identity is None else cached_identity.get("identity_name")
                    match_score = None if cached_identity is None else cached_identity.get("match_score")
                    match_second_score = None if cached_identity is None else cached_identity.get("match_second_score")
                    match_margin = None if cached_identity is None else cached_identity.get("match_margin")
                    match_decision = None if cached_identity is None else cached_identity.get("match_decision")
                    cluster_label = None if cached_identity is None else cached_identity.get("cluster_label")
                    retrieval_top_identity_id = None if cached_identity is None else cached_identity.get("retrieval_top_identity_id")
                    reid_attempted = False
                    track_mean_magnitude = None

                    track_mag_count = int(track_magnitude_count.get(track_id, 0))
                    if track_mag_count > 0:
                        track_mean_magnitude = float(track_magnitude_sum.get(track_id, 0.0) / float(track_mag_count))

                    if obs.is_live and obs.quality_pass:
                        accepted_observations += 1
                        track_accept_counts[track_id] = track_accept_counts.get(track_id, 0) + 1
                        track_magnitude_sum[track_id] = track_magnitude_sum.get(track_id, 0.0) + float(obs.magnitude)
                        track_magnitude_count[track_id] = track_magnitude_count.get(track_id, 0) + 1
                        track_mean_magnitude = float(track_magnitude_sum[track_id] / float(track_magnitude_count[track_id]))

                        if bool(args.auto_register_unknowns) and identity_id is None:
                            _collect_registration_candidate(
                                track_candidates=auto_register_candidates,
                                track_id=track_id,
                                frame_idx=frame_idx,
                                magnitude=float(obs.magnitude),
                                embedding=obs.embedding,
                                selection=str(args.auto_register_selection),
                                max_keep=max(1, int(args.auto_register_candidate_pool_size)),
                            )

                        if identity_id is not None:
                            recognized_observations += 1
                        else:
                            accept_count = int(track_accept_counts[track_id])
                            should_lookup = accept_count >= reid_min_track_frames
                            if should_lookup:
                                last_attempt = reid_last_attempt_accept_count.get(track_id)
                                if last_attempt is not None:
                                    delta = int(accept_count - int(last_attempt))
                                    if bool(args.reid_once_per_track):
                                        if int(reid_stranger_retry_interval) <= 0:
                                            should_lookup = False
                                        else:
                                            should_lookup = delta >= int(reid_stranger_retry_interval)
                                    else:
                                        interval = 1 if int(reid_stranger_retry_interval) <= 0 else int(
                                            reid_stranger_retry_interval
                                        )
                                        should_lookup = delta >= interval

                            if should_lookup:
                                pooled = pipeline.pooled_track_embedding(track_id)
                                if pooled is not None:
                                    reid_attempted = True
                                    pooled = _normalize_embedding(pooled)
                                    reid_last_attempt_accept_count[track_id] = int(accept_count)
                                    match_attempts += 1
                                    (
                                        top_identity_id,
                                        top_score,
                                        second_score,
                                        top_margin,
                                        decision,
                                    ) = _strict_identity_decision(
                                        index=index,
                                        embedding=pooled,
                                        top_k=max(1, int(args.match_topk)),
                                        min_score=float(args.match_threshold),
                                        min_margin=float(args.match_min_margin),
                                    )

                                    match_score = top_score
                                    match_second_score = second_score
                                    match_margin = top_margin
                                    retrieval_top_identity_id = int(top_identity_id) if top_identity_id is not None else None

                                    if top_identity_id is not None:
                                        identity_id = int(top_identity_id)
                                        match_score = float(top_score) if top_score is not None else None
                                        match_second_score = float(second_score) if second_score is not None else None
                                        match_margin = float(top_margin) if top_margin is not None else None
                                        identity_name = identity_name_map.get(identity_id, f"Person-{identity_id}")
                                        cluster_label = None
                                        match_decision = str(decision)
                                        recognized_observations += 1
                                        track_identity_cache[track_id] = {
                                            "identity_id": identity_id,
                                            "identity_name": identity_name,
                                            "match_score": match_score,
                                            "match_second_score": match_second_score,
                                            "match_margin": match_margin,
                                            "match_decision": match_decision,
                                            "cluster_label": None,
                                            "retrieval_top_identity_id": retrieval_top_identity_id,
                                        }
                                    else:
                                        match_reject_reasons[str(decision)] += 1

                                        cluster_decision = "pending"
                                        can_group_unknown = (
                                            track_accept_counts[track_id] >= max(1, int(args.unknown_min_track_frames))
                                            and track_mean_magnitude is not None
                                            and float(track_mean_magnitude) >= float(unknown_min_mean_magnitude)
                                        )
                                        can_auto_register = (
                                            bool(args.auto_register_unknowns)
                                            and track_accept_counts[track_id] >= int(auto_register_min_track_frames)
                                            and track_mean_magnitude is not None
                                            and float(track_mean_magnitude) >= float(auto_register_min_mean_magnitude)
                                        )
                                        should_assign_group = bool(can_group_unknown or can_auto_register)
                                        auto_registered_now = False

                                        if should_assign_group:
                                            unknown_hit = unknown_index.search(pooled, k=1)
                                            if unknown_hit and float(unknown_hit[0].score) >= float(args.unknown_group_threshold):
                                                cluster_label = int(unknown_hit[0].identity_id)
                                                cluster_decision = "matched_group"
                                                group_similarity = float(unknown_hit[0].score)
                                            else:
                                                cluster_label = int(unknown_next_group_id)
                                                unknown_next_group_id += 1
                                                cluster_decision = "new_group"
                                                group_similarity = float(unknown_hit[0].score) if unknown_hit else None

                                            unknown_index.add(identity_id=cluster_label, embedding=pooled)
                                            identity_name = f"{args.unknown_name_prefix}-{cluster_label}"

                                            st = unknown_groups.get(cluster_label)
                                            if st is None:
                                                st = {
                                                    "group_id": int(cluster_label),
                                                    "track_ids": set(),
                                                    "first_frame": int(frame_idx),
                                                    "last_frame": int(frame_idx),
                                                    "sum_track_magnitude": 0.0,
                                                    "num_track_updates": 0,
                                                    "max_similarity_to_group": None,
                                                    "prototype_sum": np.zeros((emb_dim,), dtype=np.float32),
                                                    "prototype_count": 0,
                                                    "samples": [],
                                                    "auto_identity_id": None,
                                                    "auto_identity_name": None,
                                                    "auto_registered_track_count": 0,
                                                    "last_auto_registered_frame": None,
                                                }
                                                unknown_groups[cluster_label] = st

                                            st["track_ids"].add(int(track_id))
                                            st["first_frame"] = min(int(st["first_frame"]), int(frame_idx))
                                            st["last_frame"] = max(int(st["last_frame"]), int(frame_idx))
                                            st["sum_track_magnitude"] = float(st["sum_track_magnitude"]) + float(track_mean_magnitude)
                                            st["num_track_updates"] = int(st["num_track_updates"]) + 1
                                            prev_proto_sum = np.asarray(st.get("prototype_sum"), dtype=np.float32).reshape(-1)
                                            if prev_proto_sum.size != emb_dim:
                                                prev_proto_sum = np.zeros((emb_dim,), dtype=np.float32)
                                            st["prototype_sum"] = prev_proto_sum + pooled
                                            st["prototype_count"] = int(st.get("prototype_count", 0)) + 1
                                            if group_similarity is not None:
                                                prev_sim = st.get("max_similarity_to_group")
                                                if prev_sim is None or float(group_similarity) > float(prev_sim):
                                                    st["max_similarity_to_group"] = float(group_similarity)

                                            if bool(args.auto_register_unknowns) and can_auto_register:
                                                reg_tracks = unknown_group_registered_tracks.setdefault(int(cluster_label), set())

                                                auto_identity_id_raw = st.get("auto_identity_id")
                                                if auto_identity_id_raw is None:
                                                    st["auto_identity_id"] = int(next_auto_identity_id)
                                                    next_auto_identity_id += 1
                                                    auto_registered_identity_ids.add(int(st["auto_identity_id"]))

                                                auto_identity_id = int(st["auto_identity_id"])
                                                if track_id not in reg_tracks:
                                                    selected = _select_registration_candidates(
                                                        candidates=auto_register_candidates.get(track_id, []),
                                                        selection=str(args.auto_register_selection),
                                                        max_count=max(1, int(args.auto_register_max_embeddings)),
                                                    )
                                                    if not selected:
                                                        selected = [
                                                            {
                                                                "frame_idx": int(frame_idx),
                                                                "magnitude": float(track_mean_magnitude),
                                                                "embedding": pooled,
                                                            }
                                                        ]

                                                    added_count = 0
                                                    for sel in selected:
                                                        sel_emb = np.asarray(sel.get("embedding"), dtype=np.float32).reshape(-1)
                                                        if sel_emb.size != emb_dim:
                                                            continue
                                                        index.add(identity_id=auto_identity_id, embedding=sel_emb)
                                                        added_count += 1

                                                    if added_count > 0:
                                                        reg_tracks.add(track_id)
                                                        auto_registered_tracks += 1
                                                        auto_registered_embeddings += int(added_count)
                                                        st["auto_registered_track_count"] = int(st.get("auto_registered_track_count", 0)) + 1
                                                        st["last_auto_registered_frame"] = int(frame_idx)

                                                auto_identity_name = str(
                                                    st.get("auto_identity_name")
                                                    or f"{args.auto_register_name_prefix}-{auto_identity_id}"
                                                )
                                                st["auto_identity_name"] = auto_identity_name
                                                identity_name_map[auto_identity_id] = auto_identity_name

                                                identity_id = int(auto_identity_id)
                                                identity_name = auto_identity_name
                                                recognized_observations += 1
                                                match_decision = f"{decision}|{cluster_decision}|auto_registered"
                                                auto_registered_now = True
                                                track_identity_cache[track_id] = {
                                                    "identity_id": identity_id,
                                                    "identity_name": identity_name,
                                                    "match_score": match_score,
                                                    "match_second_score": match_second_score,
                                                    "match_margin": match_margin,
                                                    "match_decision": match_decision,
                                                    "cluster_label": int(cluster_label),
                                                    "retrieval_top_identity_id": retrieval_top_identity_id,
                                                }

                                            if not auto_registered_now:
                                                match_decision = f"{decision}|{cluster_decision}"
                                                track_identity_cache[track_id] = {
                                                    "identity_id": None,
                                                    "identity_name": identity_name,
                                                    "match_score": match_score,
                                                    "match_second_score": match_second_score,
                                                    "match_margin": match_margin,
                                                    "match_decision": match_decision,
                                                    "cluster_label": int(cluster_label),
                                                    "retrieval_top_identity_id": retrieval_top_identity_id,
                                                }
                                                identity_id = None
                                        else:
                                            cluster_label = None
                                            identity_name = str(args.unknown_name_prefix)
                                            if bool(args.auto_register_unknowns) and not can_auto_register:
                                                match_decision = f"{decision}|pending_auto_register"
                                            else:
                                                match_decision = f"{decision}|pending_unknown_group"
                                            identity_id = None

                    if cluster_label is not None:
                        group_id = int(cluster_label)
                        st = unknown_groups.get(group_id)
                        if st is not None:
                            max_samples = max(1, int(args.unknown_max_samples_per_group))
                            if int(unknown_group_sample_count.get(group_id, 0)) < max_samples:
                                last_frame = int(unknown_group_last_sample_frame.get(group_id, -10**9))
                                if int(frame_idx) - last_frame >= max(0, int(args.unknown_sample_min_gap)):
                                    crop = _crop_face_patch(frame_bgr=frame_bgr, bbox_xyxy=obs.bbox_xyxy)
                                    if crop is not None:
                                        unknown_samples_dir.mkdir(parents=True, exist_ok=True)
                                        sample_name = f"group_{group_id:04d}_track_{track_id:05d}_frame_{int(frame_idx):06d}.jpg"
                                        sample_path = unknown_samples_dir / sample_name
                                        if cv2.imwrite(str(sample_path), crop):
                                            try:
                                                rel_path = sample_path.relative_to(out_unknown_manifest.parent).as_posix()
                                            except ValueError:
                                                rel_path = str(sample_path)
                                            st["samples"].append(rel_path)
                                            unknown_group_sample_count[group_id] = int(unknown_group_sample_count.get(group_id, 0)) + 1
                                            unknown_group_last_sample_frame[group_id] = int(frame_idx)

                    row = {
                        "frame_idx": int(frame_idx),
                        "track_id": track_id,
                        "bbox_xyxy": [float(v) for v in obs.bbox_xyxy],
                        "track_accept_count": int(track_accept_counts.get(track_id, 0)),
                        "track_mean_magnitude": float(track_mean_magnitude) if track_mean_magnitude is not None else None,
                        "magnitude": float(obs.magnitude),
                        "liveness_score": float(obs.liveness_score),
                        "is_live": bool(obs.is_live),
                        "quality_pass": bool(obs.quality_pass),
                        "reid_attempted": bool(reid_attempted),
                        "identity_id": identity_id,
                        "identity_name": identity_name,
                        "retrieval_top_identity_id": (
                            int(retrieval_top_identity_id)
                            if retrieval_top_identity_id is not None
                            else None
                        ),
                        "match_score": match_score,
                        "match_second_score": match_second_score,
                        "match_margin": match_margin,
                        "match_decision": match_decision,
                        "cluster_label": cluster_label,
                    }
                    fp.write(json.dumps(row, ensure_ascii=True) + "\n")
                    rows_for_draw.append(row)

                if out_video_path is not None:
                    if writer is None:
                        h, w = frame_bgr.shape[:2]
                        writer, writer_path, writer_codec = _open_annotated_writer(
                            out_video_path=out_video_path,
                            width=w,
                            height=h,
                            fps=25.0,
                        )
                        print(f"[pipeline] video_writer_codec={writer_codec} path={writer_path}")
                    for row in rows_for_draw:
                        _draw_observation(frame_bgr, row, show_track_id=bool(args.show_track_id))
                    # Draw ghost boxes for identified tracks coasting on Kalman prediction.
                    # Only for short gaps: Kalman position drifts and becomes misleading after
                    # more than ~10 missed frames, so suppress beyond that.
                    _GHOST_MAX_MISSED = 10
                    observed_tids = {int(row["track_id"]) for row in rows_for_draw}
                    for ghost_track in pipeline.track_manager.tracks.values():
                        ghost_tid = int(ghost_track.track_id)
                        if ghost_tid in observed_tids:
                            continue
                        if int(ghost_track.missed_frames) > _GHOST_MAX_MISSED:
                            continue
                        ghost_cached = track_identity_cache.get(ghost_tid)
                        if ghost_cached is None:
                            continue
                        ghost_name = ghost_cached.get("identity_name")
                        if not ghost_name:
                            continue
                        _draw_ghost_track(
                            frame_bgr,
                            ghost_track.bbox_xyxy,
                            str(ghost_name),
                            int(ghost_track.missed_frames),
                            show_track_id=bool(args.show_track_id),
                            track_id=ghost_tid,
                        )
                    writer.write(frame_bgr)

                if args.print_every > 0 and (frames_processed % int(args.print_every) == 0):
                    elapsed_sec = max(1e-6, float(time.perf_counter() - run_start_time))
                    fps = float(frames_processed) / elapsed_sec
                    print(
                        f"[progress] frame={frames_processed} det={total_detections} "
                        f"obs={total_observations} accepted={accepted_observations} "
                        f"recognized={recognized_observations} match_attempts={match_attempts} "
                        f"unknown_groups={len(unknown_groups)} fps={fps:.2f}"
                    )
        except KeyboardInterrupt:
            stop_reason = "manual_interrupt"
            print("[pipeline] interrupted by user, finalizing outputs...")

    if writer is not None:
        writer.release()
        if out_video_path is not None and writer_path is not None and writer_codec == "mp4v":
            transcoded = _transcode_mp4_to_h264(src_path=writer_path, dst_path=out_video_path)
            if transcoded:
                writer_path.unlink(missing_ok=True)
                print(f"[pipeline] transcoded annotated video to H.264: {out_video_path}")
            else:
                writer_path.replace(out_video_path)
                print(
                    "[warn] ffmpeg H.264 transcode unavailable/failed; kept mp4v output. "
                    "If VS Code cannot play it, run: ffmpeg -y -i <in.mp4> -c:v libx264 -pix_fmt yuv420p <out.mp4>"
                )

    cluster_hist: dict[str, int] = {}
    unknown_clusters_export: list[dict[str, object]] = []
    unknown_group_embeddings: dict[int, np.ndarray] = {}
    for group_id in sorted(unknown_groups.keys()):
        st = unknown_groups[group_id]
        track_ids = sorted(int(tid) for tid in st.get("track_ids", set()))
        num_updates = max(1, int(st.get("num_track_updates", 0)))
        avg_track_magnitude = float(st.get("sum_track_magnitude", 0.0)) / float(num_updates)
        samples = [str(v) for v in st.get("samples", [])]
        prototype_count = int(st.get("prototype_count", 0))
        if prototype_count > 0:
            proto_sum = np.asarray(st.get("prototype_sum"), dtype=np.float32).reshape(-1)
            if proto_sum.size == emb_dim:
                unknown_group_embeddings[int(group_id)] = _normalize_embedding(
                    proto_sum / float(max(1, prototype_count))
                )
        rec = {
            "group_id": int(group_id),
            "name": f"{args.unknown_name_prefix}-{int(group_id)}",
            "num_tracks": int(len(track_ids)),
            "track_ids": track_ids,
            "first_frame": int(st.get("first_frame", 0)),
            "last_frame": int(st.get("last_frame", 0)),
            "avg_track_magnitude": float(avg_track_magnitude),
            "max_similarity_to_group": st.get("max_similarity_to_group"),
            "prototype_count": int(prototype_count),
            "auto_identity_id": st.get("auto_identity_id"),
            "auto_identity_name": st.get("auto_identity_name"),
            "auto_registered_track_count": int(st.get("auto_registered_track_count", 0)),
            "last_auto_registered_frame": st.get("last_auto_registered_frame"),
            "samples": samples,
        }
        unknown_clusters_export.append(rec)
        cluster_hist[str(int(group_id))] = int(len(track_ids))

    unknown_manifest = {
        "manifest_path": str(out_unknown_manifest),
        "unknown_name_prefix": str(args.unknown_name_prefix),
        "group_threshold": float(args.unknown_group_threshold),
        "min_track_frames": int(max(1, int(args.unknown_min_track_frames))),
        "min_mean_magnitude": float(unknown_min_mean_magnitude),
        "auto_register_unknowns": bool(args.auto_register_unknowns),
        "auto_register_selection": str(args.auto_register_selection),
        "auto_registered_identities": int(len(auto_registered_identity_ids)),
        "auto_registered_tracks": int(auto_registered_tracks),
        "auto_registered_embeddings": int(auto_registered_embeddings),
        "num_groups": int(len(unknown_clusters_export)),
        "clusters": unknown_clusters_export,
    }
    out_unknown_manifest.write_text(json.dumps(unknown_manifest, indent=2), encoding="utf-8")
    _write_unknown_review_html(
        manifest=unknown_manifest,
        out_html=out_unknown_review_html,
        unknown_name_prefix=str(args.unknown_name_prefix),
    )
    print(f"[pipeline] wrote_unknown_manifest={out_unknown_manifest}")
    print(f"[pipeline] wrote_unknown_review_html={out_unknown_review_html}")

    if bool(args.stranger_db_use):
        stranger_session_name = str(args.stranger_db_session_name).strip() or out_jsonl.stem
        stranger_session_dir = persist_stranger_session(
            db_root=face_db_root,
            session_name=stranger_session_name,
            unknown_manifest=unknown_manifest,
            unknown_manifest_parent=out_unknown_manifest.parent,
            group_embeddings=unknown_group_embeddings,
        )
        print(f"[pipeline] wrote_stranger_session={stranger_session_dir}")

    enrolled_identities = 0
    gallery_tracks_considered = 0
    gallery_tracks_skipped_low_frames = 0
    gallery_tracks_skipped_low_quality = 0
    gallery_tracks_merged = 0
    gallery_tracks_missing_embedding = 0
    gallery_manifest_tracks: list[dict[str, object]] = []

    if out_gallery is not None:
        min_track_frames = max(1, int(args.gallery_min_track_frames))
        dedupe_threshold = float(args.gallery_dedupe_threshold)
        dedupe_index = IdentityIndex(dim=emb_dim, use_faiss=True)
        proto_sums: dict[int, np.ndarray] = {}
        proto_counts: dict[int, int] = {}
        next_gallery_id = int(args.gallery_id_offset)

        for track_id in sorted(pipeline.track_buffers.keys()):
            accepted_count = int(track_accept_counts.get(track_id, 0))
            if accepted_count < min_track_frames:
                gallery_tracks_skipped_low_frames += 1
                continue

            gallery_tracks_considered += 1

            mag_count = int(track_magnitude_count.get(track_id, 0))
            mean_mag = float(track_magnitude_sum.get(track_id, 0.0) / max(1, mag_count))
            if mean_mag < float(gallery_min_mean_magnitude):
                gallery_tracks_skipped_low_quality += 1
                gallery_manifest_tracks.append(
                    {
                        "track_id": int(track_id),
                        "accepted_count": int(accepted_count),
                        "mean_magnitude": float(mean_mag),
                        "status": "skip_low_quality",
                    }
                )
                continue

            pooled = pipeline.pooled_track_embedding(track_id)
            if pooled is None:
                gallery_tracks_missing_embedding += 1
                gallery_manifest_tracks.append(
                    {
                        "track_id": int(track_id),
                        "accepted_count": int(accepted_count),
                        "mean_magnitude": float(mean_mag),
                        "status": "skip_missing_embedding",
                    }
                )
                continue

            pooled = _normalize_embedding(pooled)
            hits = dedupe_index.search(pooled, k=1)

            merged_into_existing = False
            best_existing_score = float(hits[0].score) if hits else None
            if hits and float(hits[0].score) >= dedupe_threshold:
                assigned_id = int(hits[0].identity_id)
                merged_into_existing = True
                gallery_tracks_merged += 1
            else:
                assigned_id = int(next_gallery_id)
                next_gallery_id += 1

            dedupe_index.add(identity_id=assigned_id, embedding=pooled)

            if assigned_id in proto_sums:
                proto_sums[assigned_id] = proto_sums[assigned_id] + pooled
                proto_counts[assigned_id] = int(proto_counts[assigned_id]) + 1
            else:
                proto_sums[assigned_id] = np.asarray(pooled, dtype=np.float32)
                proto_counts[assigned_id] = 1

            gallery_manifest_tracks.append(
                {
                    "track_id": int(track_id),
                    "assigned_identity_id": int(assigned_id),
                    "accepted_count": int(accepted_count),
                    "mean_magnitude": float(mean_mag),
                    "best_existing_score": best_existing_score,
                    "status": "merged" if merged_into_existing else "new_identity",
                }
            )

        if proto_sums:
            sorted_ids = sorted(proto_sums.keys())
            emb_arr = np.stack(
                [_normalize_embedding(proto_sums[gid] / float(max(1, proto_counts[gid]))) for gid in sorted_ids],
                axis=0,
            ).astype(np.float32)
            id_arr = np.asarray(sorted_ids, dtype=np.int64)
        else:
            emb_arr = np.zeros((0, emb_dim), dtype=np.float32)
            id_arr = np.zeros((0,), dtype=np.int64)

        np.savez(out_gallery, embeddings=emb_arr, ids=id_arr)
        enrolled_identities = int(id_arr.shape[0])
        print(f"[pipeline] wrote_gallery={out_gallery} enrolled={enrolled_identities}")

        if out_gallery_manifest is not None:
            gallery_manifest = {
                "gallery_npz": str(out_gallery),
                "dedupe_threshold": float(dedupe_threshold),
                "min_track_frames": int(min_track_frames),
                "min_mean_magnitude": float(gallery_min_mean_magnitude),
                "tracks_considered": int(gallery_tracks_considered),
                "tracks_skipped_low_frames": int(gallery_tracks_skipped_low_frames),
                "tracks_skipped_low_quality": int(gallery_tracks_skipped_low_quality),
                "tracks_skipped_missing_embedding": int(gallery_tracks_missing_embedding),
                "tracks_merged": int(gallery_tracks_merged),
                "enrolled_identities": int(enrolled_identities),
                "tracks": gallery_manifest_tracks,
            }
            out_gallery_manifest.write_text(json.dumps(gallery_manifest, indent=2), encoding="utf-8")
            print(f"[pipeline] wrote_gallery_manifest={out_gallery_manifest}")

    elapsed_wall_sec = float(max(0.0, time.perf_counter() - run_start_time))
    avg_fps = float(frames_processed / elapsed_wall_sec) if elapsed_wall_sec > 1e-6 else 0.0
    print(f"[pipeline] stop_reason={stop_reason} runtime_sec={elapsed_wall_sec:.3f} avg_fps={avg_fps:.3f}")

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "source": "synthetic" if args.demo_synthetic else args.source,
        "loop_source": bool(args.loop_source),
        "face_db_root": str(face_db_root),
        "known_db_root": str(known_db_root),
        "stranger_db_root": str(stranger_db_root),
        "known_db_use": bool(args.known_db_use),
        "known_db_retrieval_mode": str(args.known_db_retrieval_mode),
        "known_db_stats": {
            "identities_total": int(known_db_stats.get("identities_total", 0)),
            "vectors_loaded": int(known_db_stats.get("vectors_loaded", 0)),
            "identities_with_embeddings": int(known_db_stats.get("identities_with_embeddings", 0)),
            "photos_total": int(known_db_stats.get("photos_total", 0)),
        },
        "known_db_refresh_from_photos": bool(args.known_db_refresh_from_photos),
        "known_db_refresh_stats": {
            "identities_scanned": int(known_db_refresh_stats.get("identities_scanned", 0)),
            "identities_with_photos": int(known_db_refresh_stats.get("identities_with_photos", 0)),
            "identities_updated": int(known_db_refresh_stats.get("identities_updated", 0)),
            "photos_total": int(known_db_refresh_stats.get("photos_total", 0)),
            "photos_embedded": int(known_db_refresh_stats.get("photos_embedded", 0)),
            "photos_failed": int(known_db_refresh_stats.get("photos_failed", 0)),
        },
        "stranger_db_use": bool(args.stranger_db_use),
        "stranger_db_session_dir": str(stranger_session_dir) if stranger_session_dir is not None else None,
        "identity_names_path": str(identity_names_path) if identity_names_path is not None else None,
        "liveness_mode": str(args.liveness_mode),
        "liveness_threshold": float(args.live_threshold),
        "liveness_tta_enabled": bool(not args.no_liveness_tta),
        "liveness_every": int(max(0, int(args.liveness_every))),
        "liveness_silent_face_model": str(silent_face_model_path) if silent_face_model_path is not None else None,
        "liveness_silent_face_device": str(silent_face_device_used) if silent_face_device_used is not None else None,
        "liveness_silent_face_live_class_index": (
            int(args.liveness_silent_face_live_class_index) if args.liveness_mode == "silent_face" else None
        ),
        "liveness_silent_face_input_color": (
            str(args.liveness_silent_face_input_color).lower() if args.liveness_mode == "silent_face" else None
        ),
        "tracker_backend": str(pipeline.track_manager.backend),
        "reid_once_per_track": bool(args.reid_once_per_track),
        "reid_min_track_frames": int(reid_min_track_frames),
        "reid_stranger_retry_interval": int(reid_stranger_retry_interval),
        "track_max_missed_frames": int(pipeline.track_manager.max_missed_frames),
        "track_n_init": int(pipeline.track_manager.deepsort_n_init),
        "track_max_iou_distance": float(pipeline.track_manager.deepsort_max_iou_distance),
        "track_max_cosine_distance": float(pipeline.track_manager.deepsort_max_cosine_distance),
        "track_nn_budget": pipeline.track_manager.deepsort_nn_budget,
        "track_nms_max_overlap": float(pipeline.track_manager.deepsort_nms_max_overlap),
        "track_gating_only_position": bool(pipeline.track_manager.deepsort_gating_only_position),
        "strict_match_threshold": float(args.match_threshold),
        "strict_match_topk": int(max(1, int(args.match_topk))),
        "strict_match_min_margin": float(args.match_min_margin),
        "match_attempts": int(match_attempts),
        "match_reject_reasons": {k: int(v) for k, v in sorted(match_reject_reasons.items())},
        "frames_processed": int(frames_processed),
        "detections": int(total_detections),
        "observations": int(total_observations),
        "accepted_observations": int(accepted_observations),
        "recognized_observations": int(recognized_observations),
        "unknown_group_threshold": float(args.unknown_group_threshold),
        "unknown_min_track_frames": int(max(1, int(args.unknown_min_track_frames))),
        "unknown_min_mean_magnitude": float(unknown_min_mean_magnitude),
        "auto_register_unknowns": bool(args.auto_register_unknowns),
        "auto_register_selection": str(args.auto_register_selection),
        "auto_register_max_embeddings": int(max(1, int(args.auto_register_max_embeddings))),
        "auto_register_min_track_frames": int(auto_register_min_track_frames),
        "auto_register_min_mean_magnitude": float(auto_register_min_mean_magnitude),
        "auto_registered_identities": int(len(auto_registered_identity_ids)),
        "auto_registered_tracks": int(auto_registered_tracks),
        "auto_registered_embeddings": int(auto_registered_embeddings),
        "unknown_groups": int(len(unknown_clusters_export)),
        "unknown_cluster_hist": cluster_hist,
        "unknown_manifest": str(out_unknown_manifest),
        "unknown_review_html": str(out_unknown_review_html),
        "enrolled_identities": int(enrolled_identities),
        "gallery_min_track_frames": int(max(1, int(args.gallery_min_track_frames))),
        "gallery_dedupe_threshold": float(args.gallery_dedupe_threshold),
        "gallery_min_mean_magnitude": float(gallery_min_mean_magnitude),
        "gallery_tracks_considered": int(gallery_tracks_considered),
        "gallery_tracks_skipped_low_frames": int(gallery_tracks_skipped_low_frames),
        "gallery_tracks_skipped_low_quality": int(gallery_tracks_skipped_low_quality),
        "gallery_tracks_skipped_missing_embedding": int(gallery_tracks_missing_embedding),
        "gallery_tracks_merged": int(gallery_tracks_merged),
        "stop_reason": str(stop_reason),
        "runtime_seconds": float(elapsed_wall_sec),
        "fps_mean": float(avg_fps),
        "output_jsonl": str(out_jsonl),
        "output_video": str(out_video_path) if out_video_path is not None else None,
        "output_gallery": str(out_gallery) if out_gallery is not None else None,
        "output_gallery_manifest": str(out_gallery_manifest) if out_gallery_manifest is not None else None,
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[summary]")
    print(json.dumps(summary, indent=2))
    print(f"WROTE {out_summary}")


if __name__ == "__main__":
    main()
