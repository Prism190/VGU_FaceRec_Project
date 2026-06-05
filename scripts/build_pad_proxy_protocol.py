#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _stable_seed_from_path(path: Path, base_seed: int) -> int:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) ^ int(base_seed)) & 0xFFFFFFFF


def _resize_face(image_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)


def _attack_print_like(face_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = face_bgr.shape[:2]

    # Simulate print loss: down/up sample + blur + quantization.
    scale = float(rng.uniform(0.45, 0.7))
    ds = cv2.resize(face_bgr, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    us = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)

    blur_k = int(rng.choice([3, 5]))
    us = cv2.GaussianBlur(us, (blur_k, blur_k), sigmaX=float(rng.uniform(0.8, 1.8)))

    levels = int(rng.choice([16, 24, 32]))
    quant = np.round(us.astype(np.float32) / (256.0 / levels)) * (256.0 / levels)

    noise = rng.normal(loc=0.0, scale=float(rng.uniform(2.0, 6.0)), size=quant.shape).astype(np.float32)
    out = np.clip(quant + noise, 0.0, 255.0).astype(np.uint8)

    # Mild contrast and saturation suppression.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(0.6, 0.9))
    hsv[..., 2] *= float(rng.uniform(0.85, 1.05))
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _attack_replay_like(face_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    out = face_bgr.astype(np.float32)

    # Gamma + slight color cast to mimic display/camera recapture.
    gamma = float(rng.uniform(1.1, 1.5))
    out = np.power(np.clip(out / 255.0, 0.0, 1.0), gamma) * 255.0
    cast = np.array(
        [
            float(rng.uniform(0.95, 1.10)),
            float(rng.uniform(0.95, 1.10)),
            float(rng.uniform(0.95, 1.10)),
        ],
        dtype=np.float32,
    )
    out *= cast.reshape(1, 1, 3)

    # Add moire-like horizontal wave pattern.
    yy = np.arange(h, dtype=np.float32).reshape(h, 1)
    wave = np.sin(2.0 * np.pi * yy / float(rng.uniform(3.0, 7.0)))
    wave = np.repeat(wave, w, axis=1)
    wave_amp = float(rng.uniform(4.0, 12.0))
    out += wave_amp * wave[..., None]

    # Add glare spot.
    cx = float(rng.uniform(0.2 * w, 0.8 * w))
    cy = float(rng.uniform(0.2 * h, 0.8 * h))
    rad = float(rng.uniform(0.18, 0.34) * min(h, w))
    yy_grid, xx_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt((xx_grid - cx) ** 2 + (yy_grid - cy) ** 2)
    glare = np.clip(1.0 - rr / max(rad, 1.0), 0.0, 1.0)
    out += float(rng.uniform(18.0, 46.0)) * glare[..., None]

    # Display border cue.
    border = int(max(2, round(min(h, w) * float(rng.uniform(0.02, 0.05)))))
    out[:border, :, :] *= float(rng.uniform(0.5, 0.85))
    out[-border:, :, :] *= float(rng.uniform(0.5, 0.85))
    out[:, :border, :] *= float(rng.uniform(0.5, 0.85))
    out[:, -border:, :] *= float(rng.uniform(0.5, 0.85))

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), sigmaX=float(rng.uniform(0.4, 1.2)))
    return out


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a proxy anti-spoof protocol from live face photos")
    parser.add_argument(
        "--live-source-root",
        default="data/face_db/known/identities",
        help="Directory containing live face photos (recursively scanned)",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/pad_proxy_protocol_v1",
        help="Output root for generated live/spoof images and manifest",
    )
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-live", type=int, default=0, help="Use only first N live photos (0 = all)")
    parser.add_argument("--seed", type=int, default=20260602)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    live_source_root = Path(args.live_source_root)
    if not live_source_root.is_absolute():
        live_source_root = (project_root / live_source_root).resolve()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()

    live_out = out_root / "live"
    spoof_out = out_root / "spoof"
    manifest_path = out_root / "manifest.csv"

    _ensure_clean_dir(live_out)
    _ensure_clean_dir(spoof_out)

    image_paths = _iter_images(live_source_root)
    if int(args.max_live) > 0:
        image_paths = image_paths[: int(args.max_live)]

    if not image_paths:
        raise RuntimeError(f"No live images found under {live_source_root}")

    rows: list[dict[str, str]] = []

    for idx, src_path in enumerate(image_paths):
        bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            continue

        face = _resize_face(bgr, int(args.image_size))
        base_name = f"img_{idx:06d}"

        live_name = f"{base_name}_live.jpg"
        live_path = live_out / live_name
        cv2.imwrite(str(live_path), face)
        rows.append(
            {
                "path": str(live_path.relative_to(out_root).as_posix()),
                "label": "1",
                "attack_type": "live",
                "source": str(src_path.relative_to(live_source_root).as_posix()),
            }
        )

        seed = _stable_seed_from_path(src_path, int(args.seed))
        rng = np.random.default_rng(seed)

        spoof_print = _attack_print_like(face, rng)
        spoof_print_name = f"{base_name}_spoof_print.jpg"
        spoof_print_path = spoof_out / spoof_print_name
        cv2.imwrite(str(spoof_print_path), spoof_print)
        rows.append(
            {
                "path": str(spoof_print_path.relative_to(out_root).as_posix()),
                "label": "0",
                "attack_type": "spoof_print",
                "source": str(src_path.relative_to(live_source_root).as_posix()),
            }
        )

        spoof_replay = _attack_replay_like(face, rng)
        spoof_replay_name = f"{base_name}_spoof_replay.jpg"
        spoof_replay_path = spoof_out / spoof_replay_name
        cv2.imwrite(str(spoof_replay_path), spoof_replay)
        rows.append(
            {
                "path": str(spoof_replay_path.relative_to(out_root).as_posix()),
                "label": "0",
                "attack_type": "spoof_replay",
                "source": str(src_path.relative_to(live_source_root).as_posix()),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["path", "label", "attack_type", "source"])
        writer.writeheader()
        writer.writerows(rows)

    n_live = sum(1 for r in rows if r["label"] == "1")
    n_spoof = sum(1 for r in rows if r["label"] == "0")

    print(f"[pad_proxy] out_root={out_root}")
    print(f"[pad_proxy] manifest={manifest_path}")
    print(f"[pad_proxy] live={n_live} spoof={n_spoof} total={len(rows)}")


if __name__ == "__main__":
    main()
