#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.manifests import build_pairs_manifest_from_rows


EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolve_lfw_image(images_root: Path, identity: str, index: str) -> str:
    stem = f"{identity}_{int(index):04d}"
    for ext in EXTS:
        path = images_root / identity / f"{stem}{ext}"
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError(f"Could not resolve image for {identity} index={index} under {images_root}")


def _parse_lfw_pairs(pairs_txt: Path, images_root: Path) -> list[dict]:
    lines = [line.strip() for line in pairs_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines and lines[0].isdigit():
        lines = lines[1:]

    rows: list[dict] = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            name, idx1, idx2 = parts
            path_a = _resolve_lfw_image(images_root, name, idx1)
            path_b = _resolve_lfw_image(images_root, name, idx2)
            rows.append({"path_a": path_a, "path_b": path_b, "is_same": 1})
        elif len(parts) == 4:
            name1, idx1, name2, idx2 = parts
            path_a = _resolve_lfw_image(images_root, name1, idx1)
            path_b = _resolve_lfw_image(images_root, name2, idx2)
            rows.append({"path_a": path_a, "path_b": path_b, "is_same": 0})
        else:
            raise ValueError(f"Unsupported LFW line format: {line}")
    return rows


def _parse_triplet_rows(protocol_txt: Path, images_root: Path) -> list[dict]:
    rows: list[dict] = []
    for line in protocol_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Expected '<path_a> <path_b> <is_same>' format: {line}")
        path_a = Path(parts[0])
        path_b = Path(parts[1])
        if not path_a.is_absolute():
            path_a = images_root / path_a
        if not path_b.is_absolute():
            path_b = images_root / path_b
        rows.append({"path_a": str(path_a.resolve()), "path_b": str(path_b.resolve()), "is_same": int(parts[2])})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pair-verification CSV manifest")
    parser.add_argument("--format", type=str, choices=["lfw", "triplet"], required=True)
    parser.add_argument("--protocol", type=str, required=True, help="Protocol text file path")
    parser.add_argument("--images-root", type=str, required=True, help="Dataset image root")
    parser.add_argument("--output-csv", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = Path(args.protocol)
    images_root = Path(args.images_root)

    if args.format == "lfw":
        rows = _parse_lfw_pairs(protocol, images_root)
    else:
        rows = _parse_triplet_rows(protocol, images_root)

    build_pairs_manifest_from_rows(rows=rows, output_csv=args.output_csv)
    print(f"Wrote {len(rows)} pairs to {args.output_csv}")


if __name__ == "__main__":
    main()
