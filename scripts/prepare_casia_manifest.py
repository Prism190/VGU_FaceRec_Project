#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.manifests import build_casia_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CASIA-WebFace train manifest")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory containing identity subfolders")
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data/manifests/casia_train.csv"),
        help="Output CSV with image_path,label",
    )
    parser.add_argument(
        "--output-id-map",
        type=str,
        default=str(PROJECT_ROOT / "data/manifests/casia_id_map.csv"),
        help="Output CSV with identity,label,num_images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_images, num_classes = build_casia_manifest(
        dataset_root=args.dataset_root,
        output_manifest_csv=args.output_manifest,
        output_identity_map_csv=args.output_id_map,
    )
    print(f"Wrote {num_images} images across {num_classes} identities")
    print(f"Manifest: {args.output_manifest}")
    print(f"ID map:   {args.output_id_map}")


if __name__ == "__main__":
    main()
