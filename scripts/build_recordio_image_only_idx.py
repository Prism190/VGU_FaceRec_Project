#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-only RecordIO idx by dropping metadata keys beyond image range"
    )
    parser.add_argument("--idx-path", required=True, help="Path to source train.idx")
    parser.add_argument("--output-idx", required=True, help="Path to output filtered idx")
    parser.add_argument(
        "--max-image-key",
        type=int,
        default=0,
        help="Largest image key to keep (if 0, infer from rec header0.label[0]-1)",
    )
    parser.add_argument(
        "--rec-path",
        default="",
        help="Path to train.rec (required when --max-image-key=0)",
    )
    return parser.parse_args()


def infer_max_image_key(rec_path: Path, idx_path: Path) -> int:
    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    import mxnet as mx

    reader = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")
    header0, _ = mx.recordio.unpack(reader.read_idx(0))
    max_image_key = int(header0.label[0]) - 1
    if max_image_key <= 0:
        raise RuntimeError(f"Unexpected header0.label[0]={header0.label[0]}")
    return max_image_key


def main() -> None:
    args = parse_args()
    idx_path = Path(args.idx_path).resolve()
    output_idx = Path(args.output_idx).resolve()

    if not idx_path.exists():
        raise FileNotFoundError(f"Missing idx file: {idx_path}")

    if args.max_image_key > 0:
        max_image_key = int(args.max_image_key)
    else:
        rec_path = Path(args.rec_path).resolve()
        if not rec_path.exists():
            raise FileNotFoundError("--rec-path is required (and must exist) when --max-image-key is not provided")
        max_image_key = infer_max_image_key(rec_path=rec_path, idx_path=idx_path)

    output_idx.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    malformed = 0

    with idx_path.open("r", encoding="utf-8", errors="strict") as fin, output_idx.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) != 2:
                malformed += 1
                continue

            try:
                key = int(parts[0])
            except ValueError:
                malformed += 1
                continue

            if key == 0 or (1 <= key <= max_image_key):
                fout.write(line)
                kept += 1
            else:
                dropped += 1

    print(f"source_idx={idx_path}")
    print(f"output_idx={output_idx}")
    print(f"max_image_key={max_image_key}")
    print(f"kept={kept} dropped={dropped} malformed={malformed}")


if __name__ == "__main__":
    main()
