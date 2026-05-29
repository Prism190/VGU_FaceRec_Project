#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train manifest from RecordIO (.rec/.idx)")
    parser.add_argument("--rec-path", required=True, help="Path to train.rec")
    parser.add_argument("--idx-path", required=True, help="Path to train.idx")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-id-map", required=True)
    parser.add_argument("--identity-prefix", default="id", help="Identity prefix in id map")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec_path = Path(args.rec_path).resolve()
    idx_path = Path(args.idx_path).resolve()
    out_manifest = Path(args.output_manifest).resolve()
    out_id_map = Path(args.output_id_map).resolve()

    if not rec_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Missing rec/idx: {rec_path} | {idx_path}")

    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    import mxnet as mx

    reader = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")
    header0, _ = mx.recordio.unpack(reader.read_idx(0))
    if not hasattr(header0.label, "__len__") or len(header0.label) < 1:
        raise RuntimeError("Unexpected RecordIO header at idx 0")

    first_header_idx = int(header0.label[0])
    if first_header_idx <= 1:
        raise RuntimeError(f"Unexpected first header idx: {first_header_idx}")

    rows: list[dict] = []
    counts: dict[int, int] = {}

    upper = first_header_idx
    if args.max_records > 0:
        upper = min(upper, 1 + int(args.max_records))

    for rec_idx in range(1, upper):
        packed = reader.read_idx(rec_idx)
        if packed is None:
            continue
        header, image_bytes = mx.recordio.unpack(packed)
        if not isinstance(header.label, (int, float)):
            continue
        if image_bytes is None or len(image_bytes) == 0:
            continue

        label = int(header.label)
        identity = f"{args.identity_prefix}_{label:07d}"
        rows.append(
            {
                "rec_idx": rec_idx,
                "rec_path": str(rec_path),
                "idx_path": str(idx_path),
                "image_path": f"recordio://{rec_idx}",
                "label": label,
                "identity": identity,
            }
        )
        counts[label] = counts.get(label, 0) + 1

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_id_map.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_manifest, index=False, quoting=csv.QUOTE_MINIMAL)

    id_rows = [
        {
            "identity": f"{args.identity_prefix}_{label:07d}",
            "label": label,
            "num_images": count,
        }
        for label, count in sorted(counts.items(), key=lambda x: x[0])
    ]
    pd.DataFrame(id_rows).to_csv(out_id_map, index=False)

    print(f"Wrote manifest: {out_manifest}")
    print(f"Wrote id map:   {out_id_map}")
    print(f"Rows: {len(rows)} | Classes: {len(counts)}")


if __name__ == "__main__":
    main()
