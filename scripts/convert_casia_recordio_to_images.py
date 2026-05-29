#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_READER = None
_MX = None
_OUTPUT_ROOT: Path | None = None
_SKIP_EXISTING = True


def _detect_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _sanitize_identity(identity: str) -> str:
    return identity.replace("/", "_").replace("\\", "_")


def _init_worker(rec_path: str, idx_path: str, output_root: str, skip_existing: bool) -> None:
    global _READER, _MX, _OUTPUT_ROOT, _SKIP_EXISTING

    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    import mxnet as mx

    _MX = mx
    _READER = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
    _OUTPUT_ROOT = Path(output_root)
    _SKIP_EXISTING = skip_existing


def _convert_one(task: tuple[int, int, str]) -> tuple[str | None, int, str, int, str | None]:
    rec_idx, label, identity = task

    assert _OUTPUT_ROOT is not None
    assert _READER is not None
    assert _MX is not None

    identity = _sanitize_identity(identity)

    packed = _READER.read_idx(int(rec_idx))
    if packed is None:
        return None, label, identity, 0, f"missing_record:{rec_idx}"

    _, image_bytes = _MX.recordio.unpack(packed)
    if image_bytes is None or len(image_bytes) == 0:
        return None, label, identity, 0, f"empty_image:{rec_idx}"

    ext = _detect_extension(image_bytes)
    out_path = _OUTPUT_ROOT / identity / f"{int(rec_idx):07d}{ext}"

    if _SKIP_EXISTING and out_path.exists():
        return str(out_path.resolve()), label, identity, 0, None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_bytes)
    return str(out_path.resolve()), label, identity, 1, None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert CASIA RecordIO to image files + image manifest")
    parser.add_argument(
        "--input-manifest",
        default=str(project_root / "data/manifests/casia_train.csv"),
        help="CSV with rec_idx/rec_path/idx_path/label/identity columns",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root / "data/processed/casia_webface_images"),
        help="Output root directory for extracted images",
    )
    parser.add_argument(
        "--output-manifest",
        default=str(project_root / "data/manifests/casia_train_images.csv"),
        help="Output CSV with image_path,label,identity",
    )
    parser.add_argument(
        "--output-id-map",
        default=str(project_root / "data/manifests/casia_id_map_images.csv"),
        help="Output identity map CSV",
    )
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    parser.add_argument("--chunksize", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_manifest = Path(args.input_manifest)
    output_root = Path(args.output_root)
    output_manifest = Path(args.output_manifest)
    output_id_map = Path(args.output_id_map)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)
    required = {"rec_idx", "rec_path", "idx_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")

    if "identity" not in df.columns:
        df["identity"] = df["label"].astype(str)

    if args.limit > 0:
        df = df.head(args.limit)

    rec_paths = df["rec_path"].dropna().astype(str).unique().tolist()
    idx_paths = df["idx_path"].dropna().astype(str).unique().tolist()
    if len(rec_paths) != 1 or len(idx_paths) != 1:
        raise ValueError("Expected a single rec_path and idx_path in input manifest")

    rec_path = rec_paths[0]
    idx_path = idx_paths[0]

    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_id_map.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (int(row.rec_idx), int(row.label), str(row.identity))
        for row in df[["rec_idx", "label", "identity"]].itertuples(index=False)
    ]

    rows: list[dict] = []
    created = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_init_worker,
        initargs=(rec_path, idx_path, str(output_root), bool(args.skip_existing)),
        mp_context=mp.get_context("fork"),
    ) as ex:
        it = ex.map(_convert_one, tasks, chunksize=max(1, int(args.chunksize)))
        for image_path, label, identity, created_flag, error in tqdm(it, total=len(tasks), desc="Convert CASIA"):
            if error is not None or image_path is None:
                failed += 1
                continue

            if created_flag:
                created += 1
            else:
                skipped += 1

            rows.append({"image_path": image_path, "label": int(label), "identity": str(identity)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_manifest, index=False, quoting=csv.QUOTE_MINIMAL)

    id_map = (
        out_df.groupby(["identity", "label"], as_index=False)
        .size()
        .rename(columns={"size": "num_images"})
        .sort_values("label")
    )
    id_map.to_csv(output_id_map, index=False)

    print(f"Converted rows written: {len(out_df)}")
    print(f"Created images: {created}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed rows: {failed}")
    print(f"Image root: {output_root}")
    print(f"Manifest: {output_manifest}")
    print(f"ID map: {output_id_map}")


if __name__ == "__main__":
    main()
