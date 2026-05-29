from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_recordio_manifest(
    rec_path: Path,
    idx_path: Path,
    output_manifest_csv: Path,
    output_identity_map_csv: Path,
) -> tuple[int, int]:
    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    import mxnet as mx

    record = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")

    header0, _ = mx.recordio.unpack(record.read_idx(0))
    if not hasattr(header0.label, "__len__") or len(header0.label) < 1:
        raise RuntimeError("Unexpected RecordIO header layout at index 0")

    first_header_idx = int(header0.label[0])
    if first_header_idx <= 1:
        raise RuntimeError(f"Unexpected first header index from RecordIO: {first_header_idx}")

    rows: list[dict] = []
    label_counts: dict[int, int] = {}

    rec_path_str = str(rec_path.resolve())
    idx_path_str = str(idx_path.resolve())

    for rec_idx in range(1, first_header_idx):
        packed = record.read_idx(rec_idx)
        if packed is None:
            continue
        header, image_bytes = mx.recordio.unpack(packed)

        if not isinstance(header.label, (int, float)):
            continue
        if image_bytes is None or len(image_bytes) == 0:
            continue

        label = int(header.label)
        label_counts[label] = label_counts.get(label, 0) + 1
        rows.append(
            {
                "rec_idx": rec_idx,
                "rec_path": rec_path_str,
                "idx_path": idx_path_str,
                "image_path": f"recordio://{rec_idx}",
                "label": label,
                "identity": str(label),
            }
        )

    id_rows = [
        {"identity": str(label), "label": label, "num_images": count}
        for label, count in sorted(label_counts.items(), key=lambda x: x[0])
    ]

    pd.DataFrame(rows).to_csv(output_manifest_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(id_rows).to_csv(output_identity_map_csv, index=False)
    return len(rows), len(label_counts)


def build_casia_manifest(
    dataset_root: str | Path,
    output_manifest_csv: str | Path,
    output_identity_map_csv: str | Path,
) -> tuple[int, int]:
    dataset_root = Path(dataset_root)
    output_manifest_csv = Path(output_manifest_csv)
    output_identity_map_csv = Path(output_identity_map_csv)
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    output_identity_map_csv.parent.mkdir(parents=True, exist_ok=True)

    identity_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    rec_path = dataset_root / "train.rec"
    idx_path = dataset_root / "train.idx"

    # InsightFace-aligned CASIA dumps are often shipped as RecordIO instead of image folders.
    if rec_path.exists() and idx_path.exists():
        return _build_recordio_manifest(
            rec_path=rec_path,
            idx_path=idx_path,
            output_manifest_csv=output_manifest_csv,
            output_identity_map_csv=output_identity_map_csv,
        )

    rows: list[dict] = []
    id_rows: list[dict] = []

    for label, identity_dir in enumerate(identity_dirs):
        count = 0
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "identity": identity_dir.name,
                }
            )
            count += 1
        id_rows.append({"identity": identity_dir.name, "label": label, "num_images": count})

    manifest_df = pd.DataFrame(rows)
    identity_df = pd.DataFrame(id_rows)

    manifest_df.to_csv(output_manifest_csv, index=False)
    identity_df.to_csv(output_identity_map_csv, index=False)

    return len(rows), len(identity_dirs)


def load_num_classes(train_manifest_csv: str | Path) -> int:
    df = pd.read_csv(train_manifest_csv, usecols=["label"])
    return int(df["label"].nunique())


def build_pairs_manifest_from_rows(rows: list[dict], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["path_a", "path_b", "is_same"]).to_csv(output_csv, index=False)
