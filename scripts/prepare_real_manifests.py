#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolve_lfw_image(images_root: Path, identity: str, index: str) -> Path:
    stem = f"{identity}_{int(index):04d}"
    for ext in IMAGE_EXTS:
        candidate = images_root / identity / f"{stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing LFW image: {identity} {index} under {images_root}")


def build_lfw_pairs(pairs_txt: Path, images_root: Path, output_csv: Path) -> int:
    lines = [line.strip() for line in pairs_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines and lines[0].isdigit():
        lines = lines[1:]

    rows: list[dict] = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            # LFW protocol includes section headers like "10 300".
            continue
        if len(parts) == 3:
            name, idx1, idx2 = parts
            rows.append(
                {
                    "path_a": str(_resolve_lfw_image(images_root, name, idx1)),
                    "path_b": str(_resolve_lfw_image(images_root, name, idx2)),
                    "is_same": 1,
                }
            )
        elif len(parts) == 4:
            name1, idx1, name2, idx2 = parts
            rows.append(
                {
                    "path_a": str(_resolve_lfw_image(images_root, name1, idx1)),
                    "path_b": str(_resolve_lfw_image(images_root, name2, idx2)),
                    "is_same": 0,
                }
            )
        else:
            raise ValueError(f"Unsupported LFW pair line: {line}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def build_ann_pairs(ann_path: Path, base_root: Path, output_csv: Path) -> int:
    rows: list[dict] = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Expected '<is_same> <path_a> <path_b>' format in {ann_path}: {line}")

        is_same_raw, rel_a, rel_b = parts
        path_a = (base_root / rel_a).resolve()
        path_b = (base_root / rel_b).resolve()
        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError(f"Missing pair files referenced by {ann_path}: {path_a} | {path_b}")

        rows.append({"path_a": str(path_a), "path_b": str(path_b), "is_same": int(is_same_raw)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def _read_indexed_path_file(path: Path, protocol_root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, rel_path = line.split(maxsplit=1)
        out[int(idx_str)] = (path.parent / rel_path).resolve()
    return out


def _read_cfp_split_file(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split(",")
        pairs.append((int(a), int(b)))
    return pairs


def build_cfp_fp_pairs(protocol_root: Path, output_csv: Path) -> int:
    profile_map = _read_indexed_path_file(protocol_root / "Pair_list_P.txt", protocol_root)
    frontal_map = _read_indexed_path_file(protocol_root / "Pair_list_F.txt", protocol_root)

    rows: list[dict] = []
    fp_split_root = protocol_root / "Split" / "FP"
    split_dirs = sorted([p for p in fp_split_root.iterdir() if p.is_dir()])

    for split_dir in split_dirs:
        same_pairs = _read_cfp_split_file(split_dir / "same.txt")
        diff_pairs = _read_cfp_split_file(split_dir / "diff.txt")

        for i, j in same_pairs:
            path_a = frontal_map[i]
            path_b = profile_map[j]
            if not path_a.exists() or not path_b.exists():
                raise FileNotFoundError(f"Missing CFP same pair files: {path_a} | {path_b}")
            rows.append({"path_a": str(path_a), "path_b": str(path_b), "is_same": 1})

        for i, j in diff_pairs:
            path_a = frontal_map[i]
            path_b = profile_map[j]
            if not path_a.exists() or not path_b.exists():
                raise FileNotFoundError(f"Missing CFP diff pair files: {path_a} | {path_b}")
            rows.append({"path_a": str(path_a), "path_b": str(path_b), "is_same": 0})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw"
    manifest_root = project_root / "data" / "manifests"

    lfw_count = build_lfw_pairs(
        pairs_txt=raw_root / "lfw" / "pairs.txt",
        images_root=raw_root / "lfw" / "lfw_funneled",
        output_csv=manifest_root / "lfw_pairs.csv",
    )

    agedb_count = build_ann_pairs(
        ann_path=raw_root / "agedb30_bundle" / "val" / "agedb_30_ann.txt",
        base_root=raw_root / "agedb30_bundle" / "val",
        output_csv=manifest_root / "agedb30_pairs.csv",
    )

    cfp_count = build_cfp_fp_pairs(
        protocol_root=raw_root / "cfp" / "cfp-dataset" / "Protocol",
        output_csv=manifest_root / "cfp_fp_pairs.csv",
    )

    print(f"LFW pairs: {lfw_count} -> {manifest_root / 'lfw_pairs.csv'}")
    print(f"AgeDB-30 pairs: {agedb_count} -> {manifest_root / 'agedb30_pairs.csv'}")
    print(f"CFP-FP pairs: {cfp_count} -> {manifest_root / 'cfp_fp_pairs.csv'}")


if __name__ == "__main__":
    main()
