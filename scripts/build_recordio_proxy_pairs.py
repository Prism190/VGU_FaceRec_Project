#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed RecordIO pair proxy set for low-FAR validation.")
    parser.add_argument("--manifest", type=str, required=True, help="Training manifest CSV with rec_idx/rec_path/idx_path/label")
    parser.add_argument("--out", type=str, required=True, help="Output pair CSV path")
    parser.add_argument("--num-pos", type=int, default=10000, help="Number of positive pairs")
    parser.add_argument("--num-neg", type=int, default=40000, help="Number of negative pairs")
    parser.add_argument("--max-images-per-id", type=int, default=8, help="Reservoir cap per identity")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    return parser.parse_args()


def _reservoir_add(
    bucket: list[tuple[int, str, str]],
    item: tuple[int, str, str],
    seen_count: int,
    cap: int,
    rng: random.Random,
) -> None:
    if len(bucket) < cap:
        bucket.append(item)
        return

    # Reservoir sampling keeps a representative subset without full-manifest memory.
    pick = rng.randint(1, seen_count)
    if pick <= cap:
        bucket[pick - 1] = item


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_to_samples: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    label_seen_count: dict[int, int] = defaultdict(int)

    required = {"rec_idx", "rec_path", "idx_path", "label"}

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty manifest: {manifest_path}")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing columns {sorted(missing)}: {manifest_path}")

        for row in reader:
            label = int(row["label"])
            rec_idx = int(row["rec_idx"])
            rec_path = str(row["rec_path"])
            idx_path = str(row["idx_path"])

            label_seen_count[label] += 1
            _reservoir_add(
                bucket=label_to_samples[label],
                item=(rec_idx, rec_path, idx_path),
                seen_count=label_seen_count[label],
                cap=int(args.max_images_per_id),
                rng=rng,
            )

    eligible_pos_labels = [label for label, items in label_to_samples.items() if len(items) >= 2]
    all_labels = [label for label, items in label_to_samples.items() if len(items) >= 1]

    if len(eligible_pos_labels) == 0:
        raise RuntimeError("No identity has at least 2 samples; cannot build positive pairs")
    if len(all_labels) < 2:
        raise RuntimeError("Need at least 2 identities to build negative pairs")

    rows: list[dict[str, object]] = []

    for _ in range(int(args.num_pos)):
        label = rng.choice(eligible_pos_labels)
        sample_a, sample_b = rng.sample(label_to_samples[label], 2)
        rows.append(
            {
                "rec_idx_a": sample_a[0],
                "rec_idx_b": sample_b[0],
                "rec_path_a": sample_a[1],
                "idx_path_a": sample_a[2],
                "rec_path_b": sample_b[1],
                "idx_path_b": sample_b[2],
                "is_same": 1,
            }
        )

    for _ in range(int(args.num_neg)):
        label_a, label_b = rng.sample(all_labels, 2)
        sample_a = rng.choice(label_to_samples[label_a])
        sample_b = rng.choice(label_to_samples[label_b])
        rows.append(
            {
                "rec_idx_a": sample_a[0],
                "rec_idx_b": sample_b[0],
                "rec_path_a": sample_a[1],
                "idx_path_a": sample_a[2],
                "rec_path_b": sample_b[1],
                "idx_path_b": sample_b[2],
                "is_same": 0,
            }
        )

    rng.shuffle(rows)

    fieldnames = [
        "rec_idx_a",
        "rec_idx_b",
        "rec_path_a",
        "idx_path_a",
        "rec_path_b",
        "idx_path_b",
        "is_same",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Built proxy pairs: {len(rows)}")
    print(f"Positives: {args.num_pos} | Negatives: {args.num_neg}")
    print(f"Eligible identities (>=2 imgs): {len(eligible_pos_labels)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
