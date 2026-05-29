#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from array import array

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

import mxnet as mx

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def sanitize_ms1m(idx_path: str, rec_path: str, out_dir: str = "clean_ms1m") -> dict[str, int | str]:
    os.makedirs(out_dir, exist_ok=True)
    out_idx = os.path.join(out_dir, "train.idx")
    out_rec = os.path.join(out_dir, "train.rec")

    print(f"[*] Opening dirty dataset: {rec_path}")
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")

    s0 = imgrec.read_idx(0)
    header0, _ = mx.recordio.unpack(s0)
    first_non_image_key = int(header0.label[0])
    identity_slots = max(0, int(header0.label[1]) - int(header0.label[0]))
    total_images = first_non_image_key - 1

    # Pass 1: validate and collect source keys that decode successfully.
    good_source_keys = array("I")

    print(f"[*] Pass 1/2: scanning {total_images} images (keys 1..{total_images})")

    iterator = range(1, first_non_image_key)
    if tqdm is not None:
        iterator = tqdm(iterator)

    for i in iterator:
        try:
            s = imgrec.read_idx(i)
            if s is None:
                continue

            header, img = mx.recordio.unpack(s)
            if img is None or len(img) == 0:
                continue

            # Force decode to catch malformed image bytes.
            _ = mx.image.imdecode(img)

            good_source_keys.append(i)
        except Exception:  # noqa: BLE001
            continue

    good_count = len(good_source_keys)
    corrupted_count = total_images - good_count

    # Pass 2: rewrite to dense keys with a correct header written once.
    print(f"[*] Pass 2/2: writing {good_count} clean images")
    imgrec_write = mx.recordio.MXIndexedRecordIO(out_idx, out_rec, "w")

    first_non_image_clean = float(good_count + 1)
    second_header_label = float((good_count + 1) + identity_slots)
    header_clean = mx.recordio.IRHeader(
        flag=0,
        label=[first_non_image_clean, second_header_label],
        id=0,
        id2=0,
    )
    imgrec_write.write_idx(0, mx.recordio.pack(header_clean, b""))

    write_iter = good_source_keys
    if tqdm is not None:
        write_iter = tqdm(good_source_keys)

    for out_key, src_key in enumerate(write_iter, start=1):
        packed = imgrec.read_idx(int(src_key))
        if packed is None:
            # Should not happen because pass 1 validated readability.
            continue
        imgrec_write.write_idx(out_key, packed)

    imgrec.close()
    imgrec_write.close()

    print("\n========================================")
    print("[*] SANITIZATION COMPLETE!")
    print(f"[*] Total Good Images Saved: {good_count}")
    print(f"[*] Total Corrupted Images Destroyed: {corrupted_count}")
    print(f"[*] Clean dataset saved to: {out_dir}")
    print("========================================")

    return {
        "out_idx": out_idx,
        "out_rec": out_rec,
        "good_images": good_count,
        "corrupted_images": corrupted_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize MS1M RecordIO by decoding and rewriting valid samples")
    parser.add_argument(
        "--idx-path",
        default="/home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4/data/raw/ms1m/faces_emore/train.idx",
        help="Path to source train.idx",
    )
    parser.add_argument(
        "--rec-path",
        default="/home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4/data/raw/ms1m/faces_emore/train.rec",
        help="Path to source train.rec",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/phongtruong/data_pool/phongtruong/fas-kd-mobilenetv4/data/raw/ms1m_clean",
        help="Output directory for clean train.idx/train.rec",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sanitize_ms1m(idx_path=args.idx_path, rec_path=args.rec_path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
