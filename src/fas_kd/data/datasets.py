from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import apply_lower_face_mask, should_apply_mask


class TrainKDDataset(Dataset):
    REQUIRED_COLUMNS = {"image_path", "label"}
    RECORDIO_COLUMNS = {"rec_idx", "rec_path", "idx_path", "label"}

    def __init__(
        self,
        manifest_csv: str | Path,
        transform,
        mask_prob: float,
        mask_fill: str,
        seed: int,
        decode_retries: int = 16,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.transform = transform
        self.mask_prob = float(mask_prob)
        self.mask_fill = mask_fill
        self.seed = int(seed)
        self.decode_retries = max(1, int(decode_retries))
        self.epoch = 0
        self.mode = "image"
        self._mx = None
        self._record_reader = None
        self._recordio_paths: tuple[str, str] | None = None

        frame = pd.read_csv(self.manifest_csv)
        columns = set(frame.columns)

        if "label" not in columns:
            raise ValueError(f"Missing label column in train manifest {self.manifest_csv}")

        unique_labels = sorted(int(x) for x in frame["label"].unique())
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        frame["label_mapped"] = frame["label"].map(self.label_to_index).astype(int)

        if self.RECORDIO_COLUMNS.issubset(columns):
            self.mode = "recordio"
            self.samples = frame[["rec_idx", "rec_path", "idx_path", "label_mapped"]].to_dict("records")
        else:
            missing = self.REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"Missing columns in train manifest {self.manifest_csv}: {sorted(missing)}")
            self.samples = frame[["image_path", "label_mapped"]].to_dict("records")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + (self.epoch * 1_000_003) + index)

    def _get_recordio_reader(self, rec_path: str, idx_path: str):
        if self._record_reader is not None and self._recordio_paths == (rec_path, idx_path):
            return self._record_reader

        import numpy as np

        if not hasattr(np, "bool"):
            np.bool = bool  # type: ignore[attr-defined]

        import mxnet as mx

        self._mx = mx
        self._record_reader = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
        self._recordio_paths = (rec_path, idx_path)
        return self._record_reader

    def _load_image(self, sample: dict) -> Image.Image:
        if self.mode == "recordio":
            rec_path = str(sample["rec_path"])
            idx_path = str(sample["idx_path"])
            rec_idx = int(sample["rec_idx"])

            reader = self._get_recordio_reader(rec_path=rec_path, idx_path=idx_path)
            packed = reader.read_idx(rec_idx)
            if packed is None:
                raise RuntimeError(f"RecordIO sample {rec_idx} could not be read from {rec_path}")

            _, image_bytes = self._mx.recordio.unpack(packed)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_path = Path(str(sample["image_path"]))
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        last_exc: Exception | None = None
        n = len(self.samples)

        # Some large RecordIO datasets contain malformed entries.
        # Retry with nearby samples instead of killing the whole training process.
        for offset in range(self.decode_retries):
            sample_index = (index + offset) % n
            sample = self.samples[sample_index]
            try:
                image = self._load_image(sample)
                clear_tensor = self.transform(image)

                rng = self._rng(sample_index)
                if should_apply_mask(self.mask_prob, rng):
                    masked_tensor = apply_lower_face_mask(clear_tensor, self.mask_fill)
                else:
                    masked_tensor = clear_tensor.clone()

                return {
                    "clear": clear_tensor,
                    "masked": masked_tensor,
                    "label": torch.tensor(int(sample["label_mapped"]), dtype=torch.long),
                }
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        raise RuntimeError(
            f"Failed to decode/transform sample index {index} after {self.decode_retries} retries"
        ) from last_exc


class PairVerificationDataset(Dataset):
    REQUIRED_COLUMNS = {"path_a", "path_b", "is_same"}
    RECORDIO_COLUMNS = {
        "rec_idx_a",
        "rec_idx_b",
        "rec_path_a",
        "idx_path_a",
        "rec_path_b",
        "idx_path_b",
        "is_same",
    }

    def __init__(self, pairs_csv: str | Path, transform) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.transform = transform
        self.mode = "image"
        self._mx = None
        self._record_readers: dict[tuple[str, str], object] = {}

        frame = pd.read_csv(self.pairs_csv)
        columns = set(frame.columns)

        if self.RECORDIO_COLUMNS.issubset(columns):
            self.mode = "recordio"
            self.samples = frame[
                [
                    "rec_idx_a",
                    "rec_idx_b",
                    "rec_path_a",
                    "idx_path_a",
                    "rec_path_b",
                    "idx_path_b",
                    "is_same",
                ]
            ].to_dict("records")
            return

        missing = self.REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Missing columns in pair manifest {self.pairs_csv}: {sorted(missing)} "
                f"(or provide RecordIO pair columns: {sorted(self.RECORDIO_COLUMNS)})"
            )
        self.samples = frame[["path_a", "path_b", "is_same"]].to_dict("records")

    def _get_recordio_reader(self, rec_path: str, idx_path: str):
        key = (str(rec_path), str(idx_path))
        reader = self._record_readers.get(key)
        if reader is not None:
            return reader

        import numpy as np

        if not hasattr(np, "bool"):
            np.bool = bool  # type: ignore[attr-defined]

        import mxnet as mx

        self._mx = mx
        reader = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")
        self._record_readers[key] = reader
        return reader

    def _load_recordio_image(self, rec_idx: int, rec_path: str, idx_path: str) -> Image.Image:
        reader = self._get_recordio_reader(rec_path=rec_path, idx_path=idx_path)
        packed = reader.read_idx(int(rec_idx))
        if packed is None:
            raise RuntimeError(f"RecordIO sample {rec_idx} could not be read from {rec_path}")

        _, image_bytes = self._mx.recordio.unpack(packed)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.samples[index]

        if self.mode == "recordio":
            image_a = self._load_recordio_image(
                rec_idx=int(row["rec_idx_a"]),
                rec_path=str(row["rec_path_a"]),
                idx_path=str(row["idx_path_a"]),
            )
            image_b = self._load_recordio_image(
                rec_idx=int(row["rec_idx_b"]),
                rec_path=str(row["rec_path_b"]),
                idx_path=str(row["idx_path_b"]),
            )
        else:
            image_a = Image.open(Path(row["path_a"])).convert("RGB")
            image_b = Image.open(Path(row["path_b"])).convert("RGB")

        return {
            "image_a": self.transform(image_a),
            "image_b": self.transform(image_b),
            "is_same": torch.tensor(int(row["is_same"]), dtype=torch.long),
        }
