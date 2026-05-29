from __future__ import annotations

from typing import Iterator

import torch
import torch.distributed as dist


class DALIWarper:
    def __init__(self, dali_iter) -> None:
        self.iter = dali_iter
        self._epoch_size = len(dali_iter)

    def __iter__(self) -> "DALIWarper":
        return self

    def __len__(self) -> int:
        return self._epoch_size

    @torch.no_grad()
    def __next__(self):
        data_dict = next(self.iter)[0]
        tensor_data = data_dict["data"]
        # Keep labels as a 1D tensor even for edge-case batch shapes.
        tensor_label: torch.Tensor = data_dict["label"].long().reshape(-1)

        if not tensor_data.is_cuda:
            tensor_data = tensor_data.cuda(non_blocking=True)
        if not tensor_label.is_cuda:
            tensor_label = tensor_label.cuda(non_blocking=True)

        return tensor_data, tensor_label

    def reset(self) -> None:
        self.iter.reset()


def create_dali_recordio_loader(
    batch_size: int,
    rec_file: str,
    idx_file: str,
    num_threads: int,
    local_rank: int,
    initial_fill: int = 32768,
    random_shuffle: bool = True,
    prefetch_queue_depth: int = 1,
    dali_aug: bool = False,
    image_size: int = 112,
    reader_name: str = "reader",
) -> Iterator:
    try:
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.pipeline import Pipeline
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    except Exception as exc:
        raise RuntimeError(
            "DALI is not installed. Install with: pip install nvidia-dali-cuda120"
        ) from exc

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def _dali_random_resize(img, resize_size):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img

    def _dali_random_gaussian_blur(img, window_size):
        return fn.gaussian_blur(img, window_size=window_size * 2 + 1)

    def _dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        return fn.hsv(img, saturation=saturate)

    def _dali_random_hsv(img, hue, saturation):
        return fn.hsv(img, hue=hue, saturation=saturation)

    def _multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(image_size * 0.5), int(image_size * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0.0, 20.0), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1.0, 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=local_rank,
        prefetch_queue_depth=prefetch_queue_depth,
    )

    condition_flip = fn.random.coin_flip(probability=0.5)

    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file,
            index_path=idx_file,
            initial_fill=initial_fill,
            num_shards=world_size,
            shard_id=rank,
            random_shuffle=random_shuffle,
            # Ensure each rank sees the same number of batches in DDP.
            pad_last_batch=True,
            name=reader_name,
        )

        # CPU decoder is slower than nvJPEG but much more tolerant to malformed images.
        images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB)

        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = _multiplexing(condition_resize, _dali_random_resize(images, size_resize), images)
            images = _multiplexing(condition_blur, _dali_random_gaussian_blur(images, window_size_blur), images)
            images = _multiplexing(condition_hsv, _dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = _dali_random_gray(images, 0.1)

        # Normalize from [0, 255] to roughly [-1, 1] to match mean/std=[0.5, 0.5, 0.5].
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=(127.5, 127.5, 127.5),
            std=(127.5, 127.5, 127.5),
            mirror=condition_flip,
        )

        pipe.set_outputs(images, labels)

    pipe.build()
    dali_iter = DALIClassificationIterator(
        pipelines=[pipe],
        reader_name=reader_name,
        auto_reset=False,
        last_batch_policy=LastBatchPolicy.DROP,
    )
    return DALIWarper(dali_iter)
