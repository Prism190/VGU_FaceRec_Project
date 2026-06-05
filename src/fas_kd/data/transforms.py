from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8) -> None:
        import cv2

        self._cv2 = cv2
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image)
        ycrcb = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2YCrCb)
        y_channel, cr, cb = self._cv2.split(ycrcb)
        y_channel = self._clahe.apply(y_channel)
        merged = self._cv2.merge([y_channel, cr, cb])
        out = self._cv2.cvtColor(merged, self._cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(out)


def build_train_transform(data_cfg: dict) -> transforms.Compose:
    ops: list = []
    image_size = int(data_cfg.get("image_size", 112))
    ops.append(transforms.Resize((image_size, image_size)))
    if data_cfg.get("use_clahe", False):
        ops.append(CLAHETransform())
    ops.extend(
        [
            transforms.RandomHorizontalFlip(p=float(data_cfg.get("hflip_prob", 0.5))),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(data_cfg: dict) -> transforms.Compose:
    ops: list = []
    image_size = int(data_cfg.get("image_size", 112))
    ops.append(transforms.Resize((image_size, image_size)))
    if data_cfg.get("use_clahe", False):
        ops.append(CLAHETransform())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ]
    )
    return transforms.Compose(ops)


def apply_lower_face_mask(image_tensor: torch.Tensor, mask_fill: str = "zero") -> torch.Tensor:
    if image_tensor.ndim != 3:
        raise ValueError("Expected CHW tensor image")
    _, height, _ = image_tensor.shape
    y_start = int(height * 0.55)
    out = image_tensor.clone()
    if mask_fill == "noise":
        noise = (torch.rand_like(out[:, y_start:, :]) * 2.0) - 1.0
        out[:, y_start:, :] = noise
    else:
        out[:, y_start:, :] = 0.0
    return out


def should_apply_mask(mask_prob: float, rng: random.Random) -> bool:
    return rng.random() < mask_prob
