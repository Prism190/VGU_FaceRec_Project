from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Linear, Module, PReLU, Sequential


class _Flatten(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class _ConvBlock(Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_c,
            out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class _LinearBlock(Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _DepthWise(Module):
    def __init__(
        self,
        c1: tuple[int, int],
        c2: tuple[int, int],
        c3: tuple[int, int],
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = _ConvBlock(c1_in, c1_out, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = _ConvBlock(c2_in, c2_out, groups=c2_in, kernel=kernel, stride=stride, padding=padding)
        self.project = _LinearBlock(c3_in, c3_out, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            return short_cut + x
        return x


class _Residual(Module):
    def __init__(
        self,
        c1: list[tuple[int, int]],
        c2: list[tuple[int, int]],
        c3: list[tuple[int, int]],
        num_block: int,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        modules = []
        for i in range(num_block):
            modules.append(
                _DepthWise(
                    c1=c1[i],
                    c2=c2[i],
                    c3=c3[i],
                    residual=True,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _MiniFASNet(Module):
    def __init__(
        self,
        keep: list[int],
        embedding_size: int = 128,
        conv6_kernel: tuple[int, int] = (7, 7),
        drop_p: float = 0.2,
        num_classes: int = 3,
        img_channel: int = 3,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)

        self.conv1 = _ConvBlock(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = _ConvBlock(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = _DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = _Residual(c1, c2, c3, num_block=4, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = _DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        c1 = [
            (keep[19], keep[20]),
            (keep[22], keep[23]),
            (keep[25], keep[26]),
            (keep[28], keep[29]),
            (keep[31], keep[32]),
            (keep[34], keep[35]),
        ]
        c2 = [
            (keep[20], keep[21]),
            (keep[23], keep[24]),
            (keep[26], keep[27]),
            (keep[29], keep[30]),
            (keep[32], keep[33]),
            (keep[35], keep[36]),
        ]
        c3 = [
            (keep[21], keep[22]),
            (keep[24], keep[25]),
            (keep[27], keep[28]),
            (keep[30], keep[31]),
            (keep[33], keep[34]),
            (keep[36], keep[37]),
        ]
        self.conv_4 = _Residual(c1, c2, c3, num_block=6, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = _DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = _Residual(c1, c2, c3, num_block=2, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = _ConvBlock(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = _LinearBlock(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = _Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


_KEEP_DICT = {
    "1.8M": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        26,
        26,
        64,
        13,
        13,
        64,
        52,
        52,
        64,
        231,
        231,
        128,
        154,
        154,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ],
    "1.8M_": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        231,
        231,
        128,
        231,
        231,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        77,
        77,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ],
}


def _mini_fasnet_v1(conv6_kernel: tuple[int, int], num_classes: int = 3) -> _MiniFASNet:
    return _MiniFASNet(_KEEP_DICT["1.8M"], conv6_kernel=conv6_kernel, num_classes=num_classes)


def _mini_fasnet_v2(conv6_kernel: tuple[int, int], num_classes: int = 3) -> _MiniFASNet:
    return _MiniFASNet(_KEEP_DICT["1.8M_"], conv6_kernel=conv6_kernel, num_classes=num_classes)


_MODEL_FACTORY: dict[str, Callable[..., _MiniFASNet]] = {
    "MiniFASNetV1": _mini_fasnet_v1,
    "MiniFASNetV2": _mini_fasnet_v2,
}


@dataclass(frozen=True)
class SilentFaceModelSpec:
    input_height: int
    input_width: int
    model_type: str
    scale: float | None


def _parse_model_name(model_path: Path) -> SilentFaceModelSpec:
    stem = model_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid Silent-Face model filename: {model_path.name}")

    patch_info = parts[-2]
    model_type = parts[-1]
    if "x" not in patch_info:
        raise ValueError(f"Missing input shape token in model filename: {model_path.name}")
    h_str, w_str = patch_info.split("x", maxsplit=1)

    scale_raw = parts[0]
    if scale_raw == "org":
        scale = None
    else:
        scale = float(scale_raw)

    return SilentFaceModelSpec(
        input_height=int(h_str),
        input_width=int(w_str),
        model_type=model_type,
        scale=scale,
    )


def _conv6_kernel(height: int, width: int) -> tuple[int, int]:
    return ((int(height) + 15) // 16, (int(width) + 15) // 16)


class SilentFaceAntiSpoof:
    """Torch inference wrapper for Silent-Face MiniFASNet anti-spoofing models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        live_class_index: int = 1,
        expect_bgr_input: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = self.model_path.resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Silent-Face model not found: {self.model_path}")

        spec = _parse_model_name(self.model_path)
        if spec.model_type not in _MODEL_FACTORY:
            supported = ", ".join(sorted(_MODEL_FACTORY.keys()))
            raise ValueError(f"Unsupported Silent-Face model type '{spec.model_type}'. Supported: {supported}")

        self.input_height = int(spec.input_height)
        self.input_width = int(spec.input_width)
        self.model_type = spec.model_type
        self.scale = spec.scale
        self.device = torch.device(device)
        self.live_class_index = int(live_class_index)
        self.expect_bgr_input = bool(expect_bgr_input)

        model = _MODEL_FACTORY[self.model_type](conv6_kernel=_conv6_kernel(self.input_height, self.input_width))
        try:
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise RuntimeError(f"Unexpected model checkpoint format: {type(state)}")

        normalized_state = {}
        for key, value in state_dict.items():
            if not isinstance(key, str):
                continue
            clean_key = key[7:] if key.startswith("module.") else key
            normalized_state[clean_key] = value

        missing, unexpected = model.load_state_dict(normalized_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "Silent-Face checkpoint key mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )

        self.model = model.to(self.device).eval()

    def _preprocess(self, face_rgb: np.ndarray) -> torch.Tensor:
        if face_rgb.ndim != 3 or face_rgb.shape[2] != 3:
            raise ValueError(f"Expected face crop shape HxWx3, got {face_rgb.shape}")

        img = np.asarray(face_rgb, dtype=np.uint8)
        if self.expect_bgr_input:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def score(self, face_rgb: np.ndarray) -> float:
        x = self._preprocess(face_rgb)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        live_idx = int(self.live_class_index)
        if live_idx < 0 or live_idx >= int(probs.shape[0]):
            live_idx = 1 if int(probs.shape[0]) > 1 else 0

        score = float(probs[live_idx].item())
        return float(np.clip(score, 0.0, 1.0))

    def __call__(self, face_rgb: np.ndarray) -> float:
        return self.score(face_rgb)
