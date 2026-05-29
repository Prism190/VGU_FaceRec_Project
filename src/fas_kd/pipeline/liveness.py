from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


def _clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    out = cv2.merge((y, cr, cb))
    out = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def build_tta_views(image_rgb: np.ndarray) -> list[np.ndarray]:
    views = [image_rgb]
    views.append(np.ascontiguousarray(np.flip(image_rgb, axis=1)))
    views.append(_clahe_rgb(image_rgb))
    return views


@dataclass
class ThresholdLivenessGate:
    infer_fn: Callable[[np.ndarray], float]
    live_threshold: float = 0.5
    use_ttda: bool = True

    def score(self, image_rgb: np.ndarray) -> float:
        if not self.use_ttda:
            return float(self.infer_fn(image_rgb))

        scores = [float(self.infer_fn(view)) for view in build_tta_views(image_rgb)]
        return float(np.mean(scores))

    def is_live(self, image_rgb: np.ndarray) -> tuple[bool, float]:
        s = self.score(image_rgb)
        return s >= self.live_threshold, s
