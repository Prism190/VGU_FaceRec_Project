from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    image_size: int = 112
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8


class FacePreprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        if cfg.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip_limit,
                tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
            )
        else:
            self._clahe = None

    def _clahe_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._clahe is None:
            return image_bgr

        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = self._clahe.apply(y)
        merged = cv2.merge((y, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _template_landmarks(self, image_size: int) -> np.ndarray:
        # InsightFace 112x112 canonical points.
        template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if image_size == 112:
            return template
        scale = float(image_size) / 112.0
        return template * scale

    def crop_center(self, image_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        h_img, w_img = image_bgr.shape[:2]
        x1i = max(0, int(x1))
        y1i = max(0, int(y1))
        x2i = min(w_img, int(x2 + 0.5))
        y2i = min(h_img, int(y2 + 0.5))
        crop = image_bgr[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            crop = image_bgr
        enhanced = self._clahe_bgr(crop)
        resized = cv2.resize(enhanced, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def align(self, image_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        if landmarks5.shape != (5, 2):
            raise ValueError(f"Expected landmarks shape (5,2), got {landmarks5.shape}")

        src = landmarks5.astype(np.float32)
        dst = self._template_landmarks(self.cfg.image_size)
        transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if transform is None:
            raise RuntimeError("Could not estimate face alignment transform")

        aligned = cv2.warpAffine(
            image_bgr,
            transform,
            (self.cfg.image_size, self.cfg.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        return aligned

    def __call__(self, image_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        enhanced = self._clahe_bgr(image_bgr)
        aligned = self.align(enhanced, landmarks5)
        return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
