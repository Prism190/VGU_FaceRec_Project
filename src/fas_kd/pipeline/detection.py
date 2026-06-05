from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import FaceDetection


@dataclass
class DetectionConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 100
    imgsz: int = 640
    enable_rescue_pass: bool = False
    rescue_conf_thres: float = 0.08
    rescue_iou_thres: float = 0.45
    rescue_imgsz: int = 1280
    rescue_min_primary_detections: int = 2
    merge_iou_thres: float = 0.55
    fallback_bbox_landmarks: bool = True


class YOLO11FaceDetector:
    def __init__(self, model_path: str, cfg: DetectionConfig | None = None) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO11FaceDetector. Install with: pip install ultralytics"
            ) from exc

        self._yolo = YOLO(model_path)
        self.cfg = cfg or DetectionConfig()

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        results = self._yolo.predict(
            source=image_bgr,
            conf=self.cfg.conf_thres,
            iou=self.cfg.iou_thres,
            max_det=self.cfg.max_det,
            imgsz=self.cfg.imgsz,
            verbose=False,
        )
        if not results:
            return []

        out: list[FaceDetection] = []
        r = results[0]

        if r.boxes is None:
            return out

        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
        scores = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.zeros((0,), dtype=np.float32)

        keypoints = None
        if getattr(r, "keypoints", None) is not None and r.keypoints.xy is not None:
            keypoints = r.keypoints.xy.cpu().numpy()

        for i in range(len(boxes)):
            b = boxes[i].astype(np.float32)
            landmarks_synthetic = False
            if keypoints is not None and i < len(keypoints) and keypoints[i].shape[0] >= 5:
                lm5 = keypoints[i][:5].astype(np.float32)
            elif self.cfg.fallback_bbox_landmarks:
                lm5 = self._landmarks_from_bbox(b)
                landmarks_synthetic = True
            else:
                continue
            out.append(
                FaceDetection(
                    bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    landmarks5=lm5,
                    score=float(scores[i]),
                    landmarks_synthetic=landmarks_synthetic,
                )
            )

        if self.cfg.merge_iou_thres > 0:
            out = self._merge_detections(out, self.cfg.merge_iou_thres)

        return out

    @staticmethod
    def _merge_detections(dets: list[FaceDetection], iou_thres: float) -> list[FaceDetection]:
        """Greedy cross-class NMS: keeps highest-score detection when two boxes overlap > iou_thres.

        YOLO applies NMS per class, so a face-age model can emit one bbox per age bucket for the
        same physical face. This pass collapses those duplicates before they hit the tracker.
        Prefers detections with real (non-synthetic) landmarks when scores are equal.
        """
        if len(dets) <= 1:
            return dets

        # Sort descending by score so we always keep the highest-confidence box.
        order = sorted(range(len(dets)), key=lambda i: (dets[i].score, not dets[i].landmarks_synthetic), reverse=True)
        keep: list[FaceDetection] = []
        suppressed = [False] * len(dets)

        for i in order:
            if suppressed[i]:
                continue
            keep.append(dets[i])
            ax1, ay1, ax2, ay2 = dets[i].bbox_xyxy
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j in order:
                if suppressed[j] or j == i:
                    continue
                bx1, by1, bx2, by2 = dets[j].bbox_xyxy
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = area_a + area_b - inter
                if union > 0 and inter / union >= iou_thres:
                    suppressed[j] = True

        return keep

    @staticmethod
    def _landmarks_from_bbox(bbox_xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.asarray(
            [
                [x1 + 0.30 * w, y1 + 0.36 * h],
                [x1 + 0.70 * w, y1 + 0.36 * h],
                [x1 + 0.50 * w, y1 + 0.56 * h],
                [x1 + 0.35 * w, y1 + 0.76 * h],
                [x1 + 0.65 * w, y1 + 0.76 * h],
            ],
            dtype=np.float32,
        )
