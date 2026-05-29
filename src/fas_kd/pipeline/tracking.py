from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import FaceDetection, TrackedFace


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


@dataclass
class TrackHistory:
    frame_indices: list[int] = field(default_factory=list)
    centers_xy: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TrackManager:
    iou_match_threshold: float = 0.3
    max_missed_frames: int = 20
    _next_track_id: int = 1
    tracks: dict[int, TrackedFace] = field(default_factory=dict)
    history: dict[int, TrackHistory] = field(default_factory=dict)

    def _bbox_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def update(self, detections: list[FaceDetection], frame_idx: int) -> list[TrackedFace]:
        matched_track_ids: set[int] = set()
        assigned_detection_ids: set[int] = set()

        # Greedy IoU assignment.
        for det_idx, det in enumerate(detections):
            best_tid = -1
            best_iou = self.iou_match_threshold
            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                score = iou_xyxy(track.bbox_xyxy, det.bbox_xyxy)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid >= 0:
                track = self.tracks[best_tid]
                track.bbox_xyxy = det.bbox_xyxy
                track.landmarks5 = det.landmarks5
                track.last_frame_idx = frame_idx
                track.missed_frames = 0
                matched_track_ids.add(best_tid)
                assigned_detection_ids.add(det_idx)

        # Spawn new tracks for unmatched detections.
        for det_idx, det in enumerate(detections):
            if det_idx in assigned_detection_ids:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self.tracks[tid] = TrackedFace(
                track_id=tid,
                bbox_xyxy=det.bbox_xyxy,
                landmarks5=det.landmarks5,
                last_frame_idx=frame_idx,
                missed_frames=0,
            )
            matched_track_ids.add(tid)

        # Increase miss count for unmatched tracks.
        dead_ids: list[int] = []
        for tid, track in self.tracks.items():
            if tid in matched_track_ids:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                dead_ids.append(tid)

        for tid in dead_ids:
            self.tracks.pop(tid, None)

        # Update motion history for interpolation.
        for tid, track in self.tracks.items():
            h = self.history.setdefault(tid, TrackHistory())
            h.frame_indices.append(frame_idx)
            h.centers_xy.append(self._bbox_center(track.bbox_xyxy))
            if len(h.frame_indices) > 100:
                h.frame_indices = h.frame_indices[-100:]
                h.centers_xy = h.centers_xy[-100:]

        return list(self.tracks.values())

    def interpolate_center(self, track_id: int, frame_idx: int) -> tuple[float, float] | None:
        h = self.history.get(track_id)
        if h is None or len(h.frame_indices) < 2:
            return None

        xs = np.asarray(h.frame_indices, dtype=np.float32)
        ys = np.asarray(h.centers_xy, dtype=np.float32)

        try:
            from scipy.interpolate import CubicSpline

            if len(xs) >= 4:
                sx = CubicSpline(xs, ys[:, 0], extrapolate=True)
                sy = CubicSpline(xs, ys[:, 1], extrapolate=True)
                return float(sx(frame_idx)), float(sy(frame_idx))
        except Exception:
            pass

        # Fallback: linear interpolation.
        x = float(np.interp(frame_idx, xs, ys[:, 0]))
        y = float(np.interp(frame_idx, xs, ys[:, 1]))
        return x, y
