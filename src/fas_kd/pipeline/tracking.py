from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import FaceDetection, TrackedFace

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None


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
    backend: str = "deepsort"
    iou_match_threshold: float = 0.3
    center_dist_match_threshold: float = 1.25
    iou_cost_weight: float = 0.75
    max_missed_frames: int = 20
    deepsort_n_init: int = 2
    deepsort_max_iou_distance: float = 0.75
    deepsort_max_cosine_distance: float = 0.25
    deepsort_nn_budget: int | None = 100
    deepsort_nms_max_overlap: float = 1.0
    deepsort_gating_only_position: bool = False
    _next_track_id: int = 1
    tracks: dict[int, TrackedFace] = field(default_factory=dict)
    history: dict[int, TrackHistory] = field(default_factory=dict)
    _deepsort: Any = field(default=None, init=False, repr=False)
    _deepsort_max_age: int = field(default=-1, init=False, repr=False)
    _deepsort_cfg_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if str(self.backend).lower() == "deepsort":
            self._ensure_deepsort()

    def _bbox_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _normalized_center_distance(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax, ay = self._bbox_center(a)
        bx, by = self._bbox_center(b)
        dist = float(np.hypot(ax - bx, ay - by))

        aw = max(1.0, float(a[2] - a[0]))
        ah = max(1.0, float(a[3] - a[1]))
        bw = max(1.0, float(b[2] - b[0]))
        bh = max(1.0, float(b[3] - b[1]))
        avg_diag = 0.5 * float(np.hypot(aw, ah) + np.hypot(bw, bh))
        return dist / max(1.0, avg_diag)

    def _ensure_deepsort(self) -> None:
        if DeepSort is None:
            raise RuntimeError(
                "TrackManager backend=deepsort requires deep-sort-realtime. "
                "Install with: python -m pip install deep-sort-realtime"
            )

        target_max_age = int(max(1, int(self.max_missed_frames)))
        target_nn_budget = None if self.deepsort_nn_budget is None else int(self.deepsort_nn_budget)
        if target_nn_budget is not None and target_nn_budget <= 0:
            target_nn_budget = None

        cfg_key = (
            target_max_age,
            int(max(1, int(self.deepsort_n_init))),
            float(self.deepsort_max_iou_distance),
            float(self.deepsort_max_cosine_distance),
            target_nn_budget,
            float(self.deepsort_nms_max_overlap),
            bool(self.deepsort_gating_only_position),
        )

        if self._deepsort is not None and self._deepsort_max_age == target_max_age and self._deepsort_cfg_key == cfg_key:
            return

        self._deepsort = DeepSort(
            max_iou_distance=float(self.deepsort_max_iou_distance),
            max_age=target_max_age,
            n_init=int(max(1, int(self.deepsort_n_init))),
            max_cosine_distance=float(self.deepsort_max_cosine_distance),
            nn_budget=target_nn_budget,
            nms_max_overlap=float(self.deepsort_nms_max_overlap),
            gating_only_position=bool(self.deepsort_gating_only_position),
            embedder=None,
        )
        self._deepsort_max_age = target_max_age
        self._deepsort_cfg_key = cfg_key

    def update(
        self,
        detections: list[FaceDetection],
        frame_idx: int,
        frame_bgr: np.ndarray | None = None,
        detection_embeddings: list[np.ndarray] | None = None,
    ) -> list[TrackedFace]:
        backend = str(self.backend).lower()
        if backend == "deepsort":
            return self._update_deepsort(
                detections=detections,
                frame_idx=frame_idx,
                frame_bgr=frame_bgr,
                detection_embeddings=detection_embeddings,
            )
        return self._update_hungarian(detections=detections, frame_idx=frame_idx)

    def _update_deepsort(
        self,
        detections: list[FaceDetection],
        frame_idx: int,
        frame_bgr: np.ndarray | None,
        detection_embeddings: list[np.ndarray] | None,
    ) -> list[TrackedFace]:
        self._ensure_deepsort()

        raw_detections: list[tuple[list[float], float, str]] = []
        det_indices: list[int] = []
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            raw_detections.append(([x1, y1, w, h], float(det.score), "face"))
            det_indices.append(int(det_idx))

        embeds = None
        if detection_embeddings is not None:
            if len(detection_embeddings) != len(detections):
                raise ValueError("detection_embeddings length must match detections length")
            embeds = []
            for emb in detection_embeddings:
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    vec = vec / norm
                embeds.append(vec)

        ds_tracks = self._deepsort.update_tracks(
            raw_detections=raw_detections,
            embeds=embeds,
            frame=frame_bgr if embeds is None else None,
            others=det_indices,
        )

        active_track_ids: set[int] = set()
        for ds_track in ds_tracks:
            tid = int(ds_track.track_id)
            active_track_ids.add(tid)

            det_idx = None
            supp = ds_track.get_det_supplementary()
            if supp is not None:
                if isinstance(supp, (list, tuple, np.ndarray)) and len(supp) > 0:
                    supp = supp[0]
                try:
                    det_idx = int(supp)
                except Exception:
                    det_idx = None

            if det_idx is not None and 0 <= det_idx < len(detections):
                det = detections[int(det_idx)]
                bbox = tuple(float(v) for v in det.bbox_xyxy)
                landmarks = np.asarray(det.landmarks5, dtype=np.float32)
                missed = 0
                last_frame_idx = int(frame_idx)
            else:
                ltrb = ds_track.to_ltrb(orig=False)
                if ltrb is None:
                    continue
                bbox = (float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]))
                prev = self.tracks.get(tid)
                if prev is None:
                    landmarks = np.zeros((5, 2), dtype=np.float32)
                    last_frame_idx = int(frame_idx)
                else:
                    landmarks = np.asarray(prev.landmarks5, dtype=np.float32)
                    last_frame_idx = int(prev.last_frame_idx)
                missed = int(getattr(ds_track, "time_since_update", 0))

            self.tracks[tid] = TrackedFace(
                track_id=tid,
                bbox_xyxy=bbox,
                landmarks5=landmarks,
                last_frame_idx=last_frame_idx,
                missed_frames=missed,
            )

        for tid in list(self.tracks.keys()):
            if tid not in active_track_ids:
                self.tracks.pop(tid, None)

        self._update_history(frame_idx=frame_idx)
        return list(self.tracks.values())

    def _update_hungarian(self, detections: list[FaceDetection], frame_idx: int) -> list[TrackedFace]:
        matched_track_ids: set[int] = set()
        assigned_detection_ids: set[int] = set()

        # Match tracks to detections with global assignment (Hungarian) when available.
        if self.tracks and detections and linear_sum_assignment is not None:
            track_ids = list(self.tracks.keys())
            n_tracks = len(track_ids)
            n_dets = len(detections)
            large_cost = np.float32(1e6)
            cost = np.full((n_tracks, n_dets), large_cost, dtype=np.float32)

            for ti, tid in enumerate(track_ids):
                track = self.tracks[tid]
                for di, det in enumerate(detections):
                    iou = iou_xyxy(track.bbox_xyxy, det.bbox_xyxy)
                    center_dist = self._normalized_center_distance(track.bbox_xyxy, det.bbox_xyxy)

                    # Accept a candidate if it has enough overlap or reasonable center continuity.
                    if iou < self.iou_match_threshold and center_dist > self.center_dist_match_threshold:
                        continue

                    assoc_cost = self.iou_cost_weight * (1.0 - iou) + (1.0 - self.iou_cost_weight) * center_dist
                    cost[ti, di] = np.float32(assoc_cost)

            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, di in zip(row_ind.tolist(), col_ind.tolist()):
                if cost[ti, di] >= large_cost:
                    continue
                tid = track_ids[ti]
                if tid in matched_track_ids or di in assigned_detection_ids:
                    continue
                det = detections[di]
                track = self.tracks[tid]
                track.bbox_xyxy = det.bbox_xyxy
                track.landmarks5 = det.landmarks5
                track.last_frame_idx = frame_idx
                track.missed_frames = 0
                matched_track_ids.add(tid)
                assigned_detection_ids.add(di)

        elif self.tracks and detections:
            # Fallback greedy assignment when scipy is unavailable.
            for det_idx, det in enumerate(detections):
                best_tid = -1
                best_score = -1e9
                for tid, track in self.tracks.items():
                    if tid in matched_track_ids:
                        continue
                    iou = iou_xyxy(track.bbox_xyxy, det.bbox_xyxy)
                    center_dist = self._normalized_center_distance(track.bbox_xyxy, det.bbox_xyxy)
                    if iou < self.iou_match_threshold and center_dist > self.center_dist_match_threshold:
                        continue
                    score = self.iou_cost_weight * iou - (1.0 - self.iou_cost_weight) * center_dist
                    if score > best_score:
                        best_score = score
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

        self._update_history(frame_idx=frame_idx)
        return list(self.tracks.values())

    def _update_history(self, frame_idx: int) -> None:
        # Update motion history for interpolation.
        for tid, track in self.tracks.items():
            h = self.history.setdefault(tid, TrackHistory())
            h.frame_indices.append(frame_idx)
            h.centers_xy.append(self._bbox_center(track.bbox_xyxy))
            if len(h.frame_indices) > 100:
                h.frame_indices = h.frame_indices[-100:]
                h.centers_xy = h.centers_xy[-100:]

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
