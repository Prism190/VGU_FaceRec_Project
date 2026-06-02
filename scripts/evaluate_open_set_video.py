#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


UNKNOWN_NAME_TOKENS = {
    "",
    "unknown",
    "unk",
    "stranger",
    "none",
    "null",
    "na",
    "n/a",
    "-1",
}


@dataclass(frozen=True)
class GTLabel:
    is_known: bool
    identity_id: int | None
    identity_name: str | None


@dataclass(frozen=True)
class ProbeItem:
    gt: GTLabel
    pred_identity_id: int | None
    pred_identity_name: str | None
    score: float


@dataclass(frozen=True)
class ScoreBuckets:
    known_total: int
    unknown_total: int
    known_correct_scores: np.ndarray
    known_wrong_scores: np.ndarray
    unknown_accept_scores: np.ndarray


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        if abs(value - round(value)) < 1e-6:
            return int(round(value))
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text else None


def _is_unknown_name(name: str | None) -> bool:
    if name is None:
        return True
    return name.strip().lower() in UNKNOWN_NAME_TOKENS


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _parse_label_payload(payload: dict[str, Any], *, default_track_id: int | None = None) -> tuple[int | None, int | None, GTLabel | None]:
    track_id = _safe_int(payload.get("track_id", default_track_id))
    frame_idx = _safe_int(payload.get("frame_idx"))
    identity_id = _safe_int(payload.get("identity_id", payload.get("id")))

    identity_name = _normalize_name(
        payload.get("identity_name", payload.get("name", payload.get("label")))
    )
    known_override = _to_bool(payload.get("is_known"))

    if known_override is None:
        is_known = identity_id is not None or (identity_name is not None and not _is_unknown_name(identity_name))
    else:
        is_known = bool(known_override)

    if not is_known:
        return track_id, frame_idx, GTLabel(is_known=False, identity_id=None, identity_name=None)

    if identity_id is None and (identity_name is None or _is_unknown_name(identity_name)):
        return track_id, frame_idx, None

    return track_id, frame_idx, GTLabel(
        is_known=True,
        identity_id=identity_id,
        identity_name=identity_name,
    )


def _parse_track_map_entry(track_key: Any, value: Any) -> tuple[int | None, GTLabel | None]:
    track_id = _safe_int(track_key)
    if track_id is None:
        return None, None

    if isinstance(value, dict):
        parsed_track_id, _, label = _parse_label_payload(value, default_track_id=track_id)
        if parsed_track_id is None:
            parsed_track_id = track_id
        return parsed_track_id, label

    scalar_id = _safe_int(value)
    if scalar_id is not None:
        return track_id, GTLabel(is_known=True, identity_id=scalar_id, identity_name=None)

    scalar_name = _normalize_name(value)
    if _is_unknown_name(scalar_name):
        return track_id, GTLabel(is_known=False, identity_id=None, identity_name=None)

    return track_id, GTLabel(is_known=True, identity_id=None, identity_name=scalar_name)


def _load_gt_json(path: Path) -> tuple[dict[int, GTLabel], dict[tuple[int, int], GTLabel], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    track_labels: dict[int, GTLabel] = {}
    observation_labels: dict[tuple[int, int], GTLabel] = {}
    warnings: list[str] = []

    def ingest_entry(entry: Any, *, mode_hint: str | None = None) -> None:
        if not isinstance(entry, dict):
            return
        track_id, frame_idx, label = _parse_label_payload(entry)
        if label is None:
            return
        if track_id is None:
            warnings.append(f"Skipped GT row without track_id: {entry}")
            return
        if mode_hint == "track":
            track_labels[int(track_id)] = label
            return
        if mode_hint == "observation":
            if frame_idx is None:
                warnings.append(f"Skipped observation GT row without frame_idx: {entry}")
                return
            observation_labels[(int(frame_idx), int(track_id))] = label
            return

        # Infer mode from available keys.
        if frame_idx is None:
            track_labels[int(track_id)] = label
        else:
            observation_labels[(int(frame_idx), int(track_id))] = label

    if isinstance(payload, dict):
        if "track_labels" in payload and isinstance(payload["track_labels"], list):
            for entry in payload["track_labels"]:
                ingest_entry(entry, mode_hint="track")

        if "observation_labels" in payload and isinstance(payload["observation_labels"], list):
            for entry in payload["observation_labels"]:
                ingest_entry(entry, mode_hint="observation")

        if "labels" in payload and isinstance(payload["labels"], list):
            for entry in payload["labels"]:
                ingest_entry(entry)

        if not track_labels and not observation_labels:
            # Treat dict as compact track map: {"12": 1001, "13": "unknown", ...}
            maybe_track_map = True
            for k in payload.keys():
                if _safe_int(k) is None:
                    maybe_track_map = False
                    break
            if maybe_track_map:
                for track_key, value in payload.items():
                    track_id, label = _parse_track_map_entry(track_key, value)
                    if track_id is None or label is None:
                        continue
                    track_labels[int(track_id)] = label
    elif isinstance(payload, list):
        for entry in payload:
            ingest_entry(entry)
    else:
        raise ValueError(f"Unsupported GT JSON payload type: {type(payload)}")

    return track_labels, observation_labels, warnings


def _load_gt_csv(path: Path) -> tuple[dict[int, GTLabel], dict[tuple[int, int], GTLabel], list[str]]:
    track_labels: dict[int, GTLabel] = {}
    observation_labels: dict[tuple[int, int], GTLabel] = {}
    warnings: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("GT CSV is missing a header row")

        for row in reader:
            row_payload = {
                "track_id": row.get("track_id"),
                "frame_idx": row.get("frame_idx"),
                "identity_id": row.get("identity_id"),
                "identity_name": row.get("identity_name", row.get("name", row.get("label"))),
                "is_known": row.get("is_known"),
            }
            track_id, frame_idx, label = _parse_label_payload(row_payload)
            if label is None:
                continue
            if track_id is None:
                warnings.append(f"Skipped CSV row without track_id: {row}")
                continue
            if frame_idx is None:
                track_labels[int(track_id)] = label
            else:
                observation_labels[(int(frame_idx), int(track_id))] = label

    return track_labels, observation_labels, warnings


def load_gt_labels(path: Path) -> tuple[dict[int, GTLabel], dict[tuple[int, int], GTLabel], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_gt_json(path)
    if suffix == ".csv":
        return _load_gt_csv(path)
    raise ValueError("--gt must be .json or .csv")


def load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows: list[dict[str, Any]] = []
    identity_name_map: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)

            track_id = _safe_int(payload.get("track_id"))
            frame_idx = _safe_int(payload.get("frame_idx"))
            if track_id is None or frame_idx is None:
                continue

            identity_id = _safe_int(payload.get("identity_id"))
            identity_name = _normalize_name(payload.get("identity_name"))
            if identity_id is not None and identity_name is not None and not _is_unknown_name(identity_name):
                identity_name_map.setdefault(int(identity_id), identity_name)

            retrieval_top_identity_id = _safe_int(payload.get("retrieval_top_identity_id"))

            score_raw = payload.get("match_score")
            score: float
            if score_raw is None:
                score = float("-inf")
            else:
                try:
                    score = float(score_raw)
                except Exception:
                    score = float("-inf")
            if not np.isfinite(score):
                score = float("-inf")

            rows.append(
                {
                    "line_idx": int(line_idx),
                    "track_id": int(track_id),
                    "frame_idx": int(frame_idx),
                    "is_live": bool(payload.get("is_live", True)),
                    "quality_pass": bool(payload.get("quality_pass", True)),
                    "identity_id": identity_id,
                    "identity_name": identity_name,
                    "retrieval_top_identity_id": retrieval_top_identity_id,
                    "score": score,
                }
            )

    return rows, identity_name_map


def _passes_probe_filter(row: dict[str, Any], probe_filter: str) -> bool:
    if probe_filter == "all":
        return True
    if probe_filter == "live":
        return bool(row.get("is_live", False))
    if probe_filter == "accepted":
        return bool(row.get("is_live", False)) and bool(row.get("quality_pass", False))
    raise ValueError(f"Unknown probe filter: {probe_filter}")


def _candidate_from_row(row: dict[str, Any], decision_source: str) -> tuple[int | None, float]:
    if decision_source == "retrieval":
        pred_id = _safe_int(row.get("retrieval_top_identity_id"))
        if pred_id is None:
            pred_id = _safe_int(row.get("identity_id"))
    elif decision_source == "accepted":
        pred_id = _safe_int(row.get("identity_id"))
    else:
        raise ValueError(f"Unknown decision source: {decision_source}")

    score = float(row.get("score", float("-inf")))
    if pred_id is None:
        score = float("-inf")
    return pred_id, score


def _select_best_candidate(
    rows: list[dict[str, Any]],
    *,
    decision_source: str,
    probe_filter: str,
) -> tuple[int | None, float]:
    best_pred_id: int | None = None
    best_score = float("-inf")

    for row in rows:
        if not _passes_probe_filter(row, probe_filter):
            continue
        pred_id, score = _candidate_from_row(row, decision_source)
        if pred_id is None:
            continue
        if score > best_score:
            best_score = float(score)
            best_pred_id = int(pred_id)

    return best_pred_id, best_score


def _is_prediction_correct(gt: GTLabel, pred_id: int | None, pred_name: str | None) -> bool:
    if not gt.is_known:
        return False

    if gt.identity_id is not None and pred_id is not None:
        return int(gt.identity_id) == int(pred_id)

    if gt.identity_name is not None and pred_name is not None:
        return str(gt.identity_name).strip().lower() == str(pred_name).strip().lower()

    return False


def _build_track_items(
    pred_rows: list[dict[str, Any]],
    gt_track_labels: dict[int, GTLabel],
    *,
    decision_source: str,
    probe_filter: str,
    id_to_name: dict[int, str],
) -> list[ProbeItem]:
    rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        rows_by_track[int(row["track_id"])].append(row)

    out: list[ProbeItem] = []
    for track_id, gt_label in sorted(gt_track_labels.items()):
        rows = rows_by_track.get(int(track_id), [])
        pred_id, score = _select_best_candidate(
            rows,
            decision_source=decision_source,
            probe_filter=probe_filter,
        )
        pred_name = id_to_name.get(int(pred_id)) if pred_id is not None else None
        out.append(
            ProbeItem(
                gt=gt_label,
                pred_identity_id=pred_id,
                pred_identity_name=pred_name,
                score=float(score),
            )
        )

    return out


def _build_observation_items(
    pred_rows: list[dict[str, Any]],
    gt_observation_labels: dict[tuple[int, int], GTLabel],
    gt_track_labels: dict[int, GTLabel],
    *,
    decision_source: str,
    probe_filter: str,
    id_to_name: dict[int, str],
) -> tuple[list[ProbeItem], str]:
    rows_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        key = (int(row["frame_idx"]), int(row["track_id"]))
        rows_by_key[key].append(row)

    out: list[ProbeItem] = []
    if gt_observation_labels:
        source = "observation_labels"
        for key, gt_label in sorted(gt_observation_labels.items()):
            pred_id, score = _select_best_candidate(
                rows_by_key.get(key, []),
                decision_source=decision_source,
                probe_filter=probe_filter,
            )
            pred_name = id_to_name.get(int(pred_id)) if pred_id is not None else None
            out.append(
                ProbeItem(
                    gt=gt_label,
                    pred_identity_id=pred_id,
                    pred_identity_name=pred_name,
                    score=float(score),
                )
            )
        return out, source

    source = "derived_from_track_labels"
    for row in pred_rows:
        track_id = int(row["track_id"])
        gt_label = gt_track_labels.get(track_id)
        if gt_label is None:
            continue
        if not _passes_probe_filter(row, probe_filter):
            continue
        pred_id, score = _candidate_from_row(row, decision_source)
        pred_name = id_to_name.get(int(pred_id)) if pred_id is not None else None
        out.append(
            ProbeItem(
                gt=gt_label,
                pred_identity_id=pred_id,
                pred_identity_name=pred_name,
                score=float(score),
            )
        )

    return out, source


def _sorted_array(values: list[float]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.sort(arr)


def _build_score_buckets(items: list[ProbeItem]) -> ScoreBuckets:
    known_total = 0
    unknown_total = 0

    known_correct_scores: list[float] = []
    known_wrong_scores: list[float] = []
    unknown_accept_scores: list[float] = []

    for item in items:
        if item.gt.is_known:
            known_total += 1
            if item.pred_identity_id is None:
                continue
            if not np.isfinite(item.score):
                continue
            is_correct = _is_prediction_correct(item.gt, item.pred_identity_id, item.pred_identity_name)
            if is_correct:
                known_correct_scores.append(float(item.score))
            else:
                known_wrong_scores.append(float(item.score))
        else:
            unknown_total += 1
            if item.pred_identity_id is None:
                continue
            if not np.isfinite(item.score):
                continue
            unknown_accept_scores.append(float(item.score))

    return ScoreBuckets(
        known_total=int(known_total),
        unknown_total=int(unknown_total),
        known_correct_scores=_sorted_array(known_correct_scores),
        known_wrong_scores=_sorted_array(known_wrong_scores),
        unknown_accept_scores=_sorted_array(unknown_accept_scores),
    )


def _count_ge(sorted_values: np.ndarray, threshold: float) -> int:
    if sorted_values.size == 0:
        return 0
    idx = np.searchsorted(sorted_values, float(threshold), side="left")
    return int(sorted_values.size - idx)


def _metrics_at_threshold(buckets: ScoreBuckets, threshold: float) -> dict[str, Any]:
    known_tp = _count_ge(buckets.known_correct_scores, threshold)
    known_mis = _count_ge(buckets.known_wrong_scores, threshold)
    known_fr = int(buckets.known_total - known_tp - known_mis)
    unknown_fa = _count_ge(buckets.unknown_accept_scores, threshold)

    if buckets.known_total > 0:
        tpir = float(known_tp / buckets.known_total)
        fnir = float(known_fr / buckets.known_total)
        misidr = float(known_mis / buckets.known_total)
    else:
        tpir = 0.0
        fnir = 0.0
        misidr = 0.0

    if buckets.unknown_total > 0:
        fpir = float(unknown_fa / buckets.unknown_total)
    else:
        fpir = 0.0

    return {
        "threshold": float(threshold),
        "known_true_positive": int(known_tp),
        "known_false_reject": int(known_fr),
        "known_misidentification": int(known_mis),
        "unknown_false_positive_identification": int(unknown_fa),
        "tpir": float(tpir),
        "fnir": float(fnir),
        "misidr": float(misidr),
        "fpir": float(fpir),
    }


def _threshold_grid(buckets: ScoreBuckets) -> np.ndarray:
    parts = []
    if buckets.known_correct_scores.size > 0:
        parts.append(buckets.known_correct_scores)
    if buckets.known_wrong_scores.size > 0:
        parts.append(buckets.known_wrong_scores)
    if buckets.unknown_accept_scores.size > 0:
        parts.append(buckets.unknown_accept_scores)

    if not parts:
        return np.asarray([float("inf"), float("-inf")], dtype=np.float64)

    all_scores = np.concatenate(parts)
    unique_scores = np.unique(all_scores)
    return np.concatenate(
        (
            np.asarray([float("inf")], dtype=np.float64),
            unique_scores[::-1],
            np.asarray([float("-inf")], dtype=np.float64),
        ),
        axis=0,
    )


def _encode_threshold(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def _pick_tpir_at_fpir(
    threshold_metrics: list[dict[str, Any]],
    target_fpir: float,
) -> dict[str, Any]:
    candidates = [m for m in threshold_metrics if float(m["fpir"]) <= float(target_fpir) + 1e-12]
    if not candidates:
        # Fallback to strictest threshold if nothing satisfies target.
        best = threshold_metrics[0]
    else:
        # Max TPIR under FPIR constraint; tie-break by lower threshold.
        best = candidates[0]
        for cand in candidates[1:]:
            if float(cand["tpir"]) > float(best["tpir"]) + 1e-12:
                best = cand
            elif abs(float(cand["tpir"]) - float(best["tpir"])) <= 1e-12:
                if float(cand["threshold"]) < float(best["threshold"]):
                    best = cand

    out = dict(best)
    out["threshold"] = _encode_threshold(float(best["threshold"]))
    out["target_fpir"] = float(target_fpir)
    return out


def _downsample_curve(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    if max_points <= 1:
        return [points[0]]

    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    out = [points[int(i)] for i in idx]
    if out[0] is not points[0]:
        out[0] = points[0]
    if out[-1] is not points[-1]:
        out[-1] = points[-1]
    return out


def evaluate_items(
    items: list[ProbeItem],
    *,
    default_threshold: float,
    fpir_targets: list[float],
    max_curve_points: int,
) -> dict[str, Any]:
    buckets = _build_score_buckets(items)
    thresholds = _threshold_grid(buckets)

    threshold_metrics: list[dict[str, Any]] = []
    for thr in thresholds:
        threshold_metrics.append(_metrics_at_threshold(buckets, float(thr)))

    default_metrics = _metrics_at_threshold(buckets, float(default_threshold))
    default_metrics["threshold"] = _encode_threshold(float(default_threshold))

    target_metrics: list[dict[str, Any]] = []
    for target in fpir_targets:
        target_metrics.append(_pick_tpir_at_fpir(threshold_metrics, target))

    curve_points: list[dict[str, Any]] = []
    for m in threshold_metrics:
        curve_points.append(
            {
                "threshold": _encode_threshold(float(m["threshold"])),
                "tpir": float(m["tpir"]),
                "fpir": float(m["fpir"]),
                "fnir": float(m["fnir"]),
                "misidr": float(m["misidr"]),
            }
        )

    curve_points = _downsample_curve(curve_points, max_points=max_curve_points)

    return {
        "num_items": int(len(items)),
        "known_probes": int(buckets.known_total),
        "unknown_probes": int(buckets.unknown_total),
        "default_threshold_metrics": default_metrics,
        "tpir_at_fpir": target_metrics,
        "curve": curve_points,
    }


def _parse_fpir_targets(value: str) -> list[float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("--fpir-targets is empty")

    out: list[float] = []
    for p in parts:
        target = float(p)
        if target < 0.0 or target > 1.0:
            raise ValueError(f"FPIR target must be in [0,1], got {target}")
        out.append(float(target))

    out = sorted(set(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-set video evaluation from pipeline JSONL and GT labels")
    parser.add_argument("--pred-jsonl", required=True, help="Pipeline prediction JSONL from run_face_pipeline.py")
    parser.add_argument("--gt", required=True, help="Ground-truth labels (.json or .csv)")
    parser.add_argument(
        "--mode",
        choices=["track", "observation", "both"],
        default="both",
        help="Evaluate at track level, observation level, or both",
    )
    parser.add_argument(
        "--decision-source",
        choices=["retrieval", "accepted"],
        default="retrieval",
        help="Use retrieval-top predictions or only accepted identities",
    )
    parser.add_argument(
        "--probe-filter",
        choices=["accepted", "live", "all"],
        default="accepted",
        help="Which rows are eligible probes before scoring",
    )
    parser.add_argument("--fpir-targets", default="0.001,0.01,0.1", help="Comma-separated FPIR targets")
    parser.add_argument("--default-threshold", type=float, default=0.46)
    parser.add_argument("--max-curve-points", type=int, default=300)
    parser.add_argument("--out", default="logs/eval_open_set_video.json")
    args = parser.parse_args()

    pred_path = Path(args.pred_jsonl).resolve()
    gt_path = Path(args.gt).resolve()
    out_path = Path(args.out).resolve()

    pred_rows, id_name_map = load_predictions(pred_path)
    gt_track_labels, gt_observation_labels, gt_warnings = load_gt_labels(gt_path)
    fpir_targets = _parse_fpir_targets(args.fpir_targets)

    if not gt_track_labels and not gt_observation_labels:
        raise RuntimeError("No GT labels loaded. Check --gt file format and required fields.")

    result: dict[str, Any] = {
        "pred_jsonl": str(pred_path),
        "gt": str(gt_path),
        "decision_source": str(args.decision_source),
        "probe_filter": str(args.probe_filter),
        "fpir_targets": [float(v) for v in fpir_targets],
        "default_threshold": float(args.default_threshold),
        "num_prediction_rows": int(len(pred_rows)),
        "gt_track_labels": int(len(gt_track_labels)),
        "gt_observation_labels": int(len(gt_observation_labels)),
        "warnings": gt_warnings,
    }

    if args.mode in {"track", "both"}:
        if not gt_track_labels:
            raise RuntimeError("Track-level evaluation requested but no track labels were found in --gt")
        track_items = _build_track_items(
            pred_rows,
            gt_track_labels,
            decision_source=str(args.decision_source),
            probe_filter=str(args.probe_filter),
            id_to_name=id_name_map,
        )
        result["track_level"] = evaluate_items(
            track_items,
            default_threshold=float(args.default_threshold),
            fpir_targets=fpir_targets,
            max_curve_points=max(10, int(args.max_curve_points)),
        )

    if args.mode in {"observation", "both"}:
        if not gt_observation_labels and not gt_track_labels:
            raise RuntimeError("Observation-level evaluation needs observation labels or track labels")
        observation_items, observation_source = _build_observation_items(
            pred_rows,
            gt_observation_labels,
            gt_track_labels,
            decision_source=str(args.decision_source),
            probe_filter=str(args.probe_filter),
            id_to_name=id_name_map,
        )
        result["observation_level"] = evaluate_items(
            observation_items,
            default_threshold=float(args.default_threshold),
            fpir_targets=fpir_targets,
            max_curve_points=max(10, int(args.max_curve_points)),
        )
        result["observation_level"]["label_source"] = str(observation_source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
