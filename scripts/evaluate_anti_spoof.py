#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.pipeline import SilentFaceAntiSpoof

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIVE_TOKENS = {"1", "live", "real", "bona_fide", "bonafide", "genuine"}
SPOOF_TOKENS = {"0", "spoof", "fake", "attack", "imposter", "impostor"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int  # 1 = live, 0 = spoof
    meta: dict[str, Any]


def _parse_label(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, np.integer)):
        return 1 if int(raw) != 0 else 0

    text = str(raw).strip().lower()
    if text in LIVE_TOKENS:
        return 1
    if text in SPOOF_TOKENS:
        return 0

    try:
        val = int(text)
    except Exception:
        return None
    return 1 if val != 0 else 0


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _samples_from_dirs(live_dir: Path, spoof_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for p in _iter_images(live_dir):
        samples.append(Sample(image_path=p, label=1, meta={"source": "live_dir"}))
    for p in _iter_images(spoof_dir):
        samples.append(Sample(image_path=p, label=0, meta={"source": "spoof_dir"}))
    return samples


def _samples_from_manifest(manifest_path: Path, root_dir: Path | None, split: str | None) -> list[Sample]:
    samples: list[Sample] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest has no header: {manifest_path}")

        for row in reader:
            if split is not None:
                row_split = str(row.get("split", "")).strip().lower()
                if row_split != str(split).strip().lower():
                    continue

            path_raw = row.get("path") or row.get("image_path") or row.get("file")
            label_raw = row.get("label")
            if path_raw is None or label_raw is None:
                continue

            label = _parse_label(label_raw)
            if label is None:
                continue

            image_path = Path(path_raw)
            if not image_path.is_absolute():
                base = root_dir if root_dir is not None else manifest_path.parent
                image_path = (base / image_path).resolve()

            meta = {k: v for k, v in row.items()}
            samples.append(Sample(image_path=image_path, label=int(label), meta=meta))

    return samples


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    idx = np.where(fpr <= float(target_far))[0]
    if idx.size == 0:
        return float(tpr[0])
    return float(tpr[idx[-1]])


def _eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) * 0.5), float(thresholds[idx])


def _metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    pred_live = scores >= float(threshold)
    true_live = labels == 1
    true_spoof = labels == 0

    n_live = int(np.sum(true_live))
    n_spoof = int(np.sum(true_spoof))

    live_pred_spoof = int(np.sum(np.logical_and(true_live, np.logical_not(pred_live))))
    spoof_pred_live = int(np.sum(np.logical_and(true_spoof, pred_live)))

    bpcer = float(live_pred_spoof / n_live) if n_live > 0 else 0.0
    apcer = float(spoof_pred_live / n_spoof) if n_spoof > 0 else 0.0
    acer = float((apcer + bpcer) * 0.5)

    return {
        "threshold": float(threshold),
        "live_total": n_live,
        "spoof_total": n_spoof,
        "live_pred_spoof": live_pred_spoof,
        "spoof_pred_live": spoof_pred_live,
        "bpcer": bpcer,
        "apcer": apcer,
        "acer": acer,
        "tpr": float(1.0 - bpcer),
        "fpr": float(apcer),
    }


def _best_acer_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    unique_scores = np.unique(scores)
    # Include infinities so edge operating points are visible.
    candidates = np.concatenate(
        (
            np.asarray([float("inf")], dtype=np.float64),
            unique_scores,
            np.asarray([float("-inf")], dtype=np.float64),
        ),
        axis=0,
    )

    best: dict[str, Any] | None = None
    for thr in candidates:
        m = _metrics_at_threshold(scores=scores, labels=labels, threshold=float(thr))
        if best is None:
            best = m
            continue
        if float(m["acer"]) < float(best["acer"]):
            best = m
            continue
        if float(m["acer"]) == float(best["acer"]) and float(m["threshold"]) < float(best["threshold"]):
            best = m

    assert best is not None
    return best


def evaluate(
    model: SilentFaceAntiSpoof,
    samples: list[Sample],
    threshold: float,
    target_fars: list[float],
) -> dict[str, Any]:
    valid_paths: list[str] = []
    scores_list: list[float] = []
    labels_list: list[int] = []
    failed_paths: list[str] = []

    for sample in samples:
        img = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            failed_paths.append(str(sample.image_path))
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score = float(model.score(rgb))
        valid_paths.append(str(sample.image_path))
        scores_list.append(score)
        labels_list.append(int(sample.label))

    if not scores_list:
        raise RuntimeError("No valid samples were scored")

    scores = np.asarray(scores_list, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.int32)

    n_live = int(np.sum(labels == 1))
    n_spoof = int(np.sum(labels == 0))

    out: dict[str, Any] = {
        "num_samples_input": int(len(samples)),
        "num_samples_scored": int(scores.shape[0]),
        "num_failed_reads": int(len(failed_paths)),
        "live_total": n_live,
        "spoof_total": n_spoof,
        "failed_paths": failed_paths[:50],
        "threshold_metrics": _metrics_at_threshold(scores, labels, threshold=float(threshold)),
        "best_acer_metrics": _best_acer_threshold(scores=scores, labels=labels),
    }

    if len(np.unique(labels)) < 2:
        out["roc_auc"] = None
        out["eer"] = None
        out["eer_threshold"] = None
        out["tar_at_far"] = {}
        out["note"] = "Only one class present in labels; ROC metrics unavailable"
        return out

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    auc = float(roc_auc_score(labels, scores))
    eer, eer_thr = _eer(fpr=fpr, tpr=tpr, thresholds=thr)

    tar_at_far = {
        f"tar_far_{target:g}": _tar_at_far(fpr=fpr, tpr=tpr, target_far=float(target))
        for target in target_fars
    }

    out["roc_auc"] = auc
    out["eer"] = eer
    out["eer_threshold"] = float(eer_thr)
    out["eer_threshold_metrics"] = _metrics_at_threshold(scores=scores, labels=labels, threshold=float(eer_thr))
    out["tar_at_far"] = tar_at_far

    # Also report BPCER at APCER targets commonly used in PAD papers.
    bpcer_at_apcer: dict[str, float] = {}
    for target in [0.01, 0.05, 0.10]:
        candidates = np.where(fpr <= target)[0]
        if candidates.size == 0:
            bpcer = 1.0
        else:
            bpcer = float(1.0 - tpr[candidates[-1]])
        bpcer_at_apcer[f"bpcer_apcer_{target:g}"] = bpcer
    out["bpcer_at_apcer"] = bpcer_at_apcer

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anti-spoof model with PAD metrics")
    parser.add_argument("--model-path", required=True, help="Path to Silent-Face .pth model")
    parser.add_argument(
        "--manifest",
        default="",
        help="CSV manifest with columns: path,label (optional split). Use this OR --live-dir/--spoof-dir",
    )
    parser.add_argument("--manifest-root", default="", help="Optional root prepended to relative manifest paths")
    parser.add_argument("--split", default="", help="Optional split filter when manifest has a split column")
    parser.add_argument("--live-dir", default="", help="Directory of live images (label=1)")
    parser.add_argument("--spoof-dir", default="", help="Directory of spoof images (label=0)")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-fars", default="0.01,0.001,0.0001")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--live-class-index", type=int, default=1)
    parser.add_argument("--input-color", choices=["bgr", "rgb"], default="bgr")
    parser.add_argument("--out", default="logs/eval_anti_spoof.json")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()

    target_fars = [float(v.strip()) for v in str(args.target_fars).split(",") if v.strip()]
    if not target_fars:
        raise RuntimeError("No target FAR values parsed from --target-fars")

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    manifest = str(args.manifest).strip()
    live_dir = str(args.live_dir).strip()
    spoof_dir = str(args.spoof_dir).strip()

    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = (PROJECT_ROOT / manifest_path).resolve()

        manifest_root = None
        if str(args.manifest_root).strip():
            manifest_root = Path(args.manifest_root)
            if not manifest_root.is_absolute():
                manifest_root = (PROJECT_ROOT / manifest_root).resolve()

        split = str(args.split).strip() or None
        samples = _samples_from_manifest(manifest_path=manifest_path, root_dir=manifest_root, split=split)
    else:
        if not live_dir or not spoof_dir:
            raise RuntimeError("Provide --manifest OR both --live-dir and --spoof-dir")
        live_dir_path = Path(live_dir)
        spoof_dir_path = Path(spoof_dir)
        if not live_dir_path.is_absolute():
            live_dir_path = (PROJECT_ROOT / live_dir_path).resolve()
        if not spoof_dir_path.is_absolute():
            spoof_dir_path = (PROJECT_ROOT / spoof_dir_path).resolve()
        samples = _samples_from_dirs(live_dir=live_dir_path, spoof_dir=spoof_dir_path)

    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]

    if not samples:
        raise RuntimeError("No evaluation samples were loaded")

    model = SilentFaceAntiSpoof(
        model_path=model_path,
        device=device,
        live_class_index=int(args.live_class_index),
        expect_bgr_input=str(args.input_color).lower() == "bgr",
    )

    metrics = evaluate(
        model=model,
        samples=samples,
        threshold=float(args.threshold),
        target_fars=target_fars,
    )

    payload = {
        "model_path": str(model_path),
        "device": str(device),
        "input_color": str(args.input_color).lower(),
        "live_class_index": int(args.live_class_index),
        "threshold": float(args.threshold),
        "target_fars": target_fars,
        **metrics,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
