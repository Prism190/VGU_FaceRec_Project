#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MPL_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from fas_kd.pipeline import DetectionConfig, FacePreprocessor, PreprocessConfig, YOLO11FaceDetector


def _load_face_image_names(face_tid_mid_path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    with face_tid_mid_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            if name in seen:
                continue
            seen.add(name)
            names.append(name)

    return names


def _load_meta_landmarks(name_5pts_path: Path) -> dict[str, tuple[np.ndarray, float]]:
    out: dict[str, tuple[np.ndarray, float]] = {}
    with name_5pts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                continue

            name = parts[0]
            coords = np.asarray([float(v) for v in parts[1:11]], dtype=np.float32).reshape(5, 2)
            score = float(parts[11])
            out[name] = (coords, score)
    return out


def _copy_meta_dir(src_meta: Path, dst_meta: Path) -> None:
    dst_meta.mkdir(parents=True, exist_ok=True)
    for src in src_meta.glob("*"):
        if not src.is_file():
            continue
        dst = dst_meta / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def _fallback_landmarks_for_image(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    box = np.asarray([0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))], dtype=np.float32)
    return YOLO11FaceDetector._landmarks_from_bbox(box)


def _clean_one_dataset(
    dataset_name: str,
    ijb_root: Path,
    output_root: Path,
    detector: YOLO11FaceDetector,
    preprocessor: FacePreprocessor,
    overwrite: bool,
    max_images: int,
    min_meta_score: float,
    jpeg_quality: int,
) -> dict[str, int | str]:
    src_dataset = ijb_root / dataset_name
    src_meta = src_dataset / "meta"
    src_loose = src_dataset / "loose_crop"

    if not src_meta.exists() or not src_loose.exists():
        raise FileNotFoundError(f"Missing expected IJB dirs under: {src_dataset}")

    prefix = dataset_name.lower()
    face_tid_mid_path = src_meta / f"{prefix}_face_tid_mid.txt"
    name_5pts_path = src_meta / f"{prefix}_name_5pts_score.txt"
    if not face_tid_mid_path.exists():
        raise FileNotFoundError(f"Missing file: {face_tid_mid_path}")
    if not name_5pts_path.exists():
        raise FileNotFoundError(f"Missing file: {name_5pts_path}")

    dst_dataset = output_root / dataset_name
    dst_meta = dst_dataset / "meta"
    dst_loose = dst_dataset / "loose_crop"
    dst_loose.mkdir(parents=True, exist_ok=True)
    _copy_meta_dir(src_meta=src_meta, dst_meta=dst_meta)

    image_names = _load_face_image_names(face_tid_mid_path=face_tid_mid_path)
    meta_landmarks = _load_meta_landmarks(name_5pts_path=name_5pts_path)

    if max_images > 0:
        image_names = image_names[:max_images]

    stats = {
        "dataset": dataset_name,
        "num_input_images": len(image_names),
        "written": 0,
        "skipped_existing": 0,
        "missing_source": 0,
        "read_fail": 0,
        "yolo_used": 0,
        "meta5pt_used": 0,
        "fallback_used": 0,
        "align_fail": 0,
        "write_fail": 0,
    }

    imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    for name in tqdm(image_names, desc=f"clean {dataset_name}"):
        src_path = src_loose / name
        dst_path = dst_loose / name

        if dst_path.exists() and not overwrite:
            stats["skipped_existing"] += 1
            continue

        if not src_path.exists():
            stats["missing_source"] += 1
            continue

        image_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            stats["read_fail"] += 1
            continue

        landmarks5: np.ndarray | None = None
        detections = detector.detect(image_bgr=image_bgr)
        if detections:
            landmarks5 = detections[0].landmarks5
            stats["yolo_used"] += 1
        else:
            meta = meta_landmarks.get(name)
            if meta is not None and float(meta[1]) >= float(min_meta_score):
                landmarks5 = meta[0]
                stats["meta5pt_used"] += 1
            else:
                landmarks5 = _fallback_landmarks_for_image(image_bgr=image_bgr)
                stats["fallback_used"] += 1

        try:
            face_rgb = preprocessor(image_bgr=image_bgr, landmarks5=landmarks5)
        except Exception:
            stats["align_fail"] += 1
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), imwrite_params)
        if ok:
            stats["written"] += 1
        else:
            stats["write_fail"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned IJB loose crops with YOLO11-based recropping")
    parser.add_argument("--ijb-root", default=str(PROJECT_ROOT / "data" / "raw" / "ijb" / "ijb"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "processed" / "ijb_clean_yolo11"))
    parser.add_argument("--datasets", nargs="+", default=["IJBB", "IJBC"], choices=["IJBB", "IJBC"])

    parser.add_argument("--detector-model", default=str(PROJECT_ROOT / "checkpoints" / "pretrained" / "yolo11n-face-age.pt"))
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--det-max", type=int, default=100)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--disable-rescue-pass", action="store_true")
    parser.add_argument("--rescue-conf", type=float, default=0.08)
    parser.add_argument("--rescue-iou", type=float, default=0.45)
    parser.add_argument("--rescue-imgsz", type=int, default=1280)
    parser.add_argument("--rescue-min-primary", type=int, default=2)
    parser.add_argument("--merge-iou", type=float, default=0.55)

    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-meta-score", type=float, default=0.0)
    parser.add_argument("--max-images", type=int, default=0, help="Process only first N images per dataset for smoke test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-json", default="", help="Optional JSON report path")
    args = parser.parse_args()

    ijb_root = Path(args.ijb_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not ijb_root.exists():
        raise FileNotFoundError(f"IJB root not found: {ijb_root}")

    det_cfg = DetectionConfig(
        conf_thres=float(args.det_conf),
        iou_thres=float(args.det_iou),
        max_det=int(args.det_max),
        imgsz=int(args.det_imgsz),
        enable_rescue_pass=not bool(args.disable_rescue_pass),
        rescue_conf_thres=float(args.rescue_conf),
        rescue_iou_thres=float(args.rescue_iou),
        rescue_imgsz=int(args.rescue_imgsz),
        rescue_min_primary_detections=int(args.rescue_min_primary),
        merge_iou_thres=float(args.merge_iou),
        fallback_bbox_landmarks=True,
    )
    detector = YOLO11FaceDetector(model_path=str(Path(args.detector_model).resolve()), cfg=det_cfg)
    preprocessor = FacePreprocessor(PreprocessConfig(image_size=int(args.image_size), use_clahe=False))

    all_stats: list[dict[str, int | str]] = []
    for dataset_name in args.datasets:
        stats = _clean_one_dataset(
            dataset_name=dataset_name,
            ijb_root=ijb_root,
            output_root=output_root,
            detector=detector,
            preprocessor=preprocessor,
            overwrite=bool(args.overwrite),
            max_images=int(args.max_images),
            min_meta_score=float(args.min_meta_score),
            jpeg_quality=int(args.jpeg_quality),
        )
        all_stats.append(stats)

    report = {
        "ijb_root": str(ijb_root),
        "output_root": str(output_root),
        "detector_model": str(Path(args.detector_model).resolve()),
        "datasets": all_stats,
    }

    print(json.dumps(report, indent=2))

    if args.report_json:
        report_path = Path(args.report_json)
        if not report_path.is_absolute():
            report_path = (PROJECT_ROOT / report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()