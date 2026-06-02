#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MPL_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from fas_kd.pipeline import DetectionConfig, FacePreprocessor, PreprocessConfig, YOLO11FaceDetector, ensure_face_db_layout
from fas_kd.utils.config import load_yaml_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IDENTITY_DIR_RE = re.compile(r"^id_(\d+)(?:__.*)?$")


def _slugify(value: str, max_len: int = 48) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower())
    text = text.strip("-")
    if not text:
        text = "person"
    return text[:max_len].strip("-") or "person"


def _is_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _download_to_dir(url: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    raw_name = Path(parsed.path).name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    stem = _slugify(Path(raw_name).stem or "image", max_len=64)
    out_path = download_dir / f"{stem}{suffix}"

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    out_path.write_bytes(data)
    return out_path


def _parse_identity_id(identity_dir: Path) -> int | None:
    m = IDENTITY_DIR_RE.match(identity_dir.name)
    if m is None:
        return None
    return int(m.group(1))


def _iter_identity_dirs(known_identities_dir: Path) -> list[Path]:
    return sorted([p for p in known_identities_dir.iterdir() if p.is_dir()])


def _resolve_identity_dir(known_identities_dir: Path, name: str, min_identity_id: int) -> tuple[int, Path, bool]:
    wanted = str(name).strip().lower()
    existing = _iter_identity_dirs(known_identities_dir)

    max_id = int(min_identity_id) - 1
    for identity_dir in existing:
        identity_id = _parse_identity_id(identity_dir)
        if identity_id is not None:
            max_id = max(max_id, int(identity_id))

        meta = _load_json(identity_dir / "meta.json")
        meta_name = str(meta.get("name") or "").strip().lower()
        if meta_name and meta_name == wanted and identity_id is not None:
            return int(identity_id), identity_dir, False

    next_id = max(int(min_identity_id), int(max_id + 1))
    out_dir = known_identities_dir / f"id_{next_id:06d}__{_slugify(name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return int(next_id), out_dir, True


def _frontal_score(landmarks5: np.ndarray) -> float:
    lm = np.asarray(landmarks5, dtype=np.float32)
    if lm.shape != (5, 2):
        return 0.0

    left_eye = lm[0]
    right_eye = lm[1]
    nose = lm[2]

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-6:
        return 0.0

    eye_mid_x = float(0.5 * (left_eye[0] + right_eye[0]))
    yaw = abs(float(nose[0]) - eye_mid_x) / eye_dist
    roll = abs(float(left_eye[1]) - float(right_eye[1])) / eye_dist

    score = 1.0 - (1.4 * yaw + 0.8 * roll)
    return float(np.clip(score, 0.0, 1.0))


def _pick_best_detection(detections: list, image_shape: tuple[int, int, int]):
    if not detections:
        return None

    h, w = image_shape[:2]
    frame_area = float(max(1, h * w))
    best = None
    best_score = -1e9
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
        frontal = _frontal_score(det.landmarks5)
        score = 0.55 * frontal + 0.30 * float(det.score) + 0.15 * float(np.clip(area_ratio / 0.12, 0.0, 1.0))
        if score > best_score:
            best_score = score
            best = det
    return best


def _draw_debug_detection(image_bgr: np.ndarray, det, out_path: Path, label: str) -> None:
    canvas = image_bgr.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 220, 20), 2)
    for pt in np.asarray(det.landmarks5, dtype=np.float32):
        px, py = int(round(float(pt[0]))), int(round(float(pt[1])))
        cv2.circle(canvas, (px, py), 3, (0, 180, 255), -1)
    cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _embedding_count(identity_dir: Path) -> int:
    emb_file = identity_dir / "embeddings.npz"
    if not emb_file.exists():
        return 0
    try:
        with np.load(emb_file) as data:
            arr = np.asarray(data.get("embeddings"), dtype=np.float32)
            if arr.ndim == 2:
                return int(arr.shape[0])
            if arr.ndim == 1:
                return 1
    except Exception:
        return 0
    return 0


def _photo_count(identity_dir: Path) -> int:
    photos_dir = identity_dir / "photos"
    if not photos_dir.exists():
        return 0
    return int(sum(1 for p in photos_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))


def _parse_entries(entries: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in entries:
        text = str(raw).strip()
        if not text or "=" not in text:
            raise ValueError(f"Invalid --entry '{raw}'. Expected format: Name=path_or_url")
        name, source = text.split("=", 1)
        name = str(name).strip()
        source = str(source).strip()
        if not name or not source:
            raise ValueError(f"Invalid --entry '{raw}'. Name and source are required")
        out.append((name, source))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Import known identities from source images with YOLO face detection")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Identity input entry as Name=path_or_url. Repeat for multiple identities.",
    )
    parser.add_argument(
        "--config",
        default="configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml",
        help="Pipeline/train config used to pick aligned face size and CLAHE setting",
    )
    parser.add_argument(
        "--detector-model",
        default="checkpoints/pretrained/yolo11n-face-age.pt",
        help="YOLO11 face detector model path",
    )
    parser.add_argument("--face-db-root", default="data/face_db", help="Face DB root folder")
    parser.add_argument(
        "--download-dir",
        default="data/raw/pipeline_demo/celebs",
        help="Folder used when downloading URL sources",
    )
    parser.add_argument(
        "--debug-dir",
        default="logs/celebs_import_debug",
        help="Folder for detector debug overlays",
    )
    parser.add_argument("--min-identity-id", type=int, default=1000)
    args = parser.parse_args()

    entries = _parse_entries(args.entry)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = load_yaml_config(str(config_path))

    detector_model = Path(args.detector_model)
    if not detector_model.is_absolute():
        detector_model = (PROJECT_ROOT / detector_model).resolve()
    if not detector_model.exists():
        raise FileNotFoundError(f"detector model not found: {detector_model}")

    face_db_root = Path(args.face_db_root)
    if not face_db_root.is_absolute():
        face_db_root = (PROJECT_ROOT / face_db_root).resolve()
    layout = ensure_face_db_layout(face_db_root)
    known_identities_dir = layout["known_identities"]

    download_dir = Path(args.download_dir)
    if not download_dir.is_absolute():
        download_dir = (PROJECT_ROOT / download_dir).resolve()
    debug_dir = Path(args.debug_dir)
    if not debug_dir.is_absolute():
        debug_dir = (PROJECT_ROOT / debug_dir).resolve()

    det_cfg = DetectionConfig(
        conf_thres=0.10,
        iou_thres=0.45,
        max_det=20,
        imgsz=960,
        enable_rescue_pass=True,
        rescue_conf_thres=0.06,
        rescue_iou_thres=0.45,
        rescue_imgsz=1280,
        rescue_min_primary_detections=1,
        merge_iou_thres=0.55,
    )
    detector = YOLO11FaceDetector(model_path=str(detector_model), cfg=det_cfg)

    pre_cfg = PreprocessConfig(
        image_size=int(cfg.get("data", {}).get("image_size", 112)),
        use_clahe=bool(cfg.get("data", {}).get("use_clahe", False)),
    )
    preprocessor = FacePreprocessor(pre_cfg)

    report: list[dict[str, object]] = []
    for name, source in entries:
        if _is_url(source):
            src_path = _download_to_dir(source, download_dir)
            source_kind = "url"
        else:
            src_path = Path(source)
            if not src_path.is_absolute():
                src_path = (PROJECT_ROOT / src_path).resolve()
            source_kind = "file"

        if not src_path.exists() or not src_path.is_file():
            raise FileNotFoundError(f"source image not found for {name}: {src_path}")

        image_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.size == 0:
            raise RuntimeError(f"failed to decode image for {name}: {src_path}")

        detections = detector.detect(image_bgr)
        best_det = _pick_best_detection(detections, image_bgr.shape)
        if best_det is None:
            raise RuntimeError(f"no face detected for {name}: {src_path}")

        face_rgb = preprocessor(image_bgr, best_det.landmarks5)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        identity_id, identity_dir, created = _resolve_identity_dir(
            known_identities_dir=known_identities_dir,
            name=name,
            min_identity_id=int(args.min_identity_id),
        )

        photos_dir = identity_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_stem = _slugify(name, max_len=32)
        out_photo = photos_dir / f"{stamp}_{file_stem}_yolo_aligned.jpg"
        suffix_idx = 1
        while out_photo.exists():
            out_photo = photos_dir / f"{stamp}_{file_stem}_yolo_aligned_{suffix_idx:02d}.jpg"
            suffix_idx += 1
        ok = cv2.imwrite(str(out_photo), face_bgr)
        if not ok:
            raise RuntimeError(f"failed to write aligned face for {name}: {out_photo}")

        debug_name = f"{identity_id:06d}_{_slugify(name)}_det.jpg"
        debug_path = debug_dir / debug_name
        _draw_debug_detection(
            image_bgr=image_bgr,
            det=best_det,
            out_path=debug_path,
            label=f"{name} conf={float(best_det.score):.3f} frontal={_frontal_score(best_det.landmarks5):.3f}",
        )

        meta_path = identity_dir / "meta.json"
        meta = _load_json(meta_path)
        created_at = meta.get("created_at") if isinstance(meta.get("created_at"), str) else _utc_now()
        meta.update(
            {
                "identity_id": int(identity_id),
                "name": str(name),
                "photo_count": int(_photo_count(identity_dir)),
                "embedding_count": int(_embedding_count(identity_dir)),
                "created_at": created_at,
                "updated_at": _utc_now(),
            }
        )
        _save_json(meta_path, meta)

        report.append(
            {
                "name": str(name),
                "identity_id": int(identity_id),
                "identity_dir": str(identity_dir),
                "created_new_identity": bool(created),
                "source": str(src_path),
                "source_kind": source_kind,
                "detections_found": int(len(detections)),
                "selected_confidence": float(best_det.score),
                "selected_frontal_score": float(_frontal_score(best_det.landmarks5)),
                "photo_written": str(out_photo),
                "debug_detection_image": str(debug_path),
            }
        )

    manifest_path = download_dir / f"known_import_manifest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"imports": report}, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"IMPORTS={len(report)}")
    for rec in report:
        print(
            f"- {rec['name']} -> id={rec['identity_id']} new={int(bool(rec['created_new_identity']))} "
            f"photo={rec['photo_written']} conf={rec['selected_confidence']:.3f} frontal={rec['selected_frontal_score']:.3f}"
        )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
