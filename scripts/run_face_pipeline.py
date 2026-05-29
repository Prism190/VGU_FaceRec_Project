#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MPL_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from fas_kd.data.transforms import build_eval_transform
from fas_kd.models.student import MobileNetV4Student
from fas_kd.pipeline import (
    DetectionConfig,
    FaceDetection,
    FacePreprocessor,
    IdentityIndex,
    IncrementalUnknownClusterer,
    MagnitudeQualityGate,
    PreprocessConfig,
    RuntimePipeline,
    ThresholdLivenessGate,
    YOLO11FaceDetector,
)
from fas_kd.utils.config import load_yaml_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_ms1m_magface_phase3_trueasym_swa_v1.yaml"


def _epoch_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def _resolve_checkpoint(cfg: dict, checkpoint_arg: str) -> Path:
    output_root = Path(cfg["experiment"]["output_root"]).resolve()
    ckpt_dir = output_root / "checkpoints"

    alias = checkpoint_arg.strip().lower()
    if alias in {"auto", "current"}:
        epoch_paths = sorted(ckpt_dir.glob("epoch_*.pt"), key=_epoch_key)
        if epoch_paths:
            return epoch_paths[-1]
        for fixed in ["latest.pt", "best.pt", "swa.pt"]:
            path = ckpt_dir / fixed
            if path.exists():
                return path
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if alias in {"latest", "best", "swa"}:
        path = ckpt_dir / f"{alias}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint alias not found: {path}")
        return path

    candidate = Path(checkpoint_arg)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def _build_student(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    student_cfg = cfg["student"]
    model = MobileNetV4Student(
        backbone_name=student_cfg["backbone_name"],
        embedding_dim=int(student_cfg.get("embedding_dim", 512)),
        pretrained=False,
        input_size=int(cfg["data"].get("image_size", 112)),
        projection_activation=str(student_cfg.get("projection_activation", "none")),
        spatial_out_channels=int(student_cfg.get("spatial_out_channels", 0)),
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("student_state", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing student keys while loading {checkpoint_path}: {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"Unexpected student keys while loading {checkpoint_path}: {unexpected[:5]}")

    model.eval().to(device)
    return model


def _build_embed_fn(model: torch.nn.Module, transform, device: torch.device, use_amp: bool):
    @torch.no_grad()
    def _embed(face_rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(face_rgb.astype(np.uint8), mode="RGB")
        x = transform(pil).unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            emb = model(x)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        return emb[0].detach().cpu().numpy().astype(np.float32)

    return _embed


def _infer_liveness_always_live(_: np.ndarray) -> float:
    return 1.0


def _infer_liveness_texture(face_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    var_lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # Heuristic score in [0,1]. This is a placeholder, not a production anti-spoof model.
    score = (var_lap - 5.0) / 60.0
    return float(np.clip(score, 0.0, 1.0))


def _load_gallery(index: IdentityIndex, gallery_npz: Path | None) -> int:
    if gallery_npz is None:
        return 0
    data = np.load(gallery_npz)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    ids = np.asarray(data["ids"], dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError(f"gallery embeddings must be 2D, got {embeddings.shape}")
    if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
        raise ValueError("gallery ids shape mismatch")

    for emb, identity in zip(embeddings, ids):
        index.add(identity_id=int(identity), embedding=emb)
    return int(ids.shape[0])


def _draw_observation(frame_bgr: np.ndarray, obs: dict) -> None:
    x1, y1, x2, y2 = [int(v) for v in obs["bbox_xyxy"]]
    is_live = bool(obs["is_live"])
    quality = bool(obs["quality_pass"])
    identity_id = obs.get("identity_id")

    if identity_id is not None:
        color = (0, 220, 0)
    elif is_live and quality:
        color = (0, 180, 255)
    else:
        color = (0, 80, 255)

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    label = (
        f"T{obs['track_id']} live={int(is_live)} q={int(quality)} "
        f"mag={obs['magnitude']:.1f}"
    )
    if identity_id is not None:
        label += f" id={identity_id} sim={obs.get('match_score', 0.0):.3f}"
    cv2.putText(frame_bgr, label, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _open_annotated_writer(
    out_video_path: Path,
    width: int,
    height: int,
    fps: float,
) -> tuple[cv2.VideoWriter, Path, str]:
    candidates: list[tuple[str, Path]] = [
        ("avc1", out_video_path),
        ("H264", out_video_path),
        ("mp4v", out_video_path.with_name(f"{out_video_path.stem}.tmp_mp4v{out_video_path.suffix}")),
    ]

    for fourcc_tag, path in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_tag),
            float(fps),
            (int(width), int(height)),
        )
        if writer.isOpened():
            return writer, path, fourcc_tag
        writer.release()

    raise RuntimeError(
        "Could not initialize video writer. Tried codecs: avc1, H264, mp4v. "
        "Install FFmpeg-backed OpenCV or use a build with H.264 support."
    )


def _transcode_mp4_to_h264(src_path: Path, dst_path: Path) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 0


def _synthetic_stream(max_frames: int) -> Iterable[tuple[int, np.ndarray, list[FaceDetection]]]:
    count = max_frames if max_frames > 0 else 180
    width = 640
    height = 480

    for frame_idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cx = 320 + int(90.0 * np.sin(frame_idx / 12.0))
        cy = 240

        cv2.circle(frame, (cx, cy), 78, (185, 170, 140), -1)
        cv2.circle(frame, (cx - 24, cy - 16), 8, (30, 30, 30), -1)
        cv2.circle(frame, (cx + 24, cy - 16), 8, (30, 30, 30), -1)
        cv2.ellipse(frame, (cx, cy + 24), (26, 12), 0, 0, 180, (40, 40, 40), 2)

        bbox = (float(cx - 80), float(cy - 96), float(cx + 80), float(cy + 96))
        landmarks = np.array(
            [
                [cx - 25.0, cy - 16.0],
                [cx + 25.0, cy - 16.0],
                [cx + 0.0, cy + 4.0],
                [cx - 20.0, cy + 36.0],
                [cx + 20.0, cy + 36.0],
            ],
            dtype=np.float32,
        )
        detections = [FaceDetection(bbox_xyxy=bbox, landmarks5=landmarks, score=0.99)]
        yield frame_idx, frame, detections


def _video_stream(source: str, detector: YOLO11FaceDetector, max_frames: int):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            yield frame_idx, frame, detections
            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the face pipeline with current student checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML training config")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="Checkpoint path or alias: auto/current/latest/best/swa",
    )
    parser.add_argument("--source", default="", help="Video path or camera index. Ignore when --demo-synthetic")
    parser.add_argument("--demo-synthetic", action="store_true", help="Run with generated synthetic frames and fake landmarks")
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to process. <=0 means full stream")

    parser.add_argument("--detector-model", default="", help="YOLO face model path for real video mode")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--det-max", type=int, default=50)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--det-disable-rescue-pass", action="store_true")
    parser.add_argument("--det-rescue-conf", type=float, default=0.08)
    parser.add_argument("--det-rescue-iou", type=float, default=0.45)
    parser.add_argument("--det-rescue-imgsz", type=int, default=1280)
    parser.add_argument("--det-rescue-min-primary", type=int, default=2)
    parser.add_argument("--det-merge-iou", type=float, default=0.55)

    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--quality-min", type=float, default=None, help="Minimum embedding norm. Default from config margin_head.feature_norm_lower")
    parser.add_argument("--quality-max", type=float, default=None, help="Maximum embedding norm. Default from config margin_head.feature_norm_upper")
    parser.add_argument("--liveness-mode", choices=["always_live", "texture"], default="always_live")
    parser.add_argument("--live-threshold", type=float, default=0.5)
    parser.add_argument("--no-liveness-tta", action="store_true")
    parser.add_argument("--liveness-every", type=int, default=15, help="Run liveness every N frames per track")

    parser.add_argument("--gallery-npz", default="", help="Optional gallery npz with arrays: embeddings, ids")
    parser.add_argument("--match-threshold", type=float, default=0.35)

    parser.add_argument("--dbscan-eps", type=float, default=0.35, help="Cosine-distance threshold for online unknown clustering")
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--unknown-max-buffer", type=int, default=2000)
    parser.add_argument("--cluster-every", type=int, default=64, help="Deprecated (ignored): kept for backward compatibility")
    parser.add_argument(
        "--cluster-interval-sec",
        type=float,
        default=2.0,
        help="Deprecated (ignored): kept for backward compatibility",
    )
    parser.add_argument(
        "--cluster-sync",
        action="store_true",
        help="Deprecated (ignored): online clustering is always incremental",
    )

    parser.add_argument("--out-jsonl", default="logs/pipeline_run.jsonl")
    parser.add_argument("--out-summary", default="", help="Optional summary JSON path")
    parser.add_argument("--out-video", default="", help="Optional annotated output video path")
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    ckpt_path = _resolve_checkpoint(cfg=cfg, checkpoint_arg=args.checkpoint)
    quality_min = (
        float(args.quality_min)
        if args.quality_min is not None
        else float(cfg.get("margin_head", {}).get("feature_norm_lower", 10.0))
    )
    quality_max = (
        float(args.quality_max)
        if args.quality_max is not None
        else float(cfg.get("margin_head", {}).get("feature_norm_upper", 120.0))
    )

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    use_amp = bool(cfg.get("system", {}).get("use_amp", True))
    model = _build_student(cfg=cfg, checkpoint_path=ckpt_path, device=device)
    transform = build_eval_transform(cfg["data"])
    embed_fn = _build_embed_fn(model=model, transform=transform, device=device, use_amp=use_amp)

    pre_cfg = PreprocessConfig(
        image_size=int(cfg["data"].get("image_size", 112)),
        use_clahe=bool(cfg["data"].get("use_clahe", False)),
    )

    liveness_infer = _infer_liveness_always_live if args.liveness_mode == "always_live" else _infer_liveness_texture

    pipeline = RuntimePipeline(
        preprocess=FacePreprocessor(pre_cfg),
        liveness_gate=ThresholdLivenessGate(
            infer_fn=liveness_infer,
            live_threshold=float(args.live_threshold),
            use_ttda=not args.no_liveness_tta,
        ),
        quality_gate=MagnitudeQualityGate(
            min_magnitude=quality_min,
            max_magnitude=quality_max,
        ),
        embed_fn=embed_fn,
        liveness_interval_frames=max(0, int(args.liveness_every)),
    )

    index = IdentityIndex(dim=int(cfg["student"].get("embedding_dim", 512)), use_faiss=True)
    gallery_path = Path(args.gallery_npz).resolve() if args.gallery_npz else None
    num_gallery = _load_gallery(index=index, gallery_npz=gallery_path)

    clusterer = IncrementalUnknownClusterer(
        eps=float(args.dbscan_eps),
        min_samples=int(args.dbscan_min_samples),
        max_buffer_size=int(args.unknown_max_buffer),
    )

    detector = None
    if not args.demo_synthetic:
        if not args.source:
            raise ValueError("--source is required unless --demo-synthetic is set")
        if not args.detector_model:
            raise ValueError("--detector-model is required unless --demo-synthetic is set")
        if args.source.startswith("/path/to/") or args.detector_model.startswith("/path/to/"):
            raise ValueError(
                "Replace placeholder paths. Example: --source data/raw/pipeline_demo/short_hamilton_clip.mp4 "
                "--detector-model checkpoints/pretrained/yolo11n-face-age.pt"
            )
        detector = YOLO11FaceDetector(
            model_path=args.detector_model,
            cfg=DetectionConfig(
                conf_thres=float(args.det_conf),
                iou_thres=float(args.det_iou),
                max_det=int(args.det_max),
                imgsz=int(args.det_imgsz),
                enable_rescue_pass=not bool(args.det_disable_rescue_pass),
                rescue_conf_thres=float(args.det_rescue_conf),
                rescue_iou_thres=float(args.det_rescue_iou),
                rescue_imgsz=int(args.det_rescue_imgsz),
                rescue_min_primary_detections=int(args.det_rescue_min_primary),
                merge_iou_thres=float(args.det_merge_iou),
            ),
        )

    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (PROJECT_ROOT / out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_video_path: Path | None = None
    if args.out_video:
        out_video_path = Path(args.out_video)
        if not out_video_path.is_absolute():
            out_video_path = (PROJECT_ROOT / out_video_path).resolve()
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

    if args.out_summary:
        out_summary = Path(args.out_summary)
        if not out_summary.is_absolute():
            out_summary = (PROJECT_ROOT / out_summary).resolve()
    else:
        out_summary = out_jsonl.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    if args.demo_synthetic:
        stream = _synthetic_stream(max_frames=int(args.max_frames))
    else:
        assert detector is not None
        stream = _video_stream(source=args.source, detector=detector, max_frames=int(args.max_frames))

    writer = None
    writer_path: Path | None = None
    writer_codec: str | None = None

    frames_processed = 0
    total_detections = 0
    total_observations = 0
    accepted_observations = 0
    recognized_observations = 0

    print(f"[pipeline] config={args.config}")
    print(f"[pipeline] checkpoint={ckpt_path}")
    print(f"[pipeline] device={device.type} amp={use_amp}")
    print(f"[pipeline] quality_gate=[{quality_min:.3f}, {quality_max:.3f}]")
    print(
        "[pipeline] detector="
        f"conf={args.det_conf:.3f} iou={args.det_iou:.3f} imgsz={args.det_imgsz} "
        f"rescue={int(not args.det_disable_rescue_pass)} rescue_conf={args.det_rescue_conf:.3f} "
        f"rescue_imgsz={args.det_rescue_imgsz}"
    )
    print("[pipeline] tracker=motion_appearance_hungarian")
    print("[pipeline] clustering_mode=incremental_online")
    if args.cluster_sync or int(args.cluster_every) != 64 or float(args.cluster_interval_sec) != 2.0:
        print("[warn] --cluster-sync/--cluster-every/--cluster-interval-sec are deprecated and ignored")
    print(f"[pipeline] gallery_size={num_gallery}")
    print(f"[pipeline] output_jsonl={out_jsonl}")

    with out_jsonl.open("w", encoding="utf-8") as fp:
        for frame_idx, frame_bgr, detections in stream:
            frames_processed += 1
            total_detections += len(detections)

            observations = pipeline.process_frame(frame_bgr=frame_bgr, detections=detections, frame_idx=frame_idx)
            total_observations += len(observations)

            rows_for_draw: list[dict] = []
            for obs in observations:
                identity_id = None
                match_score = None
                cluster_label = None

                if obs.is_live and obs.quality_pass:
                    accepted_observations += 1
                    pooled = pipeline.pooled_track_embedding(obs.track_id)
                    if pooled is not None:
                        hits = index.search(pooled, k=1)
                        if hits and hits[0].score >= float(args.match_threshold):
                            identity_id = int(hits[0].identity_id)
                            match_score = float(hits[0].score)
                            recognized_observations += 1
                        else:
                            clusterer.add(pooled)
                            latest_label = clusterer.latest_label()
                            if latest_label is not None and int(latest_label) >= 0:
                                cluster_label = int(latest_label)

                row = {
                    "frame_idx": int(frame_idx),
                    "track_id": int(obs.track_id),
                    "bbox_xyxy": [float(v) for v in obs.bbox_xyxy],
                    "magnitude": float(obs.magnitude),
                    "liveness_score": float(obs.liveness_score),
                    "is_live": bool(obs.is_live),
                    "quality_pass": bool(obs.quality_pass),
                    "identity_id": identity_id,
                    "match_score": match_score,
                    "cluster_label": cluster_label,
                }
                fp.write(json.dumps(row, ensure_ascii=True) + "\n")
                rows_for_draw.append(row)

            if out_video_path is not None:
                if writer is None:
                    h, w = frame_bgr.shape[:2]
                    writer, writer_path, writer_codec = _open_annotated_writer(
                        out_video_path=out_video_path,
                        width=w,
                        height=h,
                        fps=25.0,
                    )
                    print(f"[pipeline] video_writer_codec={writer_codec} path={writer_path}")
                for row in rows_for_draw:
                    _draw_observation(frame_bgr, row)
                writer.write(frame_bgr)

            if args.print_every > 0 and (frames_processed % int(args.print_every) == 0):
                print(
                    f"[progress] frame={frames_processed} det={total_detections} "
                    f"obs={total_observations} accepted={accepted_observations} "
                    f"recognized={recognized_observations} unknown_buf={len(clusterer)}"
                )

    if writer is not None:
        writer.release()
        if out_video_path is not None and writer_path is not None and writer_codec == "mp4v":
            transcoded = _transcode_mp4_to_h264(src_path=writer_path, dst_path=out_video_path)
            if transcoded:
                writer_path.unlink(missing_ok=True)
                print(f"[pipeline] transcoded annotated video to H.264: {out_video_path}")
            else:
                writer_path.replace(out_video_path)
                print(
                    "[warn] ffmpeg H.264 transcode unavailable/failed; kept mp4v output. "
                    "If VS Code cannot play it, run: ffmpeg -y -i <in.mp4> -c:v libx264 -pix_fmt yuv420p <out.mp4>"
                )

    final_labels = clusterer.cluster() if clusterer.buffer else np.asarray([], dtype=np.int32)
    cluster_hist: dict[str, int] = {}
    if final_labels.size > 0:
        unique, counts = np.unique(final_labels, return_counts=True)
        for label, count in zip(unique, counts):
            if int(label) < 0:
                continue
            cluster_hist[str(int(label))] = int(count)

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "source": "synthetic" if args.demo_synthetic else args.source,
        "frames_processed": int(frames_processed),
        "detections": int(total_detections),
        "observations": int(total_observations),
        "accepted_observations": int(accepted_observations),
        "recognized_observations": int(recognized_observations),
        "unknown_buffer_size": int(len(clusterer)),
        "unknown_cluster_hist": cluster_hist,
        "output_jsonl": str(out_jsonl),
        "output_video": str(out_video_path) if out_video_path is not None else None,
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[summary]")
    print(json.dumps(summary, indent=2))
    print(f"WROTE {out_summary}")


if __name__ == "__main__":
    main()
