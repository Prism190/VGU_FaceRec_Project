#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.data.transforms import build_eval_transform
from fas_kd.evaluation.ijb_template import evaluate_ijb_template_1to1
from fas_kd.models.student import MobileNetV4Student
from fas_kd.models.teacher import build_frozen_teacher
from fas_kd.utils.config import load_yaml_config


@dataclass
class ModelSpec:
    name: str
    kind: str  # student|teacher
    config_path: Path
    checkpoint_path: Path | None
    raw_metric_paths: dict[str, Path]


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _load_student(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
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
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def _parse_corrupt_path(exc: Exception) -> Path | None:
    msg = str(exc)
    m = re.search(r"cannot identify image file '([^']+)'", msg)
    if not m:
        return None
    return Path(m.group(1))


def _evaluate_with_retry(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    dataset_root: Path,
    device: torch.device,
    template_pooling: str,
    batch_size: int,
    num_workers: int,
    clean_root: Path,
    raw_root: Path,
) -> dict[str, Any]:
    transform = build_eval_transform(cfg["data"])
    target_fars = cfg.get("metrics", {}).get("target_fars", [1e-3, 1e-4, 1e-5])

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            return evaluate_ijb_template_1to1(
                model=model,
                ijb_root=dataset_root,
                transform=transform,
                device=device,
                use_amp=cfg.get("system", {}).get("use_amp", True),
                target_fars=target_fars,
                batch_size=batch_size,
                num_workers=num_workers,
                template_pooling=template_pooling,
            )
        except Exception as exc:
            bad_path = _parse_corrupt_path(exc)
            if bad_path is None:
                raise

            try:
                rel = bad_path.resolve().relative_to(clean_root.resolve())
            except Exception:
                raise

            src = (raw_root / rel).resolve()
            if not src.exists():
                raise

            bad_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, bad_path)
            print(f"[heal] replaced corrupt clean image: {bad_path} <= {src}", flush=True)
            if attempt >= max_attempts:
                raise

    raise RuntimeError("evaluation retry exhausted")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _to_num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher/phase1/2/3 raw-vs-clean IJB matrix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--template-pooling", default="magface_weighted", choices=["mean", "magface_weighted"])
    parser.add_argument("--raw-root", default="data/raw/ijb/ijb")
    parser.add_argument("--clean-root", default="data/processed/ijb_clean_yolo11")
    # MagFace iResNet100 was trained with InsightFace standard preprocessing: images in [-1, 1].
    # The training configs use input_mode=from_minus_one_to_zero_one which converts [-1,1]→[0,1]
    # before feeding the teacher — this is incorrect for MagFace checkpoints and depresses TAR@low-FAR.
    # Default here to "identity" so standalone teacher evaluation reflects the model's true capability.
    parser.add_argument(
        "--teacher-input-mode",
        default="identity",
        choices=["identity", "from_minus_one_to_zero_one"],
        help=(
            "Input mode for teacher during evaluation. "
            "Use 'identity' (default) for MagFace/InsightFace checkpoints that expect [-1,1] input. "
            "Use 'from_minus_one_to_zero_one' to reproduce training-time behaviour (converts [-1,1]→[0,1])."
        ),
    )
    parser.add_argument("--out-dir", default="logs/ijb_clean_matrix")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-clean-eval", action="store_true")
    args = parser.parse_args()

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = _resolve_path(args.raw_root)
    clean_root = _resolve_path(args.clean_root)

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    specs = [
        ModelSpec(
            name="teacher",
            kind="teacher",
            config_path=_resolve_path("configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"),
            checkpoint_path=None,
            raw_metric_paths={
                "IJBB": _resolve_path("runs/ms1m_magface_phase3_trueasym_swa_v1/logs/eval_teacher_ijbb_template_magface_weighted_20260528_021225.json"),
                "IJBC": _resolve_path("runs/ms1m_magface_phase3_trueasym_swa_v1/logs/eval_teacher_ijbc_template_magface_weighted_20260528_021225.json"),
            },
        ),
        ModelSpec(
            name="phase1",
            kind="student",
            config_path=_resolve_path("configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml"),
            checkpoint_path=_resolve_path("runs/ms1m_magface_phase1_cplus_aplus_v1/checkpoints/latest.pt"),
            raw_metric_paths={
                "IJBB": _resolve_path("runs/ms1m_magface_phase1_cplus_aplus_v1/logs/eval_latest_ijbb_template_magwpool.json"),
                "IJBC": _resolve_path("runs/ms1m_magface_phase1_cplus_aplus_v1/logs/eval_latest_ijbc_template_magwpool.json"),
            },
        ),
        ModelSpec(
            name="phase2",
            kind="student",
            config_path=_resolve_path("configs/train_ms1m_magface_phase2_occlusion_spatial_v1.yaml"),
            checkpoint_path=_resolve_path("runs/ms1m_magface_phase2_occlusion_spatial_v1/checkpoints/latest.pt"),
            raw_metric_paths={
                "IJBB": _resolve_path("runs/ms1m_magface_phase2_occlusion_spatial_v1/logs/eval_latest_ijbb_template_magwpool_20260524_225754.json"),
                "IJBC": _resolve_path("runs/ms1m_magface_phase2_occlusion_spatial_v1/logs/eval_latest_ijbc_template_magwpool_20260524_225754.json"),
            },
        ),
        ModelSpec(
            name="phase3",
            kind="student",
            config_path=_resolve_path("configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"),
            checkpoint_path=_resolve_path("runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/latest.pt"),
            raw_metric_paths={
                "IJBB": _resolve_path("runs/ms1m_magface_phase3_trueasym_swa_v1/logs/eval_latest_ijbb_template_magface_weighted_20260528_013618.json"),
                "IJBC": _resolve_path("runs/ms1m_magface_phase3_trueasym_swa_v1/logs/eval_latest_ijbc_template_magface_weighted_20260528_013618.json"),
            },
        ),
    ]

    dataset_names = ["IJBB", "IJBC"]
    matrix_rows: list[dict[str, Any]] = []

    for spec in specs:
        cfg = load_yaml_config(str(spec.config_path))

        model: torch.nn.Module
        if spec.kind == "teacher":
            teacher_cfg = dict(cfg["teacher"])
            teacher_cfg["input_mode"] = args.teacher_input_mode
            model = build_frozen_teacher(teacher_cfg).to(device)
        else:
            assert spec.checkpoint_path is not None
            model = _load_student(cfg=cfg, checkpoint_path=spec.checkpoint_path, device=device)

        for dataset in dataset_names:
            raw_metrics_path = spec.raw_metric_paths[dataset]
            if not raw_metrics_path.exists():
                raise FileNotFoundError(f"missing raw metrics for {spec.name} {dataset}: {raw_metrics_path}")
            raw_metrics = json.loads(raw_metrics_path.read_text(encoding="utf-8"))

            out_raw_path = out_dir / f"eval_{spec.name}_raw_{dataset.lower()}_{args.template_pooling}.json"
            _write_json(out_raw_path, raw_metrics)

            out_clean_path = out_dir / f"eval_{spec.name}_clean_{dataset.lower()}_{args.template_pooling}.json"
            if out_clean_path.exists() and not args.force:
                clean_metrics = json.loads(out_clean_path.read_text(encoding="utf-8"))
            elif args.skip_clean_eval:
                clean_metrics = {}
            else:
                print(f"[eval] {spec.name} clean {dataset} ...", flush=True)
                clean_metrics = _evaluate_with_retry(
                    model=model,
                    cfg=cfg,
                    dataset_root=clean_root / dataset,
                    device=device,
                    template_pooling=args.template_pooling,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    clean_root=clean_root,
                    raw_root=raw_root,
                )
                _write_json(out_clean_path, clean_metrics)
                print(f"[eval] wrote {out_clean_path}", flush=True)

            raw_auc = _to_num(raw_metrics.get("roc_auc"))
            raw_t14 = _to_num(raw_metrics.get("tar_far_1e-4"))
            raw_t15 = _to_num(raw_metrics.get("tar_far_1e-5"))

            clean_auc = _to_num(clean_metrics.get("roc_auc"))
            clean_t14 = _to_num(clean_metrics.get("tar_far_1e-4"))
            clean_t15 = _to_num(clean_metrics.get("tar_far_1e-5"))

            row = {
                "model": spec.name,
                "dataset": dataset,
                "raw": {
                    "roc_auc": raw_auc,
                    "tar_far_1e-4": raw_t14,
                    "tar_far_1e-5": raw_t15,
                    "source": str(raw_metrics_path),
                    "copied_to": str(out_raw_path),
                },
                "clean": {
                    "roc_auc": clean_auc,
                    "tar_far_1e-4": clean_t14,
                    "tar_far_1e-5": clean_t15,
                    "source": str(out_clean_path),
                },
                "delta_clean_minus_raw": {
                    "roc_auc": None if clean_auc is None or raw_auc is None else float(clean_auc - raw_auc),
                    "tar_far_1e-4": None if clean_t14 is None or raw_t14 is None else float(clean_t14 - raw_t14),
                    "tar_far_1e-5": None if clean_t15 is None or raw_t15 is None else float(clean_t15 - raw_t15),
                },
            }
            matrix_rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    matrix_payload = {
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "template_pooling": str(args.template_pooling),
        "raw_root": str(raw_root),
        "clean_root": str(clean_root),
        "rows": matrix_rows,
    }

    matrix_json = out_dir / "matrix_clean_vs_raw.json"
    _write_json(matrix_json, matrix_payload)

    lines = []
    lines.append("# IJB Clean vs Raw Matrix")
    lines.append("")
    lines.append(f"- device: {device}")
    lines.append(f"- batch_size: {args.batch_size}")
    lines.append(f"- num_workers: {args.num_workers}")
    lines.append(f"- template_pooling: {args.template_pooling}")
    lines.append("")
    lines.append("| Model | Dataset | Raw AUC | Raw TAR@1e-4 | Raw TAR@1e-5 | Clean AUC | Clean TAR@1e-4 | Clean TAR@1e-5 | dAUC | dTAR@1e-4 | dTAR@1e-5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in matrix_rows:
        d = row["delta_clean_minus_raw"]
        lines.append(
            "| "
            + f"{row['model']} | {row['dataset']} | "
            + f"{_fmt(row['raw']['roc_auc'])} | {_fmt(row['raw']['tar_far_1e-4'])} | {_fmt(row['raw']['tar_far_1e-5'])} | "
            + f"{_fmt(row['clean']['roc_auc'])} | {_fmt(row['clean']['tar_far_1e-4'])} | {_fmt(row['clean']['tar_far_1e-5'])} | "
            + f"{_fmt(_to_num(d.get('roc_auc')))} | {_fmt(_to_num(d.get('tar_far_1e-4')))} | {_fmt(_to_num(d.get('tar_far_1e-5')))} |"
        )

    matrix_md = out_dir / "matrix_clean_vs_raw.md"
    matrix_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {matrix_json}")
    print(f"WROTE {matrix_md}")


if __name__ == "__main__":
    main()
