#!/usr/bin/env python3
"""Evaluate phase3 swa.pt on InsightFace-cleaned IJB (flip-TTA on)."""
import sys, json, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fas_kd.models.student import MobileNetV4Student
from fas_kd.data.transforms import build_eval_transform
from fas_kd.evaluation.ijb_template import evaluate_ijb_template_1to1
from fas_kd.utils.config import load_yaml_config

cfg = load_yaml_config(str(PROJECT_ROOT / "configs/train_ms1m_magface_phase3_trueasym_swa_v1.yaml"))
sc = cfg["student"]
model = MobileNetV4Student(
    backbone_name=sc["backbone_name"], embedding_dim=512, pretrained=False,
    input_size=112, projection_activation=str(sc.get("projection_activation", "none")),
    spatial_out_channels=int(sc.get("spatial_out_channels", 0)),
)
ckpt = torch.load(str(PROJECT_ROOT / "runs/ms1m_magface_phase3_trueasym_swa_v1/checkpoints/swa.pt"), map_location="cpu")
model.load_state_dict(ckpt["student_state"], strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
transform = build_eval_transform(cfg["data"])

for ds in ["IJBB", "IJBC"]:
    root = PROJECT_ROOT / "data/processed/ijb_clean_insightface" / ds
    if not root.exists():
        print(f"[skip] {ds}: {root} not found")
        continue
    print(f"Evaluating phase3/swa on {ds} ...", flush=True)
    r = evaluate_ijb_template_1to1(
        model=model, ijb_root=root, transform=transform, device=device,
        use_amp=True, target_fars=[1e-3, 1e-4, 1e-5],
        batch_size=128, num_workers=4, template_pooling="magface_weighted", use_flip=True,
    )
    out = PROJECT_ROOT / f"logs/ijb_iff_phase3swa_{ds.lower()}.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"phase3/swa {ds}: AUC={r['roc_auc']:.4f}  TAR@1e-3={r['tar_far_1e-3']:.4f}  TAR@1e-4={r['tar_far_1e-4']:.4f}  TAR@1e-5={r.get('tar_far_1e-5',0):.4f}")
    print(f"  wrote {out}")
