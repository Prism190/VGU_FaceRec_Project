#!/usr/bin/env python3
"""Concrete diagnostics for the MagFace teacher: weight loading + eval transform.

Answers: did all weights load? does eval use flip TTA? what input size/range?
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.models.magface_iresnet import iresnet100
from fas_kd.utils.config import load_yaml_config

ckpt_path = PROJECT_ROOT / "checkpoints" / "pretrained" / "magface_iresnet100_ms1mv2.pth"
print(f"Checkpoint: {ckpt_path}  exists={ckpt_path.exists()}  size={ckpt_path.stat().st_size/1e6:.1f}MB")

model = iresnet100(num_classes=512)
model_keys = list(model.state_dict().keys())
print(f"Model expects {len(model_keys)} keys")

state = torch.load(ckpt_path, map_location="cpu")
raw_state = state.get("state_dict", state) if isinstance(state, dict) else state
print(f"Checkpoint has {len(raw_state)} keys")
print(f"Sample checkpoint keys: {list(raw_state.keys())[:5]}")

# Replicate teacher.py loading logic
cleaned = OrderedDict()
for key, value in raw_state.items():
    candidates = [
        key,
        key.replace("features.module.", ""),
        key.replace("module.features.", ""),
        key.replace("module.", ""),
    ]
    for candidate in candidates:
        if candidate in model.state_dict() and model.state_dict()[candidate].shape == value.shape:
            cleaned[candidate] = value
            break

missing, unexpected = model.load_state_dict(cleaned, strict=False)
print()
print("=" * 70)
print(f"LOADED:     {len(cleaned)}/{len(model_keys)} keys ({100*len(cleaned)/len(model_keys):.1f}%)")
print(f"MISSING:    {len(missing)} keys")
print(f"UNEXPECTED: {len(unexpected)} keys")
print("=" * 70)

# Critical layers
critical = ["conv1.weight", "bn1.weight", "bn2.weight", "fc.weight", "features.weight", "features.running_mean"]
print("\nCritical-layer load status:")
for c in critical:
    print(f"  {c:28s} {'LOADED' if c in cleaned else 'MISSING <-- PROBLEM'}")

if missing:
    print(f"\nFirst 15 MISSING keys: {missing[:15]}")
if unexpected:
    print(f"\nFirst 15 UNEXPECTED keys: {unexpected[:15]}")

# Eval transform inspection
print("\n" + "=" * 70)
print("EVAL TRANSFORM")
print("=" * 70)
cfg = load_yaml_config(str(PROJECT_ROOT / "configs" / "train_ms1m_magface_phase1_cplus_aplus_v1.yaml"))
data_cfg = cfg["data"]
print(f"  image_size: {data_cfg.get('image_size')}")
print(f"  mean: {data_cfg.get('mean')}  std: {data_cfg.get('std')}")
print(f"  -> output range with mean=std=0.5: [-1, 1]")

# Check if IJB eval does flip TTA
import inspect
from fas_kd.evaluation import ijb_template
src = inspect.getsource(ijb_template)
has_flip = any(t in src for t in ["flip", "fliplr", "hflip", "[..., ::-1]", "torch.flip"])
print(f"\n  IJB eval flip-TTA present: {has_flip}")
if not has_flip:
    print("  -> NO flip TTA. Official MagFace eval averages emb(img)+emb(flip(img)).")
    print("     Missing this typically costs 2-5 points of TAR@1e-4.")
