from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from fas_kd.data.datasets import PairVerificationDataset
from fas_kd.data.transforms import build_eval_transform
from fas_kd.evaluation.verification import evaluate_pair_verification


def _far_key(far: float) -> str:
    exponent = int(round(math.log10(float(far))))
    return f"tar_far_1e{exponent}"


def validate_on_sets(
    model,
    data_cfg: dict,
    val_sets: list[dict],
    batch_size: int,
    num_workers: int,
    device,
    use_amp: bool,
    target_fars: list[float],
    loader_timeout_s: float = 120.0,
) -> dict[str, Any]:
    eval_transform = build_eval_transform(data_cfg)
    out: dict[str, Any] = {"sets": {}, "aggregate": {}}

    for item in val_sets:
        name = item["name"]
        pairs_csv = Path(item["pairs_csv"])
        if not pairs_csv.exists():
            raise FileNotFoundError(f"Validation pairs file not found for {name}: {pairs_csv}")

        dataset = PairVerificationDataset(pairs_csv=pairs_csv, transform=eval_transform)
        worker_attempts = [int(num_workers)]
        if int(num_workers) > 0:
            worker_attempts.append(0)

        metrics = None
        errors: list[str] = []
        for attempt_idx, attempt_workers in enumerate(worker_attempts, start=1):
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=attempt_workers,
                pin_memory=True,
                drop_last=False,
                timeout=float(loader_timeout_s) if attempt_workers > 0 else 0.0,
            )

            try:
                metrics = evaluate_pair_verification(
                    model=model,
                    dataloader=loader,
                    device=device,
                    use_amp=use_amp,
                    target_fars=target_fars,
                )
                if attempt_workers == 0 and int(num_workers) > 0:
                    print(
                        f"[WARN] Validation for {name} succeeded only after fallback "
                        f"to num_workers=0"
                    )
                break
            except Exception as exc:
                errors.append(f"attempt={attempt_idx}, num_workers={attempt_workers}: {exc}")
                print(
                    f"[WARN] Validation failed on {name} with num_workers={attempt_workers}: {exc}"
                )

        if metrics is None:
            metrics = {
                "accuracy": 0.0,
                "roc_auc": 0.0,
                "best_threshold": 0.0,
                "num_pairs": int(len(dataset)),
                "num_scores_non_finite": 0,
                "error": " | ".join(errors) if errors else "unknown validation failure",
            }
            for far in target_fars:
                metrics[_far_key(far)] = 0.0

        out["sets"][name] = metrics

    if not out["sets"]:
        return out

    set_names = list(out["sets"].keys())
    out["aggregate"]["mean_accuracy"] = float(
        sum(out["sets"][name]["accuracy"] for name in set_names) / len(set_names)
    )
    out["aggregate"]["mean_roc_auc"] = float(
        sum(out["sets"][name]["roc_auc"] for name in set_names) / len(set_names)
    )

    for far in target_fars:
        key = _far_key(far)
        agg_key = f"mean_{key}"
        out["aggregate"][agg_key] = float(
            sum(out["sets"][name][key] for name in set_names) / len(set_names)
        )

    return out
