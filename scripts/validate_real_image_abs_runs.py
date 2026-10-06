#!/usr/bin/env python3
"""Validate tuning or final high-dimensional federated run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
REQUIRED = (
    "effective_config.json",
    "metrics.json",
    "mse_by_round.csv",
    "predictions.npz",
    "checkpoints/best_validation.pt",
    "checkpoints/final.pt",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_run(row: dict[str, str]) -> list[str]:
    run_dir = REPO_ROOT / row["final_result_dir"]
    errors = [f"missing {name}" for name in REQUIRED if not (run_dir / name).exists()]
    if errors:
        return errors
    config = load_json(run_dir / "effective_config.json")
    metrics = load_json(run_dir / "metrics.json")
    expected_config: dict[str, Any] = {
        "run_id": row["run_id"],
        "dataset": row["dataset"],
        "variant": row["method"],
        "random_seed": int(row["seed"]),
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"effective_config.{key}: expected {expected!r}, got {config.get(key)!r}")
    for key in (
        "best_validation_mse", "best_validation_round", "final_validation_mse",
        "final_test_mse", "test_mse_at_best_validation", "runtime_seconds",
    ):
        if key not in metrics or not finite(metrics[key]):
            errors.append(f"metrics.{key} missing or non-finite")
    if metrics.get("selection_metric_source") != "validation":
        errors.append("metrics.selection_metric_source is not validation")
    if metrics.get("test_mse_used_for_selection") is not False:
        errors.append("metrics.test_mse_used_for_selection is not false")

    history = read_csv(run_dir / "mse_by_round.csv")
    if not history:
        errors.append("mse_by_round.csv is empty")
    else:
        for index, item in enumerate(history):
            for key in ("round", "train_mse", "val_mse"):
                if not finite(item.get(key)):
                    errors.append(f"mse_by_round row {index} has invalid {key}")
        finite_rows = [item for item in history if finite(item.get("val_mse"))]
        if finite_rows and finite(metrics.get("best_validation_mse")):
            minimum = min(float(item["val_mse"]) for item in finite_rows)
            if abs(float(metrics["best_validation_mse"]) - minimum) > 1e-10:
                errors.append("best_validation_mse does not match history minimum")
    try:
        with np.load(run_dir / "predictions.npz") as predictions:
            required_keys = ("true_g", "best_validation_prediction", "final_prediction")
            # save_predictions_npz_compact()'s schema (opted into via a row's
            # compact_predictions_only=True) deliberately omits the full test
            # input tensor "x" to avoid the ~10 GiB-scale write across an
            # image campaign -- its absence is expected there, not a defect,
            # so it is only required when this run used the full schema.
            compact_schema = bool(config.get("compact_predictions_only", False))
            keys = required_keys if compact_schema else ("x", *required_keys)
            for key in keys:
                if key not in predictions:
                    errors.append(f"predictions.npz missing {key}")
            if all(key in predictions for key in keys):
                lengths = [predictions[key].shape[0] for key in keys]
                if len(set(lengths)) != 1 or lengths[0] == 0:
                    errors.append(f"prediction sample dimensions disagree: {lengths}")
                for key in required_keys:
                    if not np.all(np.isfinite(predictions[key])):
                        errors.append(f"predictions.npz {key} contains non-finite values")
    except Exception as exc:
        errors.append(f"cannot read predictions.npz: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(PROTOCOL_DIR / "final_manifest.csv"))
    parser.add_argument("--output", default=str(PROTOCOL_DIR / "final_validation.json"))
    args = parser.parse_args()
    manifest = Path(args.manifest)
    rows = read_csv(manifest)
    results = []
    for row in rows:
        errors = validate_run(row)
        results.append({"run_id": row["run_id"], "valid": not errors, "errors": errors})
    summary = {
        "manifest": str(manifest),
        "runs": len(rows),
        "valid": sum(item["valid"] for item in results),
        "invalid": sum(not item["valid"] for item in results),
        "complete": bool(rows) and all(item["valid"] for item in results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({key: summary[key] for key in ("runs", "valid", "invalid", "complete")}, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
