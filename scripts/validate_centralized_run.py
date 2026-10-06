#!/usr/bin/env python3
"""Validate artifacts from scripts/run_centralized_lowdim.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "effective_config.json",
    "metrics.json",
    "mse_by_round.csv",
    "predictions.npz",
    "checkpoints/best_validation.pt",
    "checkpoints/final.pt",
)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def validate(run_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (run_dir / name).exists():
            errors.append(f"missing {name}")
    if errors:
        return errors

    config = load_json(run_dir / "effective_config.json")
    metrics = load_json(run_dir / "metrics.json")
    expected_config = {
        "training_scope": "centralized",
        "uses_clients": False,
        "uses_fedavg_aggregation": False,
        "uses_client_sampling": False,
        "uses_server_learning_rate_aggregation": False,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"effective_config.{key} expected {expected!r}, got {config.get(key)!r}")

    if metrics.get("selection_metric_source") != "validation":
        errors.append("metrics.selection_metric_source is not validation")
    if metrics.get("test_mse_used_for_selection") is not False:
        errors.append("metrics.test_mse_used_for_selection is not false")

    metric_names = [
        "train_mse_final",
        "val_mse_final",
        "test_mse_final",
        "best_validation_round",
        "best_validation_mse",
        "test_mse_at_best_validation",
    ]
    for key in metric_names:
        if key not in metrics:
            errors.append(f"metrics missing {key}")
        elif not finite(metrics[key]):
            errors.append(f"metrics.{key} is not finite: {metrics[key]!r}")

    rows = []
    with (run_dir / "mse_by_round.csv").open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["round", "train_mse", "val_mse"]:
            errors.append(f"mse_by_round.csv columns expected round/train_mse/val_mse, got {reader.fieldnames}")
        for row in reader:
            rows.append(row)
    if not rows:
        errors.append("mse_by_round.csv has no rows")
    for idx, row in enumerate(rows):
        for key in ("round", "train_mse", "val_mse"):
            if key not in row or not finite(row[key]):
                errors.append(f"mse_by_round row {idx} has non-finite {key}: {row.get(key)!r}")

    if rows and "best_validation_round" in metrics and "best_validation_mse" in metrics:
        val_values = [(int(row["round"]), float(row["val_mse"])) for row in rows]
        min_round, min_value = min(val_values, key=lambda item: item[1])
        if int(metrics["best_validation_round"]) != min_round:
            errors.append(
                f"best_validation_round {metrics['best_validation_round']} != min val round {min_round}"
            )
        if abs(float(metrics["best_validation_mse"]) - min_value) > 1e-10:
            errors.append(
                f"best_validation_mse {metrics['best_validation_mse']} != min val mse {min_value}"
            )

    if config.get("log_test_mse_by_round") or config.get("test_mse_logged_by_round"):
        path = run_dir / "test_mse_by_round.csv"
        if not path.exists():
            errors.append("test_mse_by_round.csv missing while logging enabled")
        else:
            with path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if not finite(row.get("test_mse")):
                        errors.append(f"test_mse_by_round row {idx} has non-finite test_mse")

    try:
        predictions = np.load(run_dir / "predictions.npz")
        for key in ("x", "true_g", "best_validation_prediction", "final_prediction"):
            if key not in predictions:
                errors.append(f"predictions.npz missing {key}")
            elif not np.all(np.isfinite(predictions[key])):
                errors.append(f"predictions.npz {key} contains non-finite values")
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        errors.append(f"could not read predictions.npz: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--json", action="store_true", help="Print JSON validation result.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    errors = validate(run_dir)
    result = {
        "run_dir": str(run_dir),
        "valid": not errors,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        print(f"INVALID: {run_dir}")
        for error in errors:
            print(f"- {error}")
    else:
        print(f"VALID: {run_dir}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
