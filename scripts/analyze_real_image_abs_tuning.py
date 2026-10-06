#!/usr/bin/env python3
"""Select high-dimensional hyperparameters using validation metrics only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
TUNING_DIR = PROTOCOL_DIR / "tuning"
MANIFEST = TUNING_DIR / "manifest.csv"
EXPECTED_RUNS = 96
EXPECTED_SELECTIONS = 24


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def finite_float(value: Any, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value!r}")
    return value


def validation_history(run_dir: Path) -> tuple[float, bool]:
    rows = read_csv(run_dir / "mse_by_round.csv")
    if not rows:
        raise ValueError(f"empty validation history: {run_dir / 'mse_by_round.csv'}")
    values = [float(row["val_mse"]) for row in rows[-50:]]
    all_finite = all(math.isfinite(item) for item in values)
    std = statistics.pstdev(values) if all_finite and len(values) > 1 else (0.0 if all_finite else math.inf)
    diverged = any(
        str(row.get("diverged", "false")).lower() == "true"
        or str(row.get("finite", "true")).lower() != "true"
        for row in rows
    ) or not all_finite
    return std, diverged


def completion_status(manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    completed: list[str] = []
    missing: list[str] = []
    for row in manifest_rows:
        run_dir = REPO_ROOT / row["final_result_dir"]
        required = (run_dir / "metrics.json", run_dir / "mse_by_round.csv")
        (completed if all(path.exists() for path in required) else missing).append(row["run_id"])
    return {
        "expected_runs": EXPECTED_RUNS,
        "manifest_runs": len(manifest_rows),
        "completed_runs": len(completed),
        "missing_runs": len(missing),
        "complete": len(manifest_rows) == EXPECTED_RUNS and not missing,
        "missing_run_ids": missing,
    }


def load_validation_metrics(row: dict[str, str]) -> dict[str, Any]:
    run_dir = REPO_ROOT / row["final_result_dir"]
    metrics = load_json(run_dir / "metrics.json")
    last_50_std, history_diverged = validation_history(run_dir)
    best_raw = float(metrics["best_validation_mse"])
    final_raw = float(metrics["final_validation_mse"])
    diverged = (
        bool(metrics.get("diverged", False))
        or history_diverged
        or not math.isfinite(best_raw)
        or not math.isfinite(final_raw)
    )
    best = best_raw if math.isfinite(best_raw) else math.inf
    final = final_raw if math.isfinite(final_raw) else math.inf
    return {
        "run_id": row["run_id"],
        "dataset": row["dataset"],
        "method": row["method"],
        "seed": int(row["seed"]),
        "learning_rate": finite_float(row["learning_rate"], "learning_rate"),
        "weight_decay": finite_float(row["weight_decay"], "weight_decay"),
        "critic_multiplier": finite_float(row["critic_multiplier"], "critic_multiplier"),
        "best_validation_mse": best,
        "best_validation_round": int(metrics.get("best_validation_round", -1)),
        "final_validation_mse": final,
        "last_50_val_mse_std": last_50_std,
        "final_vs_best_validation_gap": final - best if math.isfinite(best) and math.isfinite(final) else math.inf,
        "diverged": diverged,
        "result_dir": row["final_result_dir"],
    }


def select(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in validation_rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) != 4:
            raise ValueError(f"expected 4 candidates for {key}, found {len(candidates)}")
        eligible = [row for row in candidates if not row["diverged"]]
        if not eligible:
            raise ValueError(f"all tuning candidates diverged for {key}")
        chosen = min(
            eligible,
            key=lambda row: (
                row["best_validation_mse"],
                row["last_50_val_mse_std"],
                row["final_vs_best_validation_gap"],
                row["learning_rate"],
                row["weight_decay"],
            ),
        )
        selected.append(dict(chosen))
    if len(selected) != EXPECTED_SELECTIONS:
        raise ValueError(f"expected {EXPECTED_SELECTIONS} selections, found {len(selected)}")
    return selected


def attach_post_selection_test_metrics(selected: list[dict[str, Any]]) -> None:
    # This is deliberately a second pass: no test metric is read until the
    # validation-only ranking above has fixed every selected configuration.
    for row in selected:
        metrics = load_json(REPO_ROOT / row["result_dir"] / "metrics.json")
        row["test_mse_at_best_validation"] = finite_float(
            metrics["test_mse_at_best_validation"], "test_mse_at_best_validation"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    manifest_rows = read_csv(MANIFEST)
    status = completion_status(manifest_rows)
    status_path = TUNING_DIR / "completion_status.json"
    with status_path.open("w") as f:
        json.dump(status, f, indent=2, sort_keys=True)
        f.write("\n")
    if args.status_only:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    if not status["complete"]:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2

    validation_rows = [load_validation_metrics(row) for row in manifest_rows]
    selected = select(validation_rows)
    attach_post_selection_test_metrics(selected)
    validation_fields = [
        "run_id", "dataset", "method", "seed", "learning_rate", "weight_decay",
        "critic_multiplier", "best_validation_mse", "best_validation_round",
        "final_validation_mse", "last_50_val_mse_std",
        "final_vs_best_validation_gap", "diverged", "result_dir",
    ]
    selected_fields = validation_fields + ["test_mse_at_best_validation"]
    write_csv(TUNING_DIR / "candidate_validation_metrics.csv", validation_rows, validation_fields)
    write_csv(TUNING_DIR / "selected_configs.csv", selected, selected_fields)
    summary = {
        **status,
        "selected_configs": len(selected),
        "selection_scope": "separate per dataset and method",
        "selection_rule": (
            "exclude diverged; lowest best_validation_mse; tie-break "
            "last_50_val_mse_std; tie-break final_vs_best_validation_gap"
        ),
        "test_mse_used_for_selection": False,
    }
    with (TUNING_DIR / "analysis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
