#!/usr/bin/env python3
"""Audit completed high-dimensional stochastic final artifacts.

The stochastic final set was completed across the original result roots and two
fresh continuation roots. This script rebuilds a single result index from the
preservation manifests, validates required artifacts, and writes a compact
five-seed aggregate summary.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1"
V1_MIGRATION_DIR = PROTOCOL_DIR / "stochastic_speedup_migration_20260719_123539"
V2_MIGRATION_DIR = PROTOCOL_DIR / "stochastic_speedup_migration_20260719_233531"
INDEX_CSV = PROTOCOL_DIR / "stochastic_final_artifact_index.csv"
SUMMARY_CSV = PROTOCOL_DIR / "stochastic_final_aggregate_summary.csv"

REQUIRED_ARTIFACTS = (
    "effective_config.json",
    "metrics.json",
    "mse_by_round.csv",
    "predictions.npz",
    "checkpoints/best_validation.pt",
    "checkpoints/final.pt",
)
ALPHA_BY_DIR = {"alpha0p1": 0.1, "alpha0p5": 0.5, "alpha1": 1.0}
METHOD_LABEL = {"fedgda_s": "FedGDA-S", "fedogda_s": "FedOGDA-S"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def finite_number(value: Any, field: str, run_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{run_id}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{run_id}: {field} is not finite: {value!r}")
    return result


def round_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in handle) - 1


def relative_result_dir(path: str | Path) -> str:
    result = Path(path)
    if result.is_absolute():
        return str(result.relative_to(REPO_ROOT))
    return str(result)


def collect_expected_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for row in read_csv(V1_MIGRATION_DIR / "preserved_old_completed_runs.csv"):
        entries.append(
            {
                "alpha": ALPHA_BY_DIR[row["alpha_dir"]],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "run_id": row["run_id"],
                "result_dir": relative_result_dir(row["old_result_dir"]),
                "source": "old_original",
            }
        )

    for row in read_csv(V2_MIGRATION_DIR / "preserved_v1_completed_runs.csv"):
        entries.append(
            {
                "alpha": float(row["alpha"]),
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "run_id": row["run_id"],
                "result_dir": relative_result_dir(row["old_result_dir"]),
                "source": "safe_speedup_v1",
            }
        )

    for row in read_csv(V2_MIGRATION_DIR / "remaining_pending_manifest_stochastic_safe_speedup_v2.csv"):
        entries.append(
            {
                "alpha": float(row["alpha"]),
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "run_id": row["run_id"],
                "result_dir": relative_result_dir(row["final_result_dir"]),
                "source": "safe_speedup_v2",
            }
        )

    return entries


def validate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    run_id = entry["run_id"]
    run_dir = REPO_ROOT / entry["result_dir"]
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"{run_id}: missing artifacts: {', '.join(missing)}")

    history_rows = round_rows(run_dir / "mse_by_round.csv")
    if history_rows != 1500:
        raise ValueError(f"{run_id}: expected 1500 round rows, found {history_rows}")

    config = load_json(run_dir / "effective_config.json")
    metrics = load_json(run_dir / "metrics.json")

    if config.get("dataset") != entry["dataset"]:
        raise ValueError(f"{run_id}: dataset mismatch in effective_config.json")
    if config.get("variant") != entry["method"]:
        raise ValueError(f"{run_id}: method mismatch in effective_config.json")
    if int(config.get("random_seed")) != int(entry["seed"]):
        raise ValueError(f"{run_id}: seed mismatch in effective_config.json")
    if metrics.get("diverged") is True:
        raise ValueError(f"{run_id}: metrics.json has diverged=true")
    if metrics.get("test_mse_used_for_selection") not in (False, None):
        raise ValueError(f"{run_id}: Test MSE was marked as used for selection")

    return {
        **entry,
        "method_label": METHOD_LABEL[entry["method"]],
        "round_rows": history_rows,
        "comm_round": config.get("comm_round"),
        "client_num_in_total": config.get("client_num_in_total"),
        "client_num_per_round": config.get("client_num_per_round"),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "critic_multiplier": config.get("critic_multiplier"),
        "server_learning_rate": config.get("server_learning_rate"),
        "gradient_clip_norm": config.get("gradient_clip_norm"),
        "best_validation_mse": finite_number(metrics.get("best_validation_mse"), "best_validation_mse", run_id),
        "best_validation_round": int(metrics["best_validation_round"]),
        "test_mse_at_best_validation": finite_number(
            metrics.get("test_mse_at_best_validation"), "test_mse_at_best_validation", run_id
        ),
        "final_test_mse": finite_number(metrics.get("final_test_mse"), "final_test_mse", run_id),
        "runtime_seconds": finite_number(metrics.get("runtime_seconds"), "runtime_seconds", run_id),
        "append_round_csv": metrics.get("append_round_csv"),
        "skip_model_selection": metrics.get("skip_model_selection"),
        "skip_gmm_eval": metrics.get("skip_gmm_eval"),
        "auxiliary_regression": metrics.get("auxiliary_regression"),
        "auxiliary_regression_epochs": metrics.get("auxiliary_regression_epochs"),
        "periodic_checkpoint_interval": metrics.get("periodic_checkpoint_interval"),
        "test_mse_used_for_selection": metrics.get("test_mse_used_for_selection"),
    }


def aggregate(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["alpha"], row["dataset"], row["method"]) for row in index_rows})
    for alpha, dataset, method in keys:
        group = [
            row
            for row in index_rows
            if (row["alpha"], row["dataset"], row["method"]) == (alpha, dataset, method)
        ]
        values = [float(row["test_mse_at_best_validation"]) for row in group]
        runtimes = [float(row["runtime_seconds"]) / 60.0 for row in group]
        summaries.append(
            {
                "alpha": alpha,
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABEL[method],
                "n": len(group),
                "test_mse_at_best_validation_mean": statistics.mean(values),
                "test_mse_at_best_validation_std": statistics.stdev(values),
                "test_mse_at_best_validation_min": min(values),
                "test_mse_at_best_validation_max": max(values),
                "learning_rates": ";".join(sorted({str(row["learning_rate"]) for row in group})),
                "runtime_minutes_mean": statistics.mean(runtimes),
                "runtime_minutes_median": statistics.median(runtimes),
            }
        )
    return summaries


def main() -> int:
    entries = collect_expected_entries()
    seen = set()
    index_rows = []
    for entry in entries:
        key = (entry["alpha"], entry["dataset"], entry["method"], entry["seed"])
        if key in seen:
            raise ValueError(f"duplicate stochastic final key: {key}")
        seen.add(key)
        index_rows.append(validate_entry(entry))

    index_rows.sort(key=lambda row: (row["alpha"], row["dataset"], row["method"], row["seed"]))
    summary_rows = aggregate(index_rows)

    if len(index_rows) != 180:
        raise ValueError(f"expected 180 valid final rows, found {len(index_rows)}")
    if len(summary_rows) != 36:
        raise ValueError(f"expected 36 aggregate rows, found {len(summary_rows)}")

    write_csv(INDEX_CSV, index_rows, list(index_rows[0]))
    write_csv(SUMMARY_CSV, summary_rows, list(summary_rows[0]))
    print(
        json.dumps(
            {
                "valid_final_runs": len(index_rows),
                "aggregate_rows": len(summary_rows),
                "index_csv": str(INDEX_CSV.relative_to(REPO_ROOT)),
                "summary_csv": str(SUMMARY_CSV.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
