#!/usr/bin/env python3
"""Validate and select high-dimensional tuning configs without Test leakage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1"
ALPHA_DIRS = ("alpha0p1", "alpha0p5", "alpha1")
REQUIRED = (
    "effective_config.json",
    "metrics.json",
    "mse_by_round.csv",
    "predictions.npz",
    "checkpoints/best_validation.pt",
    "checkpoints/final.pt",
)


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def status(rows: list[dict[str, str]]) -> dict[str, Any]:
    complete: list[str] = []
    missing: list[str] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        run_dir = REPO_ROOT / row["final_result_dir"]
        absent = [name for name in REQUIRED if not (run_dir / name).exists()]
        if absent:
            missing.append(row["run_id"])
            continue
        metrics = load_json(run_dir / "metrics.json")
        history = read_csv(run_dir / "mse_by_round.csv")
        errors = []
        if len(history) != int(row["comm_round"]):
            errors.append(f"history rows {len(history)} != {row['comm_round']}")
        if metrics.get("diverged") is True:
            errors.append("metrics.diverged is true")
        for key in ("best_validation_mse", "final_validation_mse"):
            if not finite(metrics.get(key)):
                errors.append(f"metrics.{key} is non-finite")
        if errors:
            invalid.append({"run_id": row["run_id"], "errors": errors})
        else:
            complete.append(row["run_id"])
    return {
        "expected": len(rows),
        "completed": len(complete),
        "missing": len(missing),
        "invalid": len(invalid),
        "complete": bool(rows) and not missing and not invalid,
        "missing_run_ids": missing,
        "invalid_runs": invalid,
    }


def validation_record(row: dict[str, str]) -> dict[str, Any]:
    run_dir = REPO_ROOT / row["final_result_dir"]
    metrics = load_json(run_dir / "metrics.json")
    history = read_csv(run_dir / "mse_by_round.csv")
    recent = [float(item["val_mse"]) for item in history[-50:]]
    recent_std = statistics.pstdev(recent) if len(recent) > 1 else 0.0
    best = float(metrics["best_validation_mse"])
    final = float(metrics["final_validation_mse"])
    return {
        "run_id": row["run_id"],
        "dataset": row["dataset"],
        "method": row["method"],
        "seed": int(row["seed"]),
        "alpha": float(row["alpha"]),
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
        "critic_multiplier": float(row["critic_multiplier"]),
        "best_validation_mse": best,
        "best_validation_round": int(metrics["best_validation_round"]),
        "final_validation_mse": final,
        "last_50_val_mse_std": recent_std,
        "final_vs_best_validation_gap": final - best,
        "diverged": False,
        "result_dir": row["final_result_dir"],
    }


def select(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["dataset"], record["method"]), []).append(record)
    if len(groups) != 24:
        raise ValueError(f"expected 24 dataset/method groups, found {len(groups)}")
    selected = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) != 2:
            raise ValueError(f"expected 2 effective LR candidates for {key}, found {len(candidates)}")
        selected.append(
            dict(
                min(
                    candidates,
                    key=lambda row: (
                        row["best_validation_mse"],
                        row["last_50_val_mse_std"],
                        row["final_vs_best_validation_gap"],
                        row["learning_rate"],
                    ),
                )
            )
        )
    return selected


def attach_post_selection_test(selected: list[dict[str, Any]]) -> None:
    # Deliberately separate from ``select``: every validation choice is fixed
    # before this function reads any Test MSE.
    for row in selected:
        metrics = load_json(REPO_ROOT / row["result_dir"] / "metrics.json")
        value = metrics.get("test_mse_at_best_validation")
        if not finite(value):
            raise ValueError(f"non-finite post-selection Test MSE for {row['run_id']}")
        row["test_mse_at_best_validation"] = float(value)


def materialize_final(
    base_rows: list[dict[str, str]], selected: list[dict[str, Any]]
) -> list[dict[str, str]]:
    choices = {(row["dataset"], row["method"]): row for row in selected}
    if len(base_rows) != 120 or len(choices) != 24:
        raise ValueError("final materialization requires 120 base rows and 24 selections")
    final_rows = []
    for original in base_rows:
        row = dict(original)
        choice = choices[(row["dataset"], row["method"])]
        row["learning_rate"] = str(choice["learning_rate"])
        row["weight_decay"] = str(choice["weight_decay"])
        row["critic_multiplier"] = str(choice["critic_multiplier"])
        row["learning_rate_status"] = "selected_by_seed0_validation_tuning"
        row["implementation_status"] = "ready"
        row["notes"] = row["notes"] + f" Validation source: {choice['run_id']}."
        final_rows.append(row)
    return final_rows


VALIDATION_FIELDS = [
    "run_id",
    "dataset",
    "method",
    "seed",
    "alpha",
    "learning_rate",
    "weight_decay",
    "critic_multiplier",
    "best_validation_mse",
    "best_validation_round",
    "final_validation_mse",
    "last_50_val_mse_std",
    "final_vs_best_validation_gap",
    "diverged",
    "result_dir",
]


def analyze_alpha(alpha_dir: Path, status_only: bool) -> dict[str, Any]:
    tuning_rows = read_csv(alpha_dir / "tuning_manifest.csv")
    completion = status(tuning_rows)
    with (alpha_dir / "completion_status.json").open("w") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if status_only or not completion["complete"]:
        return completion

    records = [validation_record(row) for row in tuning_rows]
    selected = select(records)
    write_csv(alpha_dir / "candidate_validation_metrics.csv", records, VALIDATION_FIELDS)
    # Test is attached only after all validation choices have been written in
    # memory and fixed.
    attach_post_selection_test(selected)
    selected_fields = VALIDATION_FIELDS + ["test_mse_at_best_validation"]
    write_csv(alpha_dir / "selected_configs.csv", selected, selected_fields)

    base_rows = read_csv(alpha_dir / "final_base_manifest.csv")
    final_rows = materialize_final(base_rows, selected)
    write_csv(alpha_dir / "final_manifest.csv", final_rows, list(base_rows[0]))
    with (alpha_dir / "final_manifest.json").open("w") as handle:
        json.dump(final_rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**completion, "selected": len(selected), "final_rows": len(final_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--alpha-dir", choices=ALPHA_DIRS, default=None)
    args = parser.parse_args()
    names = (args.alpha_dir,) if args.alpha_dir else ALPHA_DIRS
    summaries = {
        name: analyze_alpha(PROTOCOL_DIR / name, args.status_only)
        for name in names
    }
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0 if all(item["complete"] for item in summaries.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
