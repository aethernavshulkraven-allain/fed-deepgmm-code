#!/usr/bin/env python3
"""Materialize high-dimensional stochastic final manifests.

This is intentionally narrower than analyze_highdim_coauthor_tuning.py: it
only selects completed stochastic tuning rows and writes stochastic final
manifests, so the five-seed stochastic finals can proceed before deterministic
tuning is finished.
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
ALPHA_DIRS = ("alpha0p1", "alpha0p5", "alpha1")
REQUIRED = (
    "effective_config.json",
    "metrics.json",
    "mse_by_round.csv",
    "predictions.npz",
    "checkpoints/best_validation.pt",
    "checkpoints/final.pt",
)
STOCHASTIC_METHODS = {"fedgda_s", "fedogda_s"}


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


def last_50_std(history: list[dict[str, str]]) -> float:
    values = []
    for row in history[-50:]:
        try:
            value = float(row["val_mse"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) <= 1:
        return math.inf
    return statistics.pstdev(values)


def validate_and_record(row: dict[str, str]) -> dict[str, Any]:
    run_dir = REPO_ROOT / row["final_result_dir"]
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"{row['run_id']} missing artifacts: {', '.join(missing)}")
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
        raise ValueError(f"{row['run_id']}: {'; '.join(errors)}")
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
        "last_50_val_mse_std": last_50_std(history),
        "final_vs_best_validation_gap": final - best,
        "diverged": False,
        "result_dir": row["final_result_dir"],
    }


def select(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["dataset"], record["method"]), []).append(record)
    expected_groups = 6 * len(STOCHASTIC_METHODS)
    if len(groups) != expected_groups:
        raise ValueError(f"expected {expected_groups} stochastic groups, found {len(groups)}")
    selected = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) != 2:
            raise ValueError(f"expected 2 LR candidates for {key}, found {len(candidates)}")
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
    for row in selected:
        metrics = load_json(REPO_ROOT / row["result_dir"] / "metrics.json")
        value = metrics.get("test_mse_at_best_validation")
        if not finite(value):
            raise ValueError(f"non-finite Test MSE for {row['run_id']}")
        row["test_mse_at_best_validation"] = float(value)


def materialize_final(
    base_rows: list[dict[str, str]], selected: list[dict[str, Any]]
) -> list[dict[str, str]]:
    choices = {(row["dataset"], row["method"]): row for row in selected}
    final_rows = []
    for original in base_rows:
        if original["method"] not in STOCHASTIC_METHODS:
            continue
        row = dict(original)
        choice = choices[(row["dataset"], row["method"])]
        row["learning_rate"] = str(choice["learning_rate"])
        row["weight_decay"] = str(choice["weight_decay"])
        row["critic_multiplier"] = str(choice["critic_multiplier"])
        row["learning_rate_status"] = "selected_by_seed0_validation_tuning"
        row["implementation_status"] = "ready"
        row["notes"] = row["notes"] + f" Validation source: {choice['run_id']}."
        final_rows.append(row)
    expected_rows = 6 * len(STOCHASTIC_METHODS) * 5
    if len(final_rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} stochastic final rows, found {len(final_rows)}")
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


def main() -> int:
    summary = {}
    for name in ALPHA_DIRS:
        alpha_dir = PROTOCOL_DIR / name
        tuning_rows = [
            row
            for row in read_csv(alpha_dir / "tuning_manifest_stochastic.csv")
            if row["method"] in STOCHASTIC_METHODS
        ]
        records = [validate_and_record(row) for row in tuning_rows]
        selected = select(records)
        write_csv(alpha_dir / "candidate_validation_metrics_stochastic.csv", records, VALIDATION_FIELDS)
        attach_post_selection_test(selected)
        selected_fields = VALIDATION_FIELDS + ["test_mse_at_best_validation"]
        write_csv(alpha_dir / "selected_configs_stochastic.csv", selected, selected_fields)
        base_rows = read_csv(alpha_dir / "final_base_manifest.csv")
        final_rows = materialize_final(base_rows, selected)
        fieldnames = list(base_rows[0])
        write_csv(alpha_dir / "final_manifest_stochastic.csv", final_rows, fieldnames)
        with (alpha_dir / "final_manifest_stochastic.json").open("w") as handle:
            json.dump(final_rows, handle, indent=2, sort_keys=True)
            handle.write("\n")
        summary[name] = {
            "candidate_rows": len(records),
            "selected": len(selected),
            "final_rows": len(final_rows),
            "manifest": str((alpha_dir / "final_manifest_stochastic.csv").relative_to(REPO_ROOT)),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
