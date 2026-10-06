#!/usr/bin/env python3
"""Analyze the FedOGDA-S critic/weight-decay tuning pilot.

This script keeps the tuning decision validation-only. Test MSE is reported
after the selected configuration is fixed by validation metrics.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "tuning_fedogda_s" / "pilot_alpha0p5"
MANIFEST = PILOT_DIR / "manifest.csv"
BASELINE_RUNS = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "tuning_fedogda_s" / "baseline_stochastic_runs.csv"
OUTPUTS = {
    "stability_metrics_csv": PILOT_DIR / "stability_metrics.csv",
    "grid_aggregates_csv": PILOT_DIR / "grid_aggregates.csv",
    "selected_configs_csv": PILOT_DIR / "selected_configs.csv",
    "selected_seed_metrics_csv": PILOT_DIR / "selected_seed_metrics.csv",
    "baseline_comparison_csv": PILOT_DIR / "baseline_comparison.csv",
    "final_table_fedgda_s_csv": PILOT_DIR / "final_table_fedgda_s.csv",
    "final_table_current_fedogda_s_csv": PILOT_DIR / "final_table_current_fedogda_s.csv",
    "final_table_tuned_fedogda_s_csv": PILOT_DIR / "final_table_tuned_fedogda_s.csv",
    "summary_json": PILOT_DIR / "analysis_summary.json",
    "summary_md": PILOT_DIR / "analysis_summary.md",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def to_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.12g}"


def last_50_stability(mse_csv: Path) -> tuple[float, float, bool]:
    rows = read_csv(mse_csv)
    tail = rows[-50:]
    if not tail:
        return math.nan, math.nan, True
    val_mse = [to_float(row["val_mse"]) for row in tail]
    train_mse = [to_float(row["train_mse"]) for row in tail]
    diverged = any(to_bool(row.get("diverged", "false")) or not to_bool(row.get("finite", "true")) for row in rows)
    return pstdev(val_mse), pstdev(train_mse), diverged


def load_tuned_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in read_csv(MANIFEST):
        run_dir = REPO_ROOT / manifest_row["final_result_dir"]
        metrics_path = run_dir / "metrics.json"
        mse_path = run_dir / "mse_by_round.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        if not mse_path.exists():
            raise FileNotFoundError(mse_path)
        metrics = read_json(metrics_path)
        last_50_val_std, last_50_train_std, history_diverged = last_50_stability(mse_path)
        best_validation_mse = to_float(metrics["best_validation_mse"])
        final_validation_mse = to_float(metrics["final_validation_mse"])
        diverged = bool(metrics.get("diverged", False)) or history_diverged
        rows.append(
            {
                "run_id": manifest_row["run_id"],
                "method": manifest_row["method"],
                "dataset": manifest_row["dataset"],
                "alpha": to_float(manifest_row["alpha"]),
                "seed": int(manifest_row["seed"]),
                "critic_multiplier": to_float(manifest_row["critic_multiplier"]),
                "weight_decay": to_float(manifest_row["weight_decay"]),
                "best_validation_mse": best_validation_mse,
                "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
                "best_validation_round": int(metrics["best_validation_round"]),
                "final_validation_mse": final_validation_mse,
                "final_test_mse": to_float(metrics["final_test_mse"]),
                "final_vs_best_validation_gap": final_validation_mse - best_validation_mse,
                "last_50_val_mse_std": last_50_val_std,
                "last_50_train_mse_std": last_50_train_std,
                "diverged": diverged,
                "runtime_seconds": to_float(metrics.get("runtime_seconds", 0.0)),
                "result_dir": str(run_dir.relative_to(REPO_ROOT)),
            }
        )
    return rows


def aggregate_runs(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(group_keys, key))
        seeds = sorted({int(row["seed"]) for row in group})
        for hyperparameter in ("critic_multiplier", "weight_decay"):
            if hyperparameter not in item and all(hyperparameter in row for row in group):
                unique_values = sorted({float(row[hyperparameter]) for row in group})
                if len(unique_values) == 1:
                    item[hyperparameter] = unique_values[0]
        item.update(
            {
                "num_seeds": len(seeds),
                "seed_values": "|".join(str(seed) for seed in seeds),
                "diverged_count": sum(bool(row["diverged"]) for row in group),
                "mean_best_validation_mse": mean([row["best_validation_mse"] for row in group]),
                "mean_test_mse_at_best_validation": mean([row["test_mse_at_best_validation"] for row in group]),
                "std_test_mse_at_best_validation": pstdev([row["test_mse_at_best_validation"] for row in group]),
                "mean_final_validation_mse": mean([row["final_validation_mse"] for row in group]),
                "mean_final_test_mse": mean([row["final_test_mse"] for row in group]),
                "mean_final_vs_best_validation_gap": mean([row["final_vs_best_validation_gap"] for row in group]),
                "mean_last_50_val_mse_std": mean([row["last_50_val_mse_std"] for row in group]),
                "mean_last_50_train_mse_std": mean([row["last_50_train_mse_std"] for row in group]),
                "mean_runtime_seconds": mean([row["runtime_seconds"] for row in group]),
            }
        )
        out.append(item)
    return out


def select_configs(grid_rows: list[dict[str, Any]], required_seeds: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_dataset_alpha: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        row["all_seeds_present"] = int(row["num_seeds"]) == required_seeds
        row["eligible_for_selection"] = bool(row["all_seeds_present"]) and int(row["diverged_count"]) == 0
        by_dataset_alpha[(row["dataset"], row["alpha"])].append(row)

    for (_dataset, _alpha), group in sorted(by_dataset_alpha.items()):
        eligible = [row for row in group if row["eligible_for_selection"]]
        ranked = sorted(
            eligible,
            key=lambda row: (
                row["mean_best_validation_mse"],
                row["mean_last_50_val_mse_std"],
                row["mean_final_vs_best_validation_gap"],
                row["critic_multiplier"],
                row["weight_decay"],
            ),
        )
        for index, row in enumerate(ranked, start=1):
            row["selection_rank"] = index
        if ranked:
            chosen = dict(ranked[0])
            chosen["selection_rule"] = (
                "lowest mean_best_validation_mse; tie-break mean_last_50_val_mse_std; "
                "tie-break mean_final_vs_best_validation_gap; diverged_count must be 0"
            )
            selected.append(chosen)
    return selected


def load_baseline_scope(datasets: set[str], alpha: float, seeds: set[int]) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(BASELINE_RUNS):
        if row["dataset"] not in datasets:
            continue
        if abs(to_float(row["alpha"]) - alpha) > 1e-12:
            continue
        if int(row["seed"]) not in seeds:
            continue
        rows.append(
            {
                "run_id": row["run_id"],
                "method": row["method"],
                "dataset": row["dataset"],
                "alpha": to_float(row["alpha"]),
                "seed": int(row["seed"]),
                "critic_multiplier": to_float(row["critic_multiplier"]),
                "weight_decay": to_float(row["weight_decay"]),
                "best_validation_mse": to_float(row["best_validation_mse"]),
                "test_mse_at_best_validation": to_float(row["test_mse_at_best_validation"]),
                "best_validation_round": int(float(row["best_validation_round"])),
                "final_validation_mse": to_float(row["final_validation_mse"]),
                "final_test_mse": to_float(row["final_test_mse"]),
                "final_vs_best_validation_gap": to_float(row["final_vs_best_validation_gap"]),
                "last_50_val_mse_std": to_float(row["last_50_val_mse_std"]),
                "last_50_train_mse_std": to_float(row["last_50_train_mse_std"]),
                "diverged": to_bool(row["diverged"]),
                "runtime_seconds": to_float(row["runtime_seconds"]),
                "result_dir": row["result_dir"],
            }
        )
    return rows


def aggregate_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_runs(rows, ("dataset", "alpha", "method"))


def index_by(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def build_comparison(
    selected: list[dict[str, Any]],
    tuned_seed_rows: list[dict[str, Any]],
    baseline_aggregates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_idx = index_by(baseline_aggregates, "dataset", "alpha", "method")
    tuned_seed_metrics: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fedgda_table: list[dict[str, Any]] = []
    current_fedogda_table: list[dict[str, Any]] = []
    tuned_table: list[dict[str, Any]] = []

    for chosen in selected:
        dataset = chosen["dataset"]
        alpha = chosen["alpha"]
        selected_seeds = [
            row
            for row in tuned_seed_rows
            if row["dataset"] == dataset
            and abs(row["alpha"] - alpha) <= 1e-12
            and abs(row["critic_multiplier"] - chosen["critic_multiplier"]) <= 1e-12
            and abs(row["weight_decay"] - chosen["weight_decay"]) <= 1e-12
        ]
        for row in sorted(selected_seeds, key=lambda item: item["seed"]):
            tuned_seed_metrics.append(row)

        fedgda = baseline_idx[(dataset, alpha, "fedgda_s")]
        current = baseline_idx[(dataset, alpha, "fedogda_s")]
        tuned = chosen

        tuned_minus_current_test = tuned["mean_test_mse_at_best_validation"] - current["mean_test_mse_at_best_validation"]
        tuned_minus_fedgda_test = tuned["mean_test_mse_at_best_validation"] - fedgda["mean_test_mse_at_best_validation"]
        tuned_osc_minus_fedgda = tuned["mean_last_50_val_mse_std"] - fedgda["mean_last_50_val_mse_std"]
        tuned_osc_minus_current = tuned["mean_last_50_val_mse_std"] - current["mean_last_50_val_mse_std"]
        tuned_test_ratio_vs_fedgda = tuned["mean_test_mse_at_best_validation"] / fedgda["mean_test_mse_at_best_validation"]
        tuned_test_ratio_vs_current = tuned["mean_test_mse_at_best_validation"] / current["mean_test_mse_at_best_validation"]

        comparison = {
            "dataset": dataset,
            "alpha": alpha,
            "selected_critic_multiplier": tuned["critic_multiplier"],
            "selected_weight_decay": tuned["weight_decay"],
            "tuned_mean_best_validation_mse": tuned["mean_best_validation_mse"],
            "tuned_mean_test_mse_at_best_validation": tuned["mean_test_mse_at_best_validation"],
            "tuned_std_test_mse_at_best_validation": tuned["std_test_mse_at_best_validation"],
            "tuned_oscillation_score": tuned["mean_last_50_val_mse_std"],
            "tuned_diverged_count": tuned["diverged_count"],
            "current_fedogda_mean_best_validation_mse": current["mean_best_validation_mse"],
            "current_fedogda_mean_test_mse_at_best_validation": current["mean_test_mse_at_best_validation"],
            "current_fedogda_std_test_mse_at_best_validation": current["std_test_mse_at_best_validation"],
            "current_fedogda_oscillation_score": current["mean_last_50_val_mse_std"],
            "current_fedogda_diverged_count": current["diverged_count"],
            "fedgda_mean_best_validation_mse": fedgda["mean_best_validation_mse"],
            "fedgda_mean_test_mse_at_best_validation": fedgda["mean_test_mse_at_best_validation"],
            "fedgda_std_test_mse_at_best_validation": fedgda["std_test_mse_at_best_validation"],
            "fedgda_oscillation_score": fedgda["mean_last_50_val_mse_std"],
            "fedgda_diverged_count": fedgda["diverged_count"],
            "tuned_minus_current_fedogda_test_mse": tuned_minus_current_test,
            "tuned_minus_fedgda_test_mse": tuned_minus_fedgda_test,
            "tuned_test_ratio_vs_current_fedogda": tuned_test_ratio_vs_current,
            "tuned_test_ratio_vs_fedgda": tuned_test_ratio_vs_fedgda,
            "tuned_oscillation_minus_current_fedogda": tuned_osc_minus_current,
            "tuned_oscillation_minus_fedgda": tuned_osc_minus_fedgda,
            "tuned_improves_over_current_fedogda_validation": tuned["mean_best_validation_mse"] < current["mean_best_validation_mse"],
            "tuned_improves_over_current_fedogda_test": tuned["mean_test_mse_at_best_validation"] < current["mean_test_mse_at_best_validation"],
            "tuned_reduces_oscillation_vs_current_fedogda": tuned["mean_last_50_val_mse_std"] < current["mean_last_50_val_mse_std"],
            "tuned_reduces_oscillation_vs_fedgda": tuned["mean_last_50_val_mse_std"] < fedgda["mean_last_50_val_mse_std"],
            "tuned_better_test_than_fedgda": tuned["mean_test_mse_at_best_validation"] < fedgda["mean_test_mse_at_best_validation"],
            "no_tuned_divergence": int(tuned["diverged_count"]) == 0,
        }
        comparison_rows.append(comparison)

        fedgda_table.append(final_table_row("FedGDA-S baseline", fedgda, fedgda, "reference"))
        current_fedogda_table.append(final_table_row("Current FedOGDA-S baseline", current, fedgda, "win" if current["mean_test_mse_at_best_validation"] < fedgda["mean_test_mse_at_best_validation"] else "loss"))
        tuned_tally = "win" if tuned["mean_test_mse_at_best_validation"] < fedgda["mean_test_mse_at_best_validation"] else "loss"
        tuned_table.append(final_table_row("Tuned FedOGDA-S", tuned, fedgda, tuned_tally))

    return comparison_rows, tuned_seed_metrics, fedgda_table, current_fedogda_table, tuned_table


def final_table_row(method_label: str, row: dict[str, Any], fedgda_ref: dict[str, Any], win_loss: str) -> dict[str, Any]:
    return {
        "method": method_label,
        "dataset": row["dataset"],
        "alpha": row["alpha"],
        "selected_critic_multiplier": row.get("critic_multiplier", ""),
        "selected_weight_decay": row.get("weight_decay", ""),
        "mean_validation_mse": row["mean_best_validation_mse"],
        "mean_test_mse": row["mean_test_mse_at_best_validation"],
        "test_std": row["std_test_mse_at_best_validation"],
        "oscillation_score": row["mean_last_50_val_mse_std"],
        "win_loss_vs_fedgda_s": win_loss,
        "test_ratio_vs_fedgda_s": row["mean_test_mse_at_best_validation"] / fedgda_ref["mean_test_mse_at_best_validation"],
        "diverged_count": row["diverged_count"],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary_md(
    path: Path,
    selected: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    fedgda_table: list[dict[str, Any]],
    current_table: list[dict[str, Any]],
    tuned_table: list[dict[str, Any]],
) -> None:
    final_cols = [
        "dataset",
        "alpha",
        "selected_critic_multiplier",
        "selected_weight_decay",
        "mean_validation_mse",
        "mean_test_mse",
        "test_std",
        "oscillation_score",
        "win_loss_vs_fedgda_s",
    ]
    comparison_cols = [
        "dataset",
        "selected_critic_multiplier",
        "selected_weight_decay",
        "tuned_mean_best_validation_mse",
        "tuned_mean_test_mse_at_best_validation",
        "tuned_test_ratio_vs_fedgda",
        "tuned_oscillation_score",
        "tuned_improves_over_current_fedogda_test",
        "tuned_reduces_oscillation_vs_fedgda",
        "tuned_better_test_than_fedgda",
        "no_tuned_divergence",
    ]
    text = [
        "# FedOGDA-S Alpha 0.5 Tuning Pilot Analysis",
        "",
        "Scope: `fedogda_s`, datasets `abs`, `step`, `linear`, alpha `0.5`, seeds `0,1,2`, 16 critic/weight-decay configs.",
        "",
        "Selection rule: choose by validation only. Primary key is lowest mean `best_validation_mse` across seeds. Tie-breakers are lower mean `last_50_val_mse_std`, then lower mean `final_vs_best_validation_gap`. `diverged_count` must be zero. Test MSE is reported only after selection.",
        "",
        "## Selected Tuned Configs",
        "",
        markdown_table(selected, ["dataset", "alpha", "critic_multiplier", "weight_decay", "mean_best_validation_mse", "mean_test_mse_at_best_validation", "std_test_mse_at_best_validation", "mean_last_50_val_mse_std", "diverged_count"]),
        "",
        "## FedGDA-S Baseline Table",
        "",
        markdown_table(fedgda_table, final_cols),
        "",
        "## Current FedOGDA-S Baseline Table",
        "",
        markdown_table(current_table, final_cols),
        "",
        "## Tuned FedOGDA-S Table",
        "",
        markdown_table(tuned_table, final_cols),
        "",
        "## Success Criteria Check",
        "",
        markdown_table(comparison, comparison_cols),
        "",
        "Notes:",
        "",
        "- `oscillation_score` is mean `last_50_val_mse_std` across seeds.",
        "- `win_loss_vs_fedgda_s` uses post-selection mean `test_mse_at_best_validation`.",
        "- `competitive` is not thresholded here; use `tuned_test_ratio_vs_fedgda` from `baseline_comparison.csv` to choose any tolerance.",
    ]
    path.write_text("\n".join(text) + "\n")


def main() -> int:
    tuned_rows = load_tuned_runs()
    datasets = {row["dataset"] for row in tuned_rows}
    alphas = {row["alpha"] for row in tuned_rows}
    seeds = {int(row["seed"]) for row in tuned_rows}
    if alphas != {0.5}:
        raise SystemExit(f"expected only alpha=0.5, found {sorted(alphas)}")

    stability_fields = [
        "run_id",
        "method",
        "dataset",
        "alpha",
        "seed",
        "critic_multiplier",
        "weight_decay",
        "best_validation_mse",
        "test_mse_at_best_validation",
        "best_validation_round",
        "final_validation_mse",
        "final_test_mse",
        "final_vs_best_validation_gap",
        "last_50_val_mse_std",
        "last_50_train_mse_std",
        "diverged",
        "runtime_seconds",
        "result_dir",
    ]
    write_csv(OUTPUTS["stability_metrics_csv"], tuned_rows, stability_fields)

    grid_rows = aggregate_runs(tuned_rows, ("dataset", "alpha", "critic_multiplier", "weight_decay"))
    selected = select_configs(grid_rows, required_seeds=len(seeds))
    grid_fields = [
        "dataset",
        "alpha",
        "critic_multiplier",
        "weight_decay",
        "num_seeds",
        "seed_values",
        "diverged_count",
        "all_seeds_present",
        "eligible_for_selection",
        "selection_rank",
        "mean_best_validation_mse",
        "mean_test_mse_at_best_validation",
        "std_test_mse_at_best_validation",
        "mean_final_validation_mse",
        "mean_final_test_mse",
        "mean_final_vs_best_validation_gap",
        "mean_last_50_val_mse_std",
        "mean_last_50_train_mse_std",
        "mean_runtime_seconds",
    ]
    write_csv(OUTPUTS["grid_aggregates_csv"], grid_rows, grid_fields)

    selected_fields = grid_fields + ["selection_rule"]
    write_csv(OUTPUTS["selected_configs_csv"], selected, selected_fields)

    baseline_rows = load_baseline_scope(datasets, 0.5, seeds)
    baseline_aggregates = aggregate_baselines(baseline_rows)
    comparison, selected_seed_metrics, fedgda_table, current_table, tuned_table = build_comparison(
        selected, tuned_rows, baseline_aggregates
    )
    write_csv(OUTPUTS["selected_seed_metrics_csv"], selected_seed_metrics, stability_fields)

    comparison_fields = [
        "dataset",
        "alpha",
        "selected_critic_multiplier",
        "selected_weight_decay",
        "tuned_mean_best_validation_mse",
        "tuned_mean_test_mse_at_best_validation",
        "tuned_std_test_mse_at_best_validation",
        "tuned_oscillation_score",
        "tuned_diverged_count",
        "current_fedogda_mean_best_validation_mse",
        "current_fedogda_mean_test_mse_at_best_validation",
        "current_fedogda_std_test_mse_at_best_validation",
        "current_fedogda_oscillation_score",
        "current_fedogda_diverged_count",
        "fedgda_mean_best_validation_mse",
        "fedgda_mean_test_mse_at_best_validation",
        "fedgda_std_test_mse_at_best_validation",
        "fedgda_oscillation_score",
        "fedgda_diverged_count",
        "tuned_minus_current_fedogda_test_mse",
        "tuned_minus_fedgda_test_mse",
        "tuned_test_ratio_vs_current_fedogda",
        "tuned_test_ratio_vs_fedgda",
        "tuned_oscillation_minus_current_fedogda",
        "tuned_oscillation_minus_fedgda",
        "tuned_improves_over_current_fedogda_validation",
        "tuned_improves_over_current_fedogda_test",
        "tuned_reduces_oscillation_vs_current_fedogda",
        "tuned_reduces_oscillation_vs_fedgda",
        "tuned_better_test_than_fedgda",
        "no_tuned_divergence",
    ]
    write_csv(OUTPUTS["baseline_comparison_csv"], comparison, comparison_fields)

    final_fields = [
        "method",
        "dataset",
        "alpha",
        "selected_critic_multiplier",
        "selected_weight_decay",
        "mean_validation_mse",
        "mean_test_mse",
        "test_std",
        "oscillation_score",
        "win_loss_vs_fedgda_s",
        "test_ratio_vs_fedgda_s",
        "diverged_count",
    ]
    write_csv(OUTPUTS["final_table_fedgda_s_csv"], fedgda_table, final_fields)
    write_csv(OUTPUTS["final_table_current_fedogda_s_csv"], current_table, final_fields)
    write_csv(OUTPUTS["final_table_tuned_fedogda_s_csv"], tuned_table, final_fields)

    summary = {
        "scope": {
            "method": "fedogda_s",
            "datasets": sorted(datasets),
            "alpha": 0.5,
            "seeds": sorted(seeds),
            "num_runs": len(tuned_rows),
        },
        "selection_rule": "validation_only_lowest_mean_best_validation_mse_tie_last50_val_std_tie_final_vs_best_gap_no_divergence",
        "selected_configs": selected,
        "comparison": comparison,
        "outputs": {key: str(path.relative_to(REPO_ROOT)) for key, path in OUTPUTS.items()},
    }
    OUTPUTS["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary_md(OUTPUTS["summary_md"], selected, comparison, fedgda_table, current_table, tuned_table)

    print(json.dumps({
        "runs_analyzed": len(tuned_rows),
        "selected_configs": len(selected),
        "outputs": {key: str(path.relative_to(REPO_ROOT)) for key, path in OUTPUTS.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
