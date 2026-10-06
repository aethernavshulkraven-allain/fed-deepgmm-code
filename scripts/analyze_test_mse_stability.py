#!/usr/bin/env python3
"""Analyze whether completed runs have stable per-round Test MSE.

The existing result CSVs are inspected for a per-round Test MSE column. When
that column is absent, Test-MSE last-50 diagnostics are marked not evaluable and
validation-MSE last-50 diagnostics are reported as a secondary curve signal.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_MANIFEST = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv"
TUNING_MANIFEST = (
    REPO_ROOT
    / "experiments"
    / "rerun_protocol_v1"
    / "tuning_fedogda_s"
    / "pilot_alpha0p5"
    / "manifest.csv"
)
OUT_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1"
PER_RUN_CSV = OUT_DIR / "test_mse_stability_diagnostics.csv"
PAIRS_CSV = OUT_DIR / "test_mse_stability_pairs.csv"
REPORT_MD = OUT_DIR / "test_mse_stability_report.md"

TEST_MSE_CANDIDATES = {
    "test_mse",
    "test_mse_by_round",
    "test_loss",
    "mse_test",
    "test",
    "test_metric",
    "test_mse_at_round",
}
EPS = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return as_float(value)
    except (TypeError, ValueError):
        return None


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = statistics.fmean(values)
    numerator = sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


def curve_diagnostics(values: list[float]) -> dict[str, Any]:
    tail = values[-50:]
    first10 = tail[:10]
    last10 = tail[-10:]
    mean = statistics.fmean(tail)
    std = pstdev(tail)
    minimum = min(tail)
    maximum = max(tail)
    return {
        "last50_mean": mean,
        "last50_std": std,
        "last50_min": minimum,
        "last50_max": maximum,
        "last50_range": maximum - minimum,
        "last50_cv": std / max(abs(mean), EPS),
        "last50_linear_slope": slope(tail),
        "relative_drift_last50": abs(statistics.fmean(last10) - statistics.fmean(first10)) / max(abs(mean), EPS),
        "large_relative_drift_last50": abs(statistics.fmean(last10) - statistics.fmean(first10)) / max(abs(mean), EPS) > 0.20,
    }


def stability_flags(cv: float | None, final_gap: float | None) -> dict[str, Any]:
    if cv is None or final_gap is None:
        return {
            "stable_5pct": "not_evaluable",
            "stable_10pct": "not_evaluable",
            "stable_20pct": "not_evaluable",
        }
    return {
        "stable_5pct": cv <= 0.05 and final_gap <= 0.05,
        "stable_10pct": cv <= 0.10 and final_gap <= 0.10,
        "stable_20pct": cv <= 0.20 and final_gap <= 0.20,
    }


def find_test_column(columns: list[str]) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in TEST_MSE_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    for column in columns:
        lower = column.strip().lower()
        if "test" in lower and "mse" in lower:
            return column
    return ""


def mode_for_method(method: str) -> str:
    if method.endswith("_d"):
        return "deterministic"
    if method.endswith("_s"):
        return "stochastic"
    return "unknown"


def family_for_method(method: str) -> str:
    if "fedogda" in method:
        return "FedOGDA"
    if "fedgda" in method:
        return "FedGDA"
    return "other"


def load_manifest_rows(path: Path, family: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        if row.get("training_scope") != "federated":
            continue
        result_dir = row.get("final_result_dir", "")
        if not result_dir:
            continue
        metrics_path = REPO_ROOT / result_dir / "metrics.json"
        curve_path = REPO_ROOT / result_dir / "mse_by_round.csv"
        if metrics_path.exists() and curve_path.exists():
            row = dict(row)
            row["experiment_family"] = family
            rows.append(row)
    return rows


def analyze_run(row: dict[str, str]) -> dict[str, Any]:
    run_dir = REPO_ROOT / row["final_result_dir"]
    metrics = read_json(run_dir / "metrics.json")
    curve_rows = read_csv(run_dir / "mse_by_round.csv")
    columns = list(curve_rows[0].keys()) if curve_rows else []
    test_column = find_test_column(columns)
    method = row["method"]

    finite_history = all(truthy(item.get("finite", "true")) for item in curve_rows)
    history_diverged = any(truthy(item.get("diverged", "false")) for item in curve_rows)
    metric_diverged = bool(metrics.get("diverged", False))

    result: dict[str, Any] = {
        "experiment_family": row["experiment_family"],
        "run_id": row["run_id"],
        "dataset": row["dataset"],
        "alpha": row["alpha"],
        "seed": row["seed"],
        "method": method,
        "method_family": family_for_method(method),
        "mode": mode_for_method(method),
        "critic_multiplier": row.get("critic_multiplier", ""),
        "weight_decay": row.get("weight_decay", ""),
        "result_dir": row["final_result_dir"],
        "mse_by_round_columns": "|".join(columns),
        "per_round_test_mse_column": test_column,
        "has_per_round_test_mse": bool(test_column),
        "round_count": len(curve_rows),
        "history_all_finite": finite_history,
        "history_diverged": history_diverged,
        "metrics_diverged": metric_diverged,
        "numerically_stable": finite_history and not history_diverged and not metric_diverged,
        "final_test_mse_metrics": metrics.get("final_test_mse", ""),
        "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation", ""),
        "best_validation_mse": metrics.get("best_validation_mse", ""),
        "final_validation_mse_metrics": metrics.get("final_validation_mse", ""),
    }

    if test_column:
        values = [as_float(item[test_column]) for item in curve_rows]
        diag = curve_diagnostics(values)
        final_curve = values[-1]
        final_metrics = optional_float(metrics.get("final_test_mse"))
        final_for_gap = final_metrics if final_metrics is not None else final_curve
        final_gap = abs(final_for_gap - diag["last50_mean"]) / max(abs(diag["last50_mean"]), EPS)
        result.update(
            {
                "final_per_round_test_mse": final_curve,
                "last50_mean_test_mse": diag["last50_mean"],
                "last50_std_test_mse": diag["last50_std"],
                "last50_min_test_mse": diag["last50_min"],
                "last50_max_test_mse": diag["last50_max"],
                "last50_range_test_mse": diag["last50_range"],
                "last50_cv_test_mse": diag["last50_cv"],
                "final_vs_last50_mean_relative_gap_test_mse": final_gap,
                "last50_linear_slope_test_mse": diag["last50_linear_slope"],
                "relative_drift_last50_test_mse": diag["relative_drift_last50"],
                "large_relative_drift_last50_test_mse": diag["large_relative_drift_last50"],
                **stability_flags(diag["last50_cv"], final_gap),
            }
        )
    else:
        result.update(
            {
                "final_per_round_test_mse": "",
                "last50_mean_test_mse": "",
                "last50_std_test_mse": "",
                "last50_min_test_mse": "",
                "last50_max_test_mse": "",
                "last50_range_test_mse": "",
                "last50_cv_test_mse": "",
                "final_vs_last50_mean_relative_gap_test_mse": "",
                "last50_linear_slope_test_mse": "",
                "relative_drift_last50_test_mse": "",
                "large_relative_drift_last50_test_mse": "not_evaluable",
                **stability_flags(None, None),
            }
        )

    if "val_mse" in columns:
        values = [as_float(item["val_mse"]) for item in curve_rows]
        diag = curve_diagnostics(values)
        final_curve = values[-1]
        final_metrics = optional_float(metrics.get("final_validation_mse"))
        final_for_gap = final_metrics if final_metrics is not None else final_curve
        final_gap = abs(final_for_gap - diag["last50_mean"]) / max(abs(diag["last50_mean"]), EPS)
        result.update(
            {
                "fallback_curve_used": "val_mse",
                "final_per_round_val_mse": final_curve,
                "last50_mean_val_mse": diag["last50_mean"],
                "last50_std_val_mse": diag["last50_std"],
                "last50_min_val_mse": diag["last50_min"],
                "last50_max_val_mse": diag["last50_max"],
                "last50_range_val_mse": diag["last50_range"],
                "last50_cv_val_mse": diag["last50_cv"],
                "final_vs_last50_mean_relative_gap_val_mse": final_gap,
                "last50_linear_slope_val_mse": diag["last50_linear_slope"],
                "relative_drift_last50_val_mse": diag["relative_drift_last50"],
                "large_relative_drift_last50_val_mse": diag["large_relative_drift_last50"],
                "val_stable_5pct": diag["last50_cv"] <= 0.05 and final_gap <= 0.05,
                "val_stable_10pct": diag["last50_cv"] <= 0.10 and final_gap <= 0.10,
                "val_stable_20pct": diag["last50_cv"] <= 0.20 and final_gap <= 0.20,
            }
        )
    return result


def build_pairs(main_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["dataset"], row["alpha"], row["seed"], row["method"]): row
        for row in main_runs
        if row["experiment_family"] == "rerun_protocol_v1"
    }
    pairs: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in main_runs if row["experiment_family"] == "rerun_protocol_v1"}):
        for alpha in sorted({row["alpha"] for row in main_runs if row["dataset"] == dataset and row["experiment_family"] == "rerun_protocol_v1"}, key=float):
            for seed in sorted({row["seed"] for row in main_runs if row["dataset"] == dataset and row["alpha"] == alpha and row["experiment_family"] == "rerun_protocol_v1"}, key=int):
                for mode, gda_method, ogda_method in (
                    ("deterministic", "fedgda_d", "fedogda_d"),
                    ("stochastic", "fedgda_s", "fedogda_s"),
                ):
                    fedgda = by_key.get((dataset, alpha, seed, gda_method))
                    fedogda = by_key.get((dataset, alpha, seed, ogda_method))
                    if not fedgda or not fedogda:
                        continue
                    has_pair_test_curve = bool(fedgda["has_per_round_test_mse"]) and bool(fedogda["has_per_round_test_mse"])
                    row = {
                        "dataset": dataset,
                        "alpha": alpha,
                        "seed": seed,
                        "mode": mode,
                        "fedgda_method": gda_method,
                        "fedogda_method": ogda_method,
                        "has_per_round_test_mse_pair": has_pair_test_curve,
                        "fedgda_final_test_mse": fedgda["final_test_mse_metrics"],
                        "fedogda_final_test_mse": fedogda["final_test_mse_metrics"],
                        "fedogda_minus_fedgda_final_test_mse": as_float(fedogda["final_test_mse_metrics"]) - as_float(fedgda["final_test_mse_metrics"]),
                        "fedogda_lower_final_test_mse": as_float(fedogda["final_test_mse_metrics"]) < as_float(fedgda["final_test_mse_metrics"]),
                        "fedgda_last50_mean_test_mse": fedgda["last50_mean_test_mse"],
                        "fedogda_last50_mean_test_mse": fedogda["last50_mean_test_mse"],
                        "fedogda_lower_last50_mean_test_mse": "not_evaluable",
                        "fedgda_last50_std_test_mse": fedgda["last50_std_test_mse"],
                        "fedogda_last50_std_test_mse": fedogda["last50_std_test_mse"],
                        "fedogda_lower_last50_std_test_mse": "not_evaluable",
                        "fedgda_last50_cv_test_mse": fedgda["last50_cv_test_mse"],
                        "fedogda_last50_cv_test_mse": fedogda["last50_cv_test_mse"],
                        "fedogda_lower_last50_cv_test_mse": "not_evaluable",
                        "fedgda_last50_mean_val_mse": fedgda.get("last50_mean_val_mse", ""),
                        "fedogda_last50_mean_val_mse": fedogda.get("last50_mean_val_mse", ""),
                        "fedogda_lower_last50_mean_val_mse": fedogda.get("last50_mean_val_mse", math.inf) < fedgda.get("last50_mean_val_mse", math.inf),
                        "fedgda_last50_std_val_mse": fedgda.get("last50_std_val_mse", ""),
                        "fedogda_last50_std_val_mse": fedogda.get("last50_std_val_mse", ""),
                        "fedogda_lower_last50_std_val_mse": fedogda.get("last50_std_val_mse", math.inf) < fedgda.get("last50_std_val_mse", math.inf),
                        "fedgda_last50_cv_val_mse": fedgda.get("last50_cv_val_mse", ""),
                        "fedogda_last50_cv_val_mse": fedogda.get("last50_cv_val_mse", ""),
                        "fedogda_lower_last50_cv_val_mse": fedogda.get("last50_cv_val_mse", math.inf) < fedgda.get("last50_cv_val_mse", math.inf),
                        "both_numerically_stable": fedgda["numerically_stable"] and fedogda["numerically_stable"],
                    }
                    if has_pair_test_curve:
                        row.update(
                            {
                                "fedogda_lower_last50_mean_test_mse": as_float(fedogda["last50_mean_test_mse"]) < as_float(fedgda["last50_mean_test_mse"]),
                                "fedogda_lower_last50_std_test_mse": as_float(fedogda["last50_std_test_mse"]) < as_float(fedgda["last50_std_test_mse"]),
                                "fedogda_lower_last50_cv_test_mse": as_float(fedogda["last50_cv_test_mse"]) < as_float(fedgda["last50_cv_test_mse"]),
                            }
                        )
                    pairs.append(row)
    return pairs


def bool_count(rows: list[dict[str, Any]], key: str, value: Any = True) -> int:
    return sum(row.get(key) == value for row in rows)


def percent(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%" if total else "n/a"


def summarize_group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(keys, key))
        item.update(
            {
                "runs": len(group),
                "numerically_stable": bool_count(group, "numerically_stable"),
                "has_per_round_test_mse": bool_count(group, "has_per_round_test_mse"),
                "test_stable_5pct": bool_count(group, "stable_5pct"),
                "test_stable_10pct": bool_count(group, "stable_10pct"),
                "test_stable_20pct": bool_count(group, "stable_20pct"),
                "val_stable_5pct": bool_count(group, "val_stable_5pct"),
                "val_stable_10pct": bool_count(group, "val_stable_10pct"),
                "val_stable_20pct": bool_count(group, "val_stable_20pct"),
                "val_large_drift_gt20pct": bool_count(group, "large_relative_drift_last50_val_mse"),
            }
        )
        summary.append(item)
    return summary


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(per_run: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    main = [row for row in per_run if row["experiment_family"] == "rerun_protocol_v1"]
    tuning = [row for row in per_run if row["experiment_family"] == "fedogda_s_tuning_pilot_alpha0p5"]
    all_has_test = bool_count(per_run, "has_per_round_test_mse")
    main_has_test = bool_count(main, "has_per_round_test_mse")
    tuning_has_test = bool_count(tuning, "has_per_round_test_mse")
    stable_numeric = bool_count(per_run, "numerically_stable")
    main_numeric = bool_count(main, "numerically_stable")
    tuning_numeric = bool_count(tuning, "numerically_stable")

    pair_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        pair_by_mode[row["mode"]].append(row)
    pair_summary = []
    for mode, group in sorted(pair_by_mode.items()):
        pair_summary.append(
            {
                "mode": mode,
                "pairs": len(group),
                "fedogda_lower_final_test_mse": bool_count(group, "fedogda_lower_final_test_mse"),
                "fedogda_lower_last50_mean_test_mse": "not_evaluable",
                "fedogda_lower_last50_std_test_mse": "not_evaluable",
                "fedogda_lower_last50_std_val_mse_secondary": bool_count(group, "fedogda_lower_last50_std_val_mse"),
                "fedogda_lower_last50_cv_val_mse_secondary": bool_count(group, "fedogda_lower_last50_cv_val_mse"),
            }
        )

    method_summary = summarize_group(main, ("method",))
    dataset_summary = summarize_group(main, ("dataset",))
    alpha_summary = summarize_group(main, ("alpha",))
    mode_summary = summarize_group(main, ("mode",))
    tuning_method_summary = summarize_group(tuning, ("method",)) if tuning else []

    text = [
        "# Test MSE Stability Report",
        "",
        "Scope: completed existing outputs only. No training was launched and no training logic was changed.",
        "",
        "## Direct Answer",
        "",
        "All inspected completed runs are numerically stable: no non-finite history rows and no `diverged=true` metrics were found.",
        "",
        "However, the existing `mse_by_round.csv` files do **not** store per-round Test MSE. They store only:",
        "",
        "`round, train_mse, val_mse, gmm_train_objective, gmm_val_objective, gmm_eval, finite, diverged`",
        "",
        "Therefore, from the current artifacts we **cannot** compute `last50_mean_test_mse`, `last50_std_test_mse`, Test-MSE CV, Test-MSE drift, or decide whether final Test MSE is stabilized by directly inspecting a Test-MSE curve.",
        "",
        "The safest wording is:",
        "",
        "> The runs are numerically stable, but per-round Test MSE was not logged, so we cannot verify Test-MSE stabilization or report a last-50 average Test MSE from existing outputs. We can report `final_test_mse` / `test_mse_at_best_validation` as scalar held-out Test metrics, and use the last-50 validation curve as a secondary stability diagnostic. If last-50 average Test MSE is wanted, we need to add per-round Test MSE logging and rerun or at least re-evaluate checkpoints per round.",
        "",
        "## Availability Summary",
        "",
        f"- Main federated synthetic runs inspected: `{len(main)}`",
        f"- FedOGDA-S tuning pilot runs inspected: `{len(tuning)}`",
        f"- Total inspected runs: `{len(per_run)}`",
        f"- Numerically stable runs: `{stable_numeric}/{len(per_run)}`",
        f"- Runs with per-round Test MSE: `{all_has_test}/{len(per_run)}`",
        f"- Main matrix per-round Test MSE availability: `{main_has_test}/{len(main)}`",
        f"- Tuning pilot per-round Test MSE availability: `{tuning_has_test}/{len(tuning)}`",
        "",
        "## Main 144-Run Matrix: Validation-Curve Stability Secondary Signal",
        "",
        "Because Test MSE is not logged per round, the following stability counts use `val_mse` only as a secondary signal.",
        "",
        "### By Method",
        "",
        markdown_table(method_summary, ["method", "runs", "numerically_stable", "has_per_round_test_mse", "val_stable_5pct", "val_stable_10pct", "val_stable_20pct", "val_large_drift_gt20pct"]),
        "",
        "### By Dataset",
        "",
        markdown_table(dataset_summary, ["dataset", "runs", "numerically_stable", "has_per_round_test_mse", "val_stable_5pct", "val_stable_10pct", "val_stable_20pct", "val_large_drift_gt20pct"]),
        "",
        "### By Alpha",
        "",
        markdown_table(alpha_summary, ["alpha", "runs", "numerically_stable", "has_per_round_test_mse", "val_stable_5pct", "val_stable_10pct", "val_stable_20pct", "val_large_drift_gt20pct"]),
        "",
        "### By Deterministic/Stochastic Mode",
        "",
        markdown_table(mode_summary, ["mode", "runs", "numerically_stable", "has_per_round_test_mse", "val_stable_5pct", "val_stable_10pct", "val_stable_20pct", "val_large_drift_gt20pct"]),
        "",
        "## FedGDA vs FedOGDA Paired Comparisons",
        "",
        "Test last-50 comparisons are not evaluable because neither side logs per-round Test MSE. Final Test MSE comparisons are evaluable from `metrics.json`; validation oscillation comparisons are included as secondary evidence.",
        "",
        markdown_table(pair_summary, ["mode", "pairs", "fedogda_lower_final_test_mse", "fedogda_lower_last50_mean_test_mse", "fedogda_lower_last50_std_test_mse", "fedogda_lower_last50_std_val_mse_secondary", "fedogda_lower_last50_cv_val_mse_secondary"]),
        "",
        "Interpretation:",
        "",
        "- Deterministic FedOGDA has lower scalar final Test MSE than deterministic FedGDA in many pairs, but Test-curve last-50 behavior is not available.",
        "- Stochastic FedOGDA does not generally beat stochastic FedGDA on scalar final Test MSE in the existing main matrix.",
        "- FedOGDA often looks less oscillatory by validation-curve standard deviation/CV, but that is a validation-curve statement, not a Test-MSE curve statement.",
        "",
        "## FedOGDA-S Tuning Pilot",
        "",
        "The tuning pilot has the same logging limitation: no per-round Test MSE. It is numerically stable, and its validation-curve diagnostics are available.",
        "",
        markdown_table(tuning_method_summary, ["method", "runs", "numerically_stable", "has_per_round_test_mse", "val_stable_5pct", "val_stable_10pct", "val_stable_20pct", "val_large_drift_gt20pct"]) if tuning_method_summary else "No tuning runs found.",
        "",
        "Existing tuning-pilot conclusion remains: tuned FedOGDA-S improves over current FedOGDA-S and reduces validation oscillation, but it still does not beat FedGDA-S on mean scalar Test MSE for `abs`, `linear`, or `step` at alpha `0.5`.",
        "",
        "## Recommendation",
        "",
        "For the current repo outputs:",
        "",
        "- Report scalar `test_mse_at_best_validation` or `final_test_mse` only with the caveat that per-round Test-MSE stability cannot be verified.",
        "- If stabilized Test MSE or last-50 average Test MSE is specifically needed, add per-round Test MSE logging to `mse_by_round.csv` at the same evaluation frequency and rerun or re-evaluate saved per-round checkpoints.",
        "- Do not substitute last-50 validation MSE and call it Test MSE. Validation last-50 metrics are useful only as secondary stability diagnostics.",
        "",
        "## Output Files",
        "",
        f"- Per-run diagnostics: `{PER_RUN_CSV.relative_to(REPO_ROOT)}`",
        f"- Paired comparisons: `{PAIRS_CSV.relative_to(REPO_ROOT)}`",
        f"- Markdown report: `{REPORT_MD.relative_to(REPO_ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(text) + "\n")


def main() -> int:
    rows = load_manifest_rows(MAIN_MANIFEST, "rerun_protocol_v1")
    rows.extend(load_manifest_rows(TUNING_MANIFEST, "fedogda_s_tuning_pilot_alpha0p5"))
    per_run = [analyze_run(row) for row in rows]
    pairs = build_pairs(per_run)

    per_run_fields = [
        "experiment_family",
        "run_id",
        "dataset",
        "alpha",
        "seed",
        "method",
        "method_family",
        "mode",
        "critic_multiplier",
        "weight_decay",
        "result_dir",
        "mse_by_round_columns",
        "per_round_test_mse_column",
        "has_per_round_test_mse",
        "round_count",
        "history_all_finite",
        "history_diverged",
        "metrics_diverged",
        "numerically_stable",
        "final_test_mse_metrics",
        "test_mse_at_best_validation",
        "final_per_round_test_mse",
        "last50_mean_test_mse",
        "last50_std_test_mse",
        "last50_min_test_mse",
        "last50_max_test_mse",
        "last50_range_test_mse",
        "last50_cv_test_mse",
        "final_vs_last50_mean_relative_gap_test_mse",
        "last50_linear_slope_test_mse",
        "relative_drift_last50_test_mse",
        "large_relative_drift_last50_test_mse",
        "stable_5pct",
        "stable_10pct",
        "stable_20pct",
        "fallback_curve_used",
        "best_validation_mse",
        "final_validation_mse_metrics",
        "final_per_round_val_mse",
        "last50_mean_val_mse",
        "last50_std_val_mse",
        "last50_min_val_mse",
        "last50_max_val_mse",
        "last50_range_val_mse",
        "last50_cv_val_mse",
        "final_vs_last50_mean_relative_gap_val_mse",
        "last50_linear_slope_val_mse",
        "relative_drift_last50_val_mse",
        "large_relative_drift_last50_val_mse",
        "val_stable_5pct",
        "val_stable_10pct",
        "val_stable_20pct",
    ]
    pair_fields = [
        "dataset",
        "alpha",
        "seed",
        "mode",
        "fedgda_method",
        "fedogda_method",
        "has_per_round_test_mse_pair",
        "fedgda_final_test_mse",
        "fedogda_final_test_mse",
        "fedogda_minus_fedgda_final_test_mse",
        "fedogda_lower_final_test_mse",
        "fedgda_last50_mean_test_mse",
        "fedogda_last50_mean_test_mse",
        "fedogda_lower_last50_mean_test_mse",
        "fedgda_last50_std_test_mse",
        "fedogda_last50_std_test_mse",
        "fedogda_lower_last50_std_test_mse",
        "fedgda_last50_cv_test_mse",
        "fedogda_last50_cv_test_mse",
        "fedogda_lower_last50_cv_test_mse",
        "fedgda_last50_mean_val_mse",
        "fedogda_last50_mean_val_mse",
        "fedogda_lower_last50_mean_val_mse",
        "fedgda_last50_std_val_mse",
        "fedogda_last50_std_val_mse",
        "fedogda_lower_last50_std_val_mse",
        "fedgda_last50_cv_val_mse",
        "fedogda_last50_cv_val_mse",
        "fedogda_lower_last50_cv_val_mse",
        "both_numerically_stable",
    ]
    write_csv(PER_RUN_CSV, per_run, per_run_fields)
    write_csv(PAIRS_CSV, pairs, pair_fields)
    write_report(per_run, pairs)

    summary = {
        "per_run_rows": len(per_run),
        "pair_rows": len(pairs),
        "per_round_test_mse_available": sum(row["has_per_round_test_mse"] for row in per_run),
        "numerically_stable": sum(row["numerically_stable"] for row in per_run),
        "outputs": [
            str(PER_RUN_CSV.relative_to(REPO_ROOT)),
            str(PAIRS_CSV.relative_to(REPO_ROOT)),
            str(REPORT_MD.relative_to(REPO_ROOT)),
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
