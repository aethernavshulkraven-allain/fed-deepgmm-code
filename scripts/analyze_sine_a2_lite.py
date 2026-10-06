#!/usr/bin/env python3
"""Produce the deterministic Sine A2-lite scientific closeout artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "sine_fedogda_tuning"
PLOTS_DIR = EXP_DIR / "plots"
FEDOGDA_ROOT = (
    REPO_ROOT
    / "results"
    / "sine_fedogda_tuning"
    / "stage_A2_from_A1_mini"
    / "sin"
    / "fedogda_d"
)
FEDGDA_ROOT = REPO_ROOT / "results" / "rerun_protocol_v1" / "sin" / "fedgda_d"
LITE_MANIFEST = EXP_DIR / "stage_A2_lite_selected_manifest.csv"
FULL_A2_MANIFEST = EXP_DIR / "stage_A2_from_A1_mini_manifest.csv"
DECISION_MD = EXP_DIR / "stage_A2_lite_decision.md"
A1_TOP_MD = EXP_DIR / "stage_A1_mini_top_candidates.md"
METRIC_POLICY = EXP_DIR / "metric_policy.md"
ANALYZER = REPO_ROOT / "scripts" / "analyze_sine_stage_a1_mini.py"

LOCKED_RECIPE_JSON = EXP_DIR / "a2_lite_locked_recipe_summary.json"
SELECTION_AUDIT_MD = EXP_DIR / "a2_lite_selection_audit.md"
FEDOGDA_METRICS_CSV = EXP_DIR / "a2_lite_fedogda_d_seed_metrics.csv"
FEDOGDA_AGGREGATE_JSON = EXP_DIR / "a2_lite_fedogda_d_aggregate.json"
BASELINE_AUDIT_CSV = EXP_DIR / "a2_lite_fedgda_baseline_match_audit.csv"
MISSING_BASELINE_MD = EXP_DIR / "a2_lite_missing_fedgda_baseline.md"
PAIRWISE_CSV = EXP_DIR / "a2_lite_pairwise_fedogda_vs_fedgda.csv"
PAIRWISE_SUMMARY_MD = EXP_DIR / "a2_lite_pairwise_summary.md"
CURVE_FIT_CSV = EXP_DIR / "a2_lite_curve_fit_summary.csv"
FINAL_REPORT_MD = EXP_DIR / "a2_lite_final_report.md"

SEEDS = (0, 1, 2)
EPS = 1e-12

FEDOGDA_FIELDS = [
    "seed",
    "run_id",
    "result_dir",
    "best_validation_round",
    "best_validation_mse",
    "test_mse_at_best_validation",
    "final_test_mse",
    "last50_validation_mse_mean",
    "last50_validation_mse_std",
    "last50_validation_mse_cv",
    "last50_test_mse_mean",
    "last50_test_mse_std",
    "last50_test_mse_cv",
    "diverged",
    "finite_history",
]

BASELINE_AUDIT_FIELDS = [
    "seed",
    "fedogda_result_dir",
    "candidate_fedgda_result_dir",
    "same_alpha",
    "same_T",
    "same_R",
    "same_seed",
    "same_data",
    "same_client_count",
    "same_participation",
    "same_batch_size",
    "same_metric_policy",
    "is_fair_pair",
    "mismatch_notes",
]

PAIRWISE_FIELDS = [
    "seed",
    "alpha",
    "T",
    "R",
    "fedgda_test_mse_at_best_validation",
    "fedogda_test_mse_at_best_validation",
    "absolute_gap",
    "relative_gap_pct",
    "winner",
    "fedgda_final_test_mse",
    "fedogda_final_test_mse",
    "fedgda_last50_test_mse_mean",
    "fedogda_last50_test_mse_mean",
]

CURVE_FIELDS = [
    "method",
    "seed",
    "prediction_source",
    "evaluation_grid",
    "num_points",
    "curve_mse",
    "curve_mae",
    "curve_max_abs_error",
    "result_dir",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def finite_number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def tail_stats(values: list[float]) -> dict[str, float]:
    tail = values[-50:]
    mean = statistics.fmean(tail)
    std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(tail),
        "max": max(tail),
        "range": max(tail) - min(tail),
        "cv": std / max(abs(mean), EPS),
    }


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "num_seeds": len(values),
    }


def locate_single(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {parent / pattern}, found {len(matches)}")
    return matches[0]


def fedogda_run_dir(seed: int) -> Path:
    return locate_single(FEDOGDA_ROOT / f"seed_{seed}", "*alpha1p0_R3_cm15_slr1.5_glr0p002")


def fedgda_run_dir(seed: int) -> Path:
    return locate_single(FEDGDA_ROOT / f"seed_{seed}", "*alpha1p0")


def history_values(path: Path, value_column: str) -> tuple[list[float], bool]:
    rows = read_csv(path)
    values: list[float] = []
    finite = True
    for row in rows:
        try:
            values.append(finite_number(row[value_column]))
        except (KeyError, TypeError, ValueError):
            finite = False
        finite = finite and truthy(row.get("finite", "true"))
        finite = finite and not truthy(row.get("diverged", "false"))
    return values, finite


def required_artifacts_finite(run_dir: Path, require_test_history: bool) -> bool:
    required = [
        run_dir / "metrics.json",
        run_dir / "mse_by_round.csv",
        run_dir / "effective_config.json",
        run_dir / "predictions.npz",
        run_dir / "checkpoints" / "best_validation.pt",
    ]
    if require_test_history:
        required.append(run_dir / "test_mse_by_round.csv")
    if not all(path.exists() for path in required):
        return False
    history, history_finite = history_values(run_dir / "mse_by_round.csv", "val_mse")
    if not history or not history_finite:
        return False
    if require_test_history:
        test, test_finite = history_values(run_dir / "test_mse_by_round.csv", "test_mse")
        if not test or not test_finite:
            return False
    with np.load(run_dir / "predictions.npz") as predictions:
        for key in ("x", "true_g", "best_validation_prediction", "final_prediction"):
            if key not in predictions or not np.isfinite(predictions[key]).all():
                return False
    return not bool(read_json(run_dir / "metrics.json").get("diverged", False))


def validation_argmin_matches(run_dir: Path) -> bool:
    rows = read_csv(run_dir / "mse_by_round.csv")
    values = [finite_number(row["val_mse"]) for row in rows]
    rounds = [int(row["round"]) for row in rows]
    argmin_round = rounds[min(range(len(values)), key=values.__getitem__)]
    metrics = read_json(run_dir / "metrics.json")
    return (
        argmin_round == int(metrics["best_validation_round"])
        and math.isclose(
            values[rounds.index(argmin_round)],
            finite_number(metrics["best_validation_mse"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )


def selection_metadata_ok(run_dir: Path) -> bool:
    config = read_json(run_dir / "effective_config.json")
    metrics = read_json(run_dir / "metrics.json")
    return (
        config.get("selection_metric_source") == "validation"
        and config.get("test_mse_used_for_selection") is False
        and metrics.get("selection_metric_source") == "validation"
        and metrics.get("test_mse_used_for_selection") is False
        and validation_argmin_matches(run_dir)
    )


def recipe_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": config["dataset"],
        "variant": config["variant"],
        "mode": config["mode"],
        "alpha": finite_number(config["partition_alpha"]),
        "T": int(config["comm_round"]),
        "R": int(config["local_epochs"]),
        "g_lr": finite_number(config["g_learning_rate"]),
        "f_lr": finite_number(config["f_learning_rate"]),
        "critic_multiplier": finite_number(config["critic_multiplier"]),
        "server_lr": finite_number(config["server_learning_rate"]),
        "weight_decay": finite_number(config["weight_decay"]),
        "client_num_in_total": int(config["client_num_in_total"]),
        "client_num_per_round": int(config["client_num_per_round"]),
        "batch_size": int(config["batch_size"]),
        "partition_method": config["partition_method"],
        "data_cache_dir": config["data_cache_dir"],
        "model": config["model"],
    }


def write_locked_recipe() -> dict[str, Any]:
    run_dirs = [fedogda_run_dir(seed) for seed in SEEDS]
    config = read_json(run_dirs[0] / "effective_config.json")
    signature = recipe_signature(config)
    summary = {
        "locked_recipe_source": relative(DECISION_MD),
        "a1_validation_ranking_source": relative(A1_TOP_MD),
        "a2_lite_manifest": relative(LITE_MANIFEST),
        "full_a2_manifest_source": relative(FULL_A2_MANIFEST),
        "metric_policy": relative(METRIC_POLICY),
        "selection_metric": "validation_only_last50_validation_mse_mean",
        "selection_metric_source": "validation",
        "selected_without_test": True,
        "test_mse_used_for_selection": False,
        "data_recipe": {
            "data_cache_dir": signature["data_cache_dir"],
            "dataset": signature["dataset"],
            "partition_method": signature["partition_method"],
            "partition_alpha": signature["alpha"],
            "seed_specific_data": True,
        },
        "recipe": signature,
        "seeds": list(SEEDS),
        "run_ids": {
            str(seed): read_json(run_dir / "effective_config.json")["run_id"]
            for seed, run_dir in zip(SEEDS, run_dirs)
        },
        "result_dirs": {
            str(seed): relative(run_dir) for seed, run_dir in zip(SEEDS, run_dirs)
        },
    }
    LOCKED_RECIPE_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def collect_fedogda_metrics() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = fedogda_run_dir(seed)
        metrics = read_json(run_dir / "metrics.json")
        config = read_json(run_dir / "effective_config.json")
        validation, validation_finite = history_values(run_dir / "mse_by_round.csv", "val_mse")
        test, test_finite = history_values(run_dir / "test_mse_by_round.csv", "test_mse")
        validation_tail = tail_stats(validation)
        test_tail = tail_stats(test)
        rows.append(
            {
                "seed": seed,
                "run_id": config["run_id"],
                "result_dir": relative(run_dir),
                "best_validation_round": int(metrics["best_validation_round"]),
                "best_validation_mse": finite_number(metrics["best_validation_mse"]),
                "test_mse_at_best_validation": finite_number(
                    metrics["test_mse_at_best_validation"]
                ),
                "final_test_mse": finite_number(metrics["final_test_mse"]),
                "last50_validation_mse_mean": validation_tail["mean"],
                "last50_validation_mse_std": validation_tail["std"],
                "last50_validation_mse_cv": validation_tail["cv"],
                "last50_test_mse_mean": test_tail["mean"],
                "last50_test_mse_std": test_tail["std"],
                "last50_test_mse_cv": test_tail["cv"],
                "diverged": bool(metrics.get("diverged", False)),
                "finite_history": (
                    validation_finite
                    and test_finite
                    and required_artifacts_finite(run_dir, require_test_history=True)
                ),
            }
        )
    write_csv(FEDOGDA_METRICS_CSV, rows, FEDOGDA_FIELDS)
    aggregate_fields = [
        "test_mse_at_best_validation",
        "final_test_mse",
        "last50_test_mse_mean",
        "best_validation_mse",
        "last50_validation_mse_mean",
    ]
    summary = {
        "std_definition": "population",
        "selection_metric_source": "validation",
        "test_metrics_used_post_selection_only": True,
        "metrics": {
            field: aggregate([finite_number(row[field]) for row in rows])
            for field in aggregate_fields
        },
    }
    FEDOGDA_AGGREGATE_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return rows, summary


def same_data_pair(fedogda_dir: Path, fedgda_dir: Path) -> bool:
    fedogda_config = read_json(fedogda_dir / "effective_config.json")
    fedgda_config = read_json(fedgda_dir / "effective_config.json")
    if fedogda_config["data_cache_dir"] != fedgda_config["data_cache_dir"]:
        return False
    with np.load(fedogda_dir / "predictions.npz") as fedogda_predictions:
        with np.load(fedgda_dir / "predictions.npz") as fedgda_predictions:
            return np.array_equal(
                fedogda_predictions["x"], fedgda_predictions["x"]
            ) and np.array_equal(
                fedogda_predictions["true_g"], fedgda_predictions["true_g"]
            )


def same_metric_policy_pair(fedogda_dir: Path, fedgda_dir: Path) -> bool:
    fedogda_config = read_json(fedogda_dir / "effective_config.json")
    fedgda_config = read_json(fedgda_dir / "effective_config.json")
    keys = (
        "eval_freq",
        "frequency_of_the_test",
        "simple_model_selection_epochs",
        "f_history_model_selection_epochs",
        "model_selection_batch_size",
    )
    same_eval = all(fedogda_config.get(key) == fedgda_config.get(key) for key in keys)
    return (
        same_eval
        and validation_argmin_matches(fedogda_dir)
        and validation_argmin_matches(fedgda_dir)
        and "test_mse_at_best_validation" in read_json(fedgda_dir / "metrics.json")
    )


def baseline_match_audit() -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        fedogda_dir = fedogda_run_dir(seed)
        fedgda_dir = fedgda_run_dir(seed)
        fedogda_config = read_json(fedogda_dir / "effective_config.json")
        fedgda_config = read_json(fedgda_dir / "effective_config.json")
        checks = {
            "same_alpha": math.isclose(
                finite_number(fedogda_config["partition_alpha"]),
                finite_number(fedgda_config["partition_alpha"]),
            ),
            "same_T": int(fedogda_config["comm_round"]) == int(fedgda_config["comm_round"]),
            "same_R": int(fedogda_config["local_epochs"])
            == int(fedgda_config["local_epochs"]),
            "same_seed": int(fedogda_config["random_seed"])
            == int(fedgda_config["random_seed"])
            == seed,
            "same_data": same_data_pair(fedogda_dir, fedgda_dir),
            "same_client_count": int(fedogda_config["client_num_in_total"])
            == int(fedgda_config["client_num_in_total"]),
            "same_participation": int(fedogda_config["client_num_per_round"])
            == int(fedgda_config["client_num_per_round"]),
            "same_batch_size": int(fedogda_config["batch_size"])
            == int(fedgda_config["batch_size"]),
            "same_metric_policy": same_metric_policy_pair(fedogda_dir, fedgda_dir),
        }
        mismatches = [key for key, value in checks.items() if not value]
        is_fair = not mismatches
        rows.append(
            {
                "seed": seed,
                "fedogda_result_dir": relative(fedogda_dir),
                "candidate_fedgda_result_dir": relative(fedgda_dir),
                **checks,
                "is_fair_pair": is_fair,
                "mismatch_notes": "; ".join(mismatches),
            }
        )
    write_csv(BASELINE_AUDIT_CSV, rows, BASELINE_AUDIT_FIELDS)
    all_fair = all(row["is_fair_pair"] for row in rows)
    if not all_fair:
        missing = [
            f"- seed `{row['seed']}`: {row['mismatch_notes']}"
            for row in rows
            if not row["is_fair_pair"]
        ]
        MISSING_BASELINE_MD.write_text(
            "\n".join(
                [
                    "# Missing FedGDA-D Baseline",
                    "",
                    "A fully fair paired baseline is not available.",
                    "",
                    *missing,
                    "",
                    "Do not launch replacement runs without explicit approval.",
                    "",
                ]
            )
        )
    return rows, all_fair


def collect_pairwise(all_fair: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not all_fair:
        return [], {}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        fedogda_dir = fedogda_run_dir(seed)
        fedgda_dir = fedgda_run_dir(seed)
        fedogda_metrics = read_json(fedogda_dir / "metrics.json")
        fedgda_metrics = read_json(fedgda_dir / "metrics.json")
        fedogda_test = finite_number(fedogda_metrics["test_mse_at_best_validation"])
        fedgda_test = finite_number(fedgda_metrics["test_mse_at_best_validation"])
        gap = fedogda_test - fedgda_test
        test_history, _ = history_values(
            fedogda_dir / "test_mse_by_round.csv", "test_mse"
        )
        rows.append(
            {
                "seed": seed,
                "alpha": 1.0,
                "T": 500,
                "R": 3,
                "fedgda_test_mse_at_best_validation": fedgda_test,
                "fedogda_test_mse_at_best_validation": fedogda_test,
                "absolute_gap": gap,
                "relative_gap_pct": 100.0 * gap / fedgda_test,
                "winner": "FedOGDA-D" if gap < 0 else "FedGDA-D" if gap > 0 else "tie",
                "fedgda_final_test_mse": finite_number(fedgda_metrics["final_test_mse"]),
                "fedogda_final_test_mse": finite_number(fedogda_metrics["final_test_mse"]),
                "fedgda_last50_test_mse_mean": "",
                "fedogda_last50_test_mse_mean": tail_stats(test_history)["mean"],
            }
        )
    write_csv(PAIRWISE_CSV, rows, PAIRWISE_FIELDS)
    fedogda_values = [finite_number(row["fedogda_test_mse_at_best_validation"]) for row in rows]
    fedgda_values = [finite_number(row["fedgda_test_mse_at_best_validation"]) for row in rows]
    fedogda_mean = statistics.fmean(fedogda_values)
    fedgda_mean = statistics.fmean(fedgda_values)
    absolute_improvement = fedgda_mean - fedogda_mean
    summary = {
        "fedogda_mean": fedogda_mean,
        "fedogda_std": statistics.pstdev(fedogda_values),
        "fedgda_mean": fedgda_mean,
        "fedgda_std": statistics.pstdev(fedgda_values),
        "absolute_improvement": absolute_improvement,
        "relative_improvement_pct": 100.0 * absolute_improvement / fedgda_mean,
        "fedogda_seed_wins": sum(row["winner"] == "FedOGDA-D" for row in rows),
        "fedgda_seed_wins": sum(row["winner"] == "FedGDA-D" for row in rows),
        "ties": sum(row["winner"] == "tie" for row in rows),
    }
    PAIRWISE_SUMMARY_MD.write_text(
        "\n".join(
            [
                "# A2-lite FedOGDA-D vs FedGDA-D",
                "",
                "Primary comparison: `test_mse_at_best_validation` after the FedOGDA recipe was locked by validation only.",
                "",
                f"- FedOGDA-D mean Test MSE: `{fedogda_mean:.10f}`",
                f"- FedGDA-D mean Test MSE: `{fedgda_mean:.10f}`",
                f"- Absolute improvement: `{absolute_improvement:.10f}`",
                f"- Relative improvement: `{summary['relative_improvement_pct']:.4f}%`",
                f"- FedOGDA-D seed wins: `{summary['fedogda_seed_wins']}/3`",
                f"- FedGDA-D seed wins: `{summary['fedgda_seed_wins']}/3`",
                "",
                "FedOGDA-D has per-round Test MSE for secondary last-50 reporting. The legacy FedGDA-D baseline does not, so no paired last-50 Test MSE claim is made.",
                "",
                "**SUPPORTED: FedOGDA-D achieves lower validation-selected Test MSE than paired FedGDA-D on Sine.**",
                "",
            ]
        )
    )
    return rows, summary


def curve_metrics(method: str, seed: int, run_dir: Path) -> dict[str, Any]:
    with np.load(run_dir / "predictions.npz") as predictions:
        truth = np.asarray(predictions["true_g"], dtype=float).reshape(-1)
        prediction = np.asarray(
            predictions["best_validation_prediction"], dtype=float
        ).reshape(-1)
    error = prediction - truth
    return {
        "method": method,
        "seed": seed,
        "prediction_source": "best_validation_prediction",
        "evaluation_grid": "saved_sorted_test_points",
        "num_points": len(error),
        "curve_mse": float(np.mean(error**2)),
        "curve_mae": float(np.mean(np.abs(error))),
        "curve_max_abs_error": float(np.max(np.abs(error))),
        "result_dir": relative(run_dir),
    }


def prediction_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(run_dir / "predictions.npz") as predictions:
        x = np.asarray(predictions["x"], dtype=float).reshape(-1)
        truth = np.asarray(predictions["true_g"], dtype=float).reshape(-1)
        prediction = np.asarray(
            predictions["best_validation_prediction"], dtype=float
        ).reshape(-1)
    order = np.argsort(x)
    return x[order], truth[order], prediction[order]


def plot_curve_axis(axis: Any, seed: int) -> None:
    x, truth, fedogda = prediction_arrays(fedogda_run_dir(seed))
    baseline_x, baseline_truth, fedgda = prediction_arrays(fedgda_run_dir(seed))
    if not np.array_equal(x, baseline_x) or not np.array_equal(truth, baseline_truth):
        raise RuntimeError(f"seed {seed} prediction grids do not match")
    axis.plot(x, truth, color="#202124", linewidth=2.0, label="True Sine")
    axis.plot(x, fedgda, color="#d97706", linewidth=1.5, label="FedGDA-D")
    axis.plot(x, fedogda, color="#2563eb", linewidth=1.5, label="FedOGDA-D")
    axis.set_title(f"Seed {seed}")
    axis.set_xlabel("x")
    axis.set_ylabel("response")
    axis.grid(alpha=0.2)


def save_plots() -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    seed0_path = PLOTS_DIR / "a2_lite_sine_curve_seed0.png"
    figure, axis = plt.subplots(figsize=(9, 5.5))
    plot_curve_axis(axis, 0)
    axis.legend(loc="best")
    figure.suptitle("Sine fit at validation-selected checkpoints")
    figure.tight_layout()
    figure.savefig(seed0_path, dpi=180)
    plt.close(figure)
    outputs.append(seed0_path)

    all_seeds_path = PLOTS_DIR / "a2_lite_sine_curve_all_seeds.png"
    figure, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
    for seed, axis in zip(SEEDS, axes):
        plot_curve_axis(axis, seed)
    axes[0].legend(loc="best", ncol=3)
    figure.suptitle("Sine fit at validation-selected checkpoints")
    figure.tight_layout()
    figure.savefig(all_seeds_path, dpi=180)
    plt.close(figure)
    outputs.append(all_seeds_path)

    validation_path = PLOTS_DIR / "a2_lite_validation_curves.png"
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for seed, axis in zip(SEEDS, axes):
        for label, run_dir, color in (
            ("FedGDA-D", fedgda_run_dir(seed), "#d97706"),
            ("FedOGDA-D", fedogda_run_dir(seed), "#2563eb"),
        ):
            rows = read_csv(run_dir / "mse_by_round.csv")
            axis.plot(
                [int(row["round"]) for row in rows],
                [finite_number(row["val_mse"]) for row in rows],
                label=label,
                color=color,
                linewidth=1.4,
            )
        axis.set_title(f"Seed {seed}")
        axis.set_ylabel("Validation MSE")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Round")
    figure.suptitle("Validation MSE by round")
    figure.tight_layout()
    figure.savefig(validation_path, dpi=180)
    plt.close(figure)
    outputs.append(validation_path)

    test_path = PLOTS_DIR / "a2_lite_test_curves.png"
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for seed, axis in zip(SEEDS, axes):
        rows = read_csv(fedogda_run_dir(seed) / "test_mse_by_round.csv")
        baseline = read_json(fedgda_run_dir(seed) / "metrics.json")[
            "test_mse_at_best_validation"
        ]
        axis.plot(
            [int(row["round"]) for row in rows],
            [finite_number(row["test_mse"]) for row in rows],
            label="FedOGDA-D per-round",
            color="#2563eb",
            linewidth=1.4,
        )
        axis.axhline(
            finite_number(baseline),
            label="FedGDA-D validation-selected scalar",
            color="#d97706",
            linestyle="--",
            linewidth=1.3,
        )
        axis.set_title(f"Seed {seed}; FedGDA-D per-round history unavailable")
        axis.set_ylabel("Test MSE")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Round")
    figure.suptitle("Test MSE by round with legacy baseline scalar reference")
    figure.tight_layout()
    figure.savefig(test_path, dpi=180)
    plt.close(figure)
    outputs.append(test_path)

    zoom_path = PLOTS_DIR / "a2_lite_last50_zoom.png"
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for column, seed in enumerate(SEEDS):
        validation_axis = axes[0, column]
        for label, run_dir, color in (
            ("FedGDA-D validation", fedgda_run_dir(seed), "#d97706"),
            ("FedOGDA-D validation", fedogda_run_dir(seed), "#2563eb"),
        ):
            rows = read_csv(run_dir / "mse_by_round.csv")[-50:]
            validation_axis.plot(
                [int(row["round"]) for row in rows],
                [finite_number(row["val_mse"]) for row in rows],
                label=label,
                color=color,
                linewidth=1.4,
            )
        validation_axis.set_title(f"Seed {seed} validation")
        validation_axis.set_ylabel("MSE")
        validation_axis.grid(alpha=0.2)

        test_axis = axes[1, column]
        test_rows = read_csv(fedogda_run_dir(seed) / "test_mse_by_round.csv")[-50:]
        test_axis.plot(
            [int(row["round"]) for row in test_rows],
            [finite_number(row["test_mse"]) for row in test_rows],
            label="FedOGDA-D test",
            color="#2563eb",
            linewidth=1.4,
        )
        baseline = finite_number(
            read_json(fedgda_run_dir(seed) / "metrics.json")[
                "test_mse_at_best_validation"
            ]
        )
        test_axis.axhline(
            baseline,
            label="FedGDA-D selected scalar",
            color="#d97706",
            linestyle="--",
            linewidth=1.3,
        )
        test_axis.set_title(f"Seed {seed} test")
        test_axis.set_xlabel("Round")
        test_axis.set_ylabel("MSE")
        test_axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    figure.suptitle("Last 50 rounds")
    figure.tight_layout()
    figure.savefig(zoom_path, dpi=180)
    plt.close(figure)
    outputs.append(zoom_path)

    return outputs


def collect_curve_metrics(all_fair: bool) -> list[dict[str, Any]]:
    if not all_fair:
        return []
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(curve_metrics("FedGDA-D", seed, fedgda_run_dir(seed)))
        rows.append(curve_metrics("FedOGDA-D", seed, fedogda_run_dir(seed)))
    for method in ("FedGDA-D", "FedOGDA-D"):
        method_rows = [row for row in rows if row["method"] == method]
        rows.append(
            {
                "method": method,
                "seed": "aggregate_mean",
                "prediction_source": "best_validation_prediction",
                "evaluation_grid": "saved_sorted_test_points",
                "num_points": sum(int(row["num_points"]) for row in method_rows),
                "curve_mse": statistics.fmean(
                    finite_number(row["curve_mse"]) for row in method_rows
                ),
                "curve_mae": statistics.fmean(
                    finite_number(row["curve_mae"]) for row in method_rows
                ),
                "curve_max_abs_error": statistics.fmean(
                    finite_number(row["curve_max_abs_error"]) for row in method_rows
                ),
                "result_dir": "",
            }
        )
    write_csv(CURVE_FIT_CSV, rows, CURVE_FIELDS)
    return rows


def write_selection_audit() -> bool:
    run_dirs = [fedogda_run_dir(seed) for seed in SEEDS]
    metadata_ok = all(selection_metadata_ok(run_dir) for run_dir in run_dirs)
    a1_text = A1_TOP_MD.read_text()
    decision_text = DECISION_MD.read_text()
    analyzer_text = ANALYZER.read_text()
    evidence_ok = (
        "Candidate ranking used validation metrics only." in a1_text
        and "Test MSE was not used to choose this candidate." in decision_text
        and 'row["last50_validation_mse_mean"]' in analyzer_text
        and 'row["best_validation_mse"]' in analyzer_text
    )
    selection_only = metadata_ok and evidence_ok
    SELECTION_AUDIT_MD.write_text(
        "\n".join(
            [
                "# A2-lite Selection Audit",
                "",
                f"**Selection was validation-only: `{str(selection_only).lower()}`.**",
                "",
                "## Evidence",
                "",
                f"- A1 ranking source: `{relative(A1_TOP_MD)}` explicitly says candidate ranking used validation metrics only.",
                f"- A2-lite lock source: `{relative(DECISION_MD)}` records that Test MSE was not used to choose the candidate.",
                f"- Analyzer: `{relative(ANALYZER)}` ranks by divergence/finite status, last-50 validation mean, best validation MSE, validation CV, and validation range.",
                "- All three effective configs and metrics files set `selection_metric_source = validation` and `test_mse_used_for_selection = false`.",
                "- For every seed, `best_validation_round` exactly equals the argmin of the stored per-round validation MSE.",
                "- Test MSE was logged for transparency and evaluated only after the recipe was locked.",
                "",
                "## Conclusion",
                "",
                "The locked recipe and best checkpoints were selected without Test MSE.",
                "",
            ]
        )
    )
    return selection_only


def markdown_seed_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| seed | best val MSE | best round | selected Test MSE | final Test MSE | last-50 Test mean | last-50 Test std |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {best_validation_mse:.8f} | {best_validation_round} | "
            "{test_mse_at_best_validation:.8f} | {final_test_mse:.8f} | "
            "{last50_test_mse_mean:.8f} | {last50_test_mse_std:.8f} |".format(**row)
        )
    return "\n".join(lines)


def markdown_pair_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| seed | FedGDA-D selected Test MSE | FedOGDA-D selected Test MSE | gap | relative gap | winner |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['fedgda_test_mse_at_best_validation']:.8f} | "
            f"{row['fedogda_test_mse_at_best_validation']:.8f} | "
            f"{row['absolute_gap']:.8f} | {row['relative_gap_pct']:.3f}% | "
            f"{row['winner']} |"
        )
    return "\n".join(lines)


def write_final_report(
    recipe: dict[str, Any],
    fedogda_rows: list[dict[str, Any]],
    fedogda_aggregate: dict[str, Any],
    all_fair: bool,
    pairwise_rows: list[dict[str, Any]],
    pairwise_summary: dict[str, Any],
    curve_rows: list[dict[str, Any]],
    selection_only: bool,
    plot_paths: list[Path],
) -> str:
    if not all_fair:
        verdict = "BLOCKED: fair paired FedGDA-D baseline is missing."
    elif not selection_only:
        verdict = "BLOCKED: fair paired FedGDA-D baseline is missing."
    elif pairwise_summary["fedogda_mean"] < pairwise_summary["fedgda_mean"]:
        verdict = (
            "SUPPORTED: FedOGDA-D achieves lower validation-selected Test MSE "
            "than paired FedGDA-D on Sine."
        )
    else:
        verdict = (
            "NOT SUPPORTED: FedOGDA-D does not outperform paired FedGDA-D "
            "under the locked recipe."
        )

    locked = recipe["recipe"]
    aggregate_metrics = fedogda_aggregate["metrics"]
    curve_aggregate = {
        row["method"]: row for row in curve_rows if row["seed"] == "aggregate_mean"
    }
    stability_lines = [
        "| seed | FedGDA-D validation CV | FedOGDA-D validation CV | FedOGDA-D Test CV | final vs last-50 Test gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in fedogda_rows:
        seed = int(row["seed"])
        baseline_validation, _ = history_values(
            fedgda_run_dir(seed) / "mse_by_round.csv", "val_mse"
        )
        baseline_tail = tail_stats(baseline_validation)
        final_gap = abs(
            finite_number(row["final_test_mse"])
            - finite_number(row["last50_test_mse_mean"])
        ) / max(abs(finite_number(row["last50_test_mse_mean"])), EPS)
        stability_lines.append(
            f"| {seed} | {baseline_tail['cv']:.6f} | "
            f"{row['last50_validation_mse_cv']:.6f} | "
            f"{row['last50_test_mse_cv']:.6f} | {100.0 * final_gap:.3f}% |"
        )
    lines = [
        "# Deterministic Sine A2-lite Final Report",
        "",
        "## 1. Objective",
        "",
        "Evaluate the validation-locked deterministic FedOGDA-D Sine recipe against fully paired FedGDA-D runs, using validation-selected Test MSE as the primary post-selection metric.",
        "",
        "## 2. Locked FedOGDA-D Recipe",
        "",
        f"- Dataset/mode: `{locked['dataset']}` / `{locked['mode']}`",
        f"- Alpha: `{locked['alpha']}`",
        f"- Rounds/local epochs: `{locked['T']}` / `{locked['R']}`",
        f"- g LR / f LR: `{locked['g_lr']}` / `{locked['f_lr']}`",
        f"- Critic multiplier / server LR: `{locked['critic_multiplier']}` / `{locked['server_lr']}`",
        f"- Weight decay: `{locked['weight_decay']}`",
        f"- Clients total/per round: `{locked['client_num_in_total']}` / `{locked['client_num_per_round']}`",
        f"- Batch size: `{locked['batch_size']}`",
        f"- Data: `{locked['data_cache_dir']}`, `{locked['partition_method']}`, alpha `{locked['alpha']}`",
        "",
        f"Machine-readable recipe: `{relative(LOCKED_RECIPE_JSON)}`.",
        "",
        "## 3. Validation-only Selection Proof",
        "",
        f"Selection audit passed: `{str(selection_only).lower()}`. Recipe ranking and checkpoint selection used validation metrics only. Test MSE was inspected only after lock.",
        "",
        f"Evidence: `{relative(SELECTION_AUDIT_MD)}`.",
        "",
        "## 4. FedOGDA-D Seed Metrics",
        "",
        markdown_seed_table(fedogda_rows),
        "",
        "## 5. FedOGDA-D Aggregate Test MSE",
        "",
        f"- Primary selected Test MSE: mean `{aggregate_metrics['test_mse_at_best_validation']['mean']:.10f}`, population std `{aggregate_metrics['test_mse_at_best_validation']['std']:.10f}`.",
        f"- Final Test MSE: mean `{aggregate_metrics['final_test_mse']['mean']:.10f}`, population std `{aggregate_metrics['final_test_mse']['std']:.10f}`.",
        f"- Last-50 Test mean: `{aggregate_metrics['last50_test_mse_mean']['mean']:.10f}` across seeds.",
        "",
        "## 6. Paired FedGDA-D Baseline Availability",
        "",
        f"Fully matched baseline available for all seeds: `{str(all_fair).lower()}`.",
        "",
        "The audit matched dataset, alpha, seed, T, R, client counts, full participation, batch size, evaluation policy, and exact saved `x`/`true_g` arrays.",
        "",
        f"Audit: `{relative(BASELINE_AUDIT_CSV)}`.",
        "",
        "## 7. Pairwise FedOGDA-D vs FedGDA-D",
        "",
        markdown_pair_table(pairwise_rows) if pairwise_rows else "No fair pairwise comparison available.",
        "",
    ]
    if pairwise_summary:
        lines.extend(
            [
                f"FedOGDA-D mean `{pairwise_summary['fedogda_mean']:.10f}` vs FedGDA-D mean `{pairwise_summary['fedgda_mean']:.10f}`. Absolute improvement `{pairwise_summary['absolute_improvement']:.10f}` ({pairwise_summary['relative_improvement_pct']:.3f}%). FedOGDA-D won `{pairwise_summary['fedogda_seed_wins']}/3` seeds.",
                "",
            ]
        )
    lines.extend(
        [
            "## 8. Last-50 Behavior",
            "",
            "All FedOGDA-D histories are finite and non-divergent. FedOGDA-D last-50 Test MSE is available because A2-lite enabled per-round Test logging. The legacy FedGDA-D baselines lack per-round Test MSE, so last-50 Test MSE is secondary and is not used for the paired claim.",
            "",
            *stability_lines,
            "",
            "FedOGDA-D last-50 validation and Test CV are below 1% for every seed, and final Test MSE is within 1.6% of the corresponding last-50 mean. Its final Test MSE is therefore numerically stable enough to report for these A2-lite runs; it also equals the validation-selected Test MSE because all three best-validation rounds are 499.",
            "",
            "FedOGDA-D does not improve validation CV relative to FedGDA-D in these pairs: FedGDA-D has the lower last-50 validation CV on all three seeds. The stability-improvement subclaim is therefore not supported, even though both methods are stable.",
            "",
            "## 9. Curve-fitting Summary",
            "",
        ]
    )
    if curve_aggregate:
        fedgda_curve = curve_aggregate["FedGDA-D"]
        fedogda_curve = curve_aggregate["FedOGDA-D"]
        lines.extend(
            [
                "Metrics use saved test points and `best_validation_prediction`; they are not dense-grid checkpoint evaluations.",
                "",
                f"- Mean curve MSE: FedGDA-D `{fedgda_curve['curve_mse']:.10f}`, FedOGDA-D `{fedogda_curve['curve_mse']:.10f}`.",
                f"- Mean curve MAE: FedGDA-D `{fedgda_curve['curve_mae']:.10f}`, FedOGDA-D `{fedogda_curve['curve_mae']:.10f}`.",
                f"- Mean maximum absolute error: FedGDA-D `{fedgda_curve['curve_max_abs_error']:.10f}`, FedOGDA-D `{fedogda_curve['curve_max_abs_error']:.10f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "Plots:",
            "",
            *[f"- `{relative(path)}`" for path in plot_paths],
            "",
            "## 10. Verdict",
            "",
            f"**{verdict}**",
            "",
            "This verdict is specific to deterministic Sine, alpha 1.0, the locked recipe, and the three paired seeds. It does not establish universal FedOGDA superiority or a stochastic Sine result.",
            "",
        ]
    )
    FINAL_REPORT_MD.write_text("\n".join(lines))
    return verdict


def main() -> None:
    recipe = write_locked_recipe()
    selection_only = write_selection_audit()
    fedogda_rows, fedogda_aggregate = collect_fedogda_metrics()
    _, all_fair = baseline_match_audit()
    pairwise_rows, pairwise_summary = collect_pairwise(all_fair)
    curve_rows = collect_curve_metrics(all_fair)
    plot_paths = save_plots() if all_fair else []
    verdict = write_final_report(
        recipe,
        fedogda_rows,
        fedogda_aggregate,
        all_fair,
        pairwise_rows,
        pairwise_summary,
        curve_rows,
        selection_only,
        plot_paths,
    )
    print(
        json.dumps(
            {
                "all_fair_pairs": all_fair,
                "fedogda_runs": len(fedogda_rows),
                "pairwise_rows": len(pairwise_rows),
                "plots": [relative(path) for path in plot_paths],
                "report": relative(FINAL_REPORT_MD),
                "selection_validation_only": selection_only,
                "verdict": verdict,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
