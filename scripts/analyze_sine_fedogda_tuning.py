#!/usr/bin/env python3
"""Audit existing low-dimensional Sine FedGDA/FedOGDA runs.

This script is analysis-only. It reads completed Sine artifacts from the
rerun-protocol result tree and writes the pre-tuning audit files requested for
the Sine FedOGDA tuning workflow. It deliberately does not launch training or
select any hyperparameter by Test MSE.
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
MANIFEST = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv"
OUT_DIR = REPO_ROOT / "experiments" / "sine_fedogda_tuning"
RUNS_CSV = OUT_DIR / "current_sine_runs.csv"
PAIRWISE_CSV = OUT_DIR / "current_sine_pairwise.csv"
SUMMARY_MD = OUT_DIR / "current_sine_summary.md"
LOGGING_AUDIT_MD = OUT_DIR / "mse_logging_audit.md"
METRIC_POLICY_MD = OUT_DIR / "metric_policy.md"

METHODS = {"fedgda_d", "fedgda_s", "fedogda_d", "fedogda_s"}
TEST_COLUMN_CANDIDATES = {
    "test_mse",
    "test_loss",
    "mse_test",
    "test",
    "test_metric",
    "test_mse_by_round",
    "test_mse_at_round",
}
EPS = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return as_float(value)
    except (TypeError, ValueError):
        return None


def numeric_key(value: Any) -> tuple[int, float | str]:
    number = maybe_float(value)
    if number is None:
        return (1, str(value))
    return (0, number)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def mode_for_method(method: str) -> str:
    return "deterministic" if method.endswith("_d") else "stochastic"


def family_for_method(method: str) -> str:
    if "fedogda" in method:
        return "FedOGDA"
    if "fedgda" in method:
        return "FedGDA"
    return "other"


def find_test_column(columns: list[str]) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in TEST_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    for column in columns:
        lower = column.strip().lower()
        if "test" in lower and "mse" in lower:
            return column
    return ""


def last50_stats(values: list[float]) -> dict[str, float]:
    tail = values[-50:]
    tail_mean = mean(tail)
    tail_std = pstdev(tail)
    tail_min = min(tail)
    tail_max = max(tail)
    return {
        "last50_validation_mse_mean": tail_mean,
        "last50_validation_mse_std": tail_std,
        "last50_validation_mse_min": tail_min,
        "last50_validation_mse_max": tail_max,
        "last50_validation_mse_range": tail_max - tail_min,
        "last50_validation_mse_cv": tail_std / max(abs(tail_mean), EPS),
    }


def finite_history(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        if not truthy(row.get("finite", "true")):
            return False
        if truthy(row.get("diverged", "false")):
            return False
        for key, value in row.items():
            if key in {"round", "train_mse", "val_mse", "gmm_train_objective", "gmm_val_objective", "gmm_eval"}:
                try:
                    as_float(value)
                except (TypeError, ValueError):
                    return False
    return True


def checkpoint_rounds(run_dir: Path, best_round: int, final_round: int) -> dict[str, Any]:
    periodic_rounds: set[int] = set()
    for path in (run_dir / "checkpoints").glob("round_*.pt"):
        try:
            periodic_rounds.add(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    tail_start = max(0, final_round - 49)
    available_tail_states = {round_idx for round_idx in periodic_rounds if round_idx >= tail_start}
    available_tail_states.add(final_round)
    if best_round >= tail_start:
        available_tail_states.add(best_round)
    return {
        "periodic_checkpoint_rounds": "|".join(str(x) for x in sorted(periodic_rounds)),
        "tail_start_round": tail_start,
        "last50_available_checkpoint_rounds_upper_bound": "|".join(str(x) for x in sorted(available_tail_states)),
        "last50_available_checkpoint_count_upper_bound": len(available_tail_states),
        "last50_has_full_round_checkpoints": len(available_tail_states) >= min(50, final_round + 1),
    }


def load_sine_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in read_csv(MANIFEST):
        if row.get("training_scope") != "federated":
            continue
        if row.get("method") not in METHODS:
            continue
        if row.get("dataset") not in {"sin", "sine"}:
            continue
        run_dir = REPO_ROOT / row["final_result_dir"]
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "effective_config.json"
        history_path = run_dir / "mse_by_round.csv"
        predictions_path = run_dir / "predictions.npz"
        if not (metrics_path.exists() and config_path.exists() and history_path.exists()):
            continue

        metrics = read_json(metrics_path)
        config = read_json(config_path)
        history_rows = read_csv(history_path)
        columns = list(history_rows[0].keys()) if history_rows else []
        val_values = [as_float(item["val_mse"]) for item in history_rows]
        final_round = int(as_float(history_rows[-1]["round"])) if history_rows else -1
        best_round = int(metrics["best_validation_round"])
        ckpt_info = checkpoint_rounds(run_dir, best_round, final_round)
        stats = last50_stats(val_values)
        test_column = find_test_column(columns)

        run = {
            "run_id": row["run_id"],
            "dataset": config.get("dataset", row.get("dataset", "")),
            "dataset_label": "sine",
            "variant": config.get("variant", row["method"]),
            "method": row["method"],
            "method_family": family_for_method(row["method"]),
            "mode": mode_for_method(row["method"]),
            "seed": int(row["seed"]),
            "partition_alpha": as_float(row["partition_alpha"]),
            "client_num_in_total": int(config.get("client_num_in_total", row["client_num_in_total"])),
            "client_num_per_round": int(config.get("client_num_per_round", row["client_num_per_round"])),
            "batch_size": int(config.get("batch_size", row["batch_size"])),
            "comm_rounds": int(config.get("comm_round", row["comm_round"])),
            "local_epochs": int(config.get("local_epochs", config.get("epochs", row["epochs"]))),
            "g_lr": as_float(config.get("g_learning_rate", config.get("learning_rate"))),
            "f_lr": as_float(config.get("f_learning_rate", as_float(config.get("learning_rate")) * as_float(config.get("critic_multiplier")))),
            "critic_multiplier": as_float(config.get("critic_multiplier", row.get("critic_multiplier", 0.0))),
            "server_lr": as_float(config.get("server_learning_rate", row.get("server_learning_rate", 0.0))),
            "weight_decay": as_float(config.get("weight_decay", row.get("weight_decay", 0.0))),
            "best_validation_round": best_round,
            "best_validation_mse": as_float(metrics["best_validation_mse"]),
            "test_mse_at_best_validation": as_float(metrics["test_mse_at_best_validation"]),
            "final_validation_mse": as_float(metrics["final_validation_mse"]),
            "final_test_mse": as_float(metrics["final_test_mse"]),
            "diverged": bool(metrics.get("diverged", False)),
            "finite_history": finite_history(history_rows),
            "history_round_count": len(history_rows),
            "mse_by_round_columns": "|".join(columns),
            "per_round_test_mse_column": test_column,
            "has_per_round_test_mse": bool(test_column),
            "predictions_npz_exists": predictions_path.exists(),
            "result_dir": str(Path(row["final_result_dir"])),
        }
        run.update(stats)
        run.update({
            "stable_validation_cv_le_0p05": run["last50_validation_mse_cv"] <= 0.05,
            "stable_validation_range_le_1e_4": run["last50_validation_mse_range"] <= 1e-4,
        })
        run.update(ckpt_info)
        runs.append(run)
    return sorted(runs, key=lambda item: (item["mode"], item["partition_alpha"], item["seed"], item["method"]))


def make_pairs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(run["mode"], run["partition_alpha"], run["seed"], run["method"]): run for run in runs}
    pairs: list[dict[str, Any]] = []
    for mode, fedgda_method, fedogda_method in [
        ("deterministic", "fedgda_d", "fedogda_d"),
        ("stochastic", "fedgda_s", "fedogda_s"),
    ]:
        keys = sorted(
            {
                (run["partition_alpha"], run["seed"])
                for run in runs
                if run["mode"] == mode and run["method"] in {fedgda_method, fedogda_method}
            },
            key=lambda item: (item[0], item[1]),
        )
        for alpha, seed in keys:
            fedgda = by_key.get((mode, alpha, seed, fedgda_method))
            fedogda = by_key.get((mode, alpha, seed, fedogda_method))
            if not fedgda or not fedogda:
                continue
            gap = fedogda["test_mse_at_best_validation"] - fedgda["test_mse_at_best_validation"]
            final_gap = fedogda["final_test_mse"] - fedgda["final_test_mse"]
            val_gap = fedogda["best_validation_mse"] - fedgda["best_validation_mse"]
            std_gap = fedogda["last50_validation_mse_std"] - fedgda["last50_validation_mse_std"]
            cv_gap = fedogda["last50_validation_mse_cv"] - fedgda["last50_validation_mse_cv"]
            pairs.append({
                "dataset_label": "sine",
                "dataset": fedgda["dataset"],
                "mode": mode,
                "alpha": alpha,
                "seed": seed,
                "fedgda_run_id": fedgda["run_id"],
                "fedogda_run_id": fedogda["run_id"],
                "fedgda_variant": fedgda["variant"],
                "fedogda_variant": fedogda["variant"],
                "fedgda_test_mse_at_best_validation": fedgda["test_mse_at_best_validation"],
                "fedogda_test_mse_at_best_validation": fedogda["test_mse_at_best_validation"],
                "absolute_gap": gap,
                "relative_gap_pct": 100.0 * gap / max(abs(fedgda["test_mse_at_best_validation"]), EPS),
                "winner": "FedOGDA" if gap < 0 else "FedGDA" if gap > 0 else "tie",
                "fedgda_final_test_mse": fedgda["final_test_mse"],
                "fedogda_final_test_mse": fedogda["final_test_mse"],
                "final_test_absolute_gap": final_gap,
                "final_test_winner": "FedOGDA" if final_gap < 0 else "FedGDA" if final_gap > 0 else "tie",
                "fedgda_best_validation_mse": fedgda["best_validation_mse"],
                "fedogda_best_validation_mse": fedogda["best_validation_mse"],
                "best_validation_absolute_gap": val_gap,
                "fedogda_lower_best_validation_mse": val_gap < 0,
                "fedgda_last50_validation_mse_mean": fedgda["last50_validation_mse_mean"],
                "fedogda_last50_validation_mse_mean": fedogda["last50_validation_mse_mean"],
                "last50_validation_mean_gap": fedogda["last50_validation_mse_mean"] - fedgda["last50_validation_mse_mean"],
                "fedgda_last50_validation_mse_std": fedgda["last50_validation_mse_std"],
                "fedogda_last50_validation_mse_std": fedogda["last50_validation_mse_std"],
                "last50_validation_std_gap": std_gap,
                "fedogda_lower_last50_validation_std": std_gap < 0,
                "fedgda_last50_validation_mse_cv": fedgda["last50_validation_mse_cv"],
                "fedogda_last50_validation_mse_cv": fedogda["last50_validation_mse_cv"],
                "last50_validation_cv_gap": cv_gap,
                "fedogda_lower_last50_validation_cv": cv_gap < 0,
                "has_per_round_test_mse_pair": fedgda["has_per_round_test_mse"] and fedogda["has_per_round_test_mse"],
                "matched_comm_rounds": fedgda["comm_rounds"] == fedogda["comm_rounds"],
                "matched_local_epochs": fedgda["local_epochs"] == fedogda["local_epochs"],
                "matched_client_policy": (
                    fedgda["client_num_in_total"] == fedogda["client_num_in_total"]
                    and fedgda["client_num_per_round"] == fedogda["client_num_per_round"]
                    and fedgda["batch_size"] == fedogda["batch_size"]
                ),
            })
    return pairs


def group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: tuple(numeric_key(x) for x in item[0])):
        entry = {key: value for key, value in zip(keys, key_values)}
        entry.update({
            "runs": len(group),
            "mean_best_validation_mse": mean([row["best_validation_mse"] for row in group]),
            "mean_test_mse_at_best_validation": mean([row["test_mse_at_best_validation"] for row in group]),
            "std_test_mse_at_best_validation": pstdev([row["test_mse_at_best_validation"] for row in group]),
            "mean_final_test_mse": mean([row["final_test_mse"] for row in group]),
            "mean_last50_validation_mse_mean": mean([row["last50_validation_mse_mean"] for row in group]),
            "mean_last50_validation_mse_std": mean([row["last50_validation_mse_std"] for row in group]),
            "stable_cv_le_0p05_runs": sum(bool(row["stable_validation_cv_le_0p05"]) for row in group),
            "diverged_runs": sum(bool(row["diverged"]) for row in group),
        })
        summary.append(entry)
    return summary


def pair_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: tuple(numeric_key(x) for x in item[0])):
        entry = {key: value for key, value in zip(keys, key_values)}
        entry.update({
            "pairs": len(group),
            "fedogda_lower_test_mse_at_best_validation": sum(row["winner"] == "FedOGDA" for row in group),
            "fedogda_lower_final_test_mse": sum(row["final_test_winner"] == "FedOGDA" for row in group),
            "fedogda_lower_best_validation_mse": sum(bool(row["fedogda_lower_best_validation_mse"]) for row in group),
            "fedogda_lower_last50_validation_std": sum(bool(row["fedogda_lower_last50_validation_std"]) for row in group),
            "fedogda_lower_last50_validation_cv": sum(bool(row["fedogda_lower_last50_validation_cv"]) for row in group),
            "mean_test_mse_gap_fedogda_minus_fedgda": mean([row["absolute_gap"] for row in group]),
            "mean_relative_gap_pct": mean([row["relative_gap_pct"] for row in group]),
        })
        summary.append(entry)
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.8g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_summary(runs: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    method_alpha = group_summary(runs, ("mode", "method", "partition_alpha"))
    mode_summary = pair_summary(pairs, ("mode",))
    alpha_summary = pair_summary(pairs, ("alpha",))
    mode_alpha = pair_summary(pairs, ("mode", "alpha"))
    no_test_curve = sum(not row["has_per_round_test_mse"] for row in runs)
    diverged = sum(bool(row["diverged"]) for row in runs)

    lines = [
        "# Current Low-Dimensional Sine FedGDA/FedOGDA Audit",
        "",
        "Scope: completed federated Sine rows from `experiments/rerun_protocol_v1/manifest.csv` with methods `fedgda_d`, `fedgda_s`, `fedogda_d`, and `fedogda_s`.",
        "",
        "This is a pre-tuning audit only. No hyperparameter selection was made using Test MSE, and no new training was launched.",
        "",
        "## Artifact Count",
        "",
        f"- Completed Sine runs found: `{len(runs)}`.",
        f"- Diverged runs: `{diverged}`.",
        f"- Runs missing per-round Test MSE: `{no_test_curve}`.",
        f"- Paired FedOGDA-vs-FedGDA comparisons: `{len(pairs)}`.",
        "",
        "## Existing Test-MSE Baseline",
        "",
        "The table below uses scalar `test_mse_at_best_validation` from `metrics.json`. This is valid for reporting after validation-only checkpoint selection, but it is not a last-50 Test-MSE curve.",
        "",
        markdown_table(
            method_alpha,
            [
                "mode",
                "method",
                "partition_alpha",
                "runs",
                "mean_best_validation_mse",
                "mean_test_mse_at_best_validation",
                "std_test_mse_at_best_validation",
                "mean_final_test_mse",
                "mean_last50_validation_mse_std",
                "stable_cv_le_0p05_runs",
            ],
        ),
        "",
        "## Paired Summary By Mode",
        "",
        markdown_table(
            mode_summary,
            [
                "mode",
                "pairs",
                "fedogda_lower_test_mse_at_best_validation",
                "fedogda_lower_final_test_mse",
                "fedogda_lower_best_validation_mse",
                "fedogda_lower_last50_validation_std",
                "fedogda_lower_last50_validation_cv",
                "mean_test_mse_gap_fedogda_minus_fedgda",
                "mean_relative_gap_pct",
            ],
        ),
        "",
        "## Paired Summary By Alpha",
        "",
        markdown_table(
            alpha_summary,
            [
                "alpha",
                "pairs",
                "fedogda_lower_test_mse_at_best_validation",
                "fedogda_lower_final_test_mse",
                "fedogda_lower_last50_validation_std",
                "mean_test_mse_gap_fedogda_minus_fedgda",
            ],
        ),
        "",
        "## Paired Summary By Mode And Alpha",
        "",
        markdown_table(
            mode_alpha,
            [
                "mode",
                "alpha",
                "pairs",
                "fedogda_lower_test_mse_at_best_validation",
                "fedogda_lower_final_test_mse",
                "fedogda_lower_last50_validation_std",
                "fedogda_lower_last50_validation_cv",
                "mean_test_mse_gap_fedogda_minus_fedgda",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Existing deterministic Sine FedOGDA-D is close to FedGDA-D but does not clearly win by `test_mse_at_best_validation`.",
        "- Existing stochastic Sine FedOGDA-S is more stable by last-50 validation oscillation, but loses to FedGDA-S on scalar `test_mse_at_best_validation` in the current runs.",
        "- Per-round Test MSE is not present in existing `mse_by_round.csv`, so the current artifacts cannot answer whether last-50 average Test MSE favors FedOGDA.",
        "- Because the requested stop condition is met, this audit does not launch Sine tuning runs.",
        "",
        "## Output Files",
        "",
        f"- Per-run audit: `{RUNS_CSV.relative_to(REPO_ROOT)}`",
        f"- Pairwise audit: `{PAIRWISE_CSV.relative_to(REPO_ROOT)}`",
        f"- MSE logging audit: `{LOGGING_AUDIT_MD.relative_to(REPO_ROOT)}`",
        f"- Metric policy: `{METRIC_POLICY_MD.relative_to(REPO_ROOT)}`",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines))


def write_logging_audit(runs: list[dict[str, Any]]) -> None:
    columns = sorted({row["mse_by_round_columns"] for row in runs})
    has_test_count = sum(bool(row["has_per_round_test_mse"]) for row in runs)
    full_tail_ckpts = sum(bool(row["last50_has_full_round_checkpoints"]) for row in runs)
    max_tail_ckpts = max(int(row["last50_available_checkpoint_count_upper_bound"]) for row in runs)
    min_tail_ckpts = min(int(row["last50_available_checkpoint_count_upper_bound"]) for row in runs)

    lines = [
        "# Sine MSE Logging Audit",
        "",
        "This audit inspects existing completed Sine artifacts only. No training logic was changed.",
        "",
        "## Answers",
        "",
        "1. Is per-round Test MSE available for existing runs?",
        "",
        f"No. Per-round Test MSE is available in `{has_test_count}/{len(runs)}` existing Sine runs.",
        "",
        "Observed `mse_by_round.csv` column sets:",
        "",
    ]
    lines.extend(f"- `{column}`" for column in columns)
    lines.extend([
        "",
        "2. If yes, can we compute average Test MSE over last 50 rounds?",
        "",
        "No. Because per-round Test MSE is absent, the last-50 average Test MSE cannot be computed from the existing runs.",
        "",
        "3. If no, what is available?",
        "",
        "- Per-round `train_mse` and `val_mse` in `mse_by_round.csv`.",
        "- Scalar `test_mse_at_best_validation` and `final_test_mse` in `metrics.json`.",
        "- Test-point predictions for `best_validation_prediction` and `final_prediction` in `predictions.npz`.",
        "- Checkpoints for `best_validation.pt`, `final.pt`, and sparse periodic checkpoints such as `round_0.pt`, `round_200.pt`, and `round_400.pt`.",
        "",
        "4. Can last-50 validation MSE be used for tuning?",
        "",
        "Yes. Last-50 validation MSE is available for every completed Sine run and is validation-only, so it can be used for tuning and stability ranking without touching Test MSE.",
        "",
        "5. Can future runs safely log per-round Test MSE without using it for tuning?",
        "",
        "Yes, if the workflow treats it as a reporting-only diagnostic after the recipe is locked by validation. The selection code/report must not read or rank candidates by per-round Test MSE.",
        "",
        "6. Are there enough round checkpoints in the last 50 rounds to reconstruct last-50 Test MSE?",
        "",
        f"No. Across existing Sine runs, the upper-bound count of available last-50 checkpoint states ranges from `{min_tail_ckpts}` to `{max_tail_ckpts}`; runs with all last-50 model states available: `{full_tail_ckpts}/{len(runs)}`.",
        "",
        "## Required Policy Because Test Curve Is Missing",
        "",
        "- Tuning metric: `last50_validation_mse_mean`, with `best_validation_mse` as fallback/tie-break.",
        "- Stability diagnostics: last-50 validation mean/std/range/CV.",
        "- Final reporting from current artifacts: `test_mse_at_best_validation` and `final_test_mse` only.",
        "- Optional future reporting: `last50_test_mse_mean` only for future runs that log per-round Test MSE, and only after validation-only recipe lock.",
        "",
        "## Stop Condition",
        "",
        "The requested stop condition is met: per-round Test MSE is unavailable and last-50 Test MSE cannot be computed from existing Sine artifacts. Therefore no Sine tuning runs were launched by this audit.",
        "",
    ])
    LOGGING_AUDIT_MD.write_text("\n".join(lines))


def write_metric_policy() -> None:
    lines = [
        "# Sine FedOGDA Tuning Metric Policy",
        "",
        "This policy is predeclared before any new Sine tuning launch. Hyperparameter choices must be validation-only.",
        "",
        "## Primary Tuning Metric",
        "",
        "Use `last50_validation_mse_mean` when per-round validation MSE is logged, as it is for the existing Sine runs.",
        "",
        "Fallback only if last-50 validation averaging is not meaningful: `best_validation_mse`.",
        "",
        "Candidate ranking:",
        "",
        "1. lower `last50_validation_mse_mean`;",
        "2. lower `best_validation_mse`;",
        "3. lower `last50_validation_mse_cv`;",
        "4. no divergence and finite history;",
        "5. lower `last50_validation_mse_range`.",
        "",
        "## Stability Metrics",
        "",
        "Compute from validation MSE over the last 50 rounds:",
        "",
        "- `last50_validation_mse_mean`",
        "- `last50_validation_mse_std`",
        "- `last50_validation_mse_min`",
        "- `last50_validation_mse_max`",
        "- `last50_validation_mse_range`",
        "- `last50_validation_mse_cv = std / max(abs(mean), 1e-12)`",
        "",
        "Stable validation behavior is defined as either:",
        "",
        "- `last50_validation_mse_cv <= 0.05`; or",
        "- `last50_validation_mse_range <= 1e-4` when means are very small.",
        "",
        "Record both criteria; do not hide instability.",
        "",
        "## Test Reporting Metrics",
        "",
        "After a recipe is locked by validation, report:",
        "",
        "- `test_mse_at_best_validation`",
        "- `final_test_mse`",
        "- `last50_test_mse_mean`, only if per-round Test MSE exists",
        "- `last50_test_mse_std`, only if per-round Test MSE exists",
        "",
        "Do not choose the primary reported Test metric after seeing which one favors FedOGDA.",
        "",
        "Predeclared reporting rule:",
        "",
        "- If validation curves are stable, final Test MSE may be reported.",
        "- If validation curves are oscillatory, report last-50 average Test MSE only when available, plus validation-selected Test MSE.",
        "- Always include `test_mse_at_best_validation` for paired comparison.",
        "",
    ]
    METRIC_POLICY_MD.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_sine_runs()
    pairs = make_pairs(runs)

    run_fields = [
        "run_id",
        "dataset",
        "dataset_label",
        "variant",
        "method",
        "method_family",
        "mode",
        "seed",
        "partition_alpha",
        "client_num_in_total",
        "client_num_per_round",
        "batch_size",
        "comm_rounds",
        "local_epochs",
        "g_lr",
        "f_lr",
        "critic_multiplier",
        "server_lr",
        "weight_decay",
        "best_validation_round",
        "best_validation_mse",
        "test_mse_at_best_validation",
        "final_validation_mse",
        "final_test_mse",
        "diverged",
        "finite_history",
        "history_round_count",
        "last50_validation_mse_mean",
        "last50_validation_mse_std",
        "last50_validation_mse_min",
        "last50_validation_mse_max",
        "last50_validation_mse_range",
        "last50_validation_mse_cv",
        "stable_validation_cv_le_0p05",
        "stable_validation_range_le_1e_4",
        "mse_by_round_columns",
        "per_round_test_mse_column",
        "has_per_round_test_mse",
        "predictions_npz_exists",
        "periodic_checkpoint_rounds",
        "tail_start_round",
        "last50_available_checkpoint_rounds_upper_bound",
        "last50_available_checkpoint_count_upper_bound",
        "last50_has_full_round_checkpoints",
        "result_dir",
    ]
    pair_fields = [
        "dataset_label",
        "dataset",
        "mode",
        "alpha",
        "seed",
        "fedgda_run_id",
        "fedogda_run_id",
        "fedgda_variant",
        "fedogda_variant",
        "fedgda_test_mse_at_best_validation",
        "fedogda_test_mse_at_best_validation",
        "absolute_gap",
        "relative_gap_pct",
        "winner",
        "fedgda_final_test_mse",
        "fedogda_final_test_mse",
        "final_test_absolute_gap",
        "final_test_winner",
        "fedgda_best_validation_mse",
        "fedogda_best_validation_mse",
        "best_validation_absolute_gap",
        "fedogda_lower_best_validation_mse",
        "fedgda_last50_validation_mse_mean",
        "fedogda_last50_validation_mse_mean",
        "last50_validation_mean_gap",
        "fedgda_last50_validation_mse_std",
        "fedogda_last50_validation_mse_std",
        "last50_validation_std_gap",
        "fedogda_lower_last50_validation_std",
        "fedgda_last50_validation_mse_cv",
        "fedogda_last50_validation_mse_cv",
        "last50_validation_cv_gap",
        "fedogda_lower_last50_validation_cv",
        "has_per_round_test_mse_pair",
        "matched_comm_rounds",
        "matched_local_epochs",
        "matched_client_policy",
    ]

    write_csv(RUNS_CSV, runs, run_fields)
    write_csv(PAIRWISE_CSV, pairs, pair_fields)
    write_summary(runs, pairs)
    write_logging_audit(runs)
    write_metric_policy()

    print(json.dumps({
        "runs": len(runs),
        "pairs": len(pairs),
        "per_round_test_mse_runs": sum(bool(row["has_per_round_test_mse"]) for row in runs),
        "outputs": [
            str(RUNS_CSV.relative_to(REPO_ROOT)),
            str(PAIRWISE_CSV.relative_to(REPO_ROOT)),
            str(SUMMARY_MD.relative_to(REPO_ROOT)),
            str(LOGGING_AUDIT_MD.relative_to(REPO_ROOT)),
            str(METRIC_POLICY_MD.relative_to(REPO_ROOT)),
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
