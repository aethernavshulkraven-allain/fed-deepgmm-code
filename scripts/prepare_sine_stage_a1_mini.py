#!/usr/bin/env python3
"""Prepare the quota-safe deterministic Sine FedOGDA-D Stage A1-mini grid."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "sine_fedogda_tuning"
CURRENT_RUNS = OUT_DIR / "current_sine_runs.csv"
STAGE_A1_MANIFEST = OUT_DIR / "stage_A1_deterministic_manifest.csv"
ALPHA_CHOICE_MD = OUT_DIR / "stage_A1_mini_alpha_choice.md"
MINI_MANIFEST = OUT_DIR / "stage_A1_mini_deterministic_manifest.csv"
RUNTIME_ESTIMATE_JSON = OUT_DIR / "stage_A1_mini_runtime_estimate.json"
OUTPUT_ROOT = Path("results") / "sine_fedogda_tuning" / "stage_A1_mini_deterministic"
RUN_GROUP = "sine_fedogda_d_stage_A1_mini"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def alpha_token(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def lr_token(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def validation_alpha_summary() -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, str]]] = {}
    for row in read_csv(CURRENT_RUNS):
        if row["method"] != "fedogda_d" or row["mode"] != "deterministic":
            continue
        groups.setdefault(float(row["partition_alpha"]), []).append(row)
    summary = []
    for alpha, rows in sorted(groups.items()):
        summary.append({
            "alpha": alpha,
            "runs": len(rows),
            "mean_last50_validation_mse_mean": statistics.fmean(
                float(row["last50_validation_mse_mean"]) for row in rows
            ),
            "mean_best_validation_mse": statistics.fmean(
                float(row["best_validation_mse"]) for row in rows
            ),
            "mean_last50_validation_mse_std": statistics.fmean(
                float(row["last50_validation_mse_std"]) for row in rows
            ),
        })
    return summary


def select_primary_alpha(summary: list[dict[str, Any]]) -> float:
    if not summary:
        raise SystemExit("No deterministic FedOGDA-D Sine validation rows found")
    ordered = sorted(
        summary,
        key=lambda row: (
            row["mean_last50_validation_mse_mean"],
            row["mean_best_validation_mse"],
            row["mean_last50_validation_mse_std"],
        ),
    )
    return float(ordered[0]["alpha"])


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = [
        "alpha",
        "runs",
        "mean_last50_validation_mse_mean",
        "mean_best_validation_mse",
        "mean_last50_validation_mse_std",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{row[field]:.10g}" if isinstance(row[field], float) else str(row[field])
            for field in fields
        ) + " |")
    return "\n".join(lines)


def write_alpha_choice(summary: list[dict[str, Any]], primary_alpha: float) -> None:
    lines = [
        "# Stage A1-mini Primary Alpha Choice",
        "",
        "Alpha was selected using validation metrics only.",
        "",
        "Test MSE was not used for alpha selection.",
        "",
        "Selection rule: prefer lower mean `last50_validation_mse_mean`; tie-break by lower mean `best_validation_mse`, then lower mean `last50_validation_mse_std`.",
        "",
        markdown_table(summary),
        "",
        f"Selected `primary_alpha = {primary_alpha}` because it has the lowest validation-only ranking under the rule above.",
        "",
    ]
    ALPHA_CHOICE_MD.write_text("\n".join(lines))


def build_mini_manifest(primary_alpha: float) -> list[dict[str, Any]]:
    source_rows = read_csv(STAGE_A1_MANIFEST)
    fields = list(source_rows[0].keys())
    rows = []
    for source in source_rows:
        if float(source["partition_alpha"]) != primary_alpha:
            continue
        if float(source["server_learning_rate"]) != 1.5:
            continue
        if float(source["critic_multiplier"]) not in {10.0, 15.0}:
            continue
        if int(source["epochs"]) not in {2, 3}:
            continue
        if float(source["learning_rate"]) not in {0.0005, 0.001, 0.002}:
            continue
        row = dict(source)
        run_id = (
            f"stage_A1_mini_sin_fedogda_d_seed0_alpha{alpha_token(primary_alpha)}"
            f"_R{row['epochs']}_cm{lr_token(float(row['critic_multiplier']))}"
            f"_slr{lr_token(float(row['server_learning_rate']))}"
            f"_glr{lr_token(float(row['learning_rate']))}"
        )
        row.update({
            "run_id": run_id,
            "run_group": RUN_GROUP,
            "output_root": str(OUTPUT_ROOT),
            "final_result_dir": str(OUTPUT_ROOT / "sin" / "fedogda_d" / "seed_0" / run_id),
            "run_status": "not_started",
            "log_test_mse_by_round": True,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
            "selected_without_test": True,
            "notes": (
                "Stage A1-mini deterministic Sine FedOGDA-D quota-safe grid; "
                "candidate ranking validation-only; Test MSE logged for reporting only."
            ),
        })
        rows.append(row)
    rows.sort(key=lambda row: (
        float(row["partition_alpha"]),
        int(row["epochs"]),
        float(row["critic_multiplier"]),
        float(row["server_learning_rate"]),
        float(row["learning_rate"]),
    ))
    if len(rows) != 12:
        raise SystemExit(f"Expected 12 mini rows, found {len(rows)}")
    write_csv(MINI_MANIFEST, rows, fields)
    return rows


def write_runtime_estimate(row_count: int) -> dict[str, Any]:
    prior_48_gpu_hours = 51.5
    estimated_gpu_hours = prior_48_gpu_hours * row_count / 48.0
    estimate = {
        "basis": "previous_estimate_48_candidates_approximately_51p5_gpu_hours",
        "previous_grid_candidates": 48,
        "previous_grid_gpu_hours": prior_48_gpu_hours,
        "mini_grid_candidates": row_count,
        "estimated_gpu_hours": estimated_gpu_hours,
        "estimated_wall_clock_hours_on_two_gpus": estimated_gpu_hours / 2.0,
        "stop_if_exceeds_gpu_hours": 18.0,
        "safe_to_launch": estimated_gpu_hours <= 18.0,
    }
    RUNTIME_ESTIMATE_JSON.write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n")
    return estimate


def main() -> None:
    summary = validation_alpha_summary()
    primary_alpha = select_primary_alpha(summary)
    write_alpha_choice(summary, primary_alpha)
    rows = build_mini_manifest(primary_alpha)
    estimate = write_runtime_estimate(len(rows))
    print(json.dumps({
        "primary_alpha": primary_alpha,
        "alpha_choice": str(ALPHA_CHOICE_MD.relative_to(REPO_ROOT)),
        "manifest": str(MINI_MANIFEST.relative_to(REPO_ROOT)),
        "runtime_estimate": str(RUNTIME_ESTIMATE_JSON.relative_to(REPO_ROOT)),
        "rows": len(rows),
        **estimate,
    }, indent=2, sort_keys=True))
    if not estimate["safe_to_launch"]:
        raise SystemExit("Runtime estimate exceeds safe threshold")


if __name__ == "__main__":
    main()
