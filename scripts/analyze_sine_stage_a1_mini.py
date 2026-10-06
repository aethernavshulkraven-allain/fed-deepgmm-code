#!/usr/bin/env python3
"""Analyze deterministic Sine FedOGDA-D Stage A1-mini tuning results."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "sine_fedogda_tuning"
MANIFEST = OUT_DIR / "stage_A1_mini_deterministic_manifest.csv"
RESULTS_CSV = OUT_DIR / "stage_A1_mini_results.csv"
TOP_MD = OUT_DIR / "stage_A1_mini_top_candidates.md"
A2_MANIFEST = OUT_DIR / "stage_A2_from_A1_mini_manifest.csv"
EXISTING_RUNS = OUT_DIR / "current_sine_runs.csv"


RESULT_FIELDS = [
    "rank_validation_only",
    "run_id",
    "alpha",
    "seed",
    "T",
    "R",
    "g_lr",
    "f_lr",
    "critic_multiplier",
    "server_lr",
    "best_validation_mse",
    "best_validation_round",
    "last50_validation_mse_mean",
    "last50_validation_mse_std",
    "last50_validation_mse_cv",
    "last50_validation_mse_range",
    "test_mse_at_best_validation",
    "final_test_mse",
    "last50_test_mse_mean",
    "last50_test_mse_std",
    "last50_test_mse_min",
    "last50_test_mse_max",
    "last50_test_mse_range",
    "last50_test_mse_cv",
    "last50_test_mse_status",
    "diverged",
    "finite_history",
    "validated",
    "result_dir",
]


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


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def last50(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"last50_{prefix}_mean": "",
            f"last50_{prefix}_std": "",
            f"last50_{prefix}_min": "",
            f"last50_{prefix}_max": "",
            f"last50_{prefix}_range": "",
            f"last50_{prefix}_cv": "",
        }
    tail = values[-50:]
    tail_mean = mean(tail)
    tail_std = pstdev(tail)
    tail_min = min(tail)
    tail_max = max(tail)
    return {
        f"last50_{prefix}_mean": tail_mean,
        f"last50_{prefix}_std": tail_std,
        f"last50_{prefix}_min": tail_min,
        f"last50_{prefix}_max": tail_max,
        f"last50_{prefix}_range": tail_max - tail_min,
        f"last50_{prefix}_cv": tail_std / max(abs(tail_mean), 1e-12),
    }


def finite_history(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        if not truthy(row.get("finite", "true")) or truthy(row.get("diverged", "false")):
            return False
        for key in ("round", "train_mse", "val_mse", "gmm_train_objective", "gmm_val_objective", "gmm_eval"):
            try:
                as_float(row[key])
            except (KeyError, TypeError, ValueError):
                return False
    return True


def test_history(run_dir: Path) -> tuple[str, dict[str, Any]]:
    path = run_dir / "test_mse_by_round.csv"
    if not path.exists():
        return "unavailable", last50([], "test_mse")
    rows = read_csv(path)
    values = []
    for row in rows:
        if not truthy(row.get("finite", "true")) or truthy(row.get("diverged", "false")):
            return "nonfinite_or_diverged", last50([], "test_mse")
        values.append(as_float(row["test_mse"]))
    return "available", last50(values, "test_mse")


def existing_fedgda_baseline_exists(alpha: float, R: int) -> bool:
    seeds = set()
    for row in read_csv(EXISTING_RUNS):
        if row["method"] != "fedgda_d" or row["mode"] != "deterministic":
            continue
        if abs(float(row["partition_alpha"]) - alpha) > 1e-12:
            continue
        if int(row["comm_rounds"]) != 500 or int(row["local_epochs"]) != R:
            continue
        seeds.add(int(row["seed"]))
    return seeds == {0, 1, 2}


def completed_rows() -> list[dict[str, Any]]:
    rows = []
    for item in read_csv(MANIFEST):
        run_dir = REPO_ROOT / item["final_result_dir"]
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "mse_by_round.csv"
        config_path = run_dir / "effective_config.json"
        if not (metrics_path.exists() and history_path.exists() and config_path.exists()):
            continue
        metrics = read_json(metrics_path)
        config = read_json(config_path)
        history = read_csv(history_path)
        val_values = [as_float(row["val_mse"]) for row in history]
        val_stats = last50(val_values, "validation_mse")
        test_status, test_stats = test_history(run_dir)
        config_selection_ok = (
            not bool(config.get("test_mse_used_for_selection", False))
            and str(config.get("selection_metric_source", "validation")).lower() == "validation"
        )
        metrics_selection_ok = (
            not bool(metrics.get("test_mse_used_for_selection", False))
            and str(metrics.get("selection_metric_source", "validation")).lower() == "validation"
        )
        row = {
            "run_id": item["run_id"],
            "alpha": as_float(item["partition_alpha"]),
            "seed": int(item["seed"]),
            "T": int(config["comm_round"]),
            "R": int(config["local_epochs"]),
            "g_lr": as_float(config["g_learning_rate"]),
            "f_lr": as_float(config["f_learning_rate"]),
            "critic_multiplier": as_float(config["critic_multiplier"]),
            "server_lr": as_float(config["server_learning_rate"]),
            "best_validation_mse": as_float(metrics["best_validation_mse"]),
            "best_validation_round": int(metrics["best_validation_round"]),
            "test_mse_at_best_validation": as_float(metrics["test_mse_at_best_validation"]),
            "final_test_mse": as_float(metrics["final_test_mse"]),
            "last50_test_mse_status": test_status,
            "diverged": bool(metrics.get("diverged", False)),
            "finite_history": finite_history(history),
            "validated": config_selection_ok and metrics_selection_ok and test_status == "available",
            "result_dir": item["final_result_dir"],
        }
        row.update(val_stats)
        row.update(test_stats)
        rows.append(row)
    rows.sort(key=lambda row: (
        bool(row["diverged"]),
        not bool(row["finite_history"]),
        row["last50_validation_mse_mean"],
        row["best_validation_mse"],
        row["last50_validation_mse_cv"],
        row["last50_validation_mse_range"],
    ))
    for index, row in enumerate(rows, start=1):
        row["rank_validation_only"] = index
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "No completed candidates found."
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        formatted = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                formatted.append(f"{value:.8g}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def make_a2_manifest(top_rows: list[dict[str, Any]]) -> None:
    if len(top_rows) < 3:
        return
    source_rows = {row["run_id"]: row for row in read_csv(MANIFEST)}
    fields = list(next(iter(source_rows.values())).keys())
    out_rows = []
    for candidate in top_rows[:3]:
        for seed in [0, 1, 2]:
            source = dict(source_rows[candidate["run_id"]])
            alpha_token = str(candidate["alpha"]).replace(".", "p")
            g_lr_token = f"{candidate['g_lr']:.8g}".replace(".", "p")
            run_id = (
                f"stage_A2_from_A1_mini_sin_fedogda_d_seed{seed}_alpha{alpha_token}"
                f"_R{candidate['R']}_cm{candidate['critic_multiplier']:.8g}"
                f"_slr{candidate['server_lr']:.8g}_glr{g_lr_token}"
            )
            baseline_exists = existing_fedgda_baseline_exists(candidate["alpha"], candidate["R"])
            if baseline_exists:
                baseline_note = "matched FedGDA-D baseline exists for same alpha/T/R/seeds"
            else:
                baseline_note = "paired FedGDA-D baseline will need to be run later for fairness"
            source.update({
                "run_id": run_id,
                "run_group": "sine_fedogda_d_stage_A2_from_A1_mini",
                "seed": seed,
                "comm_round": 500,
                "output_root": str(Path("results") / "sine_fedogda_tuning" / "stage_A2_from_A1_mini"),
                "run_status": "not_started",
                "final_result_dir": str(
                    Path("results") / "sine_fedogda_tuning" / "stage_A2_from_A1_mini"
                    / "sin" / "fedogda_d" / f"seed_{seed}" / run_id
                ),
                "notes": (
                    "Prepared from validation-only Stage A1-mini top candidates; "
                    "do not launch without explicit instruction; "
                    f"{baseline_note}."
                ),
            })
            out_rows.append(source)
    write_csv(A2_MANIFEST, out_rows, fields)


def write_top_markdown(rows: list[dict[str, Any]]) -> None:
    top = rows[:3]
    fields = [
        "rank_validation_only",
        "run_id",
        "alpha",
        "R",
        "g_lr",
        "critic_multiplier",
        "server_lr",
        "last50_validation_mse_mean",
        "best_validation_mse",
        "last50_validation_mse_cv",
        "test_mse_at_best_validation",
        "last50_test_mse_mean",
        "last50_test_mse_status",
        "diverged",
        "finite_history",
        "validated",
    ]
    lines = [
        "# Stage A1-mini Deterministic Sine Top Candidates",
        "",
        "Candidate ranking used validation metrics only.",
        "",
        "Test MSE columns were reported for transparency but were not used for selection.",
        "",
        f"Completed candidates analyzed: `{len(rows)}`.",
        "",
        markdown_table(top, fields),
        "",
        "Ranking key:",
        "",
        "1. no divergence;",
        "2. finite history;",
        "3. lower `last50_validation_mse_mean`;",
        "4. lower `best_validation_mse`;",
        "5. lower `last50_validation_mse_cv`;",
        "6. lower `last50_validation_mse_range`.",
        "",
    ]
    if top:
        lines.extend(["## FedGDA-D Baseline Fairness Notes", ""])
        for row in top:
            exists = existing_fedgda_baseline_exists(row["alpha"], row["R"])
            status = "exists" if exists else "must be run later"
            lines.append(f"- Rank {row['rank_validation_only']} candidate `R={row['R']}`, alpha `{row['alpha']}`: matched FedGDA-D baseline {status}.")
        lines.append("")
    TOP_MD.write_text("\n".join(lines))


def main() -> None:
    rows = completed_rows()
    write_csv(RESULTS_CSV, rows, RESULT_FIELDS)
    write_top_markdown(rows)
    make_a2_manifest(rows[:3])
    print(json.dumps({
        "completed_candidates": len(rows),
        "results_csv": str(RESULTS_CSV.relative_to(REPO_ROOT)),
        "top_candidates_md": str(TOP_MD.relative_to(REPO_ROOT)),
        "stage_A2_manifest": str(A2_MANIFEST.relative_to(REPO_ROOT)) if A2_MANIFEST.exists() else "",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
