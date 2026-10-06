#!/usr/bin/env python3
"""Aggregate the completed 120-run high-dimensional experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
MANIFEST = PROTOCOL_DIR / "final_manifest.csv"
ANALYSIS_DIR = PROTOCOL_DIR / "analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def value(raw: Any, name: str) -> float:
    number = float(raw)
    if not math.isfinite(number):
        raise ValueError(f"{name} is non-finite: {raw!r}")
    return number


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def load_runs(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    runs = []
    for row in manifest_rows:
        run_dir = REPO_ROOT / row["final_result_dir"]
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = load_json(metrics_path)
        runs.append({
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "method": row["method"],
            "seed": int(row["seed"]),
            "alpha": value(row["alpha"], "alpha"),
            "g_function": "abs",
            "learning_rate": value(row["learning_rate"], "learning_rate"),
            "weight_decay": value(row["weight_decay"], "weight_decay"),
            "critic_multiplier": value(row["critic_multiplier"], "critic_multiplier"),
            "best_validation_mse": value(metrics["best_validation_mse"], "best_validation_mse"),
            "best_validation_round": int(metrics["best_validation_round"]),
            "test_mse_at_best_validation": value(
                metrics["test_mse_at_best_validation"], "test_mse_at_best_validation"
            ),
            "final_validation_mse": value(metrics["final_validation_mse"], "final_validation_mse"),
            "final_test_mse": value(metrics["final_test_mse"], "final_test_mse"),
            "diverged": bool(metrics.get("diverged", False)),
            "runtime_seconds": value(metrics.get("runtime_seconds", 0), "runtime_seconds"),
            "result_dir": row["final_result_dir"],
        })
    return runs


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["method"])].append(run)
    output = []
    for (dataset, method), group in sorted(groups.items()):
        seeds = sorted(run["seed"] for run in group)
        if seeds != [0, 1, 2, 3, 4]:
            raise ValueError(f"{dataset}/{method} has seeds {seeds}, expected [0, 1, 2, 3, 4]")
        configs = {(run["learning_rate"], run["weight_decay"], run["critic_multiplier"]) for run in group}
        if len(configs) != 1:
            raise ValueError(f"{dataset}/{method} did not use a fixed config across seeds")
        tests = [run["test_mse_at_best_validation"] for run in group]
        vals = [run["best_validation_mse"] for run in group]
        lr, wd, critic = next(iter(configs))
        output.append({
            "dataset": dataset,
            "method": method,
            "num_seeds": 5,
            "learning_rate": lr,
            "weight_decay": wd,
            "critic_multiplier": critic,
            "diverged_count": sum(run["diverged"] for run in group),
            "mean_best_validation_mse": statistics.fmean(vals),
            "std_best_validation_mse": statistics.pstdev(vals),
            "mean_test_mse_at_best_validation": statistics.fmean(tests),
            "std_test_mse_at_best_validation": statistics.pstdev(tests),
            "mean_runtime_seconds": statistics.fmean(run["runtime_seconds"] for run in group),
        })
    if len(output) != 24:
        raise ValueError(f"expected 24 dataset+method aggregates, found {len(output)}")
    return output


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Method | Test MSE at best validation (mean ± std) | Diverged |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['mean_test_mse_at_best_validation']:.6g} ± "
            f"{row['std_test_mse_at_best_validation']:.6g} | {row['diverged_count']}/3 |"
        )
    return "\n".join(lines)


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit("final_manifest.csv is missing; materialize selected configs first")
    manifest_rows = read_csv(MANIFEST)
    if len(manifest_rows) != 120:
        raise SystemExit(f"expected 120 final rows, found {len(manifest_rows)}")
    runs = load_runs(manifest_rows)
    summaries = aggregate(runs)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    run_fields = list(runs[0])
    summary_fields = list(summaries[0])
    write_csv(ANALYSIS_DIR / "run_metrics.csv", runs, run_fields)
    write_csv(ANALYSIS_DIR / "method_summary.csv", summaries, summary_fields)
    report = (
        "# High-dimensional fixed-abs results\n\n"
        "Protocol: 6 scenarios × 4 methods × 5 seeds, alpha = 0.5, g(x) = |x|.\n\n"
        "Hyperparameters were selected by validation metrics only. Test MSE below is evaluated "
        "at the validation-selected checkpoint.\n\n"
        + markdown_table(summaries)
        + "\n"
    )
    (ANALYSIS_DIR / "report.md").write_text(report)
    summary = {
        "runs": len(runs),
        "aggregates": len(summaries),
        "g_function": "abs",
        "alpha": 0.5,
        "test_mse_used_for_selection": False,
        "diverged_runs": sum(run["diverged"] for run in runs),
    }
    with (ANALYSIS_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
