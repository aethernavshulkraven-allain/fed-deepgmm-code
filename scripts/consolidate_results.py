#!/usr/bin/env python3
"""Consolidate result artifacts without modifying the artifacts themselves.

The generated ledger is deliberately provenance-first.  In particular, tuning,
smoke, failed, golden, and primary runs are labelled separately, and aggregates
never combine distinct hyperparameter configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RUN_FIELDS = [
    "artifact_status",
    "family",
    "phase",
    "dataset",
    "method",
    "mode",
    "training_scope",
    "seed",
    "partition_alpha",
    "run_id",
    "comm_round_or_iterations",
    "local_epochs",
    "batch_size",
    "g_learning_rate",
    "f_learning_rate",
    "server_learning_rate",
    "critic_multiplier",
    "weight_decay",
    "gradient_clip_norm",
    "best_validation_mse",
    "best_validation_round",
    "test_mse_at_best_validation",
    "final_validation_mse",
    "final_test_mse",
    "runtime_seconds",
    "diverged",
    "selection_metric_source",
    "test_mse_used_for_selection",
    "test_mse_logged_by_round",
    "has_mse_by_round",
    "has_predictions",
    "has_best_validation_checkpoint",
    "config_path",
    "metrics_path",
]

CONFIG_KEYS = [
    "family",
    "phase",
    "artifact_status",
    "dataset",
    "method",
    "mode",
    "training_scope",
    "partition_alpha",
    "comm_round_or_iterations",
    "local_epochs",
    "batch_size",
    "g_learning_rate",
    "f_learning_rate",
    "server_learning_rate",
    "critic_multiplier",
    "weight_decay",
    "gradient_clip_norm",
]

METRIC_NAMES = [
    "best_validation_mse",
    "test_mse_at_best_validation",
    "final_validation_mse",
    "final_test_mse",
    "runtime_seconds",
]

SUPPLEMENTARY_NAME = re.compile(r"(?:result|summary|report|status)", re.IGNORECASE)
SUPPLEMENTARY_SUFFIXES = {".csv", ".json", ".md", ".png"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def infer_status(parts: tuple[str, ...], run_id: str) -> str:
    lowered = [part.lower() for part in parts]
    run_lower = run_id.lower()
    if "_failed" in lowered:
        return "archived_failure"
    if "_golden" in lowered:
        return "golden_reference"
    if any("smoke" in part for part in lowered) or "smoke" in run_lower:
        return "smoke"
    if any("tuning" in part for part in lowered):
        return "tuning"
    return "primary"


def infer_phase(parts: tuple[str, ...]) -> str:
    if not parts:
        return ""
    family = parts[0]
    if family in {"centralized_lowdim_v1_tuning", "rerun_protocol_v1_tuning", "sine_fedogda_tuning"}:
        return parts[1] if len(parts) > 1 else ""
    if family in {"_failed", "_smoke", "_data_certification"}:
        return parts[1] if len(parts) > 1 else ""
    return ""


def infer_method(config: dict[str, Any], metrics_path: Path) -> str:
    variant = config.get("variant") or config.get("method")
    if variant:
        return str(variant).lower()
    for part in reversed(metrics_path.parts):
        if re.fullmatch(r"(?:fed)?(?:o?gda|sgda|oadam)(?:_[ds])?", part.lower()):
            return part.lower()
    return str(config.get("algorithm", "unknown")).lower()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scan_runs(root: Path, results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        metrics = load_json(metrics_path)
        run_dir = metrics_path.parent
        config_path = run_dir / "effective_config.json"
        config = load_json(config_path) if config_path.exists() else {}
        result_parts = metrics_path.relative_to(results_dir).parts[:-1]
        run_id = str(config.get("run_id") or run_dir.name)
        seed = config.get("seed", config.get("random_seed"))
        row: dict[str, Any] = {
            "artifact_status": infer_status(result_parts, run_id),
            "family": result_parts[0] if result_parts else "results",
            "phase": infer_phase(result_parts),
            "dataset": str(config.get("dataset", "unknown")).lower(),
            "method": infer_method(config, metrics_path),
            "mode": config.get("mode", ""),
            "training_scope": config.get("training_scope", "federated" if "fed" in infer_method(config, metrics_path) else ""),
            "seed": seed,
            "partition_alpha": config.get("partition_alpha", ""),
            "run_id": run_id,
            "comm_round_or_iterations": config.get("comm_round", config.get("iterations", "")),
            "local_epochs": config.get("local_epochs", config.get("epochs", "")),
            "batch_size": config.get("batch_size", ""),
            "g_learning_rate": config.get("g_learning_rate", config.get("learning_rate", "")),
            "f_learning_rate": config.get("f_learning_rate", ""),
            "server_learning_rate": config.get("server_learning_rate", ""),
            "critic_multiplier": config.get("critic_multiplier", ""),
            "weight_decay": config.get("weight_decay", ""),
            "gradient_clip_norm": config.get("gradient_clip_norm", ""),
            "best_validation_mse": metrics.get("best_validation_mse", ""),
            "best_validation_round": metrics.get("best_validation_round", ""),
            "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation", ""),
            "final_validation_mse": metrics.get("final_validation_mse", metrics.get("val_mse_final", "")),
            "final_test_mse": metrics.get("final_test_mse", metrics.get("test_mse_final", "")),
            "runtime_seconds": metrics.get("runtime_seconds", ""),
            "diverged": metrics.get("diverged", ""),
            "selection_metric_source": metrics.get("selection_metric_source", config.get("selection_metric_source", "")),
            "test_mse_used_for_selection": metrics.get("test_mse_used_for_selection", config.get("test_mse_used_for_selection", "")),
            "test_mse_logged_by_round": metrics.get("test_mse_logged_by_round", config.get("test_mse_logged_by_round", "")),
            "has_mse_by_round": (run_dir / "mse_by_round.csv").exists(),
            "has_predictions": (run_dir / "predictions.npz").exists(),
            "has_best_validation_checkpoint": (run_dir / "checkpoints" / "best_validation.pt").exists(),
            "config_path": rel(config_path, root) if config_path.exists() else "",
            "metrics_path": rel(metrics_path, root),
        }
        rows.append(row)
    return rows


def normalized_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(normalized_key(row[field]) for field in CONFIG_KEYS)
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item = dict(zip(CONFIG_KEYS, key))
        seeds = sorted({normalized_key(member["seed"]) for member in members if normalized_key(member["seed"])})
        item.update(
            {
                "number_of_runs": len(members),
                "number_of_seeds": len(seeds),
                "seeds": "|".join(seeds),
                "diverged_runs": sum(member["diverged"] is True for member in members),
            }
        )
        for metric in METRIC_NAMES:
            values = [number for member in members if (number := finite_number(member[metric])) is not None]
            item[f"mean_{metric}"] = statistics.fmean(values) if values else ""
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        output.append(item)
    return output


def scan_supplementary_files(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Catalog derived summaries and legacy result files not represented as runs."""
    rows: list[dict[str, Any]] = []
    for base_name in ("experiments", "fedgmm"):
        base = root / base_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPLEMENTARY_SUFFIXES:
                continue
            if output_dir == path or output_dir in path.parents:
                continue
            if not SUPPLEMENTARY_NAME.search(path.name):
                continue
            rows.append(
                {
                    "category": "derived_experiment_summary" if base_name == "experiments" else "legacy_repository_result",
                    "format": path.suffix.lower().lstrip("."),
                    "size_bytes": path.stat().st_size,
                    "path": rel(path, root),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    number = finite_number(value)
    return "" if number is None else f"{number:.6g}"


def write_report(
    path: Path,
    runs: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    supplementary: list[dict[str, Any]],
) -> None:
    statuses = Counter(row["artifact_status"] for row in runs)
    families = Counter(row["family"] for row in runs)
    primary = [row for row in aggregates if row["artifact_status"] == "primary"]
    primary_table = []
    for row in primary:
        primary_table.append(
            [
                row["family"],
                row["dataset"],
                row["method"],
                row["partition_alpha"] or "—",
                row["number_of_runs"],
                f"{fmt(row['mean_test_mse_at_best_validation'])} ± {fmt(row['std_test_mse_at_best_validation'])}",
                f"{fmt(row['mean_best_validation_mse'])} ± {fmt(row['std_best_validation_mse'])}",
                row["diverged_runs"],
            ]
        )

    text = [
        "# Consolidated result index",
        "",
        f"Generated from local artifacts at `{datetime.now(timezone.utc).isoformat()}`.",
        "",
        "This index does not claim paper reproduction. Current synthetic data is reproducible but is not verified as paper-aligned. Hyperparameter candidates are never selected using Test MSE; `test_mse_at_best_validation` is reported only for the checkpoint chosen by validation.",
        "",
        "## Inventory",
        "",
        markdown_table(["artifact status", "runs"], [[key, statuses[key]] for key in sorted(statuses)]),
        "",
        markdown_table(["result family", "runs"], [[key, families[key]] for key in sorted(families)]),
        "",
        "## Files",
        "",
        "- `all_runs.csv`: every discovered `metrics.json`, joined to its `effective_config.json` when present.",
        "- `aggregates_by_exact_config.csv`: seed aggregates with all scientifically relevant configuration fields in the grouping key.",
        "- `primary_aggregates.csv`: the exact-config aggregate restricted to primary artifacts.",
        "- `supplementary_result_files.csv`: existing derived reports/summaries and legacy result files outside the run directories.",
        "- `inventory.json`: machine-readable counts and generation metadata.",
        "",
        "Tuning, smoke, archived failures, and golden references remain in the ledger but are excluded from the primary table. Existing tuning-selection reports remain authoritative because their choices are validation-driven.",
        f"The supplementary catalog contains {len(supplementary)} files. It is an index, not a second metric source: many entries are derived from the runs in `all_runs.csv`.",
        "",
        "## Primary result aggregates",
        "",
        markdown_table(
            ["family", "dataset", "method", "alpha", "runs", "test MSE @ best val", "best val MSE", "diverged"],
            primary_table,
        ),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/consolidated_results"))
    args = parser.parse_args()

    root = Path.cwd().resolve()
    results_dir = (root / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    runs = scan_runs(root, results_dir)
    aggregates = aggregate(runs)
    primary = [row for row in aggregates if row["artifact_status"] == "primary"]
    supplementary = scan_supplementary_files(root, output_dir)

    write_csv(output_dir / "all_runs.csv", runs, RUN_FIELDS)
    write_csv(output_dir / "aggregates_by_exact_config.csv", aggregates)
    write_csv(output_dir / "primary_aggregates.csv", primary)
    write_csv(
        output_dir / "supplementary_result_files.csv",
        supplementary,
        ["category", "format", "size_bytes", "path"],
    )
    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": rel(results_dir, root),
        "metrics_files": len(runs),
        "exact_config_aggregates": len(aggregates),
        "supplementary_result_files": len(supplementary),
        "counts_by_artifact_status": dict(sorted(Counter(row["artifact_status"] for row in runs).items())),
        "counts_by_family": dict(sorted(Counter(row["family"] for row in runs).items())),
        "missing_effective_config": sum(not row["config_path"] for row in runs),
        "diverged_true": sum(row["diverged"] is True for row in runs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "README.md", runs, aggregates, supplementary)
    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
