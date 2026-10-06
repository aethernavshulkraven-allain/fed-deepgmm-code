#!/usr/bin/env python3
"""Run full-gradient data-only preflights from a rerun protocol manifest."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from scripts.preflight_full_gradient import (  # noqa: E402
    FullGradientPreflightError,
    run_preflight,
)


DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "preflight" / "full_gradient" / "rerun_protocol_v1"
SUPPORTED_DATASETS = (
    "abs",
    "step",
    "linear",
    "sin",
    "femnist_x",
    "femnist_z",
    "femnist_xz",
    "cifar10_x",
    "cifar10_z",
    "cifar10_xz",
)

FIELDNAMES = (
    "run_id",
    "dataset",
    "method",
    "seed",
    "alpha",
    "status",
    "checked_client_count",
    "sample_count_min",
    "sample_count_max",
    "sample_count_sum",
    "preflight_run_dir",
    "preflight_report_path",
    "effective_config_path",
    "error",
)


def _bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _selected_rows(rows: list[dict[str, str]], only_dataset: str | None, only_method: str | None) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if row.get("training_scope") != "federated":
            continue
        if not _bool_from_csv(row.get("preflight_required")):
            continue
        if only_dataset is not None and row.get("dataset") != only_dataset:
            continue
        if only_method is not None and row.get("method") != only_method:
            continue
        selected.append(row)
    return selected


def _row_to_args(row: dict[str, str], output_dir: Path, overwrite: bool) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=row["dataset"],
        model=row.get("model", "lr") or "lr",
        federated_optimizer=row.get("federated_optimizer", "FedAvg") or "FedAvg",
        client_id_list="[]",
        client_num_in_total=int(row["client_num_in_total"]),
        client_num_per_round=int(row["client_num_per_round"]),
        comm_round=int(row["comm_round"]),
        epochs=int(row["epochs"]),
        frequency_of_the_test=1,
        random_seed=int(row["seed"]),
        partition_method=row.get("partition_method", "hetero") or "hetero",
        partition_alpha=float(row["partition_alpha"]),
        data_cache_dir=row.get("data_cache_dir", "data") or "data",
        batch_size=int(row["batch_size"]),
        client_optimizer=row["client_optimizer"],
        learning_rate=0.001,
        weight_decay=0.1,
        critic_multiplier=float(row.get("critic_multiplier") or 10.0),
        server_learning_rate=float(row.get("server_learning_rate") or 1.5),
        gradient_clip_norm=float(row.get("gradient_clip_norm") or 1.0),
        simple_model_selection_epochs=int(row.get("simple_model_selection_epochs") or 100),
        f_history_model_selection_epochs=int(row.get("f_history_model_selection_epochs") or 60),
        model_selection_batch_size=int(row.get("model_selection_batch_size") or 200),
        output_dir=str(output_dir),
        run_id=f"{row['run_id']}_full_gradient_preflight",
        using_gpu=False,
        gpu_id=0,
        enable_legacy_outputs=False,
        overwrite=overwrite,
        scenario_name=None,
    )


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    return csv_path, json_path


def run_rows(rows: list[dict[str, str]], output_dir: Path, overwrite: bool, rounds_to_check: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        args = _row_to_args(row, output_dir, overwrite)
        captured = io.StringIO()
        print(
            f"[{index}/{len(rows)}] preflight {row['dataset']} {row['method']} "
            f"seed={row['seed']} alpha={row['alpha']}"
        )
        try:
            with redirect_stdout(captured), redirect_stderr(captured):
                report = run_preflight(
                    args,
                    rounds_to_check=rounds_to_check,
                    client_detail_limit=5,
                )
            results.append({
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "alpha": row["alpha"],
                "status": report["status"],
                "checked_client_count": report["checked_client_count"],
                "sample_count_min": report["sample_count_min"],
                "sample_count_max": report["sample_count_max"],
                "sample_count_sum": report["sample_count_sum"],
                "preflight_run_dir": report["run_dir"],
                "preflight_report_path": report["preflight_report_path"],
                "effective_config_path": report["effective_config_path"],
                "error": "",
            })
        except FullGradientPreflightError as exc:
            results.append({
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "alpha": row["alpha"],
                "status": "failed",
                "checked_client_count": "",
                "sample_count_min": "",
                "sample_count_max": "",
                "sample_count_sum": "",
                "preflight_run_dir": "",
                "preflight_report_path": "",
                "effective_config_path": "",
                "error": str(exc),
            })
        except Exception as exc:  # pragma: no cover - exercised during operational failures
            results.append({
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "alpha": row["alpha"],
                "status": "error",
                "checked_client_count": "",
                "sample_count_min": "",
                "sample_count_max": "",
                "sample_count_sum": "",
                "preflight_run_dir": "",
                "preflight_report_path": "",
                "effective_config_path": "",
                "error": f"{type(exc).__name__}: {exc}\n{captured.getvalue()[-2000:]}",
            })
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic federated full-gradient preflights.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--only-dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--only-method", choices=["fedgda_d", "fedogda_d"], default=None)
    parser.add_argument("--rounds-to-check", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    rows = _selected_rows(_load_rows(manifest_path), args.only_dataset, args.only_method)
    results = run_rows(rows, output_dir, bool(args.overwrite), int(args.rounds_to_check))
    csv_path, json_path = _write_summary(results, output_dir)
    counts = {
        "selected": len(rows),
        "passed": sum(row["status"] == "passed" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "error": sum(row["status"] == "error" for row in results),
        "summary_csv": str(csv_path.relative_to(REPO_ROOT)),
        "summary_json": str(json_path.relative_to(REPO_ROOT)),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0 if counts["selected"] == counts["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
