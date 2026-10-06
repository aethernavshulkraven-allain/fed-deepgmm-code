#!/usr/bin/env python3
"""Prepare runnable manifests for centralized low-dimensional DeepGMM baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "centralized_baselines"
DATASETS = ("abs", "step", "linear", "sin")
METHODS = ("gda", "sgda", "oadam")
SEEDS = (0, 1, 2)


FIELDNAMES = [
    "run_id",
    "training_scope",
    "dataset",
    "method",
    "seed",
    "output_dir",
    "iterations",
    "batch_size",
    "g_lr",
    "f_lr",
    "weight_decay",
    "gradient_clip_norm",
    "log_test_mse_by_round",
    "selection_metric_source",
    "test_mse_used_for_selection",
    "status",
    "command",
]


def defaults_for(method: str, smoke: bool) -> dict[str, object]:
    return {
        "iterations": 2 if smoke else 500,
        "batch_size": 0 if method == "gda" else 256,
        "g_lr": 0.001,
        "f_lr": 0.01,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "log_test_mse_by_round": False,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
    }


def command_for(row: dict[str, object]) -> str:
    parts = [
        "python",
        "scripts/run_centralized_lowdim.py",
        "--dataset", str(row["dataset"]),
        "--method", str(row["method"]),
        "--seed", str(row["seed"]),
        "--output-dir", str(row["output_dir"]),
        "--iterations", str(row["iterations"]),
        "--batch-size", str(row["batch_size"]),
        "--g-lr", str(row["g_lr"]),
        "--f-lr", str(row["f_lr"]),
        "--weight-decay", str(row["weight_decay"]),
        "--gradient-clip-norm", str(row["gradient_clip_norm"]),
    ]
    if str(row["log_test_mse_by_round"]).lower() == "true":
        parts.append("--log-test-mse-by-round")
    return " ".join(parts)


def make_row(dataset: str, method: str, seed: int, smoke: bool) -> dict[str, object]:
    run_id = f"centralized_lowdim_{dataset}_{method}_seed{seed}"
    root = "results/centralized_lowdim_v1_smoke" if smoke else "results/centralized_lowdim_v1"
    output_dir = f"{root}/{dataset}/{method}/seed_{seed}"
    row = {
        "run_id": run_id,
        "training_scope": "centralized",
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "output_dir": output_dir,
        "status": "ready_for_smoke" if smoke else "ready_not_launched",
    }
    row.update(defaults_for(method, smoke))
    row["command"] = command_for(row)
    return row


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    full_rows = [
        make_row(dataset, method, seed, smoke=False)
        for dataset in DATASETS
        for method in METHODS
        for seed in SEEDS
    ]
    smoke_rows = [
        make_row("abs", method, 0, smoke=True)
        for method in METHODS
    ]
    full_path = out_dir / "centralized_lowdim_manifest.csv"
    smoke_path = out_dir / "centralized_smoke_manifest_runnable.csv"
    write_manifest(full_path, full_rows)
    write_manifest(smoke_path, smoke_rows)
    print(f"wrote {len(full_rows)} rows to {full_path}")
    print(f"wrote {len(smoke_rows)} rows to {smoke_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
