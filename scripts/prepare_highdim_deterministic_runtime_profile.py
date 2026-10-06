#!/usr/bin/env python3
"""Prepare a short deterministic runtime-profiling campaign.

Purpose: measure where wall time actually goes in a 1000-client deterministic
round, so the protocol-v2 GPU-hour projection can be refreshed from measured
per-phase costs instead of extrapolated from the cheap ``femnist_z`` gate.

This is a diagnostic campaign. It runs a handful of rounds per scenario at the
protocol-v2 candidate configuration and is not a tuning or production queue.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_deterministic.csv"
)
CAMPAIGN_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_runtime_profile_20260805"
)
OUTPUT_ROOT = REPO_ROOT / "results/highdim_deterministic_runtime_profile_20260805"

# Cheapest cell (matches the completed gate), a CNN-g cell, and the worst case
# (CNN g and CNN f on CIFAR-10). These bracket the per-round cost of the queue.
SCENARIOS = ("femnist_z", "femnist_x", "cifar10_xz")
METHOD = "fedgda_d"
COMM_ROUND = 6

# The alpha0p5 deterministic source manifest predates these columns, so they
# must be appended explicitly. Writing rows against the source fieldnames alone
# silently drops them and the run falls back to defaults (auxiliary_regression
# on, periodic checkpoints every 200 rounds), which is not the v2 configuration.
EXTRA_FIELDS = (
    "auxiliary_regression",
    "auxiliary_regression_epochs",
    "append_round_csv",
    "periodic_checkpoint_interval",
    "log_test_mse_by_round",
    "test_mse_used_for_selection",
    "selection_metric_source",
    "objective_mode",
    "aggregation_weighting",
)

# Paired arms so the auxiliary-regression cost is measured directly on the
# expensive image scenarios, not just the cheap scalar-g femnist_z cell.
AUX_ARMS = (("auxon", True), ("auxoff", False))

# Protocol-v2 FedGDA-D candidate.
LEARNING_RATE = 0.03
CRITIC_MULTIPLIER = 3.0


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows: list[dict[str, str]], dataset: str) -> dict[str, str]:
    matches = [
        row for row in rows if row["dataset"] == dataset and row["method"] == METHOD
    ]
    if not matches:
        raise RuntimeError(f"No source row for {dataset}/{METHOD}")
    return matches[0]


def build_row(source: dict[str, str], dataset: str, arm: str, aux: bool) -> dict[str, str]:
    run_id = f"det_profile_{dataset}_{METHOD}_seed0_alpha0p5_{arm}_r{COMM_ROUND}"
    row = dict(source)
    row.update(
        {
            "run_id": run_id,
            "protocol_version": "highdim_deterministic_runtime_profile_v1",
            "run_group": "highdim_deterministic_runtime_profile_20260805",
            "training_scope": "federated",
            "dataset": dataset,
            "seed": "0",
            "alpha": "0.5",
            "partition_alpha": "0.5",
            "client_num_in_total": "1000",
            "client_num_per_round": "1000",
            "comm_round": str(COMM_ROUND),
            "epochs": "3",
            "batch_size": "0",
            "learning_rate": str(LEARNING_RATE),
            "learning_rate_status": "runtime_profile_probe",
            "weight_decay": "0.001",
            "critic_multiplier": str(CRITIC_MULTIPLIER),
            "server_learning_rate": "1.5",
            "gradient_clip_norm": "1.0",
            "run_status": "not_started",
            "implementation_status": "ready",
            "preflight_required": "False",
            "preflight_status": "not_required",
            "auxiliary_regression": str(aux),
            "auxiliary_regression_epochs": "3" if aux else "0",
            "append_round_csv": "True",
            "periodic_checkpoint_interval": "0",
            "log_test_mse_by_round": "False",
            "test_mse_used_for_selection": "False",
            "selection_metric_source": "validation",
            "objective_mode": "legacy",
            "aggregation_weighting": "sample_size",
            "output_root": str(OUTPUT_ROOT),
            "final_result_dir": str(
                OUTPUT_ROOT / dataset / METHOD / "seed_0" / run_id
            ),
            "notes": (
                "Runtime-profiling probe only. Not a tuning candidate; "
                "too few rounds to carry any scientific selection meaning."
            ),
        }
    )
    return row


def main() -> int:
    fieldnames, rows = load_source()
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)

    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    out_rows = [
        build_row(reference_row(rows, ds), ds, arm, aux)
        for ds in SCENARIOS
        for arm, aux in AUX_ARMS
    ]

    manifest_path = CAMPAIGN_DIR / "profile_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "campaign_dir": str(CAMPAIGN_DIR),
        "manifest": str(manifest_path),
        "output_root": str(OUTPUT_ROOT),
        "scenarios": list(SCENARIOS),
        "method": METHOD,
        "comm_round": COMM_ROUND,
        "learning_rate": LEARNING_RATE,
        "critic_multiplier": CRITIC_MULTIPLIER,
        "aux_arms": [arm for arm, _ in AUX_ARMS],
        "purpose": (
            "Measure per-phase deterministic round cost to refresh the "
            "protocol-v2 GPU-hour projection and locate optimization targets."
        ),
        "rows": len(out_rows),
    }
    summary_path = CAMPAIGN_DIR / "setup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(out_rows)} rows -> {manifest_path}")
    print(f"Wrote summary      -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
