#!/usr/bin/env python3
"""Prepare the adopted 10-client deterministic tuning screen (2026-08-13).

6 scenarios x 2 methods x 3 learning rates x 2 critic multipliers = 72 runs,
seed 0, alpha 0.5, 150 rounds. This is the tuning-side "Screen" stage: cheap
enough to run in full, its winners (by validation MSE) get frozen and carried
into the final evaluation stage (500 rounds x 5 seeds x 3 alphas), which is
generated separately once these results are in -- final configs depend on
selecting a winner per (scenario, method) cell from this screen's output.

client_num_in_total = client_num_per_round = 10, full participation,
batch_size=0 (full local batch), matching deterministic_10client_proposal.md.
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
    / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
)
OUTPUT_ROOT = REPO_ROOT / "results/highdim_deterministic_screen_20260813"

SCENARIOS = ("femnist_z", "femnist_x", "femnist_xz", "cifar10_z", "cifar10_x", "cifar10_xz")
METHODS = ("fedgda_d", "fedogda_d")

LEARNING_RATES = {
    "fedgda_d": (0.003, 0.01, 0.03),
    "fedogda_d": (0.001, 0.003, 0.01),
}

# Group Z/XZ (g or f is an MLP against a CNN) vs Group X (CNN g, small MLP f
# -- critic-starved, needs a larger multiplier). Per the decided config.
X_SCENARIOS = {"femnist_x", "cifar10_x"}
CRITIC_MULTIPLIERS = {
    "z_xz": (1.0, 5.0),
    "x": (5.0, 10.0),
}

COMM_ROUND = 150
CLIENT_COUNT = 10

EXTRA_FIELDS = (
    "auxiliary_regression",
    "auxiliary_regression_epochs",
    "objective_lambda_1",
    "append_round_csv",
    "periodic_checkpoint_interval",
    "log_test_mse_by_round",
    "test_mse_used_for_selection",
    "selection_metric_source",
    "objective_mode",
    "aggregation_weighting",
)


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows: list[dict[str, str]], dataset: str, method: str) -> dict[str, str]:
    matches = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
    if not matches:
        raise RuntimeError(f"No source row for {dataset}/{method}")
    return matches[0]


def common_row(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "protocol_version": "highdim_deterministic_screen_v1",
            "run_group": "highdim_deterministic_screen_20260813",
            "training_scope": "federated",
            "seed": "0",
            "alpha": "0.5",
            "partition_alpha": "0.5",
            "client_num_in_total": str(CLIENT_COUNT),
            "client_num_per_round": str(CLIENT_COUNT),
            "comm_round": str(COMM_ROUND),
            "epochs": "3",
            "batch_size": "0",
            "weight_decay": "0.001",
            "server_learning_rate": "1.5",
            "gradient_clip_norm": "1.0",
            "objective_lambda_1": "0.1",
            "run_status": "not_started",
            "implementation_status": "ready",
            "preflight_required": "False",
            "preflight_status": "not_required",
            "auxiliary_regression": "False",
            "auxiliary_regression_epochs": "0",
            "append_round_csv": "True",
            "periodic_checkpoint_interval": "0",
            "log_test_mse_by_round": "False",
            "test_mse_used_for_selection": "False",
            "selection_metric_source": "validation",
            "objective_mode": "legacy",
            "aggregation_weighting": "sample_size",
        }
    )
    return row


def make_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for dataset in SCENARIOS:
        cm_group = "x" if dataset in X_SCENARIOS else "z_xz"
        for method in METHODS:
            source = reference_row(source_rows, dataset, method)
            for learning_rate in LEARNING_RATES[method]:
                for critic_multiplier in CRITIC_MULTIPLIERS[cm_group]:
                    row = common_row(source)
                    run_id = (
                        f"det_screen_{dataset}_{method}_seed0_alpha0p5"
                        f"_lr{token(learning_rate)}_cm{token(critic_multiplier)}"
                    )
                    row.update(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "method": method,
                            "learning_rate": f"{learning_rate:g}",
                            "learning_rate_status": "screen_candidate",
                            "critic_multiplier": f"{critic_multiplier:g}",
                            "output_root": str(OUTPUT_ROOT),
                            "final_result_dir": str(
                                OUTPUT_ROOT / dataset / method / "seed_0" / run_id
                            ),
                            "notes": (
                                "Deterministic tuning screen (150 rounds, N=10 full "
                                "participation). Select by validation MSE only; do not "
                                "inspect Test MSE. Winners per (scenario, method) cell "
                                "carry forward into the frozen final-evaluation config."
                            ),
                        }
                    )
                    output.append(row)
    return output


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    source_fields, source_rows = load_source()
    fieldnames = source_fields + [field for field in EXTRA_FIELDS if field not in source_fields]
    rows = make_rows(source_rows)
    write_manifest(CAMPAIGN_DIR / "screen_manifest.csv", fieldnames, rows)

    summary = {
        "campaign": "highdim_deterministic_screen_20260813",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "output_root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
        "runs": len(rows),
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "learning_rates": LEARNING_RATES,
        "critic_multipliers": CRITIC_MULTIPLIERS,
        "seed": 0,
        "alpha": 0.5,
        "comm_round": COMM_ROUND,
        "client_num_in_total": CLIENT_COUNT,
        "client_num_per_round": CLIENT_COUNT,
        "batch_size": 0,
        "auxiliary_regression": False,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "next_step": (
            "Score by validation MSE per (scenario, method) cell, freeze the winning "
            "lr/critic_multiplier, then generate the final-evaluation manifest "
            "(500 rounds, 3 alphas, 3 seeds each -- Option 2) from the winners."
        ),
    }
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN_DIR / "setup_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
