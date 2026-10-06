#!/usr/bin/env python3
"""Prepare the deterministic tuning screen for FedEG and FedEG-Double.

Mirrors deterministic_screen_20260813 (the FedGDA-D/FedOGDA-D screen):
6 scenarios x 3 learning rates x 2 critic multipliers = 36 runs per method,
72 runs total across both new methods. Seed 0, alpha 0.5, 150 rounds,
auxiliary_regression off, N=10 full participation.

Reuses FedOGDA-D's learning-rate grid for both new methods, per the decided
plan -- both fed_eg and fed_eg_double are extra-gradient variants, closer in
character to OGDA's optimistic update than to plain SGD.

client_optimizer values used here (see experiment_utils.CLIENT_OPTIMIZER_CHOICES):
  fed_eg        -- extra-gradient at the server only; clients run plain SGD.
  fed_eg_double -- extra-gradient at both the server AND each client locally.

Designed to run standalone on a machine with no gpu-broker (e.g. a
colleague's own GPU): this script and run_manifest.py have no gpurun
dependency, only scripts/run_manifest.py + main.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_deterministic.csv"
)
DEFAULT_CAMPAIGN_DIR = (
    REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/fedeg_screen"
)

SCENARIOS = ("femnist_z", "femnist_x", "femnist_xz", "cifar10_z", "cifar10_x", "cifar10_xz")

METHODS = {
    "fed_eg_d": {"client_optimizer": "fed_eg", "method_label": "FedEG-D"},
    "fed_eg_double_d": {"client_optimizer": "fed_eg_double", "method_label": "FedEG-Double-D"},
}

# FedOGDA-D's grid, reused here per the decided plan.
LEARNING_RATES = (0.001, 0.003, 0.01)

# Same scenario grouping used for FedGDA-D/FedOGDA-D.
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


def reference_row(rows: list[dict[str, str]], dataset: str) -> dict[str, str]:
    """Any method's row for this dataset works -- model/client-count/data
    fields don't vary by method, only the fields we override below do."""
    matches = [row for row in rows if row["dataset"] == dataset]
    if not matches:
        raise RuntimeError(f"No source row for dataset={dataset}")
    return matches[0]


def common_row(source: dict[str, str], output_root: Path) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "protocol_version": "highdim_fedeg_screen_v1",
            "run_group": "highdim_fedeg_screen",
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


def make_rows(source_rows: list[dict[str, str]], output_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for dataset in SCENARIOS:
        cm_group = "x" if dataset in X_SCENARIOS else "z_xz"
        source = reference_row(source_rows, dataset)
        for method, method_info in METHODS.items():
            for learning_rate in LEARNING_RATES:
                for critic_multiplier in CRITIC_MULTIPLIERS[cm_group]:
                    row = common_row(source, output_root)
                    run_id = (
                        f"det_screen_{dataset}_{method}_seed0_alpha0p5"
                        f"_lr{token(learning_rate)}_cm{token(critic_multiplier)}"
                    )
                    row.update(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "method": method,
                            "method_label": method_info["method_label"],
                            "client_optimizer": method_info["client_optimizer"],
                            "learning_rate": f"{learning_rate:g}",
                            "learning_rate_status": "screen_candidate",
                            "critic_multiplier": f"{critic_multiplier:g}",
                            "output_root": str(output_root),
                            "final_result_dir": str(
                                output_root / dataset / method / "seed_0" / run_id
                            ),
                            "notes": (
                                "FedEG/FedEG-Double deterministic tuning screen (150 "
                                "rounds, N=10 full participation). Learning-rate grid "
                                "reused from FedOGDA-D. Select by validation MSE only; "
                                "do not inspect Test MSE."
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir", default=str(DEFAULT_CAMPAIGN_DIR),
        help="Where to write the manifest and summary (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Where run results are written (default: results/<campaign-dir-name>)",
    )
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    output_root = (
        Path(args.output_root) if args.output_root
        else REPO_ROOT / "results" / campaign_dir.name
    )

    source_fields, source_rows = load_source()
    fieldnames = source_fields + [f for f in EXTRA_FIELDS if f not in source_fields]
    rows = make_rows(source_rows, output_root)
    write_manifest(campaign_dir / "screen_manifest.csv", fieldnames, rows)

    summary = {
        "campaign": campaign_dir.name,
        "output_root": str(output_root),
        "runs": len(rows),
        "scenarios": list(SCENARIOS),
        "methods": {m: v["client_optimizer"] for m, v in METHODS.items()},
        "learning_rates": list(LEARNING_RATES),
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
            "Run scripts/score_highdim_screen_winners.py against this manifest once "
            "all rows complete, review any grid-boundary flags, then build finals "
            "with scripts/prepare_highdim_finals_from_winners.py."
        ),
    }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    with (campaign_dir / "setup_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
