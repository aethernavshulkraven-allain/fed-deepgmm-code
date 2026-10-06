#!/usr/bin/env python3
"""Prepare the final-evaluation manifest for the 2026-08-13 deterministic
campaign (Option 2 seed schedule).

Uses the 12 (scenario, method) winners frozen by
deterministic_screen_20260813 (screen + boundary-expansion re-screen,
validation MSE only -- test MSE was never inspected during tuning).

6 scenarios x 2 methods x 3 alphas {0.1, 0.5, 1.0} x 3 seeds {0, 1, 2}
= 108 runs, 500 rounds each. Test MSE is read only after this config is
frozen, per the decided protocol -- it is not used for selection anywhere
in this script.
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
    / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
)
OUTPUT_ROOT = REPO_ROOT / "results/highdim_deterministic_finals_20260813"

# Frozen per (dataset, method) from deterministic_screen_20260813, selected by
# validation MSE across the union of the original screen (72 runs) and the
# boundary-expansion re-screen (19 runs). See screen winners in
# deterministic_screen_20260813/final_winners.json (copied alongside this
# manifest for provenance).
WINNERS: dict[tuple[str, str], dict[str, float]] = {
    ("cifar10_x", "fedgda_d"):   {"lr": 0.1,   "cm": 20.0, "val_mse": 0.14588661986283688},
    ("cifar10_x", "fedogda_d"):  {"lr": 0.01,  "cm": 10.0, "val_mse": 0.14435996505851953},
    ("cifar10_xz", "fedgda_d"):  {"lr": 0.01,  "cm": 5.0,  "val_mse": 0.137273},
    ("cifar10_xz", "fedogda_d"): {"lr": 0.003, "cm": 1.0,  "val_mse": 0.144610},
    ("cifar10_z", "fedgda_d"):   {"lr": 0.01,  "cm": 5.0,  "val_mse": 0.182017},
    ("cifar10_z", "fedogda_d"):  {"lr": 0.003, "cm": 1.0,  "val_mse": 0.233771},
    ("femnist_x", "fedgda_d"):   {"lr": 0.03,  "cm": 5.0,  "val_mse": 0.117750},
    ("femnist_x", "fedogda_d"):  {"lr": 0.01,  "cm": 20.0, "val_mse": 0.131510},
    ("femnist_xz", "fedgda_d"):  {"lr": 0.1,   "cm": 5.0,  "val_mse": 0.097181},
    ("femnist_xz", "fedogda_d"): {"lr": 0.01,  "cm": 1.0,  "val_mse": 0.137973},
    ("femnist_z", "fedgda_d"):   {"lr": 0.03,  "cm": 1.0,  "val_mse": 0.043800},
    ("femnist_z", "fedogda_d"):  {"lr": 0.01,  "cm": 1.0,  "val_mse": 0.165541},
}

SCENARIOS = ("femnist_z", "femnist_x", "femnist_xz", "cifar10_z", "cifar10_x", "cifar10_xz")
METHODS = ("fedgda_d", "fedogda_d")
ALPHAS = (0.1, 0.5, 1.0)
SEEDS = (0, 1, 2)  # Option 2: uniform 3 seeds per alpha
COMM_ROUND = 500
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


def alpha_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def lr_token(value: float) -> str:
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
            "protocol_version": "highdim_deterministic_finals_v1",
            "run_group": "highdim_deterministic_finals_20260813",
            "training_scope": "federated",
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
        for method in METHODS:
            winner = WINNERS[(dataset, method)]
            source = reference_row(source_rows, dataset, method)
            for alpha in ALPHAS:
                for seed in SEEDS:
                    row = common_row(source)
                    run_id = (
                        f"det_final_{dataset}_{method}_seed{seed}_alpha{alpha_token(alpha)}"
                        f"_lr{lr_token(winner['lr'])}_cm{lr_token(winner['cm'])}"
                    )
                    row.update(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "method": method,
                            "seed": str(seed),
                            "alpha": f"{alpha:g}",
                            "partition_alpha": f"{alpha:g}",
                            "learning_rate": f"{winner['lr']:g}",
                            "learning_rate_status": "frozen_final",
                            "critic_multiplier": f"{winner['cm']:g}",
                            "output_root": str(OUTPUT_ROOT),
                            "final_result_dir": str(
                                OUTPUT_ROOT / dataset / method / f"seed_{seed}" / run_id
                            ),
                            "notes": (
                                f"Frozen final-evaluation run. Config selected by validation "
                                f"MSE during deterministic_screen_20260813 (screen val_mse="
                                f"{winner['val_mse']:.6f}); test MSE for THIS run is read only "
                                f"after that selection, never used to pick lr/critic_multiplier."
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
    write_manifest(CAMPAIGN_DIR / "finals_manifest.csv", fieldnames, rows)

    with (CAMPAIGN_DIR / "frozen_winners.json").open("w") as handle:
        json.dump(
            {f"{ds}/{m}": w for (ds, m), w in WINNERS.items()},
            handle, indent=2, sort_keys=True,
        )
        handle.write("\n")

    summary = {
        "campaign": "highdim_deterministic_finals_20260813",
        "parent_campaigns": [
            "highdim_deterministic_screen_20260813",
            "highdim_deterministic_screen_20260813 (boundary expansion)",
        ],
        "seed_schedule": "option_2_uniform_3_seeds",
        "runs": len(rows),
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "comm_round": COMM_ROUND,
        "client_num_in_total": CLIENT_COUNT,
        "client_num_per_round": CLIENT_COUNT,
        "batch_size": 0,
        "auxiliary_regression": False,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "note": (
            "Config frozen from deterministic_screen_20260813; test MSE for these runs "
            "is reported only after selection, never used to pick hyperparameters."
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
