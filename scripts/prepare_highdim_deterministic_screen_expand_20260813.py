#!/usr/bin/env python3
"""Boundary-expansion re-screen for the 2026-08-13 deterministic tuning screen.

8 of the screen's 12 (scenario, method) cells had their winning learning_rate
and/or critic_multiplier sitting at the edge of the tested grid, with a clear
monotonic trend still improving at that edge (not a marginal/noisy touch --
see deterministic_screen_20260813/boundary_review.json for the full
candidate-by-candidate evidence). Per the decided protocol ("expand only if
the best value lies clearly at the grid boundary"), this adds one extra rung
on each flagged axis for those 8 cells only -- 19 targeted candidates, same
150-round / N=10 / aux-off screen protocol as the original.

Cells needing both axes expanded get the two single-axis extensions plus the
new-lr x new-cm corner, to check for interaction rather than assuming the
axes are independent.
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

COMM_ROUND = 150
CLIENT_COUNT = 10

# (dataset, method) -> list of (learning_rate, critic_multiplier) new candidates
NEW_CANDIDATES: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("femnist_z", "fedgda_d"):   [(0.1, 1.0), (0.1, 5.0)],
    ("femnist_z", "fedogda_d"):  [(0.03, 1.0), (0.03, 5.0)],
    ("femnist_xz", "fedgda_d"):  [(0.1, 5.0), (0.03, 10.0), (0.1, 10.0)],
    ("femnist_x", "fedgda_d"):   [(0.1, 5.0), (0.1, 10.0)],
    ("femnist_x", "fedogda_d"):  [(0.03, 10.0), (0.01, 20.0), (0.03, 20.0)],
    ("cifar10_x", "fedgda_d"):   [(0.1, 10.0), (0.03, 20.0), (0.1, 20.0)],
    ("cifar10_x", "fedogda_d"):  [(0.03, 10.0), (0.01, 20.0), (0.03, 20.0)],
    ("cifar10_z", "fedgda_d"):   [(0.01, 10.0)],
}

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
            "protocol_version": "highdim_deterministic_screen_expand_v1",
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
    for (dataset, method), candidates in NEW_CANDIDATES.items():
        source = reference_row(source_rows, dataset, method)
        for learning_rate, critic_multiplier in candidates:
            row = common_row(source)
            run_id = (
                f"det_screen_expand_{dataset}_{method}_seed0_alpha0p5"
                f"_lr{token(learning_rate)}_cm{token(critic_multiplier)}"
            )
            row.update(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "method": method,
                    "learning_rate": f"{learning_rate:g}",
                    "learning_rate_status": "screen_boundary_expansion_candidate",
                    "critic_multiplier": f"{critic_multiplier:g}",
                    "output_root": str(OUTPUT_ROOT),
                    "final_result_dir": str(
                        OUTPUT_ROOT / dataset / method / "seed_0" / run_id
                    ),
                    "notes": (
                        "Boundary-expansion candidate: the original 150-round screen's "
                        "winner for this cell sat at the edge of the tested lr/critic_"
                        "multiplier grid with a still-improving trend. Select by "
                        "validation MSE only; do not inspect Test MSE."
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
    write_manifest(CAMPAIGN_DIR / "screen_expand_manifest.csv", fieldnames, rows)

    summary = {
        "campaign": "highdim_deterministic_screen_20260813_expand",
        "parent_campaign": "highdim_deterministic_screen_20260813",
        "reason": (
            "8 of 12 cells from the original screen had their winning lr/critic_"
            "multiplier at the grid edge with a clearly still-improving trend, "
            "not a marginal touch. Re-screening those cells with one extra rung "
            "per flagged axis before freezing any config for the final stage."
        ),
        "runs": len(rows),
        "cells_expanded": [f"{d}/{m}" for d, m in NEW_CANDIDATES.keys()],
        "comm_round": COMM_ROUND,
        "client_num_in_total": CLIENT_COUNT,
        "client_num_per_round": CLIENT_COUNT,
        "batch_size": 0,
        "auxiliary_regression": False,
        "next_step": (
            "Combine with the original screen's 63 valid results, re-pick the winner "
            "per (scenario, method) cell across the union, re-check boundaries, then "
            "build the Option 2 final-evaluation manifest from the frozen winners."
        ),
    }
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN_DIR / "screen_expand_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
