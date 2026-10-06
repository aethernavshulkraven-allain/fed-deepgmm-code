#!/usr/bin/env python3
"""Build a final-evaluation manifest from a frozen winners.json (as written
by scripts/score_highdim_screen_winners.py).

500 rounds, 3 alphas {0.1, 0.5, 1.0} x 3 seeds {0, 1, 2} per (dataset,
method) cell -- the same "Option 2" uniform-3-seed schedule used for the
FedGDA-D/FedOGDA-D finals, so all methods stay comparable.

Usage:
  python scripts/prepare_highdim_finals_from_winners.py \\
      --winners experiments/highdim_coauthor_protocol_v1/fedeg_screen/winners.json \\
      --campaign-dir experiments/highdim_coauthor_protocol_v1/fedeg_finals
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

METHOD_CLIENT_OPTIMIZER = {
    "fed_eg_d": "fed_eg",
    "fed_eg_double_d": "fed_eg_double",
    "fedgda_d": "sgd",
    "fedogda_d": "ogda",
}
METHOD_LABEL = {
    "fed_eg_d": "FedEG-D",
    "fed_eg_double_d": "FedEG-Double-D",
    "fedgda_d": "FedGDA-D",
    "fedogda_d": "FedOGDA-D",
}

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


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows: list[dict[str, str]], dataset: str) -> dict[str, str]:
    matches = [row for row in rows if row["dataset"] == dataset]
    if not matches:
        raise RuntimeError(f"No source row for dataset={dataset}")
    return matches[0]


def common_row(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "protocol_version": "highdim_finals_from_winners_v1",
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


def make_rows(source_rows, winners, output_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    by_dataset_method: dict[tuple[str, str], dict] = {}
    for key, winner in winners.items():
        by_dataset_method[(winner["dataset"], winner["method"])] = winner

    for (dataset, method), winner in sorted(by_dataset_method.items()):
        source = reference_row(source_rows, dataset)
        client_optimizer = METHOD_CLIENT_OPTIMIZER.get(method)
        if client_optimizer is None:
            raise RuntimeError(
                f"Unknown method {method!r} -- add it to METHOD_CLIENT_OPTIMIZER "
                f"in this script before running."
            )
        for alpha in ALPHAS:
            for seed in SEEDS:
                row = common_row(source)
                run_id = (
                    f"det_final_{dataset}_{method}_seed{seed}_alpha{token(alpha)}"
                    f"_lr{token(winner['lr'])}_cm{token(winner['cm'])}"
                )
                row.update(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "method": method,
                        "method_label": METHOD_LABEL.get(method, method),
                        "client_optimizer": client_optimizer,
                        "seed": str(seed),
                        "alpha": f"{alpha:g}",
                        "partition_alpha": f"{alpha:g}",
                        "learning_rate": f"{winner['lr']:g}",
                        "learning_rate_status": "frozen_final",
                        "critic_multiplier": f"{winner['cm']:g}",
                        "output_root": str(output_root),
                        "final_result_dir": str(
                            output_root / dataset / method / f"seed_{seed}" / run_id
                        ),
                        "notes": (
                            f"Frozen final-evaluation run. Config selected by validation "
                            f"MSE during the tuning screen (val_mse={winner['val_mse']:.6f}); "
                            f"test MSE for this run is read only after that selection."
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--winners", required=True, help="winners.json from score_highdim_screen_winners.py")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    output_root = (
        Path(args.output_root) if args.output_root
        else REPO_ROOT / "results" / campaign_dir.name
    )

    with open(args.winners) as handle:
        winners = json.load(handle)
    if any(w.get("at_boundary") for w in winners.values()):
        boundary_cells = [k for k, w in winners.items() if w.get("at_boundary")]
        print(f"WARNING: {len(boundary_cells)} winner(s) are still flagged at_boundary: "
              f"{boundary_cells}. Proceeding anyway -- re-run the screen scorer after "
              f"expanding the grid first if you haven't reviewed these yet.\n")

    source_fields, source_rows = load_source()
    fieldnames = source_fields + [f for f in EXTRA_FIELDS if f not in source_fields]
    rows = make_rows(source_rows, winners, output_root)
    write_manifest(campaign_dir / "finals_manifest.csv", fieldnames, rows)

    summary = {
        "campaign": campaign_dir.name,
        "output_root": str(output_root),
        "runs": len(rows),
        "cells": len(winners),
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "comm_round": COMM_ROUND,
        "seed_schedule": "option_2_uniform_3_seeds",
        "winners_source": str(args.winners),
    }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    with (campaign_dir / "setup_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
