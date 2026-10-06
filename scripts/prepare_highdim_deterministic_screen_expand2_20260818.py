#!/usr/bin/env python3
"""Second boundary-expansion re-screen (2026-08-18), applying the amended
rule in BOUNDARY_RULE_AMENDMENT_20260818.md: one grid rung per axis where
the Psi rank-1 candidate (from psi_rescore.json) touches the tested max,
no recursion, resolved by 500-round/3-seed Rank/Confirm afterward rather
than another screen-stage calculation.

17 candidates across 11 flagged cells. Same 150-round / N=10 / aux-off
screen protocol as both prior screens.
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
PSI_RESCORE_PATH = CAMPAIGN_DIR / "psi_rescore.json"

COMM_ROUND = 150
CLIENT_COUNT = 10

# Method-specific learning-rate step ratio, matching each method's own
# existing grid spacing (FedGDA-D {0.003,0.01,0.03}: ~3.33x; FedOGDA-D
# {0.001,0.003,0.01}: 3x) -- same ratios used for the first expansion round.
LR_STEP = {"fedgda_d": 10.0 / 3.0, "fedogda_d": 3.0}
CM_STEP = 2.0
KNOWN_LR_ROUNDING = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

# Ground truth, mirrored from run_manifest.METHOD_TO_OPTIMIZER /
# experiment_utils.get_effective_config()'s algorithm map. reference_row()
# used to match on dataset only, so every row (regardless of intended
# method) silently inherited client_optimizer/method_label from whichever
# source row for that dataset came first in the CSV (always fedgda_d/sgd) --
# this is what caused the 2026-08-18 mislabeled-FedOGDA incident. Asserted
# explicitly here so a future reference-row regression fails loudly at
# generation time instead of a run_manifest.py "missing artifacts" masking
# a silently-wrong optimizer 150 rounds later.
METHOD_TO_OPTIMIZER = {"fedgda_d": "sgd", "fedogda_d": "ogda"}
METHOD_LABEL = {"fedgda_d": "FedGDA-D", "fedogda_d": "FedOGDA-D"}

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


def round_lr(value: float) -> float:
    for candidate in KNOWN_LR_ROUNDING:
        if abs(candidate - value) < value * 0.05:
            return candidate
    return round(value, 6)


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows: list[dict[str, str]], dataset: str, method: str) -> dict[str, str]:
    matches = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
    if not matches:
        raise RuntimeError(f"No source row for dataset={dataset}, method={method}")
    return matches[0]


def new_candidates_for_cell(method: str, winner_lr: float, winner_cm: float, flags: list[str]):
    axes = {"lr" if f.startswith("lr") else "cm" for f in flags}
    new_lr = round_lr(winner_lr * LR_STEP[method])
    new_cm = winner_cm * CM_STEP
    if axes == {"lr", "cm"}:
        return [(new_lr, winner_cm), (winner_lr, new_cm), (new_lr, new_cm)]
    if axes == {"lr"}:
        return [(new_lr, winner_cm)]
    if axes == {"cm"}:
        return [(winner_lr, new_cm)]
    raise RuntimeError(f"Unexpected boundary flags: {flags}")


def common_row(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "protocol_version": "highdim_deterministic_screen_expand2_v1",
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


def main() -> int:
    psi = json.load(open(PSI_RESCORE_PATH))
    _, source_rows = load_source()
    source_fields, _ = load_source()
    fieldnames = source_fields + [f for f in EXTRA_FIELDS if f not in source_fields]

    rows: list[dict[str, str]] = []
    plan = []
    for key, entry in sorted(psi.items()):
        if not entry["at_boundary"]:
            continue
        dataset, method = entry["dataset"], entry["method"]
        winner = entry["rank_1"]
        candidates = new_candidates_for_cell(method, winner["lr"], winner["cm"], entry["boundary_detail"])
        plan.append({"dataset": dataset, "method": method, "candidates": candidates})
        source = reference_row(source_rows, dataset, method)
        for lr, cm in candidates:
            row = common_row(source)
            run_id = (
                f"det_screen_expand2_{dataset}_{method}_seed0_alpha0p5"
                f"_lr{token(lr)}_cm{token(cm)}"
            )
            row.update(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "method": method,
                    "client_optimizer": METHOD_TO_OPTIMIZER[method],
                    "method_label": METHOD_LABEL[method],
                    "learning_rate": f"{lr:g}",
                    "learning_rate_status": "screen_boundary_expansion2_candidate",
                    "critic_multiplier": f"{cm:g}",
                    "output_root": str(OUTPUT_ROOT),
                    "final_result_dir": str(OUTPUT_ROOT / dataset / method / "seed_0" / run_id),
                    "notes": (
                        "Second boundary-expansion candidate (rule amended 2026-08-18): "
                        "one grid rung, no further screen-stage expansion -- resolved by "
                        "500-round/3-seed Rank/Confirm, not another screen calculation."
                    ),
                }
            )
            rows.append(row)

    manifest_path = CAMPAIGN_DIR / "screen_expand2_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "campaign": "highdim_deterministic_screen_20260813_expand2",
        "rule_source": "BOUNDARY_RULE_AMENDMENT_20260818.md",
        "runs": len(rows),
        "cells_flagged": len(plan),
        "plan": plan,
    }
    with (CAMPAIGN_DIR / "screen_expand2_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
