#!/usr/bin/env python3
"""Prepare the guarded high-dimensional deterministic learning gate."""

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
    / "experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802"
)
OUTPUT_ROOT = REPO_ROOT / "results/highdim_deterministic_learning_gate_20260802"
METHODS = ("fedgda_d", "fedogda_d")
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03)
CRITIC_MULTIPLIERS = (1.0, 3.0, 10.0)
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


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["dataset"] == "femnist_z"
        and row["method"] == method
        and float(row["learning_rate"]) == 0.001
        and float(row["weight_decay"]) == 0.001
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one reference row for {method}, found {len(matches)}")
    return matches[0]


def common_row(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "protocol_version": "highdim_deterministic_learning_gate_v1",
            "run_group": "highdim_deterministic_learning_gate_20260802",
            "training_scope": "federated",
            "dataset": "femnist_z",
            "seed": "0",
            "alpha": "0.5",
            "partition_alpha": "0.5",
            "client_num_in_total": "1000",
            "client_num_per_round": "1000",
            "epochs": "3",
            "batch_size": "0",
            "weight_decay": "0.001",
            "server_learning_rate": "1.5",
            "gradient_clip_norm": "1.0",
            "run_status": "not_started",
            "implementation_status": "ready",
            "preflight_required": "False",
            "preflight_status": "not_required",
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


def make_equivalence_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for method in METHODS:
        source = reference_row(rows, method)
        for auxiliary in (True, False):
            row = common_row(source)
            suffix = "auxon" if auxiliary else "auxoff"
            run_id = f"det_gate_equiv_femnist_z_{method}_seed0_alpha0p5_{suffix}_r10"
            row.update(
                {
                    "run_id": run_id,
                    "comm_round": "10",
                    "learning_rate": "0.003",
                    "learning_rate_status": "equivalence_probe",
                    "critic_multiplier": "10.0",
                    "auxiliary_regression": str(auxiliary),
                    "auxiliary_regression_epochs": "3" if auxiliary else "0",
                    "output_root": str(OUTPUT_ROOT / "equivalence"),
                    "final_result_dir": str(
                        OUTPUT_ROOT
                        / "equivalence"
                        / "femnist_z"
                        / method
                        / "seed_0"
                        / run_id
                    ),
                    "notes": (
                        "Non-production 10-round paired auxiliary-regression equivalence "
                        "and timing probe; validation-only configuration."
                    ),
                }
            )
            output.append(row)
    return output


def make_gate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for method in METHODS:
        source = reference_row(rows, method)
        for learning_rate in LEARNING_RATES:
            for critic_multiplier in CRITIC_MULTIPLIERS:
                row = common_row(source)
                run_id = (
                    f"det_gate_femnist_z_{method}_seed0_alpha0p5"
                    f"_lr{token(learning_rate)}_cm{token(critic_multiplier)}"
                )
                row.update(
                    {
                        "run_id": run_id,
                        "comm_round": "150",
                        "learning_rate": f"{learning_rate:g}",
                        "learning_rate_status": "validation_learning_gate_candidate",
                        "critic_multiplier": f"{critic_multiplier:g}",
                        "auxiliary_regression": "False",
                        "auxiliary_regression_epochs": "0",
                        "output_root": str(OUTPUT_ROOT / "gate"),
                        "final_result_dir": str(
                            OUTPUT_ROOT
                            / "gate"
                            / "femnist_z"
                            / method
                            / "seed_0"
                            / run_id
                        ),
                        "notes": (
                            "Non-production deterministic learning-gate candidate. Select by "
                            "validation MSE only; do not use Test MSE for hyperparameter selection."
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
    equivalence_rows = make_equivalence_rows(source_rows)
    gate_rows = make_gate_rows(source_rows)
    write_manifest(CAMPAIGN_DIR / "equivalence_manifest.csv", fieldnames, equivalence_rows)
    write_manifest(CAMPAIGN_DIR / "gate_manifest.csv", fieldnames, gate_rows)

    summary = {
        "campaign": "highdim_deterministic_learning_gate_20260802",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "output_root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
        "equivalence_runs": len(equivalence_rows),
        "gate_runs": len(gate_rows),
        "methods": list(METHODS),
        "dataset": "femnist_z",
        "alpha": 0.5,
        "seed": 0,
        "equivalence_rounds": 10,
        "gate_rounds": 150,
        "learning_rates": list(LEARNING_RATES),
        "critic_multipliers": list(CRITIC_MULTIPLIERS),
        "client_num_in_total": 1000,
        "client_num_per_round": 1000,
        "local_epochs": 3,
        "batch_size": 0,
        "weight_decay": 0.001,
        "server_learning_rate": 1.5,
        "gradient_clip_norm": 1.0,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "gate_auxiliary_regression": False,
    }
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN_DIR / "setup_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
