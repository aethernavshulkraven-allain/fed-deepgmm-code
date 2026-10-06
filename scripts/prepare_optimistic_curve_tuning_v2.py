#!/usr/bin/env python3
"""Prepare the corrected optimistic curve-fitting tuning screen.

Unlike v1, this manifest tunes `objective_lambda_1`, the actual lambda passed
to `OptimalMomentObjective`. `weight_decay` is kept only as a compatibility
field for the launcher/config schema.
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "optimistic_curve_screen_v2"
OUT_DIR = REPO_ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
MANIFEST = OUT_DIR / "manifest.csv"
SETUP_SUMMARY = OUT_DIR / "setup_summary.json"
OUTPUT_ROOT = Path("results") / "curve_fitting_tuning" / SCREEN_NAME

FIELDS = [
    "run_id",
    "protocol_version",
    "run_group",
    "training_scope",
    "method",
    "method_label",
    "dataset",
    "seed",
    "alpha",
    "output_root",
    "final_result_dir",
    "implementation_status",
    "run_status",
    "preflight_required",
    "preflight_status",
    "model",
    "federated_optimizer",
    "client_optimizer",
    "client_num_in_total",
    "client_num_per_round",
    "comm_round",
    "epochs",
    "batch_size",
    "partition_method",
    "partition_alpha",
    "data_cache_dir",
    "learning_rate",
    "learning_rate_status",
    "weight_decay",
    "objective_lambda_1",
    "critic_multiplier",
    "server_learning_rate",
    "gradient_clip_norm",
    "simple_model_selection_epochs",
    "f_history_model_selection_epochs",
    "model_selection_batch_size",
    "using_gpu",
    "gpu_id",
    "log_test_mse_by_round",
    "test_mse_used_for_selection",
    "selection_metric_source",
    "selected_without_test",
    "notes",
]


def token(value: Any) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def base_row(
    *,
    run_id: str,
    dataset: str,
    alpha: float,
    comm_round: int,
    epochs: int,
    learning_rate: float,
    critic_multiplier: float,
    objective_lambda_1: float,
    notes: str,
) -> dict[str, Any]:
    method = "fedogda_s"
    result_dir = OUTPUT_ROOT / dataset / method / "seed_0" / run_id
    return {
        "run_id": run_id,
        "protocol_version": SCREEN_NAME,
        "run_group": SCREEN_NAME,
        "training_scope": "federated",
        "method": method,
        "method_label": "FedOGDA-S",
        "dataset": dataset,
        "seed": 0,
        "alpha": alpha,
        "output_root": str(OUTPUT_ROOT),
        "final_result_dir": str(result_dir),
        "implementation_status": "ready",
        "run_status": "not_started",
        "preflight_required": False,
        "preflight_status": "not_required",
        "model": "lr",
        "federated_optimizer": "FedAvg",
        "client_optimizer": "ogda",
        "client_num_in_total": 1000,
        "client_num_per_round": 10,
        "comm_round": comm_round,
        "epochs": epochs,
        "batch_size": 256,
        "partition_method": "hetero",
        "partition_alpha": alpha,
        "data_cache_dir": "data",
        "learning_rate": learning_rate,
        "learning_rate_status": "corrected_objective_lambda_screen",
        "weight_decay": 0.0,
        "objective_lambda_1": objective_lambda_1,
        "critic_multiplier": critic_multiplier,
        "server_learning_rate": 1.5,
        "gradient_clip_norm": 1.0,
        "simple_model_selection_epochs": 100,
        "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200,
        "using_gpu": True,
        "gpu_id": "",
        "log_test_mse_by_round": True,
        "test_mse_used_for_selection": False,
        "selection_metric_source": "validation",
        "selected_without_test": True,
        "notes": notes,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for learning_rate, critic_multiplier, objective_lambda_1 in product(
        [0.002, 0.005, 0.01],
        [10.0, 15.0],
        [0.03, 0.1],
    ):
        run_id = (
            "curvefit_sin_fedogda_s_seed0_alpha1p0"
            f"_T500_R3_batch256_glr{token(learning_rate)}"
            f"_cm{token(critic_multiplier)}_lam{token(objective_lambda_1)}_slr1p5"
        )
        rows.append(
            base_row(
                run_id=run_id,
                dataset="sin",
                alpha=1.0,
                comm_round=500,
                epochs=3,
                learning_rate=learning_rate,
                critic_multiplier=critic_multiplier,
                objective_lambda_1=objective_lambda_1,
                notes=(
                    "Corrected Sine FedOGDA-S screen. Tunes the real OptimalMomentObjective "
                    "lambda plus generator/critic learning-rate balance; validation-only selection."
                ),
            )
        )

    step_settings = [
        (0.005, 15.0, 0.1),
        (0.005, 15.0, 0.3),
        (0.005, 15.0, 1.0),
        (0.01, 15.0, 0.3),
        (0.01, 15.0, 1.0),
        (0.02, 15.0, 0.3),
    ]
    for learning_rate, critic_multiplier, objective_lambda_1 in step_settings:
        run_id = (
            "curvefit_step_fedogda_s_seed0_alpha0p5"
            f"_T1500_R7_batch256_glr{token(learning_rate)}"
            f"_cm{token(critic_multiplier)}_lam{token(objective_lambda_1)}_slr1p5"
        )
        rows.append(
            base_row(
                run_id=run_id,
                dataset="step",
                alpha=0.5,
                comm_round=1500,
                epochs=7,
                learning_rate=learning_rate,
                critic_multiplier=critic_multiplier,
                objective_lambda_1=objective_lambda_1,
                notes=(
                    "Corrected Step FedOGDA-S screen around the reproduced reference recipe. "
                    "Existing lr=0.01/lambda=0.1 reference is included during analysis, not relaunched."
                ),
            )
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    rows = build_rows()
    write_csv(MANIFEST, rows)
    summary = {
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "output_root": str(OUTPUT_ROOT),
        "rows": len(rows),
        "datasets": {
            "sin": sum(row["dataset"] == "sin" for row in rows),
            "step": sum(row["dataset"] == "step" for row in rows),
        },
        "selection_rule": (
            "validation-only: finite/non-diverged; lowest best_validation_mse; "
            "tie lower last-50 validation std; tie lower final-vs-best validation gap"
        ),
        "actual_tuned_regularizer": "objective_lambda_1",
        "compatibility_only_fields": ["weight_decay"],
        "post_selection_only": ["test_mse_at_best_validation", "curve_mse", "curve_mae", "visual curve"],
    }
    SETUP_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
