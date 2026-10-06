#!/usr/bin/env python3
"""Prepare the FedOGDA-S focused v3 tuning screen.

This screen tunes only knobs that are wired into the federated DeepGMM path:
generator learning rate, critic multiplier, server learning rate, and the
objective regularizer lambda. ``weight_decay`` is kept at zero because it is a
launcher/config compatibility field for these runs.
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_focused_v3"
OUT_DIR = REPO_ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
MANIFEST = OUT_DIR / "manifest.csv"
SETUP_SUMMARY = OUT_DIR / "setup_summary.json"
OUTPUT_ROOT = Path("results") / "curve_fitting_tuning" / SCREEN_NAME

FIELDS = [
    "run_id",
    "protocol_version",
    "run_group",
    "stage",
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


def run_id(
    *,
    stage: str,
    dataset: str,
    seed: int,
    alpha: float,
    comm_round: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    critic_multiplier: float,
    objective_lambda_1: float,
    server_learning_rate: float,
) -> str:
    return (
        f"v3_{stage}_{dataset}_fedogda_s_seed{seed}_alpha{token(alpha)}"
        f"_T{comm_round}_R{epochs}_batch{batch_size}"
        f"_glr{token(learning_rate)}_cm{token(critic_multiplier)}"
        f"_lam{token(objective_lambda_1)}_slr{token(server_learning_rate)}"
    )


def row(
    *,
    stage: str,
    dataset: str,
    alpha: float,
    seed: int,
    comm_round: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    critic_multiplier: float,
    objective_lambda_1: float,
    server_learning_rate: float,
    notes: str,
) -> dict[str, Any]:
    method = "fedogda_s"
    rid = run_id(
        stage=stage,
        dataset=dataset,
        seed=seed,
        alpha=alpha,
        comm_round=comm_round,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        critic_multiplier=critic_multiplier,
        objective_lambda_1=objective_lambda_1,
        server_learning_rate=server_learning_rate,
    )
    result_dir = OUTPUT_ROOT / dataset / method / f"seed_{seed}" / rid
    return {
        "run_id": rid,
        "protocol_version": SCREEN_NAME,
        "run_group": SCREEN_NAME,
        "stage": stage,
        "training_scope": "federated",
        "method": method,
        "method_label": "FedOGDA-S",
        "dataset": dataset,
        "seed": seed,
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
        "batch_size": batch_size,
        "partition_method": "hetero",
        "partition_alpha": alpha,
        "data_cache_dir": "data",
        "learning_rate": learning_rate,
        "learning_rate_status": "fedogda_s_focused_v3",
        "weight_decay": 0.0,
        "objective_lambda_1": objective_lambda_1,
        "critic_multiplier": critic_multiplier,
        "server_learning_rate": server_learning_rate,
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

    for learning_rate, critic_multiplier, objective_lambda_1, server_learning_rate in product(
        [0.004, 0.005, 0.0075, 0.01],
        [6.0, 8.0, 10.0, 12.0],
        [0.01, 0.03],
        [1.0, 1.5],
    ):
        rows.append(
            row(
                stage="screen_sine",
                dataset="sin",
                alpha=1.0,
                seed=0,
                comm_round=500,
                epochs=3,
                batch_size=256,
                learning_rate=learning_rate,
                critic_multiplier=critic_multiplier,
                objective_lambda_1=objective_lambda_1,
                server_learning_rate=server_learning_rate,
                notes=(
                    "FedOGDA-S focused v3 Sine seed-0 screen. Validation-only selection; "
                    "test MSE is post-selection only."
                ),
            )
        )

    for learning_rate, critic_multiplier, objective_lambda_1, server_learning_rate in product(
        [0.003, 0.005, 0.0075],
        [10.0, 15.0, 20.0],
        [0.05, 0.1],
        [1.0, 1.5, 2.0],
    ):
        rows.append(
            row(
                stage="screen_step_proxy",
                dataset="step",
                alpha=0.5,
                seed=0,
                comm_round=600,
                epochs=7,
                batch_size=256,
                learning_rate=learning_rate,
                critic_multiplier=critic_multiplier,
                objective_lambda_1=objective_lambda_1,
                server_learning_rate=server_learning_rate,
                notes=(
                    "FedOGDA-S focused v3 Step proxy screen. Full Step confirmation "
                    "uses validation-selected proxy configs only."
                ),
            )
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({field: item.get(field, "") for field in FIELDS})


def main() -> None:
    rows = build_rows()
    write_csv(MANIFEST, rows)
    summary = {
        "screen_name": SCREEN_NAME,
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "output_root": str(OUTPUT_ROOT),
        "rows": len(rows),
        "stages": {
            "screen_sine": sum(item["stage"] == "screen_sine" for item in rows),
            "screen_step_proxy": sum(item["stage"] == "screen_step_proxy" for item in rows),
        },
        "selection_rule": (
            "validation-only: finite/non-diverged; lowest best_validation_mse; "
            "tie lower last_50_val_mse_std; tie lower final_vs_best_validation_gap"
        ),
        "tuned_fields": [
            "learning_rate",
            "critic_multiplier",
            "objective_lambda_1",
            "server_learning_rate",
        ],
        "fixed_fields": {"weight_decay": 0.0},
        "post_selection_only": [
            "test_mse_at_best_validation",
            "curve_mse",
            "curve_mae",
            "curve_corr",
            "visual curve",
        ],
    }
    SETUP_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
