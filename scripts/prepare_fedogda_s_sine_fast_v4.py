#!/usr/bin/env python3
"""Prepare the fast Sine FedOGDA-S v4 Stage A boundary probe."""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_sine_fast_v4"
EXP_DIR = REPO_ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
MANIFEST = EXP_DIR / "stage_a_manifest.csv"
SETUP_SUMMARY = EXP_DIR / "setup_summary.json"
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
    "append_round_csv",
    "periodic_checkpoint_interval",
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


def make_run_id(
    *,
    stage: str,
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
        f"v4_{stage}_sin_fedogda_s_seed{seed}_alpha{token(alpha)}"
        f"_T{comm_round}_R{epochs}_batch{batch_size}"
        f"_glr{token(learning_rate)}_cm{token(critic_multiplier)}"
        f"_lam{token(objective_lambda_1)}_slr{token(server_learning_rate)}"
    )


def make_row(
    *,
    stage: str,
    seed: int,
    learning_rate: float,
    critic_multiplier: float,
    objective_lambda_1: float,
    server_learning_rate: float,
    comm_round: int = 500,
    epochs: int = 3,
) -> dict[str, Any]:
    alpha = 1.0
    batch_size = 256
    method = "fedogda_s"
    rid = make_run_id(
        stage=stage,
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
    result_dir = OUTPUT_ROOT / "sin" / method / f"seed_{seed}" / rid
    return {
        "run_id": rid,
        "protocol_version": SCREEN_NAME,
        "run_group": SCREEN_NAME,
        "stage": stage,
        "training_scope": "federated",
        "method": method,
        "method_label": "FedOGDA-S",
        "dataset": "sin",
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
        "learning_rate_status": SCREEN_NAME,
        "weight_decay": 0.0,
        "objective_lambda_1": objective_lambda_1,
        "critic_multiplier": critic_multiplier,
        "server_learning_rate": server_learning_rate,
        "gradient_clip_norm": 1.0,
        "simple_model_selection_epochs": 100,
        "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200,
        "append_round_csv": True,
        "periodic_checkpoint_interval": 0,
        "using_gpu": True,
        "gpu_id": "",
        "log_test_mse_by_round": False,
        "test_mse_used_for_selection": False,
        "selection_metric_source": "validation",
        "selected_without_test": True,
        "notes": (
            "Fast v4 Sine FedOGDA-S boundary probe. Validation-only selection; "
            "Test MSE and curve diagnostics are post-selection readouts only."
        ),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for learning_rate, critic_multiplier, objective_lambda_1 in product(
        [0.0125, 0.015],
        [5.0, 6.0, 7.0, 8.0, 9.0],
        [0.005, 0.01],
    ):
        key = (learning_rate, critic_multiplier, objective_lambda_1, 1.5)
        seen.add(key)
        rows.append(
            make_row(
                stage="stage_a_boundary_probe",
                seed=0,
                learning_rate=learning_rate,
                critic_multiplier=critic_multiplier,
                objective_lambda_1=objective_lambda_1,
                server_learning_rate=1.5,
            )
        )
    for critic_multiplier, objective_lambda_1 in product([7.0, 9.0], [0.005, 0.01]):
        key = (0.01, critic_multiplier, objective_lambda_1, 1.5)
        if key in seen:
            continue
        rows.append(
            make_row(
                stage="stage_a_boundary_probe",
                seed=0,
                learning_rate=0.01,
                critic_multiplier=critic_multiplier,
                objective_lambda_1=objective_lambda_1,
                server_learning_rate=1.5,
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
        "screen_name": SCREEN_NAME,
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "output_root": str(OUTPUT_ROOT),
        "rows": len(rows),
        "stage_a_rows": len(rows),
        "baseline_to_beat": {
            "learning_rate": 0.01,
            "critic_multiplier": 8.0,
            "objective_lambda_1": 0.01,
            "server_learning_rate": 1.5,
            "best_validation_mse": 0.02964276923734608,
            "test_mse_at_best_validation": 0.030104425463267824,
            "curve_mae": 0.13467988046524487,
            "amp_ratio": 0.619,
        },
        "selection_rule": (
            "validation-only: finite/non-diverged; lowest best_validation_mse; "
            "tie lower last_50_val_mse_std; tie lower final_vs_best_validation_gap"
        ),
        "speed_policy": {
            "log_test_mse_by_round": False,
            "periodic_checkpoint_interval": 0,
            "weight_decay": 0.0,
        },
    }
    SETUP_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SETUP_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
