#!/usr/bin/env python3
"""Prepare matched FedSGDA/FedOGDA-S Sine tuning at alpha=0.5."""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

from prepare_fedogda_s_sine_fast_v4 import FIELDS


ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "sine_alpha0p5_paired_v1"
EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
OUTPUT_ROOT = Path("results") / "curve_fitting_tuning" / SCREEN_NAME
STAGE_A_MANIFEST = EXP_DIR / "stage_a_manifest.csv"

METHOD_INFO = {
    "fedgda_s": ("FedGDA-S", "sgd"),
    "fedogda_s": ("FedOGDA-S", "ogda"),
}


def token(value: Any) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def make_row(
    *,
    method: str,
    stage: str,
    seed: int,
    learning_rate: float,
    critic_multiplier: float,
    objective_lambda_1: float,
    server_learning_rate: float,
    comm_round: int,
) -> dict[str, Any]:
    if method not in METHOD_INFO:
        raise ValueError(f"unsupported method: {method}")
    method_label, client_optimizer = METHOD_INFO[method]
    alpha = 0.5
    epochs = 3
    batch_size = 256
    run_id = (
        f"sine05_v1_{stage}_{method}_seed{seed}_alpha0p5"
        f"_T{comm_round}_R{epochs}_batch{batch_size}"
        f"_glr{token(learning_rate)}_cm{token(critic_multiplier)}"
        f"_lam{token(objective_lambda_1)}_slr{token(server_learning_rate)}"
    )
    result_dir = OUTPUT_ROOT / "sin" / method / f"seed_{seed}" / run_id
    return {
        "run_id": run_id,
        "protocol_version": SCREEN_NAME,
        "run_group": SCREEN_NAME,
        "stage": stage,
        "training_scope": "federated",
        "method": method,
        "method_label": method_label,
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
        "client_optimizer": client_optimizer,
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
            "Matched Sine alpha=0.5 paired tuning. Validation-only configuration "
            "and checkpoint selection; test metrics are post-selection readouts."
        ),
    }


def stage_a_configs() -> list[tuple[float, float, float, float]]:
    configs = {
        (learning_rate, critic_multiplier, objective_lambda_1, 1.5)
        for learning_rate, critic_multiplier, objective_lambda_1 in product(
            [0.005, 0.0075, 0.01],
            [7.0, 10.0],
            [0.005, 0.01],
        )
    }
    configs.update(
        {
            # Transfer the best alpha=1.0 Sine neighborhood.
            (0.01, 7.0, 0.005, 1.75),
            (0.0125, 7.0, 0.005, 1.5),
            (0.01, 5.0, 0.005, 1.5),
            # Include the current alpha=0.5 preset so tuning cannot silently
            # discard the existing baseline.
            (0.003, 10.0, 0.1, 1.5),
        }
    )
    return sorted(configs)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    configs = stage_a_configs()
    rows = [
        make_row(
            method=method,
            stage="stage_a_screen",
            seed=0,
            learning_rate=config[0],
            critic_multiplier=config[1],
            objective_lambda_1=config[2],
            server_learning_rate=config[3],
            comm_round=500,
        )
        for method in METHOD_INFO
        for config in configs
    ]
    write_manifest(STAGE_A_MANIFEST, rows)
    summary = {
        "screen_name": SCREEN_NAME,
        "alpha": 0.5,
        "methods": sorted(METHOD_INFO),
        "stage_a_configs_per_method": len(configs),
        "stage_a_rows": len(rows),
        "manifest": str(STAGE_A_MANIFEST.relative_to(ROOT)),
        "output_root": str(OUTPUT_ROOT),
        "selection_rule": (
            "validation-only: lowest best_validation_mse; tie lower "
            "last_50_val_mse_std; tie lower final_vs_best_validation_gap"
        ),
        "stage_plan": {
            "stage_a": "16 matched seed-0 configurations/method, 500 rounds",
            "stage_b": "top 2/method, server learning-rate refinement, 500 rounds",
            "stage_c": "top 2/method, seeds 0-2, 1000 rounds",
        },
    }
    (EXP_DIR / "setup_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
