#!/usr/bin/env python3
"""Prepare focused optimistic curve-fitting tuning manifests.

The grid is intentionally small and validation-driven:

* Sine: tune FedOGDA-S with lower regularization, because the current tuned
  deterministic FedOGDA-D is numerically better but visually too smooth.
* Step: tune FedOGDA-S around the reproduced reference recipe, excluding the
  already-completed reference row.
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "curve_fitting_tuning" / "optimistic_curve_screen_v1"
MANIFEST = OUT_DIR / "manifest.csv"
SETUP_SUMMARY = OUT_DIR / "setup_summary.json"
OUTPUT_ROOT = Path("results") / "curve_fitting_tuning" / "optimistic_curve_screen_v1"
RUN_GROUP = "optimistic_curve_screen_v1"

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
    method: str,
    method_label: str,
    dataset: str,
    alpha: float,
    comm_round: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    critic_multiplier: float,
    server_learning_rate: float,
    notes: str,
) -> dict[str, Any]:
    result_dir = OUTPUT_ROOT / dataset / method / "seed_0" / run_id
    return {
        "run_id": run_id,
        "protocol_version": "optimistic_curve_tuning_v1",
        "run_group": RUN_GROUP,
        "training_scope": "federated",
        "method": method,
        "method_label": method_label,
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
        "batch_size": batch_size,
        "partition_method": "hetero",
        "partition_alpha": alpha,
        "data_cache_dir": "data",
        "learning_rate": learning_rate,
        "learning_rate_status": "focused_validation_screen",
        "weight_decay": weight_decay,
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

    # Sine stochastic optimistic screen: current FedOGDA-S uses wd=0.1 and
    # looks as smooth as the deterministic run. Lower wd is the main probe.
    for epochs, learning_rate, weight_decay in product([3, 7], [0.001, 0.002, 0.005], [0.001, 0.01]):
        run_id = (
            "curvefit_sin_fedogda_s_seed0_alpha1p0"
            f"_T500_R{epochs}_batch256_glr{token(learning_rate)}"
            f"_wd{token(weight_decay)}_cm15_slr1p5"
        )
        rows.append(
            base_row(
                run_id=run_id,
                method="fedogda_s",
                method_label="FedOGDA-S",
                dataset="sin",
                alpha=1.0,
                comm_round=500,
                epochs=epochs,
                batch_size=256,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                critic_multiplier=15.0,
                server_learning_rate=1.5,
                notes=(
                    "Sine FedOGDA-S lower-regularization screen; rank by validation only. "
                    "Test and curve metrics are post-selection diagnostics."
                ),
            )
        )

    # Step stochastic optimistic screen around the earlier reference recipe. The exact
    # reproduced reference row glr=0.01/wd=0.02 is already completed elsewhere,
    # so this screen tests nearby settings without overwriting it.
    for learning_rate, weight_decay in product([0.005, 0.02, 0.03, 0.05], [0.005, 0.02]):
        run_id = (
            "curvefit_step_fedogda_s_seed0_alpha0p5"
            f"_T1500_R7_batch256_glr{token(learning_rate)}"
            f"_wd{token(weight_decay)}_cm15_slr1p5"
        )
        rows.append(
            base_row(
                run_id=run_id,
                method="fedogda_s",
                method_label="FedOGDA-S",
                dataset="step",
                alpha=0.5,
                comm_round=1500,
                epochs=7,
                batch_size=256,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                critic_multiplier=15.0,
                server_learning_rate=1.5,
                notes=(
                    "Step FedOGDA-S screen around reproduced reference recipe; rank by validation only. "
                    "Existing glr=0.01/wd=0.02 reference is included during analysis, not relaunched."
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
            "tie lower last-50 validation std from mse_by_round; tie lower final-vs-best validation gap"
        ),
        "post_selection_only": ["test_mse_at_best_validation", "curve_mse", "curve_mae", "visual curve"],
    }
    SETUP_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
