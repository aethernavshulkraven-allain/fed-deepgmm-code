#!/usr/bin/env python3
"""Create the deterministic Sine FedOGDA-D Stage A1 tuning manifest."""

from __future__ import annotations

import csv
import json
import statistics
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RUNS = REPO_ROOT / "experiments" / "sine_fedogda_tuning" / "current_sine_runs.csv"
OUT_DIR = REPO_ROOT / "experiments" / "sine_fedogda_tuning"
MANIFEST = OUT_DIR / "stage_A1_deterministic_manifest.csv"
SETUP_SUMMARY = OUT_DIR / "stage_A1_deterministic_setup_summary.json"
OUTPUT_ROOT = Path("results") / "sine_fedogda_tuning" / "stage_A1_deterministic"
RUN_GROUP = "sine_fedogda_d_stage_A1"


BASE_FIELDS = [
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def alpha_token(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def lr_token(value: float) -> str:
    text = f"{value:.8g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def current_fedogda_d_config() -> dict[str, Any]:
    candidates = [
        row for row in read_csv(CURRENT_RUNS)
        if row["method"] == "fedogda_d" and row["mode"] == "deterministic"
    ]
    if not candidates:
        raise SystemExit("No current deterministic Sine FedOGDA-D rows found")
    configs = []
    for row in candidates:
        config_path = REPO_ROOT / row["result_dir"] / "effective_config.json"
        configs.append(read_json(config_path))

    keys = [
        "dataset",
        "model",
        "federated_optimizer",
        "client_optimizer",
        "client_num_in_total",
        "client_num_per_round",
        "batch_size",
        "partition_method",
        "data_cache_dir",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "simple_model_selection_epochs",
        "f_history_model_selection_epochs",
        "model_selection_batch_size",
    ]
    merged = {}
    for key in keys:
        values = {str(config.get(key)) for config in configs}
        if len(values) != 1:
            raise SystemExit(f"Current FedOGDA-D configs disagree on {key}: {sorted(values)}")
        merged[key] = configs[0].get(key)
    return merged


def existing_runtime_estimate_seconds() -> float:
    runtimes = []
    for row in read_csv(CURRENT_RUNS):
        if row["method"] == "fedogda_d" and row["mode"] == "deterministic":
            metrics = read_json(REPO_ROOT / row["result_dir"] / "metrics.json")
            runtimes.append(float(metrics["runtime_seconds"]))
    return statistics.fmean(runtimes) * 200.0 / 500.0


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current = current_fedogda_d_config()
    current_g_lr = float(current["learning_rate"])
    full_grid_count = 2 * 2 * 3 * 3 * 4
    estimated_200s = existing_runtime_estimate_seconds()
    estimated_full_hours_2gpu = estimated_200s * full_grid_count / 2.0 / 3600.0

    # This fallback was pre-approved in case 144 is too large. Existing Sine
    # runtime estimates put the full grid at roughly multi-day wall-clock time.
    alphas = [0.5, 1.0]
    epochs = [2, 3]
    critics = [10.0, 15.0]
    server_lrs = [1.5, 2.0]
    learning_rates = [current_g_lr / 2.0, current_g_lr, current_g_lr * 2.0]
    grid_kind = "fallback_48"

    rows = []
    for alpha, local_epochs, critic, server_lr, g_lr in product(
        alphas, epochs, critics, server_lrs, learning_rates
    ):
        run_id = (
            f"stage_A1_sin_fedogda_d_seed0_alpha{alpha_token(alpha)}"
            f"_R{local_epochs}_cm{lr_token(critic)}_slr{lr_token(server_lr)}"
            f"_glr{lr_token(g_lr)}"
        )
        result_dir = OUTPUT_ROOT / "sin" / "fedogda_d" / "seed_0" / run_id
        rows.append({
            "run_id": run_id,
            "protocol_version": "sine_fedogda_tuning_v1",
            "run_group": RUN_GROUP,
            "training_scope": "federated",
            "method": "fedogda_d",
            "method_label": "FedOGDA-D",
            "dataset": current["dataset"],
            "seed": 0,
            "alpha": alpha,
            "output_root": str(OUTPUT_ROOT),
            "final_result_dir": str(result_dir),
            "implementation_status": "ready",
            "run_status": "not_started",
            "preflight_required": False,
            "preflight_status": "not_required",
            "model": current["model"],
            "federated_optimizer": current["federated_optimizer"],
            "client_optimizer": current["client_optimizer"],
            "client_num_in_total": current["client_num_in_total"],
            "client_num_per_round": current["client_num_per_round"],
            "comm_round": 200,
            "epochs": local_epochs,
            "batch_size": current["batch_size"],
            "partition_method": current["partition_method"],
            "partition_alpha": alpha,
            "data_cache_dir": current["data_cache_dir"],
            "learning_rate": g_lr,
            "learning_rate_status": "stage_A1_grid_from_current_fedogda_d",
            "weight_decay": current["weight_decay"],
            "critic_multiplier": critic,
            "server_learning_rate": server_lr,
            "gradient_clip_norm": current["gradient_clip_norm"],
            "simple_model_selection_epochs": current["simple_model_selection_epochs"],
            "f_history_model_selection_epochs": current["f_history_model_selection_epochs"],
            "model_selection_batch_size": current["model_selection_batch_size"],
            "using_gpu": True,
            "gpu_id": "",
            "log_test_mse_by_round": True,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
            "selected_without_test": True,
            "notes": (
                "Stage A1 deterministic Sine FedOGDA-D fallback grid; "
                "candidate ranking validation-only; Test MSE logged for post-selection reporting only."
            ),
        })

    summary = {
        "grid_kind": grid_kind,
        "rows": len(rows),
        "current_g_lr": current_g_lr,
        "alphas": alphas,
        "epochs_R": epochs,
        "critic_multiplier": critics,
        "server_learning_rate": server_lrs,
        "learning_rate": learning_rates,
        "estimated_seconds_per_200_round_run_from_existing_sine_fedogda_d": estimated_200s,
        "estimated_full_144_hours_on_two_gpus": estimated_full_hours_2gpu,
        "estimated_fallback_48_hours_on_two_gpus": estimated_200s * len(rows) / 2.0 / 3600.0,
        "selection_metric": "validation_only",
        "selected_without_test": True,
        "test_mse_used_for_selection": False,
    }
    return rows, summary


def main() -> None:
    rows, summary = build_manifest()
    write_csv(MANIFEST, rows, BASE_FIELDS)
    SETUP_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "setup_summary": str(SETUP_SUMMARY.relative_to(REPO_ROOT)),
        **summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
