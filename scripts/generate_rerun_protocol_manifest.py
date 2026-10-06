#!/usr/bin/env python3
"""Generate the rerun_protocol_v1 final-run manifest.

The manifest is intentionally declarative: federated rows are launch-ready,
while centralized rows are recorded as intended final runs but marked pending
until the true centralized DeepGMM runner is verified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1"
CSV_PATH = MANIFEST_DIR / "manifest.csv"
JSON_PATH = MANIFEST_DIR / "manifest.json"

PROTOCOL_VERSION = "rerun_protocol_v1"
OUTPUT_ROOT = "results/rerun_protocol_v1"
DATASETS = ("abs", "step", "linear", "sin")
SEEDS = (0, 1, 2)
FEDERATED_ALPHAS = (0.1, 0.5, 1.0)

FEDERATED_METHODS = {
    "fedgda_d": {
        "method_label": "FedGDA-D",
        "client_optimizer": "sgd",
        "stochasticity": "deterministic",
        "client_num_in_total": 1000,
        "client_num_per_round": 1000,
        "batch_size": 0,
        "implementation_status": "ready_for_preflight",
        "preflight_required": True,
    },
    "fedgda_s": {
        "method_label": "FedGDA-S",
        "client_optimizer": "sgd",
        "stochasticity": "stochastic",
        "client_num_in_total": 1000,
        "client_num_per_round": 10,
        "batch_size": 256,
        "implementation_status": "ready_pending_stochastic_smoke",
        "preflight_required": False,
    },
    "fedogda_d": {
        "method_label": "FedOGDA-D",
        "client_optimizer": "ogda",
        "stochasticity": "deterministic",
        "client_num_in_total": 1000,
        "client_num_per_round": 1000,
        "batch_size": 0,
        "implementation_status": "ready_for_preflight",
        "preflight_required": True,
    },
    "fedogda_s": {
        "method_label": "FedOGDA-S",
        "client_optimizer": "ogda",
        "stochasticity": "stochastic",
        "client_num_in_total": 1000,
        "client_num_per_round": 10,
        "batch_size": 256,
        "implementation_status": "ready_pending_stochastic_smoke",
        "preflight_required": False,
    },
}

CENTRALIZED_METHODS = {
    "gda_d": {
        "method_label": "DeepGMM-GDA",
        "stochasticity": "deterministic",
    },
    "sgda_s": {
        "method_label": "DeepGMM-SGDA",
        "stochasticity": "stochastic",
    },
    "oadam_d": {
        "method_label": "DeepGMM-OAdam-D",
        "stochasticity": "deterministic",
    },
    "oadam_s": {
        "method_label": "DeepGMM-OAdam-S",
        "stochasticity": "stochastic",
    },
}

FIELDNAMES = (
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
    "notes",
)


def alpha_label(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def final_result_dir(dataset: str, method: str, seed: int, run_id: str, training_scope: str) -> str:
    if training_scope == "federated":
        return f"{OUTPUT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}"
    return f"{OUTPUT_ROOT}/centralized/{dataset}/{method}/seed_{seed}/{run_id}"


def federated_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for alpha in FEDERATED_ALPHAS:
            alpha_text = alpha_label(alpha)
            for seed in SEEDS:
                for method, spec in FEDERATED_METHODS.items():
                    run_id = f"{PROTOCOL_VERSION}_{dataset}_{method}_seed{seed}_alpha{alpha_text}"
                    rows.append({
                        "run_id": run_id,
                        "protocol_version": PROTOCOL_VERSION,
                        "run_group": f"federated_alpha_{alpha_text}",
                        "training_scope": "federated",
                        "method": method,
                        "method_label": spec["method_label"],
                        "dataset": dataset,
                        "seed": seed,
                        "alpha": alpha,
                        "output_root": OUTPUT_ROOT,
                        "final_result_dir": final_result_dir(dataset, method, seed, run_id, "federated"),
                        "implementation_status": spec["implementation_status"],
                        "run_status": "not_started",
                        "preflight_required": spec["preflight_required"],
                        "preflight_status": "not_started" if spec["preflight_required"] else "not_required",
                        "model": "lr",
                        "federated_optimizer": "FedAvg",
                        "client_optimizer": spec["client_optimizer"],
                        "client_num_in_total": spec["client_num_in_total"],
                        "client_num_per_round": spec["client_num_per_round"],
                        "comm_round": 500,
                        "epochs": 3,
                        "batch_size": spec["batch_size"],
                        "partition_method": "hetero",
                        "partition_alpha": alpha,
                        "data_cache_dir": "data",
                        "learning_rate": "",
                        "learning_rate_status": "to_be_selected_from_validation_tuning_grid",
                        "weight_decay": "",
                        "critic_multiplier": 10.0,
                        "server_learning_rate": 1.5,
                        "gradient_clip_norm": 1.0,
                        "simple_model_selection_epochs": 100,
                        "f_history_model_selection_epochs": 60,
                        "model_selection_batch_size": 200,
                        "using_gpu": True,
                        "gpu_id": "",
                        "notes": "Federated rerun protocol; alpha sweep included; no data regeneration.",
                    })
    return rows


def centralized_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for method, spec in CENTRALIZED_METHODS.items():
                run_id = f"{PROTOCOL_VERSION}_{dataset}_{method}_seed{seed}"
                rows.append({
                    "run_id": run_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "run_group": "centralized_pending_runner_verification",
                    "training_scope": "centralized",
                    "method": method,
                    "method_label": spec["method_label"],
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": "na",
                    "output_root": OUTPUT_ROOT,
                    "final_result_dir": final_result_dir(dataset, method, seed, run_id, "centralized"),
                    "implementation_status": "blocked_pending_true_centralized_runner_verification",
                    "run_status": "blocked",
                    "preflight_required": False,
                    "preflight_status": "not_required",
                    "model": "lr",
                    "federated_optimizer": "na",
                    "client_optimizer": "na",
                    "client_num_in_total": "na",
                    "client_num_per_round": "na",
                    "comm_round": "na",
                    "epochs": "na",
                    "batch_size": "na",
                    "partition_method": "na",
                    "partition_alpha": "na",
                    "data_cache_dir": "data",
                    "learning_rate": "",
                    "learning_rate_status": "pending_centralized_runner_verification",
                    "weight_decay": "",
                    "critic_multiplier": "na",
                    "server_learning_rate": "na",
                    "gradient_clip_norm": "na",
                    "simple_model_selection_epochs": "na",
                    "f_history_model_selection_epochs": "na",
                    "model_selection_batch_size": "na",
                    "using_gpu": True,
                    "gpu_id": "",
                    "notes": "Intended final centralized row; do not launch until true centralized DeepGMM/OAdam implementation is verified.",
                })
    return rows


def generate_rows() -> list[dict[str, object]]:
    return federated_rows() + centralized_rows()


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    rows = generate_rows()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, CSV_PATH)
    write_json(rows, JSON_PATH)
    counts = {
        "total": len(rows),
        "federated": sum(row["training_scope"] == "federated" for row in rows),
        "centralized": sum(row["training_scope"] == "centralized" for row in rows),
        "deterministic_federated_preflights": sum(
            row["training_scope"] == "federated"
            and row["preflight_required"] is True
            for row in rows
        ),
        "csv": str(CSV_PATH.relative_to(REPO_ROOT)),
        "json": str(JSON_PATH.relative_to(REPO_ROOT)),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
