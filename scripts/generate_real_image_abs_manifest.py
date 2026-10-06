#!/usr/bin/env python3
"""Generate the abs real-image federated run manifest.

This protocol keeps the federated optimization settings from the rerun
manifest, but limits the data matrix to FEMNIST-digits and CIFAR-10 image
scenarios with ``g_function=abs`` and ``partition_alpha=0.5``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
CSV_PATH = MANIFEST_DIR / "manifest.csv"
JSON_PATH = MANIFEST_DIR / "manifest.json"
SUMMARY_PATH = MANIFEST_DIR / "setup_summary.json"

PROTOCOL_VERSION = "rerun_protocol_v1_real_images_abs_alpha0p5"
OUTPUT_ROOT = "results/rerun_protocol_v1_real_images_abs_alpha0p5"
DATASETS = (
    "femnist_x",
    "femnist_z",
    "femnist_xz",
    "cifar10_x",
    "cifar10_z",
    "cifar10_xz",
)
SEEDS = (0, 1, 2, 3, 4)
FEDERATED_ALPHA = 0.5
G_FUNCTION = "abs"

FEDERATED_METHODS = {
    "fedgda_d": {
        "method_label": "FedGDA-D",
        "client_optimizer": "sgd",
        "client_num_in_total": 1000,
        "client_num_per_round": 1000,
        "batch_size": 0,
        "implementation_status": "ready_for_preflight",
        "preflight_required": True,
    },
    "fedgda_s": {
        "method_label": "FedGDA-S",
        "client_optimizer": "sgd",
        "client_num_in_total": 1000,
        "client_num_per_round": 10,
        "batch_size": 256,
        "implementation_status": "ready_pending_stochastic_smoke",
        "preflight_required": False,
    },
    "fedogda_d": {
        "method_label": "FedOGDA-D",
        "client_optimizer": "ogda",
        "client_num_in_total": 1000,
        "client_num_per_round": 1000,
        "batch_size": 0,
        "implementation_status": "ready_for_preflight",
        "preflight_required": True,
    },
    "fedogda_s": {
        "method_label": "FedOGDA-S",
        "client_optimizer": "ogda",
        "client_num_in_total": 1000,
        "client_num_per_round": 10,
        "batch_size": 256,
        "implementation_status": "ready_pending_stochastic_smoke",
        "preflight_required": False,
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


def final_result_dir(dataset: str, method: str, seed: int, run_id: str) -> str:
    return f"{OUTPUT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}"


def generate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    alpha_text = alpha_label(FEDERATED_ALPHA)
    for dataset in DATASETS:
        for seed in SEEDS:
            for method, spec in FEDERATED_METHODS.items():
                run_id = f"{PROTOCOL_VERSION}_{dataset}_{method}_seed{seed}_alpha{alpha_text}"
                rows.append({
                    "run_id": run_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "run_group": f"real_images_abs_alpha_{alpha_text}",
                    "training_scope": "federated",
                    "method": method,
                    "method_label": spec["method_label"],
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": FEDERATED_ALPHA,
                    "output_root": OUTPUT_ROOT,
                    "final_result_dir": final_result_dir(dataset, method, seed, run_id),
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
                    "partition_alpha": FEDERATED_ALPHA,
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
                    "notes": (
                        "Real-image abs protocol; FEMNIST uses TFF Federated EMNIST "
                        "only_digits=True; CIFAR-10 uses the standard torchvision "
                        "CIFAR10 source; no data regeneration during launch."
                    ),
                })
    return rows


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


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "g_function": G_FUNCTION,
        "femnist_only_digits": True,
        "alpha": FEDERATED_ALPHA,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "methods": list(FEDERATED_METHODS),
        "total_rows": len(rows),
        "federated_rows": len(rows),
        "centralized_rows": 0,
        "data_generation_command": (
            "python "
            "scripts/prepare_real_image_abs_data.py"
        ),
        "dry_run_command_template": (
            "python "
            "scripts/run_manifest.py "
            "--manifest experiments/rerun_protocol_v1_real_images_abs_alpha0p5/manifest.csv "
            "--config-dir experiments/rerun_protocol_v1_real_images_abs_alpha0p5/generated_configs "
            "--output-root results/rerun_protocol_v1_real_images_abs_alpha0p5 "
            "--default-learning-rate <LR> --default-weight-decay <WD> --dry-run"
        ),
    }
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    rows = generate_rows()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, CSV_PATH)
    write_json(rows, JSON_PATH)
    write_summary(rows, SUMMARY_PATH)
    counts = {
        "total": len(rows),
        "federated": len(rows),
        "centralized": 0,
        "csv": str(CSV_PATH.relative_to(REPO_ROOT)),
        "json": str(JSON_PATH.relative_to(REPO_ROOT)),
        "setup_summary": str(SUMMARY_PATH.relative_to(REPO_ROOT)),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
