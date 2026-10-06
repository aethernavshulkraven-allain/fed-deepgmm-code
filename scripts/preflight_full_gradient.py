#!/usr/bin/env python3
"""Preflight deterministic full-gradient FedML synthetic-data runs.

This script intentionally stops before model creation/training.  It loads data
through the same synthetic FedML data path used by local runs, then checks the
full-gradient invariants needed for deterministic federated experiments:

* every round uses all clients,
* selected clients have exactly one local train batch,
* that batch contains the whole local client dataset,
* the effective config still records ``batch_size: 0``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from fedml.data.data_loader import load_synthetic_data  # noqa: E402
from fedgmm.sp_decentralized_mnist_lr_example.experiment_utils import (  # noqa: E402
    build_run_dir,
    client_indices_for_round,
    get_effective_config,
    json_safe,
    prepare_run_dir,
    write_effective_config,
)


ZOO_DATASETS = {"abs", "step", "linear", "sin"}


@dataclass
class FullGradientPreflightError(Exception):
    errors: list[str]

    def __str__(self) -> str:
        return "\n".join(self.errors)


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flatten_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    flattened: dict[str, Any] = {}
    for section_values in raw.values():
        if isinstance(section_values, dict):
            flattened.update(section_values)
    return flattened


def _default_config() -> dict[str, Any]:
    return {
        "dataset": "abs",
        "model": "lr",
        "federated_optimizer": "FedAvg",
        "client_id_list": "[]",
        "client_num_in_total": 1000,
        "client_num_per_round": 1000,
        "comm_round": 1,
        "epochs": 3,
        "frequency_of_the_test": 1,
        "random_seed": 0,
        "partition_method": "hetero",
        "partition_alpha": 0.5,
        "data_cache_dir": "data",
        "batch_size": 0,
        "client_optimizer": "sgd",
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "critic_multiplier": 10.0,
        "server_learning_rate": 1.5,
        "gradient_clip_norm": 1.0,
        "simple_model_selection_epochs": 100,
        "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200,
        "output_dir": os.path.join("experiments", "preflight", "full_gradient"),
        "run_id": "preflight_full_gradient",
        "using_gpu": False,
        "gpu_id": 0,
        "enable_legacy_outputs": False,
        "overwrite": False,
        "scenario_name": None,
    }


def _config_from_args(args: argparse.Namespace) -> SimpleNamespace:
    config = _default_config()
    if args.config is not None:
        config.update(_flatten_yaml_config(args.config))

    explicit_updates = {
        "dataset": args.dataset,
        "random_seed": args.seed,
        "partition_alpha": args.partition_alpha,
        "client_num_in_total": args.client_num_in_total,
        "client_num_per_round": args.client_num_per_round,
        "batch_size": args.batch_size,
        "client_optimizer": args.client_optimizer,
        "output_dir": args.output_dir,
        "run_id": args.run_id,
        "overwrite": args.overwrite,
        "using_gpu": args.using_gpu,
        "gpu_id": args.gpu_id,
    }
    for key, value in explicit_updates.items():
        if value is not None:
            config[key] = value

    config["dataset"] = str(config["dataset"])
    if config["dataset"].lower() == "sine":
        config["dataset"] = "sin"
    config["random_seed"] = int(config["random_seed"])
    config["partition_alpha"] = float(config["partition_alpha"])
    config["client_num_in_total"] = int(config["client_num_in_total"])
    config["client_num_per_round"] = int(config["client_num_per_round"])
    config["batch_size"] = int(config["batch_size"])
    config["gpu_id"] = int(config["gpu_id"])
    config["using_gpu"] = bool(config["using_gpu"])
    config["overwrite"] = bool(config["overwrite"])
    return SimpleNamespace(**{key: _coerce_scalar(value) for key, value in config.items()})


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_example_cwd() -> str:
    original_cwd = os.getcwd()
    os.chdir(EXAMPLE_ROOT)
    return original_cwd


def _batch_tensors(batch: Any) -> list[Any]:
    if isinstance(batch, dict):
        return list(batch.values())
    if isinstance(batch, (list, tuple)):
        return list(batch)
    return [batch]


def infer_batch_sample_count(batch: Any) -> int:
    """Return the shared leading dimension for all tensor-like batch fields."""
    sizes: list[int] = []
    for item in _batch_tensors(batch):
        shape = getattr(item, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        sizes.append(int(shape[0]))
    if not sizes:
        raise ValueError("Could not infer batch sample count")
    first = sizes[0]
    mismatched = sorted(set(sizes))
    if len(mismatched) != 1:
        raise ValueError(f"Batch fields have inconsistent leading dimensions: {mismatched}")
    return first


def _loader_to_list(loader_or_batches: Any) -> list[Any]:
    if isinstance(loader_or_batches, list):
        return loader_or_batches
    return list(loader_or_batches)


def validate_full_gradient_invariants(
    args: SimpleNamespace,
    dataset: list[Any],
    effective_config: dict[str, Any],
    *,
    rounds_to_check: int = 1,
    client_detail_limit: int = 20,
) -> dict[str, Any]:
    errors: list[str] = []
    if int(args.batch_size) != 0:
        errors.append(f"Expected restored args.batch_size=0, got {args.batch_size}")
    if int(args.client_num_per_round) != int(args.client_num_in_total):
        errors.append(
            "Deterministic full-gradient runs require "
            f"client_num_per_round == client_num_in_total, got "
            f"{args.client_num_per_round} != {args.client_num_in_total}"
        )
    if effective_config.get("batch_size") != 0:
        errors.append(
            f"effective_config must record batch_size=0, got {effective_config.get('batch_size')}"
        )
    if effective_config.get("mode") != "deterministic":
        errors.append(
            f"effective_config must record deterministic mode, got {effective_config.get('mode')}"
        )

    (
        _train_data_num,
        _test_data_num,
        _val_data_num,
        _train_data_global,
        _test_data_global,
        _val_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        _test_data_local_dict,
        _val_data_local_dict,
        _class_num,
    ) = dataset

    selected_client_count = 0
    checked_client_count = 0
    sample_counts: list[int] = []
    detail_rows: list[dict[str, Any]] = []
    failed_clients: list[dict[str, Any]] = []

    for round_idx in range(int(rounds_to_check)):
        selected_clients = client_indices_for_round(
            int(args.random_seed),
            round_idx,
            int(args.client_num_in_total),
            int(args.client_num_per_round),
        )
        selected_client_count += len(selected_clients)
        expected_clients = list(range(int(args.client_num_in_total)))
        if selected_clients != expected_clients:
            errors.append(
                f"Round {round_idx} did not select all clients in order "
                f"(selected {len(selected_clients)} clients)."
            )

        for client_id in selected_clients:
            checked_client_count += 1
            batches = _loader_to_list(train_data_local_dict[client_id])
            sample_count = int(train_data_local_num_dict[client_id])
            sample_counts.append(sample_count)
            batch_count = len(batches)
            actual_batch_size = None
            client_error = None
            if batch_count != 1:
                client_error = (
                    f"client {client_id} round {round_idx}: expected 1 train batch, "
                    f"got {batch_count}"
                )
            else:
                try:
                    actual_batch_size = infer_batch_sample_count(batches[0])
                    if actual_batch_size != sample_count:
                        client_error = (
                            f"client {client_id} round {round_idx}: expected full local "
                            f"batch size {sample_count}, got {actual_batch_size}"
                        )
                except ValueError as exc:
                    client_error = f"client {client_id} round {round_idx}: {exc}"

            row = {
                "round": round_idx,
                "client_id": int(client_id),
                "sample_count": sample_count,
                "num_train_batches": batch_count,
                "actual_batch_size": actual_batch_size,
            }
            if len(detail_rows) < client_detail_limit:
                detail_rows.append(row)
            if client_error is not None:
                failed_clients.append({**row, "error": client_error})
                errors.append(client_error)

    report = {
        "status": "passed" if not errors else "failed",
        "dataset": args.dataset,
        "seed": int(args.random_seed),
        "partition_alpha": float(args.partition_alpha),
        "client_num_in_total": int(args.client_num_in_total),
        "client_num_per_round": int(args.client_num_per_round),
        "configured_batch_size": int(args.batch_size),
        "effective_config_batch_size": effective_config.get("batch_size"),
        "effective_config_mode": effective_config.get("mode"),
        "rounds_checked": int(rounds_to_check),
        "selected_client_count": selected_client_count,
        "checked_client_count": checked_client_count,
        "sample_count_min": min(sample_counts) if sample_counts else None,
        "sample_count_max": max(sample_counts) if sample_counts else None,
        "sample_count_sum": sum(sample_counts),
        "client_detail_limit": int(client_detail_limit),
        "client_details": detail_rows,
        "failed_clients": failed_clients,
        "errors": errors,
    }
    if errors:
        raise FullGradientPreflightError(errors)
    return report


def write_preflight_report(run_dir: str, report: dict[str, Any]) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "full_gradient_preflight.json")
    with open(path, "w") as f:
        json.dump(json_safe(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run_preflight(args: SimpleNamespace, *, rounds_to_check: int, client_detail_limit: int) -> dict[str, Any]:
    _seed_everything(int(args.random_seed))
    original_cwd = _ensure_example_cwd()
    try:
        dataset, _class_num = load_synthetic_data(args)
    finally:
        os.chdir(original_cwd)

    effective_config = get_effective_config(args)
    run_dir = build_run_dir(
        args.output_dir,
        effective_config["dataset"],
        effective_config["variant"],
        effective_config["random_seed"],
        effective_config["run_id"],
    )
    prepare_run_dir(run_dir, overwrite=bool(args.overwrite))
    effective_config_path = write_effective_config(run_dir, effective_config)
    try:
        report = validate_full_gradient_invariants(
            args,
            dataset,
            effective_config,
            rounds_to_check=rounds_to_check,
            client_detail_limit=client_detail_limit,
        )
    except FullGradientPreflightError as exc:
        report = {
            "status": "failed",
            "dataset": args.dataset,
            "seed": int(args.random_seed),
            "partition_alpha": float(args.partition_alpha),
            "client_num_in_total": int(args.client_num_in_total),
            "client_num_per_round": int(args.client_num_per_round),
            "configured_batch_size": int(args.batch_size),
            "effective_config_batch_size": effective_config.get("batch_size"),
            "effective_config_mode": effective_config.get("mode"),
            "rounds_checked": int(rounds_to_check),
            "errors": exc.errors,
        }
        report_path = write_preflight_report(run_dir, report)
        report["run_dir"] = run_dir
        report["effective_config_path"] = effective_config_path
        report["preflight_report_path"] = report_path
        raise

    report["run_dir"] = run_dir
    report["effective_config_path"] = effective_config_path
    report_path = write_preflight_report(run_dir, report)
    report["preflight_report_path"] = report_path
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight deterministic full-gradient FedML batching.")
    parser.add_argument("--config", default=None, help="Optional FedML YAML config to load.")
    parser.add_argument("--dataset", choices=sorted(ZOO_DATASETS | {"sine"}), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--partition-alpha", type=float, default=None)
    parser.add_argument("--client-num-in-total", type=int, default=None)
    parser.add_argument("--client-num-per-round", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--client-optimizer", choices=["sgd", "ogda"], default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--using-gpu", action="store_true", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--rounds-to-check", type=int, default=1)
    parser.add_argument("--client-detail-limit", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    cli_args = parser.parse_args(argv)
    args = _config_from_args(cli_args)
    try:
        report = run_preflight(
            args,
            rounds_to_check=int(cli_args.rounds_to_check),
            client_detail_limit=int(cli_args.client_detail_limit),
        )
    except FullGradientPreflightError as exc:
        print("FULL-GRADIENT PREFLIGHT FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(json_safe({
        "status": report["status"],
        "run_dir": report["run_dir"],
        "effective_config_path": report["effective_config_path"],
        "preflight_report_path": report["preflight_report_path"],
        "checked_client_count": report["checked_client_count"],
        "sample_count_min": report["sample_count_min"],
        "sample_count_max": report["sample_count_max"],
        "effective_config_batch_size": report["effective_config_batch_size"],
    }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
