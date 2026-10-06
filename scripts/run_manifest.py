#!/usr/bin/env python3
"""Queue federated runs from a rerun protocol manifest.

This launcher is intentionally conservative:

* centralized rows are skipped until their runner is verified,
* real launches require concrete learning-rate and weight-decay values,
* dry-runs generate YAML configs and print commands without training,
* completed runs can be skipped after artifact validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment_utils import config_checksum  # noqa: E402
from verify_protocol_hashes import verify_hashes  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv"
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "generated_configs"
DEFAULT_MAIN = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example" / "main.py"
FEDML_LOG_DIR_ENV = "FEDGMM_FEDML_LOG_DIR"
FEDML_TRACE_DIR_ENV = "FEDGMM_FEDML_TRACE_DIR"
# Read by main.py's top-level exception handler so a pretraining_failure.json
# it writes can record hashes of this job's own stdout/stderr as evidence.
STDOUT_LOG_ENV = "FEDGMM_JOB_STDOUT_LOG"
STDERR_LOG_ENV = "FEDGMM_JOB_STDERR_LOG"

SUPPORTED_FEDERATED_METHODS = {
    "fedgda_d", "fedgda_s", "fedogda_d", "fedogda_s",
    "fed_eg_d", "fed_eg_s", "fed_eg_double_d", "fed_eg_double_s",
    "fed_zo_eg_d", "fed_zo_eg_s",
}

# Ground truth for method<->optimizer, mirrored from the `algorithm` map in
# experiment_utils.get_effective_config() (the training code's own inverse
# derivation of client_optimizer -> output-directory variant). Kept here so
# run_manifest.py can catch a mismatch BEFORE launching a job, not after a
# full run completes and writes real artifacts to a variant-derived path
# that silently differs from what the manifest's "method" column implied.
# See QUARANTINE_20260819_mislabeled_fedogda_expand2/ for the incident this
# check exists to prevent: a template-lookup bug left ten "fedogda_d" rows
# with client_optimizer="sgd", so they trained and completed in full as
# fedgda_d and landed in the fedgda_d results tree under fedogda_d run_ids.
METHOD_TO_OPTIMIZER = {
    "fedgda_d": "sgd", "fedgda_s": "sgd",
    "fedogda_d": "ogda", "fedogda_s": "ogda",
    "fed_eg_d": "fed_eg", "fed_eg_s": "fed_eg",
    "fed_eg_double_d": "fed_eg_double", "fed_eg_double_s": "fed_eg_double",
    "fed_zo_eg_d": "fed_zo_eg", "fed_zo_eg_s": "fed_zo_eg",
}

# Cosmetic (not read by any training code path), but a wrong method_label is
# the same template-copy bug showing up in reports/dashboards even where it
# doesn't affect what actually trains -- checked for the same reason.
METHOD_LABEL = {
    "fedgda_d": "FedGDA-D", "fedgda_s": "FedGDA-S",
    "fedogda_d": "FedOGDA-D", "fedogda_s": "FedOGDA-S",
    "fed_eg_d": "FedEG-D", "fed_eg_s": "FedEG-S",
    "fed_eg_double_d": "FedEG-Double-D", "fed_eg_double_s": "FedEG-Double-S",
    "fed_zo_eg_d": "FedZOEG-D", "fed_zo_eg_s": "FedZOEG-S",
}
EXPECTED_ARTIFACTS = (
    "effective_config.json",
    "mse_by_round.csv",
    "metrics.json",
    "predictions.npz",
    os.path.join("checkpoints", "best_validation.pt"),
    os.path.join("checkpoints", "final.pt"),
)


class ManifestLaunchError(Exception):
    pass


@dataclass
class Job:
    row: dict[str, str]
    config: dict[str, Any]
    config_path: Path
    run_dir: Path
    command: list[str]
    env: dict[str, str]
    gpu_id: int


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).strip().lower() == "na"


def _as_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ManifestLaunchError(f"{field} must be numeric, got {value!r}")
    if not math.isfinite(number):
        raise ManifestLaunchError(f"{field} must be finite, got {value!r}")
    return number


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ManifestLaunchError(f"{field} must be an integer, got {value!r}")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_only(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ManifestLaunchError(f"--only must be KEY=VALUE, got {value!r}")
        key, expected = value.split("=", 1)
        if not key or expected == "":
            raise ManifestLaunchError(f"--only must be KEY=VALUE, got {value!r}")
        filters[key] = expected
    return filters


def _row_matches_filters(row: dict[str, str], filters: dict[str, str]) -> bool:
    return all(str(row.get(key, "")) == expected for key, expected in filters.items())


def select_rows(rows: list[dict[str, str]], filters: dict[str, str], include_blocked: bool) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("training_scope") != "federated":
            continue
        if row.get("method") not in SUPPORTED_FEDERATED_METHODS:
            continue
        if not include_blocked and str(row.get("run_status", "")).lower() == "blocked":
            continue
        if not include_blocked and "blocked" in str(row.get("implementation_status", "")).lower():
            continue
        if not _row_matches_filters(row, filters):
            continue
        selected.append(row)
    return selected


def _config_value(row: dict[str, str], key: str, default: Any = None) -> Any:
    value = row.get(key)
    if _blank(value):
        return default
    return value


def _resolve_learning_rate(row: dict[str, str], default_learning_rate: float | None) -> float:
    value = _config_value(row, "learning_rate")
    if value is None:
        if default_learning_rate is None:
            raise ManifestLaunchError(f"{row['run_id']} is missing learning_rate")
        return float(default_learning_rate)
    return _as_float(value, "learning_rate")


def _resolve_weight_decay(row: dict[str, str], default_weight_decay: float | None) -> float:
    value = _config_value(row, "weight_decay")
    if value is None:
        if default_weight_decay is None:
            raise ManifestLaunchError(f"{row['run_id']} is missing weight_decay")
        return float(default_weight_decay)
    return _as_float(value, "weight_decay")


def assert_single_output_root(rows: list[dict[str, str]]) -> None:
    """Refuse a manifest whose rows declare DIFFERENT output_root values.

    run_manifest routes every run under one --output-root; the ``output_root``
    column campaign builders write is not read when routing. A manifest whose
    rows all name the same root is fine -- that is just a relocated campaign,
    which is supported. But rows naming *different* roots mean the author
    intended per-row separation that will silently collapse into one directory,
    which is exactly how an 18-run two-arm campaign ended up writing every arm
    to the same place. Refuse that, and let the caller launch once per root.

    Receives the rows actually being launched, after --only selection and
    --limit, so selecting a single arm out of a mixed-root manifest is allowed
    and blocked or excluded rows cannot veto a run they take no part in.
    """
    declared = set()
    for row in rows:
        value = str(row.get("output_root", "") or "").strip()
        if value:
            path = Path(value)
            declared.add(str((path if path.is_absolute() else REPO_ROOT / path).resolve()))
    if len(declared) > 1:
        raise ManifestLaunchError(
            "manifest rows declare "
            f"{len(declared)} different output_root values ({sorted(declared)}), but "
            "run_manifest routes every row under a single --output-root, so the "
            "column is ignored and all rows would land together. Split the manifest "
            "and invoke run_manifest once per output root."
        )


def _run_dir(output_root: Path, row: dict[str, str]) -> Path:
    return (
        output_root
        / row["dataset"]
        / row["method"]
        / f"seed_{int(row['seed'])}"
        / row["run_id"]
    )


def _log_dir(row: dict[str, str]) -> Path:
    return REPO_ROOT / "logs" / "fedml" / row["method"] / row["run_id"]


def _trace_dir(row: dict[str, str]) -> Path:
    return REPO_ROOT / "logs" / "fedml_trace" / row["method"] / row["run_id"]


LEGACY_TRAJECTORY_REGISTRY = (
    REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/legacy_batchnorm_trajectories_20260822.json"
)
_legacy_ineligible_run_ids_cache: frozenset[str] | None = None


def _legacy_ineligible_run_ids() -> frozenset[str]:
    """Run IDs the BatchNorm-legacy audit has recorded as ineligible.

    A manifest row's own `scientific_status` column defaults to "eligible"
    when the column is absent entirely -- true of every pre-fix screen/
    expand/finals/v2 manifest, since that column didn't exist when they were
    written. Relying on that column alone would let those legacy manifests
    launch again via a direct run_manifest.py invocation that doesn't go
    through one of the retired wrapper scripts. Cross-checking run_id
    against the audit registry closes that gap centrally, independent of
    what any given manifest CSV does or doesn't say.
    """
    global _legacy_ineligible_run_ids_cache
    if _legacy_ineligible_run_ids_cache is not None:
        return _legacy_ineligible_run_ids_cache
    try:
        registry = json.loads(LEGACY_TRAJECTORY_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        _legacy_ineligible_run_ids_cache = frozenset()
        return _legacy_ineligible_run_ids_cache
    _legacy_ineligible_run_ids_cache = frozenset(
        str(record["run_id"])
        for record in registry.get("records", [])
        if str(record.get("scientific_status", "")) != "eligible"
    )
    return _legacy_ineligible_run_ids_cache


def _require_launch_eligible(row: dict[str, str]) -> None:
    run_id = str(row.get("run_id", "<unknown>"))
    scientific_status = str(_config_value(row, "scientific_status", "eligible"))
    if scientific_status != "eligible":
        raise ManifestLaunchError(
            f"run {run_id} is not launch-eligible: scientific_status={scientific_status!r}"
        )
    if run_id in _legacy_ineligible_run_ids():
        raise ManifestLaunchError(
            f"run {run_id} is a registered pre-BatchNorm-fix legacy trajectory "
            f"({LEGACY_TRAJECTORY_REGISTRY.relative_to(REPO_ROOT)}); not launch-eligible"
        )
    # A manifest's "alpha" column is a human-readable label for the Dirichlet
    # partition concentration; "partition_alpha" is the only field
    # build_config() actually feeds into the training process (see
    # partition_alpha below). A generator that updates one column without the
    # other silently launches at a different alpha than its own run_id/label
    # claims -- this check catches that class of bug before launch rather
    # than after the run has already trained at the wrong alpha.
    alpha_label, partition_alpha = row.get("alpha"), row.get("partition_alpha")
    if not _blank(alpha_label) and not _blank(partition_alpha):
        alpha_value = _as_float(alpha_label, "alpha")
        partition_alpha_value = _as_float(partition_alpha, "partition_alpha")
        if not math.isclose(alpha_value, partition_alpha_value, rel_tol=1e-9, abs_tol=1e-12):
            raise ManifestLaunchError(
                f"run {run_id}: alpha label {alpha_value:g} != executable "
                f"partition_alpha {partition_alpha_value:g} -- this run would train "
                "at a different alpha than its label claims"
            )


def build_config(
    row: dict[str, str],
    *,
    output_root: Path,
    gpu_id: int,
    default_learning_rate: float | None,
    default_weight_decay: float | None,
    override_comm_round: int | None,
    override_epochs: int | None,
    override_simple_model_selection_epochs: int | None,
    override_f_history_model_selection_epochs: int | None,
    override_model_selection_batch_size: int | None,
    override_model_selection_max_samples: int | None,
    override_skip_model_selection: bool | None,
    override_skip_gmm_eval: bool | None,
    override_auxiliary_regression: bool | None,
    override_auxiliary_regression_epochs: int | None,
    override_append_round_csv: bool | None,
    override_periodic_checkpoint_interval: int | None,
    override_dataloader_num_workers: int | None,
    override_dataloader_pin_memory: bool | None,
) -> dict[str, Any]:
    _require_launch_eligible(row)
    expected_optimizer = METHOD_TO_OPTIMIZER.get(row["method"])
    if expected_optimizer is None:
        raise ManifestLaunchError(f"{row['run_id']}: unknown method {row['method']!r}, cannot verify client_optimizer")
    if row.get("client_optimizer") != expected_optimizer:
        raise ManifestLaunchError(
            f"{row['run_id']}: method={row['method']!r} requires client_optimizer="
            f"{expected_optimizer!r}, got {row.get('client_optimizer')!r}"
        )
    expected_label = METHOD_LABEL.get(row["method"])
    actual_label = _config_value(row, "method_label")
    if actual_label is not None and actual_label != expected_label:
        raise ManifestLaunchError(
            f"{row['run_id']}: method={row['method']!r} requires method_label="
            f"{expected_label!r}, got {actual_label!r}"
        )
    learning_rate = _resolve_learning_rate(row, default_learning_rate)
    weight_decay = _resolve_weight_decay(row, default_weight_decay)
    test_mse_used_for_selection = _truthy(_config_value(row, "test_mse_used_for_selection", False))
    selection_metric_source = str(_config_value(row, "selection_metric_source", "validation"))
    if test_mse_used_for_selection:
        raise ManifestLaunchError(f"{row['run_id']} attempts to use Test MSE for selection")
    if selection_metric_source.lower() != "validation":
        raise ManifestLaunchError(f"{row['run_id']} selection_metric_source must be validation")
    comm_round = int(override_comm_round) if override_comm_round is not None else _as_int(row["comm_round"], "comm_round")
    epochs = int(override_epochs) if override_epochs is not None else _as_int(row["epochs"], "epochs")
    skip_model_selection = (
        _truthy(_config_value(row, "skip_model_selection", False))
        if override_skip_model_selection is None
        else bool(override_skip_model_selection)
    )
    if override_skip_gmm_eval is not None:
        skip_gmm_eval = bool(override_skip_gmm_eval)
    elif skip_model_selection:
        skip_gmm_eval = True
    else:
        skip_gmm_eval = _truthy(_config_value(row, "skip_gmm_eval", False))
    if skip_model_selection:
        skip_gmm_eval = True
    auxiliary_regression = (
        _truthy(_config_value(row, "auxiliary_regression", False))
        if override_auxiliary_regression is None
        else bool(override_auxiliary_regression)
    )
    auxiliary_regression_epochs_default = epochs if auxiliary_regression else 0
    auxiliary_regression_epochs = (
        int(override_auxiliary_regression_epochs)
        if override_auxiliary_regression_epochs is not None
        else _as_int(
            _config_value(row, "auxiliary_regression_epochs", auxiliary_regression_epochs_default),
            "auxiliary_regression_epochs",
        )
    )
    append_round_csv = (
        _truthy(_config_value(row, "append_round_csv", True))
        if override_append_round_csv is None
        else bool(override_append_round_csv)
    )
    periodic_checkpoint_interval = (
        int(override_periodic_checkpoint_interval)
        if override_periodic_checkpoint_interval is not None
        else _as_int(_config_value(row, "periodic_checkpoint_interval", 200), "periodic_checkpoint_interval")
    )
    dataloader_num_workers = (
        int(override_dataloader_num_workers)
        if override_dataloader_num_workers is not None
        else _as_int(_config_value(row, "dataloader_num_workers", 0), "dataloader_num_workers")
    )
    if dataloader_num_workers < 0:
        raise ManifestLaunchError("dataloader_num_workers must be >= 0")
    if auxiliary_regression_epochs < 0:
        raise ManifestLaunchError("auxiliary_regression_epochs must be >= 0")
    if periodic_checkpoint_interval < 0:
        raise ManifestLaunchError("periodic_checkpoint_interval must be >= 0")
    gmm_eval_proxy = (
        "negative_val_mse"
        if skip_gmm_eval
        else str(_config_value(row, "gmm_eval_proxy", "approx_psi"))
    )
    dataloader_pin_memory = (
        _truthy(_config_value(row, "dataloader_pin_memory", False))
        if override_dataloader_pin_memory is None
        else bool(override_dataloader_pin_memory)
    )
    server_buffer_policy = str(
        _config_value(row, "server_buffer_policy", "direct_client_aggregate")
    )
    if server_buffer_policy != "direct_client_aggregate":
        raise ManifestLaunchError(
            "server_buffer_policy must be 'direct_client_aggregate', "
            f"got {server_buffer_policy!r}"
        )
    return {
        "dataset": row["dataset"],
        "model": _config_value(row, "model", "lr"),
        "federated_optimizer": _config_value(row, "federated_optimizer", "FedAvg"),
        "client_id_list": "[]",
        "client_num_in_total": _as_int(row["client_num_in_total"], "client_num_in_total"),
        "client_num_per_round": _as_int(row["client_num_per_round"], "client_num_per_round"),
        "comm_round": comm_round,
        "epochs": epochs,
        "batch_size": _as_int(row["batch_size"], "batch_size"),
        "client_optimizer": row["client_optimizer"],
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "critic_multiplier": _as_float(_config_value(row, "critic_multiplier", 10.0), "critic_multiplier"),
        "server_learning_rate": _as_float(_config_value(row, "server_learning_rate", 1.5), "server_learning_rate"),
        "server_buffer_policy": server_buffer_policy,
        "eg_predictor_server_lr": _as_float(
            _config_value(row, "eg_predictor_server_lr", _config_value(row, "server_learning_rate", 1.5)),
            "eg_predictor_server_lr",
        ),
        "eg_corrector_server_lr": _as_float(
            _config_value(row, "eg_corrector_server_lr", _config_value(row, "server_learning_rate", 1.5)),
            "eg_corrector_server_lr",
        ),
        # Default to the same phase's g coefficient, so an absent column
        # reproduces the previous shared-coefficient behaviour exactly.
        "eg_predictor_critic_server_lr": _as_float(
            _config_value(row, "eg_predictor_critic_server_lr",
                          _config_value(row, "eg_predictor_server_lr",
                                        _config_value(row, "server_learning_rate", 1.5))),
            "eg_predictor_critic_server_lr",
        ),
        "eg_corrector_critic_server_lr": _as_float(
            _config_value(row, "eg_corrector_critic_server_lr",
                          _config_value(row, "eg_corrector_server_lr",
                                        _config_value(row, "server_learning_rate", 1.5))),
            "eg_corrector_critic_server_lr",
        ),
        "zo_mu": _as_float(_config_value(row, "zo_mu", 1e-3), "zo_mu"),
        "zo_num_directions": _as_int(_config_value(row, "zo_num_directions", 1), "zo_num_directions"),
        "stop_on_numerical_failure": _truthy(_config_value(row, "stop_on_numerical_failure", False)),
        "validation_deterioration_window": _as_int(_config_value(row, "validation_deterioration_window", 0), "validation_deterioration_window"),
        "validation_deterioration_min_rounds": _as_int(_config_value(row, "validation_deterioration_min_rounds", 100), "validation_deterioration_min_rounds"),
        "validation_deterioration_patience": _as_int(_config_value(row, "validation_deterioration_patience", 50), "validation_deterioration_patience"),
        "validation_deterioration_factor": _as_float(_config_value(row, "validation_deterioration_factor", 5.0), "validation_deterioration_factor"),
        "validation_deterioration_absolute_delta": _as_float(_config_value(row, "validation_deterioration_absolute_delta", 0.1), "validation_deterioration_absolute_delta"),
        "client_execution_mode": str(_config_value(row, "client_execution_mode", "sp")),
        "enable_multiprocessing": _truthy(_config_value(row, "enable_multiprocessing", False)),
        "multiprocessing_num_workers": _as_int(_config_value(row, "multiprocessing_num_workers", 0), "multiprocessing_num_workers"),
        "multiprocessing_gpu_ids": str(_config_value(row, "multiprocessing_gpu_ids", "")),
        "multiprocessingsinglegpu_num_workers": _as_int(_config_value(row, "multiprocessingsinglegpu_num_workers", 2), "multiprocessingsinglegpu_num_workers"),
        "multiprocessingsinglegpu_gpu_id": _as_int(_config_value(row, "multiprocessingsinglegpu_gpu_id", gpu_id), "multiprocessingsinglegpu_gpu_id"),
        "objective_lambda_1": _as_float(_config_value(row, "objective_lambda_1", 0.1), "objective_lambda_1"),
        "gradient_clip_norm": _as_float(_config_value(row, "gradient_clip_norm", 1.0), "gradient_clip_norm"),
        "simple_model_selection_epochs": (
            int(override_simple_model_selection_epochs)
            if override_simple_model_selection_epochs is not None
            else _as_int(_config_value(row, "simple_model_selection_epochs", 100), "simple_model_selection_epochs")
        ),
        "f_history_model_selection_epochs": (
            int(override_f_history_model_selection_epochs)
            if override_f_history_model_selection_epochs is not None
            else _as_int(_config_value(row, "f_history_model_selection_epochs", 60), "f_history_model_selection_epochs")
        ),
        "model_selection_batch_size": (
            int(override_model_selection_batch_size)
            if override_model_selection_batch_size is not None
            else _as_int(_config_value(row, "model_selection_batch_size", 200), "model_selection_batch_size")
        ),
        "model_selection_max_samples": (
            int(override_model_selection_max_samples)
            if override_model_selection_max_samples is not None
            else _as_int(_config_value(row, "model_selection_max_samples", 0), "model_selection_max_samples")
        ),
        "log_test_mse_by_round": _truthy(_config_value(row, "log_test_mse_by_round", False)),
        "compact_predictions_only": _truthy(_config_value(row, "compact_predictions_only", False)),
        "test_mse_used_for_selection": test_mse_used_for_selection,
        "selection_metric_source": selection_metric_source,
        "skip_model_selection": skip_model_selection,
        "skip_gmm_eval": skip_gmm_eval,
        "gmm_eval_proxy": gmm_eval_proxy,
        "auxiliary_regression": auxiliary_regression,
        "auxiliary_regression_epochs": auxiliary_regression_epochs,
        "auxiliary_regression_state_device": str(_config_value(row, "auxiliary_regression_state_device", "device")),
        "append_round_csv": append_round_csv,
        "periodic_checkpoint_interval": periodic_checkpoint_interval,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "frequency_of_the_test": 1,
        "random_seed": _as_int(row["seed"], "seed"),
        "partition_method": _config_value(row, "partition_method", "hetero"),
        "partition_alpha": _as_float(row["partition_alpha"], "partition_alpha"),
        "data_cache_dir": _config_value(row, "data_cache_dir", "data"),
        "run_id": row["run_id"],
        "output_dir": str(output_root),
        "using_gpu": _truthy(_config_value(row, "using_gpu", True)),
        "gpu_id": int(gpu_id),
        "enable_legacy_outputs": False,
        "overwrite": False,
        "require_multibatch_stochastic": _truthy(
            _config_value(row, "require_multibatch_stochastic", False)
        ),
        # 0/unset = unlimited = pre-existing behavior. See experiment_utils.py
        # and my_model_trainer_classification.py's train_gmm.
        "max_local_steps_per_round": _as_int(
            _config_value(row, "max_local_steps_per_round", 0), "max_local_steps_per_round"
        ),
        "variant": row["method"],
        # eicu_semisynth-specific; absent/default for every other row, so this
        # is a pure addition with no effect on existing (non-eicu) manifests.
        "scenario_name": _config_value(row, "scenario_name", ""),
        "objective_mode": _config_value(row, "objective_mode", "legacy"),
        "aggregation_weighting": _config_value(row, "aggregation_weighting", "sample_size"),
        "input_dim_g": _as_int(_config_value(row, "input_dim_g", 0), "input_dim_g"),
        "input_dim_f": _as_int(_config_value(row, "input_dim_f", 0), "input_dim_f"),
        "hidden_widths": str(_config_value(row, "hidden_widths", "64,64")),
        "model_activation": str(
            _config_value(row, "model_activation", "leaky_relu")
        ),
        # protocol_v1.md S7.1: scenario_seed (which DGP/scenario artifact) and
        # optimizer_seed (this run's random_seed) must be recorded separately
        # even though row["seed"] continues to mean "optimizer seed" for path
        #/backward-compatibility purposes (_run_dir, the seed-mismatch check
        # below, and every pre-Study-A manifest already key off row["seed"]).
        # Absent scenario_seed defaults to row["seed"] so non-Study-A
        # manifests keep today's conflated-but-consistent behavior exactly.
        "scenario_seed": _as_int(_config_value(row, "scenario_seed", row["seed"]), "scenario_seed"),
        "optimizer_seed": _as_int(_config_value(row, "optimizer_seed", row["seed"]),
                                  "optimizer_seed"),
        # Narrow, explicit initialization override: a path to another run's
        # post-model-selection initial g/f state. Absent means no override.
        "init_state_override": (str(_config_value(row, "init_state_override", "")) or None),
        "init_g_state_override": (str(_config_value(row, "init_g_state_override", "")) or None),
        "init_f_state_override": (str(_config_value(row, "init_f_state_override", "")) or None),
        "checkpoint_rounds": (str(_config_value(row, "checkpoint_rounds", "")) or None),
        "local_bn_mode": str(_config_value(row, "local_bn_mode", "batch")),
        "calibration_panel_size": _as_int(
            _config_value(row, "calibration_panel_size", 512), "calibration_panel_size"),
        "seed_pair_id": _config_value(row, "seed_pair_id", ""),
        "campaign_role": _config_value(row, "campaign_role", ""),
        "scenario_checksum": _config_value(row, "scenario_checksum", ""),
        "protocol_version": _config_value(row, "protocol_version", ""),
        "role": _config_value(row, "role", ""),
        "g0": _config_value(row, "g0", ""),
        "alignment_label": _config_value(row, "alignment_label", ""),
        # Checkpoint selection always actually runs on primary_val_mse (see
        # _validate_round_curve's required_numeric and fedavg_api.py's
        # evaluate()), which is a pooled validation MSE, not an equal-client
        # one -- equal_client_val_mse is populated only for eICU runs, whose
        # manifest generators already set this metadata explicitly per row.
        # "pooled_validation_mse" is the accurate default for every other
        # dataset (FEMNIST/CIFAR/MNIST/zoo/...), whose validation data
        # carries no client_id to compute an equal-client statistic from.
        "primary_selection_metric": _config_value(
            row, "primary_selection_metric", "pooled_validation_mse"
        ),
        "selection_source": _config_value(row, "selection_source", "validation_only"),
        "scenario_scope": _config_value(row, "scenario_scope", ""),
        "study_claim": _config_value(row, "study_claim", ""),
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = {
        "common_args": {
            "training_type": "simulation",
            "using_mlops": False,
            "random_seed": config["random_seed"],
            "run_id": config["run_id"],
            "config_version": "release",
            "output_dir": config["output_dir"],
            "overwrite": config.get("overwrite", False),
            **({"protocol_version": config["protocol_version"]} if config.get("protocol_version") else {}),
            **({"role": config["role"]} if config.get("role") else {}),
            **({"alignment_label": config["alignment_label"]} if config.get("alignment_label") else {}),
            **({"study_claim": config["study_claim"]} if config.get("study_claim") else {}),
        },
        "data_args": {
            "dataset": config["dataset"],
            "data_cache_dir": config["data_cache_dir"],
            "partition_method": config["partition_method"],
            "partition_alpha": config["partition_alpha"],
            "dataloader_num_workers": config["dataloader_num_workers"],
            "dataloader_pin_memory": config["dataloader_pin_memory"],
            **({"scenario_name": config["scenario_name"]} if config.get("scenario_name") else {}),
            **({"scenario_checksum": config["scenario_checksum"]} if config.get("scenario_checksum") else {}),
            **({"scenario_scope": config["scenario_scope"]} if config.get("scenario_scope") else {}),
            **({"g0": config["g0"]} if config.get("g0") else {}),
        },
        "model_args": {
            "model": config["model"],
            **({"input_dim_g": config["input_dim_g"]} if config.get("input_dim_g") else {}),
            **({"input_dim_f": config["input_dim_f"]} if config.get("input_dim_f") else {}),
            "hidden_widths": config["hidden_widths"],
            "model_activation": config["model_activation"],
        },
        "train_args": {
            "federated_optimizer": config["federated_optimizer"],
            "client_id_list": config["client_id_list"],
            "client_num_in_total": config["client_num_in_total"],
            "client_num_per_round": config["client_num_per_round"],
            "comm_round": config["comm_round"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "client_optimizer": config["client_optimizer"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "critic_multiplier": config["critic_multiplier"],
            "server_learning_rate": config["server_learning_rate"],
            "server_buffer_policy": config["server_buffer_policy"],
            "eg_predictor_server_lr": config["eg_predictor_server_lr"],
            "eg_corrector_server_lr": config["eg_corrector_server_lr"],
            "eg_predictor_critic_server_lr": config["eg_predictor_critic_server_lr"],
            "eg_corrector_critic_server_lr": config["eg_corrector_critic_server_lr"],
            "zo_mu": config["zo_mu"],
            "zo_num_directions": config["zo_num_directions"],
            "stop_on_numerical_failure": config.get("stop_on_numerical_failure", False),
            "validation_deterioration_window": config.get("validation_deterioration_window", 0),
            "validation_deterioration_min_rounds": config.get("validation_deterioration_min_rounds", 100),
            "validation_deterioration_patience": config.get("validation_deterioration_patience", 50),
            "validation_deterioration_factor": config.get("validation_deterioration_factor", 5.0),
            "validation_deterioration_absolute_delta": config.get("validation_deterioration_absolute_delta", 0.1),
            "client_execution_mode": config["client_execution_mode"],
            "enable_multiprocessing": config["enable_multiprocessing"],
            "multiprocessing_num_workers": config["multiprocessing_num_workers"],
            "multiprocessing_gpu_ids": config["multiprocessing_gpu_ids"],
            "multiprocessingsinglegpu_num_workers": config["multiprocessingsinglegpu_num_workers"],
            "multiprocessingsinglegpu_gpu_id": config["multiprocessingsinglegpu_gpu_id"],
            "objective_lambda_1": config["objective_lambda_1"],
            "gradient_clip_norm": config["gradient_clip_norm"],
            "simple_model_selection_epochs": config["simple_model_selection_epochs"],
            "f_history_model_selection_epochs": config["f_history_model_selection_epochs"],
            "model_selection_batch_size": config["model_selection_batch_size"],
            "model_selection_max_samples": config["model_selection_max_samples"],
            "enable_legacy_outputs": config["enable_legacy_outputs"],
            "require_multibatch_stochastic": config["require_multibatch_stochastic"],
            "max_local_steps_per_round": config["max_local_steps_per_round"],
            "log_test_mse_by_round": config["log_test_mse_by_round"],
            "compact_predictions_only": config["compact_predictions_only"],
            "test_mse_used_for_selection": config["test_mse_used_for_selection"],
            "selection_metric_source": config["selection_metric_source"],
            "primary_selection_metric": config["primary_selection_metric"],
            "selection_source": config["selection_source"],
            "skip_model_selection": config["skip_model_selection"],
            "skip_gmm_eval": config["skip_gmm_eval"],
            "gmm_eval_proxy": config["gmm_eval_proxy"],
            "auxiliary_regression": config["auxiliary_regression"],
            "auxiliary_regression_epochs": config["auxiliary_regression_epochs"],
            "auxiliary_regression_state_device": config["auxiliary_regression_state_device"],
            "append_round_csv": config["append_round_csv"],
            "periodic_checkpoint_interval": config["periodic_checkpoint_interval"],
            "objective_mode": config["objective_mode"],
            "aggregation_weighting": config["aggregation_weighting"],
            "scenario_seed": config["scenario_seed"],
            "optimizer_seed": config["optimizer_seed"],
            "init_state_override": config["init_state_override"],
            "init_g_state_override": config["init_g_state_override"],
            "init_f_state_override": config["init_f_state_override"],
            "checkpoint_rounds": config["checkpoint_rounds"],
            "local_bn_mode": config["local_bn_mode"],
            "calibration_panel_size": config["calibration_panel_size"],
            **({"seed_pair_id": config["seed_pair_id"]} if config.get("seed_pair_id") else {}),
            **({"campaign_role": config["campaign_role"]} if config.get("campaign_role") else {}),
        },
        "validation_args": {
            "frequency_of_the_test": config["frequency_of_the_test"],
        },
        "device_args": {
            "using_gpu": config["using_gpu"],
            "gpu_id": config["gpu_id"],
        },
        "comm_args": {
            "backend": "sp",
        },
        "tracking_args": {
            "enable_tracking": False,
            "enable_wandb": False,
            "run_name": f"{config['dataset']}_{config['variant']}_seed_{config['random_seed']}_{config['run_id']}",
        },
    }
    with path.open("w") as f:
        for section, values in sections.items():
            f.write(f"{section}:\n")
            for key, value in values.items():
                f.write(f"  {key}: {_yaml_scalar(value)}\n")
            f.write("\n")


def _json_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _finite_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(number) else None


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestLaunchError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestLaunchError(f"{path.name} must contain a JSON object")
    return value


def _expected_value(
    expected_config: dict[str, Any] | None,
    config_field: str,
    row: dict[str, str],
    row_field: str,
) -> Any:
    """Prefer an explicit expected config value, otherwise use the manifest.

    Scorers often know only a strict subset of the resolved configuration.  A
    missing key in that subset must not erase the corresponding manifest
    constraint by turning its expected value into ``None``.
    """

    if expected_config is not None:
        value = expected_config.get(config_field)
        if not _blank(value):
            return value
    return row.get(row_field)


def _validate_effective_config(
    config: dict[str, Any],
    row: dict[str, str],
    expected_config: dict[str, Any] | None = None,
) -> None:
    exact_fields = {
        "dataset": "dataset",
        "variant": "method",
        "run_id": "run_id",
        "client_optimizer": "client_optimizer",
    }
    for config_field, row_field in exact_fields.items():
        expected = _expected_value(expected_config, config_field, row, row_field)
        if str(config.get(config_field)) != str(expected):
            raise ManifestLaunchError(
                f"{config_field} mismatch: {config.get(config_field)!r} != {expected!r}"
            )

    integer_fields = {
        "random_seed": "seed",
        "client_num_in_total": "client_num_in_total",
        "client_num_per_round": "client_num_per_round",
        "comm_round": "comm_round",
        "epochs": "epochs",
        "batch_size": "batch_size",
    }
    for config_field, row_field in integer_fields.items():
        expected = _expected_value(expected_config, config_field, row, row_field)
        if _blank(expected):
            continue
        try:
            actual_int = int(config.get(config_field))
            expected_int = int(str(expected))
        except (TypeError, ValueError) as exc:
            raise ManifestLaunchError(
                f"{config_field} is not an integer in config or manifest"
            ) from exc
        if actual_int != expected_int:
            raise ManifestLaunchError(
                f"{config_field} mismatch: {actual_int} != {expected_int}"
            )

    numeric_fields = {
        "learning_rate": "learning_rate",
        "critic_multiplier": "critic_multiplier",
        "weight_decay": "weight_decay",
        "server_learning_rate": "server_learning_rate",
        "partition_alpha": "partition_alpha",
        "eg_predictor_server_lr": "eg_predictor_server_lr",
        "eg_corrector_server_lr": "eg_corrector_server_lr",
        "eg_predictor_critic_server_lr": "eg_predictor_critic_server_lr",
        "eg_corrector_critic_server_lr": "eg_corrector_critic_server_lr",
    }
    for config_field, row_field in numeric_fields.items():
        expected = _expected_value(expected_config, config_field, row, row_field)
        if _blank(expected):
            continue
        try:
            actual_float = float(config.get(config_field))
            expected_float = float(str(expected))
        except (TypeError, ValueError) as exc:
            raise ManifestLaunchError(
                f"{config_field} is not numeric in config or manifest"
            ) from exc
        if not math.isclose(actual_float, expected_float, rel_tol=1e-12, abs_tol=0.0):
            raise ManifestLaunchError(
                f"{config_field} mismatch: {actual_float} != {expected_float}"
            )

    expected_buffer_policy = _expected_value(
        expected_config,
        "server_buffer_policy",
        row,
        "server_buffer_policy",
    )
    if not _blank(expected_buffer_policy):
        if str(config.get("server_buffer_policy")) != str(expected_buffer_policy):
            raise ManifestLaunchError(
                "server_buffer_policy mismatch: "
                f"{config.get('server_buffer_policy')!r} != {expected_buffer_policy!r}"
            )
    if bool(config.get("test_mse_used_for_selection", False)):
        raise ManifestLaunchError("config.test_mse_used_for_selection must be false")
    if str(config.get("selection_metric_source", "validation")).lower() != "validation":
        raise ManifestLaunchError("config.selection_metric_source must be validation")


def _required_bn_fields(dataset: str) -> tuple[str, ...]:
    """Return BatchNorm telemetry required by an image dataset's model path."""

    normalized = dataset.lower()
    if not normalized.startswith(("mnist_", "femnist_", "cifar10_", "cifar_")):
        return ()
    if normalized.endswith("_xz"):
        return ("g_bn_min_running_var", "f_bn_min_running_var")
    if normalized.endswith("_x"):
        return ("g_bn_min_running_var",)
    if normalized.endswith("_z"):
        return ("f_bn_min_running_var",)
    return ()


def _validate_round_curve(
    path: Path,
    comm_round: int,
    dataset: str,
) -> tuple[list[dict[str, str]], bool]:
    try:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise ManifestLaunchError(f"cannot read {path.name}: {exc}") from exc
    if len(rows) != comm_round:
        raise ManifestLaunchError(
            f"{path.name} row count {len(rows)} != comm_round {comm_round}"
        )

    terminal_ineligible = False
    required_bn_fields = _required_bn_fields(dataset)
    # equal_client_val_mse is populated only for eICU runs (val_global carries
    # a client_id there and nowhere else -- see fedavg_api.py's evaluate()) --
    # legacy datasets like this campaign's FEMNIST/CIFAR10 leave it blank by
    # design, so it is checked separately below as an optional numeric field
    # rather than required here.
    required_numeric = (
        "train_mse",
        "val_mse",
        "primary_val_mse",
        "train_moment_violation",
        "val_moment_violation",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
    )
    for index, curve_row in enumerate(rows):
        try:
            round_index = int(curve_row.get("round", ""))
        except (TypeError, ValueError) as exc:
            raise ManifestLaunchError(f"{path.name}[{index}].round is not an integer") from exc
        if round_index != index:
            raise ManifestLaunchError(
                f"{path.name}[{index}].round is {round_index}; expected {index}"
            )

        finite_text = str(curve_row.get("finite", ""))
        diverged_text = str(curve_row.get("diverged", ""))
        if finite_text not in {"True", "False"}:
            raise ManifestLaunchError(f"{path.name}[{index}].finite is not a boolean")
        if diverged_text not in {"True", "False"}:
            raise ManifestLaunchError(f"{path.name}[{index}].diverged is not a boolean")
        if finite_text == "False" or diverged_text == "True":
            terminal_ineligible = True

        for field in required_numeric:
            raw_value = curve_row.get(field)
            if _blank(raw_value):
                raise ManifestLaunchError(f"{path.name}[{index}].{field} is blank")
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ManifestLaunchError(
                    f"{path.name}[{index}].{field} is not numeric"
                ) from exc
            if not math.isfinite(number):
                terminal_ineligible = True

        for field in ("g_bn_min_running_var", "f_bn_min_running_var"):
            raw_value = curve_row.get(field)
            if _blank(raw_value):
                if field in required_bn_fields:
                    raise ManifestLaunchError(
                        f"{path.name}[{index}].{field} is blank for {dataset}"
                    )
                continue
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ManifestLaunchError(
                    f"{path.name}[{index}].{field} is not numeric"
                ) from exc
            if not math.isfinite(number) or number < 0.0:
                terminal_ineligible = True

        equal_client_val_mse = curve_row.get("equal_client_val_mse")
        if _blank(equal_client_val_mse) and dataset.lower().startswith("eicu"):
            raise ManifestLaunchError(
                f"{path.name}[{index}].equal_client_val_mse is blank for {dataset}"
            )
        if not _blank(equal_client_val_mse):
            try:
                number = float(equal_client_val_mse)
            except (TypeError, ValueError) as exc:
                raise ManifestLaunchError(
                    f"{path.name}[{index}].equal_client_val_mse is not numeric"
                ) from exc
            if not math.isfinite(number):
                terminal_ineligible = True
    return rows, terminal_ineligible


def load_certification_ledger(path: Path | None) -> dict[str, dict[str, str]]:
    """A certification ledger maps an original run_id to the run_id/run_dir
    of an independent reproduction that produced real, process-authored
    pretraining_failure.json evidence for it (closeout plan SS6.2). Unlike
    an earlier design, nothing is ever copied or synthesized into the
    original run's own directory -- validate_pretraining_failure_artifact
    always validates evidence in the directory the failing process itself
    actually wrote to."""
    if path is None:
        return {}
    with path.open() as handle:
        ledger = json.load(handle)
    if not isinstance(ledger, dict):
        raise ManifestLaunchError(f"{path} must contain a JSON object")
    for run_id, entry in ledger.items():
        if not isinstance(entry, dict) or "certified_run_id" not in entry or "certified_run_dir" not in entry:
            raise ManifestLaunchError(
                f"certification ledger entry for {run_id!r} must have "
                "certified_run_id and certified_run_dir"
            )
    return ledger


def resolve_certified_run(
    run_id: str, row: dict[str, str], run_dir: Path, ledger: dict[str, dict[str, str]]
) -> tuple[Path, dict[str, str]]:
    """If run_id has a certification-ledger entry, returns the independent
    reproduction's own directory and a row carrying its own real run_id --
    so validate_artifacts' exact run_id match succeeds against that
    directory's real, process-authored evidence -- instead of the original
    row/run_dir (whose directory was never touched and still contains only
    what the original process actually wrote). Otherwise returns
    (run_dir, row) unchanged."""
    entry = ledger.get(run_id)
    if entry is None:
        return run_dir, row
    certified_run_dir = Path(entry["certified_run_dir"])
    if not certified_run_dir.is_absolute():
        certified_run_dir = REPO_ROOT / certified_run_dir
    certified_row = {**row, "run_id": entry["certified_run_id"]}
    return certified_run_dir, certified_row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_recorded_log_digest(run_dir: Path, log_name: str, recorded_digest: Any) -> None:
    """A recorded stdout/stderr sha256 is only evidence if it is independently
    recomputable from the actual log file next to it, right now -- otherwise
    a certification claiming a specific hash could never be checked against
    anything. If the log file is genuinely absent (e.g. this job's stdout/
    stderr were never captured), the recorded digest must be null to match;
    a non-null digest with no file to recompute it from is itself a defect."""
    log_path = run_dir / log_name
    if not log_path.exists():
        if recorded_digest is not None:
            raise ManifestLaunchError(
                f"pretraining_failure.json records {log_name}_sha256={recorded_digest!r} "
                f"but {log_name} is not present in {run_dir} to recompute it from"
            )
        return
    if not isinstance(recorded_digest, str) or not recorded_digest.strip():
        raise ManifestLaunchError(
            f"pretraining_failure.json {log_name}_sha256 must be a non-empty string "
            f"when {log_name} is present"
        )
    recomputed = _sha256_file(log_path)
    if recomputed != recorded_digest:
        raise ManifestLaunchError(
            f"pretraining_failure.json {log_name}_sha256 does not match a fresh "
            f"recomputation from {log_name}: recorded {recorded_digest!r}, "
            f"recomputed {recomputed!r}"
        )


def validate_pretraining_failure_artifact(
    run_dir: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    """Validates pretraining_failure.json -- the only artifact that may
    classify a run whose training process never wrote round-curve artifacts
    as terminal_pretraining_ineligible rather than an unexplained process
    failure. Verifies exact configuration equivalence against the manifest
    row (dataset, method, seed, learning rate, critic multiplier, partition
    alpha, horizon, ...) via the same _validate_effective_config() the
    normal completed-run path uses, and independently recomputes the
    stdout/stderr digests rather than trusting the values the failing
    process itself recorded."""
    payload = _load_json_object(run_dir / "pretraining_failure.json")
    if payload.get("schema_version") != 1:
        raise ManifestLaunchError("pretraining_failure.json schema_version must be 1")
    if str(payload.get("run_id")) != str(row["run_id"]):
        raise ManifestLaunchError(
            "pretraining_failure.json run_id does not match manifest row"
        )
    if payload.get("failure_phase") != "model_selection":
        raise ManifestLaunchError(
            "pretraining_failure.json failure_phase must be 'model_selection'"
        )
    if payload.get("federated_rounds_started") != 0:
        raise ManifestLaunchError(
            "pretraining_failure.json federated_rounds_started must be 0"
        )
    epochs_attempted = payload.get("model_selection_epochs_attempted")
    if (
        not isinstance(epochs_attempted, int)
        or isinstance(epochs_attempted, bool)
        or epochs_attempted < 0
    ):
        raise ManifestLaunchError(
            "pretraining_failure.json model_selection_epochs_attempted must be "
            "a non-negative integer"
        )
    if not isinstance(payload.get("per_epoch_finite_status"), list):
        raise ManifestLaunchError(
            "pretraining_failure.json per_epoch_finite_status must be a list"
        )
    best_score = payload.get("best_model_selection_score")
    if best_score is not None and (
        not isinstance(best_score, (int, float)) or isinstance(best_score, bool)
    ):
        raise ManifestLaunchError(
            "pretraining_failure.json best_model_selection_score must be numeric or null"
        )
    terminal_reason = payload.get("terminal_reason")
    if not isinstance(terminal_reason, str) or not terminal_reason.strip():
        raise ManifestLaunchError(
            "pretraining_failure.json terminal_reason must be a non-empty string"
        )
    traceback_text = payload.get("traceback")
    if not isinstance(traceback_text, str) or not traceback_text.strip():
        raise ManifestLaunchError(
            "pretraining_failure.json traceback must be a non-empty string"
        )
    checksum = payload.get("effective_config_checksum")
    if not isinstance(checksum, str) or not checksum.strip():
        raise ManifestLaunchError(
            "pretraining_failure.json effective_config_checksum must be a non-empty string"
        )
    effective_config = _load_json_object(run_dir / "effective_config.json")
    recomputed_checksum = config_checksum(effective_config)
    if checksum != recomputed_checksum:
        raise ManifestLaunchError(
            "pretraining_failure.json effective_config_checksum does not match a fresh "
            f"recomputation from effective_config.json: recorded {checksum!r}, "
            f"recomputed {recomputed_checksum!r}"
        )
    # A pretraining failure happens before federated round 0, so
    # _validate_round_curve never runs for this artifact -- this is the only
    # place that proves the reproduction actually used the same dataset,
    # method, seed, learning rate, critic multiplier, partition alpha, and
    # horizon as the row it is meant to be certifying, rather than merely
    # being internally self-consistent.
    _validate_effective_config(effective_config, row)
    hash_bundle_id = payload.get("hash_bundle_id")
    if not isinstance(hash_bundle_id, str) or not hash_bundle_id.strip():
        raise ManifestLaunchError(
            "pretraining_failure.json hash_bundle_id must be a non-empty string, "
            f"got {hash_bundle_id!r}"
        )
    # A non-empty string alone proves nothing -- it could name a bundle that
    # was never generated, or one whose recorded hashes no longer match the
    # actual source tree. Resolve it and run the real hash verifier so this
    # is provenance, not just a label.
    bundle_path = Path(hash_bundle_id)
    if bundle_path.is_absolute() or ".." in bundle_path.parts:
        raise ManifestLaunchError(
            f"pretraining_failure.json hash_bundle_id has an unsafe path: {hash_bundle_id!r}"
        )
    bundle_path = REPO_ROOT / bundle_path
    if not bundle_path.is_file():
        raise ManifestLaunchError(
            f"pretraining_failure.json hash_bundle_id does not exist: {hash_bundle_id!r}"
        )
    # A clean relative path can still resolve (e.g. through a symlink) to
    # somewhere outside REPO_ROOT -- the unsafe-path check above only rejects
    # ".." literally appearing in the string, not what the path actually
    # resolves to on disk.
    resolved_bundle_path = bundle_path.resolve()
    resolved_repo_root = REPO_ROOT.resolve()
    if resolved_bundle_path != resolved_repo_root and resolved_repo_root not in resolved_bundle_path.parents:
        raise ManifestLaunchError(
            f"pretraining_failure.json hash_bundle_id resolves outside the repository: "
            f"{hash_bundle_id!r} -> {resolved_bundle_path}"
        )
    # The path alone is a mutable label -- a bundle file swapped in-place at
    # the same path would still verify against its own (now-different)
    # recorded hashes. Binding to the bundle file's own content hash, taken
    # at the moment the failure was recorded, closes that gap.
    bundle_sha256 = payload.get("hash_bundle_sha256")
    if not isinstance(bundle_sha256, str) or not bundle_sha256.strip():
        raise ManifestLaunchError(
            "pretraining_failure.json hash_bundle_sha256 must be a non-empty string, "
            f"got {bundle_sha256!r}"
        )
    actual_bundle_sha256 = hashlib.sha256(resolved_bundle_path.read_bytes()).hexdigest()
    if bundle_sha256 != actual_bundle_sha256:
        raise ManifestLaunchError(
            f"pretraining_failure.json hash_bundle_sha256 does not match {hash_bundle_id!r}'s "
            f"current content: recorded {bundle_sha256!r}, actual {actual_bundle_sha256!r}"
        )
    try:
        verify_hashes(bundle_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestLaunchError(
            f"pretraining_failure.json hash_bundle_id {hash_bundle_id!r} failed verification: {exc}"
        ) from exc
    _validate_recorded_log_digest(run_dir, "stdout.log", payload.get("stdout_sha256"))
    _validate_recorded_log_digest(run_dir, "stderr.log", payload.get("stderr_sha256"))
    return {
        "run_dir": str(run_dir),
        "test_mse_at_best_validation": None,
        "final_test_mse": None,
        "best_validation_round": None,
        "terminal_ineligible": True,
        "terminal_pretraining_ineligible": True,
        "terminal_reason": terminal_reason,
    }


def validate_artifacts(
    run_dir: Path,
    row: dict[str, str],
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (run_dir / "pretraining_failure.json").exists():
        return validate_pretraining_failure_artifact(run_dir, row)
    missing = [name for name in EXPECTED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise ManifestLaunchError(f"missing artifacts: {', '.join(missing)}")
    config = _load_json_object(run_dir / "effective_config.json")
    metrics = _load_json_object(run_dir / "metrics.json")
    _validate_effective_config(config, row, expected_config)
    try:
        comm_round = int(config.get("comm_round"))
    except (TypeError, ValueError) as exc:
        raise ManifestLaunchError("config.comm_round is not an integer") from exc
    if comm_round < 1:
        raise ManifestLaunchError("config.comm_round must be positive")
    _, terminal_ineligible = _validate_round_curve(
        run_dir / "mse_by_round.csv", comm_round, str(config.get("dataset", ""))
    )
    if str(metrics.get("run_status")) != "completed":
        raise ManifestLaunchError("metrics.run_status must be 'completed'")
    try:
        rounds_completed = int(metrics.get("rounds_completed"))
    except (TypeError, ValueError) as exc:
        raise ManifestLaunchError("metrics.rounds_completed is not an integer") from exc
    if rounds_completed != comm_round:
        raise ManifestLaunchError(
            f"metrics.rounds_completed {rounds_completed} != comm_round {comm_round}"
        )
    terminal_ineligible = terminal_ineligible or bool(metrics.get("diverged", False))
    if str(metrics.get("server_buffer_policy")) != str(
        config.get("server_buffer_policy")
    ):
        raise ManifestLaunchError(
            "metrics.server_buffer_policy must match effective_config.json"
        )
    if str(metrics.get("server_buffer_policy")) != "direct_client_aggregate":
        raise ManifestLaunchError(
            "metrics.server_buffer_policy must be 'direct_client_aggregate'"
        )

    if "nonfinite_first_round" not in metrics:
        raise ManifestLaunchError("metrics.nonfinite_first_round is required")
    nonfinite_first_round = metrics.get("nonfinite_first_round")
    if nonfinite_first_round is not None:
        if isinstance(nonfinite_first_round, bool):
            raise ManifestLaunchError("metrics.nonfinite_first_round must be an integer or null")
        try:
            first_round = int(nonfinite_first_round)
        except (TypeError, ValueError) as exc:
            raise ManifestLaunchError(
                "metrics.nonfinite_first_round must be an integer or null"
            ) from exc
        if first_round < 0 or first_round >= comm_round:
            raise ManifestLaunchError(
                "metrics.nonfinite_first_round is outside the completed round range"
            )
        terminal_ineligible = True

    nonfinite_diagnostics = metrics.get("nonfinite_diagnostics")
    if not isinstance(nonfinite_diagnostics, list):
        raise ManifestLaunchError("metrics.nonfinite_diagnostics must be a list")
    if nonfinite_diagnostics:
        terminal_ineligible = True

    for field in _required_bn_fields(str(config.get("dataset", ""))):
        raw_value = metrics.get(field)
        if raw_value is None:
            raise ManifestLaunchError(f"metrics.{field} is required for this dataset")
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ManifestLaunchError(f"metrics.{field} is not numeric") from exc
        if not math.isfinite(number) or number < 0.0:
            terminal_ineligible = True
    if bool(config.get("log_test_mse_by_round", False)):
        test_curve_path = run_dir / "test_mse_by_round.csv"
        if not test_curve_path.exists():
            raise ManifestLaunchError("missing test_mse_by_round.csv")
        with test_curve_path.open("r", newline="") as f:
            test_rows = list(csv.DictReader(f))
        if len(test_rows) != int(config.get("comm_round")):
            raise ManifestLaunchError(
                f"test_mse_by_round row count {len(test_rows)} != comm_round {config.get('comm_round')}"
            )
        if bool(metrics.get("test_mse_used_for_selection", False)):
            raise ManifestLaunchError("metrics.test_mse_used_for_selection must be false")
        if str(metrics.get("selection_metric_source", "validation")).lower() != "validation":
            raise ManifestLaunchError("metrics.selection_metric_source must be validation")
        for index, test_row in enumerate(test_rows):
            try:
                test_round = int(test_row.get("round", ""))
            except (TypeError, ValueError) as exc:
                raise ManifestLaunchError(
                    f"test_mse_by_round[{index}].round is not an integer"
                ) from exc
            if test_round != index:
                raise ManifestLaunchError(
                    f"test_mse_by_round[{index}].round is {test_round}; expected {index}"
                )
            try:
                test_mse = float(test_row.get("test_mse"))
            except (TypeError, ValueError):
                raise ManifestLaunchError(f"test_mse_by_round[{index}].test_mse is not numeric")
            if not math.isfinite(test_mse):
                terminal_ineligible = True
            finite_text = str(test_row.get("finite"))
            diverged_text = str(test_row.get("diverged"))
            if finite_text not in {"True", "False"}:
                raise ManifestLaunchError(f"test_mse_by_round[{index}].finite is not a boolean")
            if diverged_text not in {"True", "False"}:
                raise ManifestLaunchError(f"test_mse_by_round[{index}].diverged is not a boolean")
            if finite_text == "False" or diverged_text == "True":
                terminal_ineligible = True
    if not _json_numbers_are_finite(metrics):
        terminal_ineligible = True
    return {
        "run_dir": str(run_dir),
        "test_mse_at_best_validation": _finite_or_none(
            metrics.get("test_mse_at_best_validation")
        ),
        "final_test_mse": _finite_or_none(metrics.get("final_test_mse")),
        "best_validation_round": _finite_or_none(metrics.get("best_validation_round")),
        "terminal_ineligible": terminal_ineligible,
        "terminal_pretraining_ineligible": False,
        "terminal_reason": metrics.get("failure_reason") or (
            "sticky divergence/nonfinite evidence" if terminal_ineligible else None
        ),
    }


def has_completed_artifacts(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in EXPECTED_ARTIFACTS)


def has_any_artifacts(run_dir: Path) -> bool:
    return run_dir.exists() and any(run_dir.iterdir())


def assign_job_gpu(job: Job, gpu_id: int) -> None:
    """Assign a physical GPU to a job immediately before launch.

    GPU assignment must happen after resume/skip decisions. Otherwise a skipped
    job can consume a round-robin slot and two live jobs can accidentally land
    on the same GPU.
    """

    job.gpu_id = int(gpu_id)
    job.config["gpu_id"] = int(gpu_id)
    write_config(job.config_path, job.config)


def build_jobs(
    rows: list[dict[str, str]],
    *,
    python_executable: str,
    main_path: Path,
    config_dir: Path,
    output_root: Path,
    gpu_ids: list[int],
    default_learning_rate: float | None,
    default_weight_decay: float | None,
    override_comm_round: int | None,
    override_epochs: int | None,
    override_simple_model_selection_epochs: int | None,
    override_f_history_model_selection_epochs: int | None,
    override_model_selection_batch_size: int | None,
    override_model_selection_max_samples: int | None,
    override_skip_model_selection: bool | None,
    override_skip_gmm_eval: bool | None,
    override_auxiliary_regression: bool | None,
    override_auxiliary_regression_epochs: int | None,
    override_append_round_csv: bool | None,
    override_periodic_checkpoint_interval: int | None,
    override_dataloader_num_workers: int | None,
    override_dataloader_pin_memory: bool | None,
) -> tuple[list[Job], list[dict[str, str]]]:
    jobs: list[Job] = []
    skipped: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        try:
            config = build_config(
                row,
                output_root=output_root,
                gpu_id=gpu_id,
                default_learning_rate=default_learning_rate,
                default_weight_decay=default_weight_decay,
                override_comm_round=override_comm_round,
                override_epochs=override_epochs,
                override_simple_model_selection_epochs=override_simple_model_selection_epochs,
                override_f_history_model_selection_epochs=override_f_history_model_selection_epochs,
                override_model_selection_batch_size=override_model_selection_batch_size,
                override_model_selection_max_samples=override_model_selection_max_samples,
                override_skip_model_selection=override_skip_model_selection,
                override_skip_gmm_eval=override_skip_gmm_eval,
                override_auxiliary_regression=override_auxiliary_regression,
                override_auxiliary_regression_epochs=override_auxiliary_regression_epochs,
                override_append_round_csv=override_append_round_csv,
                override_periodic_checkpoint_interval=override_periodic_checkpoint_interval,
                override_dataloader_num_workers=override_dataloader_num_workers,
                override_dataloader_pin_memory=override_dataloader_pin_memory,
            )
        except ManifestLaunchError as exc:
            skipped.append({"run_id": row.get("run_id", ""), "reason": str(exc)})
            continue
        config_path = config_dir / row["dataset"] / row["method"] / f"seed_{int(row['seed'])}" / f"{row['run_id']}.yaml"
        run_dir = _run_dir(output_root, row)
        command = [python_executable, str(main_path), "--cf", str(config_path)]
        env = dict(os.environ)
        env[FEDML_LOG_DIR_ENV] = str(_log_dir(row))
        env[FEDML_TRACE_DIR_ENV] = str(_trace_dir(row))
        env[STDOUT_LOG_ENV] = str(run_dir / "stdout.log")
        env[STDERR_LOG_ENV] = str(run_dir / "stderr.log")
        jobs.append(Job(row=row, config=config, config_path=config_path, run_dir=run_dir, command=command, env=env, gpu_id=gpu_id))
    return jobs, skipped


def dry_run(jobs: list[Job], skipped: list[dict[str, str]], limit: int | None) -> None:
    shown = jobs if limit is None else jobs[:limit]
    for job in shown:
        write_config(job.config_path, job.config)
        print(f"{job.row['run_id']} -> {job.run_dir}")
        print(f"gpu_id={job.gpu_id}")
        print(f"config={job.config_path}")
        print(" ".join(job.command))
    if skipped:
        print("Skipped unlaunchable rows:")
        for item in skipped[:20]:
            print(f"  {item['run_id']}: {item['reason']}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
    print(json.dumps({
        "dry_run": True,
        "launchable": len(jobs),
        "skipped_unlaunchable": len(skipped),
        "shown": len(shown),
    }, indent=2, sort_keys=True))


def run_jobs(
    jobs: list[Job],
    *,
    gpu_ids: list[int],
    max_parallel: int,
    resume_skip_completed: bool,
    overwrite_incomplete: bool,
    stop_on_failure: bool,
    results_json: Path | None = None,
) -> list[dict[str, Any]]:
    # If results_json is given, written after every job resolves (not just
    # once at the end) -- so a hard kill mid-manifest (e.g. broker
    # preemption) still leaves an accurate results file for whatever
    # completed before the kill, instead of losing the launcher's own
    # bookkeeping for a manifest that was mostly finished. The training
    # artifacts themselves (metrics.json, checkpoints) are always written
    # directly to run_dir by the training process on completion regardless
    # of this -- this only protects the summary/bookkeeping layer.
    pending = list(jobs)
    active: list[tuple[Job, subprocess.Popen[str], Any, Any]] = []
    results: list[dict[str, Any]] = []
    stop_starting = False
    parallel_limit = min(int(max_parallel), len(gpu_ids))
    attempt_id = f"{time.time_ns()}-{os.getpid()}"
    ledger_path = _attempt_ledger_path(results_json) if results_json is not None else None
    if ledger_path is not None:
        _append_attempt_event(ledger_path, {
            "attempt_id": attempt_id,
            "event": "invocation_started",
            "job_count": len(jobs),
            "timestamp_ns": time.time_ns(),
        })

    def record(entry: dict[str, Any]) -> None:
        resolved_entry = {
            **entry,
            "attempt_id": attempt_id,
            "resolved_timestamp_ns": time.time_ns(),
        }
        results.append(resolved_entry)
        if ledger_path is not None:
            _append_attempt_event(ledger_path, {
                **resolved_entry,
                "event": "job_resolved",
            })
        if results_json is not None:
            _write_results(results_json, results)

    while pending or active:
        while pending and len(active) < parallel_limit and not stop_starting:
            active_gpu_ids = {active_job.gpu_id for active_job, *_rest in active}
            free_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in active_gpu_ids]
            if not free_gpu_ids:
                break
            job = pending.pop(0)
            already_resolved = has_completed_artifacts(job.run_dir) or (
                job.run_dir / "pretraining_failure.json"
            ).exists()
            if resume_skip_completed and already_resolved:
                try:
                    validation = validate_artifacts(
                        job.run_dir, job.row, expected_config=job.config
                    )
                    status = (
                        "skipped_terminal_ineligible"
                        if validation["terminal_ineligible"]
                        else "skipped_completed"
                    )
                    record({"run_id": job.row["run_id"], "status": status, **validation})
                    print(f"SKIP {status.removeprefix('skipped_')} {job.row['run_id']}")
                    continue
                except ManifestLaunchError as exc:
                    record({
                        "run_id": job.row["run_id"],
                        "status": "failed_existing_artifacts",
                        "error": str(exc),
                        "run_dir": str(job.run_dir),
                    })
                    print(
                        f"FAIL {job.row['run_id']} completed artifacts are invalid and were preserved: {exc}"
                    )
                    if stop_on_failure:
                        stop_starting = True
                    continue
            if has_any_artifacts(job.run_dir):
                if not overwrite_incomplete:
                    record({
                        "run_id": job.row["run_id"],
                        "status": "failed_incomplete_artifacts",
                        "error": "run directory contains incomplete artifacts; pass --overwrite-incomplete to rerun",
                        "run_dir": str(job.run_dir),
                    })
                    print(f"FAIL {job.row['run_id']} incomplete artifacts require --overwrite-incomplete")
                    if stop_on_failure:
                        stop_starting = True
                    continue
                try:
                    archived_run_dir = _archive_partial_run(job.run_dir, attempt_id)
                except (ManifestLaunchError, OSError) as exc:
                    record({
                        "run_id": job.row["run_id"],
                        "status": "failed_partial_archive",
                        "error": str(exc),
                        "run_dir": str(job.run_dir),
                    })
                    print(f"FAIL {job.row['run_id']} could not archive partial artifacts: {exc}")
                    if stop_on_failure:
                        stop_starting = True
                    continue
                job.config["overwrite"] = False
                if ledger_path is not None:
                    _append_attempt_event(ledger_path, {
                        "attempt_id": attempt_id,
                        "event": "partial_archived",
                        "run_id": job.row["run_id"],
                        "source": str(job.run_dir),
                        "destination": str(archived_run_dir),
                        "timestamp_ns": time.time_ns(),
                    })
                print(f"ARCHIVE partial {job.row['run_id']} -> {archived_run_dir}")
            job.run_dir.parent.mkdir(parents=True, exist_ok=True)
            job.run_dir.mkdir(parents=True, exist_ok=True)
            _log_dir(job.row).mkdir(parents=True, exist_ok=True)
            _trace_dir(job.row).mkdir(parents=True, exist_ok=True)
            assign_job_gpu(job, free_gpu_ids[0])
            print(f"START {job.row['run_id']} gpu={job.gpu_id}")
            if ledger_path is not None:
                _append_attempt_event(ledger_path, {
                    "attempt_id": attempt_id,
                    "command": job.command,
                    "config": str(job.config_path),
                    "event": "job_started",
                    "gpu_id": job.gpu_id,
                    "run_dir": str(job.run_dir),
                    "run_id": job.row["run_id"],
                    "timestamp_ns": time.time_ns(),
                })
            # Captured so a pretraining_failure.json written by the job can
            # record stdout/stderr hashes as evidence (closeout plan SS4.2) --
            # previously discarded entirely.
            stdout_handle = open(job.run_dir / "stdout.log", "w")
            stderr_handle = open(job.run_dir / "stderr.log", "w")
            process = subprocess.Popen(
                job.command,
                cwd=str(REPO_ROOT),
                env=job.env,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            active.append((job, process, stdout_handle, stderr_handle))

        still_active: list[tuple[Job, subprocess.Popen[str], Any, Any]] = []
        for job, process, stdout_handle, stderr_handle in active:
            returncode = process.poll()
            if returncode is None:
                still_active.append((job, process, stdout_handle, stderr_handle))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if returncode != 0:
                try:
                    pretraining_validation = validate_artifacts(
                        job.run_dir, job.row, expected_config=job.config
                    )
                except ManifestLaunchError:
                    pretraining_validation = None
                if pretraining_validation is not None and pretraining_validation.get(
                    "terminal_pretraining_ineligible"
                ):
                    record({
                        "run_id": job.row["run_id"],
                        "status": "terminal_ineligible",
                        "returncode": returncode,
                        **pretraining_validation,
                    })
                    print(f"TERMINAL_PRETRAINING_INELIGIBLE {job.row['run_id']}")
                else:
                    # No valid pretraining_failure.json -- an unexplained
                    # process failure, not a certified terminal outcome.
                    record({
                        "run_id": job.row["run_id"],
                        "status": "failed_process",
                        "returncode": returncode,
                        "run_dir": str(job.run_dir),
                    })
                    print(f"FAIL {job.row['run_id']} returncode={returncode}")
                if stop_on_failure:
                    stop_starting = True
                continue
            try:
                validation = validate_artifacts(
                    job.run_dir, job.row, expected_config=job.config
                )
                status = "terminal_ineligible" if validation["terminal_ineligible"] else "passed"
                record({"run_id": job.row["run_id"], "status": status, **validation})
                print(f"{status.upper()} {job.row['run_id']}")
            except ManifestLaunchError as exc:
                record({
                    "run_id": job.row["run_id"],
                    "status": "failed_validation",
                    "error": str(exc),
                    "run_dir": str(job.run_dir),
                })
                print(f"VALIDATION FAIL {job.row['run_id']}: {exc}")
                if stop_on_failure:
                    stop_starting = True
        active = still_active
        if stop_starting and not active:
            for job in pending:
                record({
                    "run_id": job.row["run_id"],
                    "status": "not_started_after_failure",
                    "run_dir": str(job.run_dir),
                })
            pending = []
        if active:
            time.sleep(5)

    if ledger_path is not None:
        _append_attempt_event(ledger_path, {
            "attempt_id": attempt_id,
            "event": "invocation_resolved",
            "result_count": len(results),
            "timestamp_ns": time.time_ns(),
        })
    return results


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, path)


def _attempt_ledger_path(results_json: Path) -> Path:
    return results_json.with_name(f"{results_json.stem}_attempts.jsonl")


def _append_attempt_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _archive_partial_run(run_dir: Path, attempt_id: str) -> Path:
    archive_root = run_dir.parent / "_interrupted_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{run_dir.name}.{attempt_id}"
    if destination.exists():
        raise ManifestLaunchError(f"partial-run archive already exists: {destination}")
    shutil.move(str(run_dir), str(destination))
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue federated runs from a protocol manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--main", default=str(DEFAULT_MAIN))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-root", default="results/rerun_protocol_v1")
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--only", action="append", default=[], help="Filter rows with KEY=VALUE; repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-skip-completed", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--default-learning-rate", type=float, default=None)
    parser.add_argument("--default-weight-decay", type=float, default=None)
    parser.add_argument("--override-comm-round", type=int, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-simple-model-selection-epochs", type=int, default=None)
    parser.add_argument("--override-f-history-model-selection-epochs", type=int, default=None)
    parser.add_argument("--override-model-selection-batch-size", type=int, default=None)
    parser.add_argument("--override-model-selection-max-samples", type=int, default=None)
    parser.add_argument("--skip-model-selection", dest="override_skip_model_selection", action="store_true", default=None)
    parser.add_argument("--skip-gmm-eval", dest="override_skip_gmm_eval", action="store_true", default=None)
    parser.add_argument("--disable-auxiliary-regression", dest="override_auxiliary_regression", action="store_false", default=None)
    parser.add_argument("--enable-auxiliary-regression", dest="override_auxiliary_regression", action="store_true")
    parser.add_argument("--override-auxiliary-regression-epochs", type=int, default=None)
    parser.add_argument("--rewrite-round-csv", dest="override_append_round_csv", action="store_false", default=None)
    parser.add_argument("--override-periodic-checkpoint-interval", type=int, default=None)
    parser.add_argument("--override-dataloader-num-workers", type=int, default=None)
    parser.add_argument("--dataloader-pin-memory", dest="override_dataloader_pin_memory", action="store_true", default=None)
    parser.add_argument("--results-json", default="experiments/rerun_protocol_v1/launcher_results.json")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    parser.add_argument("--keep-going", dest="stop_on_failure", action="store_false")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    main_path = Path(args.main)
    config_dir = Path(args.config_dir)
    output_root = Path(args.output_root)
    results_json = Path(args.results_json)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not main_path.is_absolute():
        main_path = REPO_ROOT / main_path
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if not results_json.is_absolute():
        results_json = REPO_ROOT / results_json

    gpu_ids = [int(item.strip()) for item in str(args.gpu_ids).split(",") if item.strip() != ""]
    if not gpu_ids:
        raise SystemExit("--gpu-ids must contain at least one GPU id")
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")

    filters = _parse_only(args.only)
    manifest_rows = _load_rows(manifest_path)
    selected = select_rows(manifest_rows, filters, bool(args.include_blocked))
    # An --only filter that matches nothing is essentially never intentional,
    # and silently succeeding with zero jobs is the worst possible outcome:
    # the run looks clean (exit 0, no error) while training nothing at all.
    # The concrete trap this guards is numeric formatting -- manifests write
    # alpha with %g, so alpha=1.0 lands in the CSV as the single character
    # "1", and _row_matches_filters compares strings exactly, making
    # `--only alpha=1.0` select 0 rows. Report the values actually present
    # for each filtered column so the mismatch is obvious immediately.
    if filters and not selected:
        details = []
        for key, expected in sorted(filters.items()):
            present = sorted({str(row.get(key, "")) for row in manifest_rows})
            if not any(key in row for row in manifest_rows):
                details.append(f"{key}={expected!r} (no such column in this manifest)")
            else:
                details.append(f"{key}={expected!r} (values present: {present})")
        raise SystemExit(
            "--only matched 0 of "
            f"{len(manifest_rows)} manifest rows, so there is nothing to run: "
            + "; ".join(details)
            + ". Note --only compares strings exactly."
        )
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    # Validate the rows actually being launched, after --only and --limit. A
    # mixed-root manifest is legitimate when the caller selects a single arm
    # from it, so checking every row here would reject valid filtered launches
    # (and let blocked or excluded rows veto a run they take no part in).
    assert_single_output_root(selected)
    jobs, skipped = build_jobs(
        selected,
        python_executable=args.python,
        main_path=main_path,
        config_dir=config_dir,
        output_root=output_root,
        gpu_ids=gpu_ids,
        default_learning_rate=args.default_learning_rate,
        default_weight_decay=args.default_weight_decay,
        override_comm_round=args.override_comm_round,
        override_epochs=args.override_epochs,
        override_simple_model_selection_epochs=args.override_simple_model_selection_epochs,
        override_f_history_model_selection_epochs=args.override_f_history_model_selection_epochs,
        override_model_selection_batch_size=args.override_model_selection_batch_size,
        override_model_selection_max_samples=args.override_model_selection_max_samples,
        override_skip_model_selection=args.override_skip_model_selection,
        override_skip_gmm_eval=args.override_skip_gmm_eval,
        override_auxiliary_regression=args.override_auxiliary_regression,
        override_auxiliary_regression_epochs=args.override_auxiliary_regression_epochs,
        override_append_round_csv=args.override_append_round_csv,
        override_periodic_checkpoint_interval=args.override_periodic_checkpoint_interval,
        override_dataloader_num_workers=args.override_dataloader_num_workers,
        override_dataloader_pin_memory=args.override_dataloader_pin_memory,
    )
    if args.dry_run:
        dry_run(jobs, skipped, None)
        return 0
    if skipped:
        print("Refusing real launch because selected rows are missing required hyperparameters:", file=sys.stderr)
        for item in skipped[:20]:
            print(f"  {item['run_id']}: {item['reason']}", file=sys.stderr)
        return 1
    if not jobs:
        print("No launchable jobs selected.", file=sys.stderr)
        return 1
    results = run_jobs(
        jobs,
        gpu_ids=gpu_ids,
        max_parallel=int(args.max_parallel),
        resume_skip_completed=bool(args.resume_skip_completed),
        overwrite_incomplete=bool(args.overwrite_incomplete),
        stop_on_failure=bool(args.stop_on_failure),
        results_json=results_json,
    )
    _write_results(results_json, results)
    resolved_statuses = {
        "passed",
        "skipped_completed",
        "terminal_ineligible",
        "skipped_terminal_ineligible",
    }
    failed = [row for row in results if row.get("status") not in resolved_statuses]
    print(json.dumps({
        "jobs": len(jobs),
        "results_json": str(results_json.relative_to(REPO_ROOT)),
        "passed": sum(row.get("status") == "passed" for row in results),
        "skipped_completed": sum(row.get("status") == "skipped_completed" for row in results),
        "terminal_ineligible": sum(
            row.get("status") in {"terminal_ineligible", "skipped_terminal_ineligible"}
            for row in results
        ),
        "failed": len(failed),
    }, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
