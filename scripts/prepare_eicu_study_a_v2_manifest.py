#!/usr/bin/env python3
"""Create Study A v2 tuning or frozen 105-row final manifests.

The tuning stage is exactly 3 g0 x 2 federated methods x the local-LR
multipliers x the server LRs given in ``protocol_v2.json`` ("tuning"), all on
scenario seed 11.  The final stage is materialized only after a
validation-selected hyperparameter JSON exists and contains 30 primary
federated, 45 centralized, and 30 aggregation-ablation rows.

The grid and the round budget are read from the protocol at import time -- see
``load_protocol`` -- so that amending the protocol is sufficient to change what
this script emits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

G0_VARIANTS = ("linear", "interaction", "mlp")
FEDERATED_METHODS = ("fedgda_s", "fedogda_s")
CENTRALIZED_METHODS = ("gda_d", "sgda_s", "oadam_s")
TUNING_PAIR = ("tuning_01", 11, 1011)
FINAL_SEED_PAIRS = (
    ("confirmatory_01", 101, 1101),
    ("confirmatory_02", 102, 1102),
    ("confirmatory_03", 103, 1103),
    ("confirmatory_04", 104, 1104),
    ("confirmatory_05", 105, 1105),
)
PROTOCOL_VERSION = "eicu_study_a_v2_offhours"
PROTOCOL_PATH = REPO_ROOT / "experiments" / PROTOCOL_VERSION / "protocol_v2.json"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Read the pre-registered protocol.

    The grid and horizon below are *derived* from this file rather than
    duplicated as literals. Study A v2's first campaign was invalidated
    because the protocol and the code that generated its manifests drifted
    apart, so a stale constant here is exactly the failure mode to avoid.
    """
    with path.open() as fh:
        return json.load(fh)


_PROTOCOL = load_protocol()

BASE_LEARNING_RATE = float(_PROTOCOL["tuning"]["base_learning_rate"])
LOCAL_LR_MULTIPLIERS = tuple(float(m) for m in _PROTOCOL["tuning"]["local_lr_multipliers"])
SERVER_LEARNING_RATES = tuple(
    float(s) for s in _PROTOCOL["tuning"]["server_learning_rates"]
)
TUNING_ROUNDS = int(_PROTOCOL["training"]["tuning_rounds"])
FINAL_ROUNDS = int(_PROTOCOL["training"]["final_rounds"])
WEIGHT_DECAY = 0.01
CRITIC_MULTIPLIER = 10.0
PRIMARY_METRIC = "equal_client_validation_mse"
TIE_BREAKER = "equal_client_validation_moment_violation_at_best_validation"

FIELDNAMES = [
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
    "require_multibatch_stochastic",
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
    "scenario_name",
    "objective_mode",
    "aggregation_weighting",
    "input_dim_g",
    "input_dim_f",
    "hidden_widths",
    "model_activation",
    "g0",
    "skip_model_selection",
    "scenario_seed",
    "optimizer_seed",
    "seed_pair_id",
    "campaign_role",
    "scenario_checksum",
    "role",
    "g0_display_label",
    "alignment_label",
    "primary_selection_metric",
    "test_mse_used_for_selection",
    "selection_source",
    "study_claim",
    "scenario_scope",
    "scenario_metadata_path",
    "config_path",
    "result_path",
]


def load_metadata(scenario_dir: Path, g0: str, scenario_seed: int) -> dict[str, Any]:
    path = scenario_dir / f"{g0}_scenario_seed{scenario_seed}_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing certified scenario metadata: {path}")
    with path.open() as handle:
        metadata = json.load(handle)
    if not metadata.get("certification_passed"):
        raise ValueError(f"scenario is not certified: {path}")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"scenario has wrong protocol_version: {path}")
    return metadata


def _federated_row(
    *,
    scenario_dir: Path,
    output_root: Path,
    g0: str,
    method: str,
    seed_pair_id: str,
    scenario_seed: int,
    optimizer_seed: int,
    role: str,
    run_id: str,
    rounds: int,
    learning_rate: float,
    server_learning_rate: float,
    aggregation_weighting: str,
    learning_rate_status: str,
    notes: str = "",
) -> dict[str, Any]:
    metadata = load_metadata(scenario_dir, g0, scenario_seed)
    dataset = scenario_dir.name
    campaign_role = (
        "aggregation_ablation" if role == "aggregation_ablation" else ""
    )
    optimizer = "ogda" if method == "fedogda_s" else "sgd"
    return {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "run_group": role,
        "training_scope": "federated",
        "method": method,
        "method_label": "FedOGDA" if method == "fedogda_s" else "FedGDA",
        "dataset": dataset,
        "seed": optimizer_seed,
        "alpha": "",
        "output_root": str(output_root),
        "final_result_dir": "",
        "implementation_status": "implemented",
        "run_status": "not_started",
        "preflight_required": "true",
        "preflight_status": "certified_scenario",
        "model": "lr",
        "federated_optimizer": "FedAvg",
        "client_optimizer": optimizer,
        "client_num_in_total": int(metadata["n_clients"]),
        "client_num_per_round": int(metadata["n_clients"]),
        "comm_round": rounds,
        "epochs": 1,
        "batch_size": 4,
        "require_multibatch_stochastic": "true",
        "partition_method": "natural",
        "partition_alpha": 0.0,
        "data_cache_dir": str(scenario_dir.parent),
        "learning_rate": learning_rate,
        "learning_rate_status": learning_rate_status,
        "weight_decay": WEIGHT_DECAY,
        "critic_multiplier": CRITIC_MULTIPLIER,
        "server_learning_rate": server_learning_rate,
        "gradient_clip_norm": 1.0,
        "simple_model_selection_epochs": 100,
        "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200,
        "using_gpu": "false",
        "gpu_id": 0,
        "notes": notes,
        "scenario_name": f"{g0}_scenario_seed{scenario_seed}",
        "objective_mode": "paper_aligned",
        "aggregation_weighting": aggregation_weighting,
        "input_dim_g": int(metadata["input_dim"]),
        "input_dim_f": int(metadata["instrument_dim"]),
        "hidden_widths": "32,32",
        "model_activation": "relu",
        "g0": g0,
        "g0_display_label": metadata["g0_display_label"],
        "skip_model_selection": "true",
        "scenario_seed": scenario_seed,
        "optimizer_seed": optimizer_seed,
        "seed_pair_id": seed_pair_id,
        "campaign_role": campaign_role,
        "scenario_checksum": metadata["scenario_checksum_sha256"],
        "role": role,
        "alignment_label": {
            "tuning": "validation_only_tuning",
            "confirmatory": "primary_extension",
            "aggregation_ablation": "non_paper_aligned_aggregation_ablation",
        }[role],
        "primary_selection_metric": PRIMARY_METRIC,
        "test_mse_used_for_selection": "false",
        "selection_source": "validation_only",
        "study_claim": "semi_synthetic_benchmark_no_clinical_claim",
        "scenario_scope": metadata["scenario_scope"],
        "scenario_metadata_path": str(
            scenario_dir
            / f"{g0}_scenario_seed{scenario_seed}_metadata.json"
        ),
        "config_path": "",
        "result_path": str(
            Path(dataset) / method / f"seed_{optimizer_seed}" / run_id
        ),
    }


def generate_tuning(scenario_dir: Path, output_root: Path) -> list[dict[str, Any]]:
    pair_id, scenario_seed, optimizer_seed = TUNING_PAIR
    rows = []
    for g0 in G0_VARIANTS:
        for method in FEDERATED_METHODS:
            for multiplier in LOCAL_LR_MULTIPLIERS:
                for server_lr in SERVER_LEARNING_RATES:
                    run_id = (
                        f"tuning_{g0}_{method}_{pair_id}_"
                        f"lrm{multiplier}_slr{server_lr}"
                    ).replace(".", "p")
                    rows.append(
                        _federated_row(
                            scenario_dir=scenario_dir,
                            output_root=output_root,
                            g0=g0,
                            method=method,
                            seed_pair_id=pair_id,
                            scenario_seed=scenario_seed,
                            optimizer_seed=optimizer_seed,
                            role="tuning",
                            run_id=run_id,
                            rounds=TUNING_ROUNDS,
                            learning_rate=BASE_LEARNING_RATE * multiplier,
                            server_learning_rate=server_lr,
                            aggregation_weighting="uniform_clients",
                            learning_rate_status="validation_tuning_candidate",
                            notes=(
                                f"local_lr_multiplier={multiplier};"
                                f"tie_breaker={TIE_BREAKER}"
                            ),
                        )
                    )
    return rows


def _selected_key(g0: str, method: str) -> str:
    return f"{g0}:{method}"


def load_selected(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        selected = json.load(handle)
    required = {
        _selected_key(g0, method)
        for g0 in G0_VARIANTS
        for method in FEDERATED_METHODS
    }
    missing = required - set(selected)
    if missing:
        raise ValueError(
            "selected hyperparameters are incomplete; missing "
            + ", ".join(sorted(missing))
        )
    return selected


def _centralized_row(
    *,
    scenario_dir: Path,
    output_root: Path,
    g0: str,
    method: str,
    seed_pair_id: str,
    scenario_seed: int,
    optimizer_seed: int,
) -> dict[str, Any]:
    metadata = load_metadata(scenario_dir, g0, scenario_seed)
    dataset = scenario_dir.name
    run_id = f"centralized_{g0}_{method}_{seed_pair_id}"
    return {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "run_group": "centralized_baseline",
        "training_scope": "centralized",
        "method": method,
        "method_label": method.upper(),
        "dataset": dataset,
        "seed": optimizer_seed,
        "alpha": "",
        "output_root": str(output_root),
        "implementation_status": "implemented",
        "run_status": "not_started",
        "preflight_required": "true",
        "preflight_status": "certified_scenario",
        "model": "lr",
        "client_optimizer": {
            "gda_d": "sgd",
            "sgda_s": "sgd",
            "oadam_s": "oadam",
        }[method],
        "client_num_in_total": int(metadata["n_clients"]),
        "client_num_per_round": int(metadata["n_clients"]),
        "comm_round": FINAL_ROUNDS,
        "epochs": 1,
        "batch_size": {"gda_d": 0, "sgda_s": 256, "oadam_s": 256}[method],
        "require_multibatch_stochastic": "",
        "partition_method": "not_applicable_centralized",
        "partition_alpha": 0.0,
        "data_cache_dir": str(scenario_dir.parent),
        "learning_rate": BASE_LEARNING_RATE,
        "learning_rate_status": "fixed_centralized_reference",
        "weight_decay": WEIGHT_DECAY,
        "critic_multiplier": CRITIC_MULTIPLIER,
        "server_learning_rate": "",
        "gradient_clip_norm": 1.0,
        "using_gpu": "false",
        "gpu_id": 0,
        "notes": "centralized diagnostic baseline; no federated aggregation",
        "scenario_name": f"{g0}_scenario_seed{scenario_seed}",
        "objective_mode": "paper_aligned",
        "aggregation_weighting": "none",
        "input_dim_g": int(metadata["input_dim"]),
        "input_dim_f": int(metadata["instrument_dim"]),
        "hidden_widths": "32,32",
        "model_activation": "relu",
        "g0": g0,
        "g0_display_label": metadata["g0_display_label"],
        "skip_model_selection": "true",
        "scenario_seed": scenario_seed,
        "optimizer_seed": optimizer_seed,
        "seed_pair_id": seed_pair_id,
        "campaign_role": "",
        "scenario_checksum": metadata["scenario_checksum_sha256"],
        "role": "centralized_baseline",
        "alignment_label": "centralized_reference",
        "primary_selection_metric": PRIMARY_METRIC,
        "test_mse_used_for_selection": "false",
        "selection_source": "validation_only",
        "study_claim": "semi_synthetic_benchmark_no_clinical_claim",
        "scenario_scope": metadata["scenario_scope"],
        "scenario_metadata_path": str(
            scenario_dir
            / f"{g0}_scenario_seed{scenario_seed}_metadata.json"
        ),
        "config_path": "",
        "result_path": str(
            Path("centralized")
            / f"{g0}_{seed_pair_id}"
            / method
            / f"seed_{optimizer_seed}"
        ),
    }


def generate_final(
    scenario_dir: Path, output_root: Path, selected: dict[str, Any]
) -> list[dict[str, Any]]:
    primary = []
    centralized = []
    ablation = []
    for g0 in G0_VARIANTS:
        for pair_id, scenario_seed, optimizer_seed in FINAL_SEED_PAIRS:
            for method in FEDERATED_METHODS:
                choice = selected[_selected_key(g0, method)]
                common = dict(
                    scenario_dir=scenario_dir,
                    output_root=output_root,
                    g0=g0,
                    method=method,
                    seed_pair_id=pair_id,
                    scenario_seed=scenario_seed,
                    optimizer_seed=optimizer_seed,
                    rounds=FINAL_ROUNDS,
                    learning_rate=float(choice["learning_rate"]),
                    server_learning_rate=float(choice["server_learning_rate"]),
                    learning_rate_status="frozen_from_validation_tuning",
                )
                primary.append(
                    _federated_row(
                        **common,
                        role="confirmatory",
                        run_id=f"confirmatory_{g0}_{method}_{pair_id}",
                        aggregation_weighting="uniform_clients",
                        notes=f"selected_tuning_run={choice.get('run_id', '')}",
                    )
                )
                ablation.append(
                    _federated_row(
                        **common,
                        role="aggregation_ablation",
                        run_id=f"ablation_{g0}_{method}_sample_size_{pair_id}",
                        aggregation_weighting="sample_size",
                        notes=(
                            "sample-size aggregation ablation;"
                            f"selected_tuning_run={choice.get('run_id', '')}"
                        ),
                    )
                )
            for method in CENTRALIZED_METHODS:
                centralized.append(
                    _centralized_row(
                        scenario_dir=scenario_dir,
                        output_root=output_root,
                        g0=g0,
                        method=method,
                        seed_pair_id=pair_id,
                        scenario_seed=scenario_seed,
                        optimizer_seed=optimizer_seed,
                    )
                )
    rows = primary + centralized + ablation
    if (len(primary), len(centralized), len(ablation), len(rows)) != (
        30,
        45,
        30,
        105,
    ):
        raise RuntimeError("internal error: final matrix is not 30+45+30=105")
    return rows


def write_manifest(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    with output.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("tuning", "final"), required=True)
    parser.add_argument(
        "--scenario-dir",
        default=str(
            REPO_ROOT
            / "fedgmm"
            / "sp_decentralized_mnist_lr_example"
            / "data"
            / "eicu_semisynth_offhours_v2_demo"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "eicu_study_a_v2_offhours_demo"),
    )
    parser.add_argument("--selected-hyperparameters", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenario_dir = Path(args.scenario_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if args.stage == "tuning":
        rows = generate_tuning(scenario_dir, output_root)
    else:
        if not args.selected_hyperparameters:
            raise ValueError(
                "--selected-hyperparameters is required for --stage final; "
                "final configurations must not be guessed before validation tuning"
            )
        selected = load_selected(Path(args.selected_hyperparameters).resolve())
        rows = generate_final(scenario_dir, output_root, selected)
    output = Path(args.out).resolve()
    write_manifest(rows, output)
    print(f"stage={args.stage} rows={len(rows)}")
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
