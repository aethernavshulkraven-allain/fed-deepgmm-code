#!/usr/bin/env python3
"""Run true centralized low-dimensional DeepGMM baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Every model here is a tiny 1-2 hidden-layer MLP on at most a few thousand
# rows; PyTorch's CPU default is one BLAS thread per core, which on a shared
# multi-user machine means one process can grab dozens of cores for matmuls
# too small to benefit from that parallelism (measured: ~29 cores per process,
# almost pure thread-contention overhead, no speedup). Must be set before
# numpy/torch are imported -- the BLAS thread pool is initialized at import.
_DEFAULT_CPU_THREADS = "4"
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_env_var, _DEFAULT_CPU_THREADS)

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from experiment_utils import (  # noqa: E402
    OBJECTIVE_MODE_CHOICES,
    PAPER_ALIGNED_LAMBDA,
    check_eicu_objective_mode,
    config_checksum,
    equal_client_moment_violation,
    equal_client_structural_mse,
    json_safe,
    moment_violation,
    predictions_for_split,
    prepare_run_dir,
    save_predictions_npz,
    state_is_finite,
    structural_mse,
    structural_mse_from_predictions,
    write_effective_config,
    write_metrics,
    write_per_client_metrics,
    write_test_mse_by_round,
)
from game_objectives.simple_moment_objective import (  # noqa: E402
    OptimalMomentObjective,
    PaperAlignedMomentObjective,
)
from models.mlp_model import MLPModel  # noqa: E402
from optimizers.Customsgd import CustomSGD  # noqa: E402
from optimizers.oadam import OAdam  # noqa: E402
from scenarios.abstract_scenario import AbstractScenario  # noqa: E402


DATASET_ALIASES = {
    "absolute": "abs",
    "sine": "sin",
}
LOWDIM_DATASETS = {"abs", "step", "linear", "sin"}
# eicu_semisynth is not a single scenario file (unlike the zoo datasets): the
# actual file is data/eicu_semisynth/<scenario_name>.npz, so which scenario to
# load is a separate --scenario-name argument, not encoded in --dataset.
EICU_DATASETS = {"eicu_semisynth", "eicu_semisynth_offhours_v2_demo"}
METHODS = {"gda", "sgda", "oadam"}
# protocol_v1.md S6.2: the canonical campaign method label suffixes
# deterministic/stochastic exactly like the federated fedgda_s/fedogda_s
# convention. Recorded alongside the bare "method"/"algorithm" fields (which
# stay bare for backward compatibility with every pre-existing zoo
# centralized result) so eICU campaign tooling has the protocol-exact label
# without renaming anything consumers already depend on.
CANONICAL_METHOD_LABEL = {"gda": "gda_d", "sgda": "sgda_s", "oadam": "oadam_s"}

# Matches model_hub.py's eicu* branch; centralized baselines must use the same
# architecture as the federated runs they are compared against.
EICU_HIDDEN_WIDTHS = [64, 64]
ZOO_HIDDEN_WIDTHS = [20, 20]


def canonical_dataset(dataset: str) -> str:
    value = DATASET_ALIASES.get(dataset.lower(), dataset.lower())
    if value not in LOWDIM_DATASETS and not value.startswith("eicu"):
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if args.no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.gpu_id is not None:
        return torch.device(f"cuda:{int(args.gpu_id)}")
    return torch.device("cuda")


def move_split(split: Any, device: torch.device) -> Any:
    for name in ("x", "z", "y", "g", "w"):
        tensor = getattr(split, name, None)
        if torch.is_tensor(tensor):
            setattr(split, name, tensor.to(device).double())
    return split


def load_pooled_splits(
    dataset: str, data_dir: Path, device: torch.device, scenario_name: str | None = None
) -> tuple[Any, Any, Any, Path]:
    if dataset.startswith("eicu"):
        if not scenario_name:
            raise ValueError(f"--scenario-name is required for --dataset {dataset!r}")
        scenario_path = data_dir / dataset / f"{scenario_name}.npz"
    else:
        scenario_path = data_dir / "zoo" / f"{dataset}.npz"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario file: {scenario_path}")
    scenario = AbstractScenario(filename=str(scenario_path))
    scenario.to_tensor()
    train = move_split(scenario.get_dataset("train"), device)
    val = move_split(scenario.get_dataset("dev"), device)
    test = move_split(scenario.get_dataset("test"), device)
    return train, val, test, scenario_path


def build_models(
    device: torch.device,
    input_dim_g: int,
    input_dim_f: int,
    hidden_widths: list[int],
    activation: type[torch.nn.Module] = nn.LeakyReLU,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Dimensions are derived from the loaded data (train.x/.z widths), not
    hardcoded -- the zoo scenarios are always 1-D treatment / 2-D instrument,
    but eicu_semisynth packs [D, W] / [Z, W] and its width depends on the
    cohort's covariate count.
    """
    g = MLPModel(input_dim=input_dim_g, layer_widths=hidden_widths, activation=activation).double()
    f = MLPModel(input_dim=input_dim_f, layer_widths=hidden_widths, activation=activation).double()
    g.initialize()
    f.initialize()
    return g.to(device), f.to(device)


def build_optimizer(method: str, model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if method in {"gda", "sgda"}:
        return CustomSGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    if method == "oadam":
        return OAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported method {method!r}")


def choose_batch_indices(split: Any, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    n = int(split.y.shape[0])
    if batch_size <= 0 or batch_size >= n:
        return torch.arange(n, device=split.y.device)
    perm = torch.randperm(n, generator=generator, device=split.y.device)
    return perm[:batch_size]


def capture_state(g: torch.nn.Module, f: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "g_state_dict": {key: value.detach().cpu().clone() for key, value in g.state_dict().items()},
        "f_state_dict": {key: value.detach().cpu().clone() for key, value in f.state_dict().items()},
    }


def load_state(g: torch.nn.Module, f: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    g.load_state_dict(copy.deepcopy(state["g_state_dict"]))
    f.load_state_dict(copy.deepcopy(state["f_state_dict"]))


def finite_state_and_metrics(g: torch.nn.Module, f: torch.nn.Module, values: list[float]) -> bool:
    return (
        state_is_finite(g.state_dict())
        and state_is_finite(f.state_dict())
        and all(math.isfinite(float(value)) for value in values)
    )


def write_mse_csv(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = run_dir / "mse_by_round.csv"
    fieldnames = ["round", "train_mse", "val_mse", "primary_val_mse"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def save_checkpoint(
    path: Path,
    round_idx: int,
    checkpoint_type: str,
    state: dict[str, dict[str, torch.Tensor]],
    metrics: dict[str, Any],
    effective_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "round": int(round_idx),
        "checkpoint_type": checkpoint_type,
        "state": {
            "g": state["g_state_dict"],
            "f": state["f_state_dict"],
        },
        "g_state_dict": state["g_state_dict"],
        "f_state_dict": state["f_state_dict"],
        "metrics": json_safe(metrics),
        "effective_config": json_safe(effective_config),
    }, path)


def default_batch_size(method: str, raw_batch_size: int | None) -> int:
    if raw_batch_size is not None:
        return int(raw_batch_size)
    if method == "gda":
        return 0
    return 256


def default_output_dir(dataset: str, method: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "centralized_lowdim_v1" / dataset / method / f"seed_{seed}"


def build_effective_config(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    batch_size: int,
    output_dir: Path,
    device: torch.device,
    scenario_path: Path,
    input_dim_g: int,
    input_dim_f: int,
    hidden_widths: list[int],
    objective_mode: str,
) -> dict[str, Any]:
    mode = "deterministic" if batch_size <= 0 else "stochastic"
    objective_name = "PaperAlignedMomentObjective" if objective_mode == "paper_aligned" else "OptimalMomentObjective"
    scenario_seed = int(getattr(args, "scenario_seed", args.seed) or args.seed)
    return {
        "training_scope": "centralized",
        "method": method,
        "algorithm": method,
        "variant": method,
        "mode": mode,
        "dataset": dataset,
        "seed": int(args.seed),
        "random_seed": int(args.seed),
        # "none": no client aggregation step to weight (the campaign
        # validator hard-checks this literal value for federated: false
        # roles). protocol_v1.md's descriptive aggregation_mode field for
        # this case is the more explicit not_applicable_centralized;
        # aggregation_weighting itself is simply inapplicable.
        "aggregation_weighting": "none",
        "objective_mode": objective_mode,
        "scenario_seed": scenario_seed,
        "optimizer_seed": int(args.seed),
        "seed_pair_id": str(getattr(args, "seed_pair_id", "") or ""),
        "campaign_role": str(getattr(args, "campaign_role", "") or ""),
        "scenario_checksum": str(getattr(args, "scenario_checksum", "") or ""),
        "protocol_version": str(getattr(args, "protocol_version", "") or ""),
        "role": str(getattr(args, "role", "") or ""),
        "scenario_name": str(getattr(args, "scenario_name", "") or ""),
        "g0": str(getattr(args, "g0", "") or ""),
        "alignment_label": str(getattr(args, "alignment_label", "") or ""),
        "primary_selection_metric": str(
            getattr(args, "primary_selection_metric", "equal_client_validation_mse")
            or "equal_client_validation_mse"
        ),
        "selection_source": str(
            getattr(args, "selection_source", "validation_only") or "validation_only"
        ),
        "scenario_scope": str(getattr(args, "scenario_scope", "") or ""),
        "study_claim": str(getattr(args, "study_claim", "") or ""),
        "campaign_method_label": CANONICAL_METHOD_LABEL.get(method, method),
        "iterations": int(args.iterations),
        "batch_size": int(batch_size),
        "g_learning_rate": float(args.g_lr),
        "f_learning_rate": float(args.f_lr),
        "learning_rate": float(args.g_lr),
        "weight_decay": float(args.weight_decay),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "test_mse_logged_by_round": bool(args.log_test_mse_by_round),
        "log_test_mse_by_round": bool(args.log_test_mse_by_round),
        "uses_clients": False,
        "uses_fedavg_aggregation": False,
        "uses_client_sampling": False,
        "uses_server_learning_rate_aggregation": False,
        "objective": objective_name,
        "input_dim_g": int(input_dim_g),
        "input_dim_f": int(input_dim_f),
        "hidden_widths": list(hidden_widths),
        "model_activation": str(
            getattr(args, "model_activation", "leaky_relu")
        ),
        "model": f"MLPModel(g:{input_dim_g}->{hidden_widths}, f:{input_dim_f}->{hidden_widths})",
        "data_path": str(scenario_path.relative_to(REPO_ROOT) if scenario_path.is_relative_to(REPO_ROOT) else scenario_path),
        "output_dir": str(output_dir),
        "run_id": str(args.run_id or f"centralized_{dataset}_{method}_seed{args.seed}"),
        "device": str(device),
        "overwrite": bool(args.overwrite),
    }


def run(args: argparse.Namespace) -> Path:
    method = args.method.lower()
    if method not in METHODS:
        raise ValueError(f"Unsupported method {args.method!r}")
    dataset = canonical_dataset(args.dataset)
    batch_size = default_batch_size(method, args.batch_size)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(dataset, method, args.seed)
    output_dir = output_dir.resolve()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if method == "gda" and batch_size > 0:
        raise ValueError("gda is deterministic; use --batch-size 0")
    if method in {"sgda", "oadam"} and batch_size <= 0:
        raise ValueError(f"{method} requires a positive minibatch --batch-size")

    prepare_run_dir(str(output_dir), overwrite=args.overwrite)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    objective_mode = str(getattr(args, "objective_mode", "legacy"))
    if objective_mode not in OBJECTIVE_MODE_CHOICES:
        raise ValueError(f"objective_mode must be one of {OBJECTIVE_MODE_CHOICES}, got {objective_mode!r}")
    check_eicu_objective_mode(dataset, objective_mode)

    set_seed(args.seed)
    device = resolve_device(args)
    data_dir = Path(args.data_dir).resolve()
    train, val, test, scenario_path = load_pooled_splits(
        dataset, data_dir, device, scenario_name=getattr(args, "scenario_name", None)
    )
    # Hospital membership survives untouched on these splits exactly as it
    # does in FedAvgAPI: move_split only moves x/z/y/g/w, never client_id.
    # None for every non-eICU scenario -- the gate this whole block hangs on.
    train_client_id = getattr(train, "client_id", None)
    val_client_id = getattr(val, "client_id", None)
    test_client_id = getattr(test, "client_id", None)
    input_dim_g = int(train.x.shape[1])
    input_dim_f = int(train.z.shape[1])
    if dataset.startswith("eicu"):
        hidden_widths = [
            int(piece.strip())
            for piece in str(getattr(args, "hidden_widths", "64,64")).strip("[]").split(",")
            if piece.strip()
        ]
        if not hidden_widths or any(width <= 0 for width in hidden_widths):
            raise ValueError("--hidden-widths must contain positive integers")
        activation_choices = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
        }
        model_activation = str(
            getattr(args, "model_activation", "leaky_relu")
        )
        if model_activation not in activation_choices:
            raise ValueError(
                f"--model-activation must be one of {sorted(activation_choices)}"
            )
        activation = activation_choices[model_activation]
    else:
        hidden_widths = ZOO_HIDDEN_WIDTHS
        activation = nn.LeakyReLU
    g, f = build_models(
        device, input_dim_g, input_dim_f, hidden_widths, activation=activation
    )
    objective = (
        PaperAlignedMomentObjective(lambda_1=PAPER_ALIGNED_LAMBDA)
        if objective_mode == "paper_aligned"
        else OptimalMomentObjective()
    )
    g_optimizer = build_optimizer(method, g, args.g_lr, args.weight_decay)
    f_optimizer = build_optimizer(method, f, args.f_lr, args.weight_decay)
    effective_config = build_effective_config(
        args, dataset, method, batch_size, output_dir, device, scenario_path,
        input_dim_g, input_dim_f, hidden_widths, objective_mode,
    )
    write_effective_config(str(output_dir), effective_config)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    mse_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_validation_mse = float("inf")
    best_validation_round = -1
    diverged = False
    finite_history = True
    start_time = time.time()
    final_round = -1

    for round_idx in range(int(args.iterations)):
        final_round = round_idx
        g.train()
        f.train()
        if objective_mode == "paper_aligned":
            # Frozen theta~ = g's parameters before this iteration's update --
            # centralized training has no multi-step "communication round" the
            # way federated clients do, so "one iteration" is the natural
            # analogue of "one round" here.
            objective.set_theta_tilde(g)
        idx = choose_batch_indices(train, batch_size, generator)
        x_batch = train.x[idx]
        z_batch = train.z[idx]
        y_batch = train.y[idx]

        g_obj, f_obj = objective.calc_objective(g, f, x_batch, z_batch, y_batch)
        g_optimizer.zero_grad()
        f_optimizer.zero_grad()
        g_obj.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(g.parameters(), float(args.gradient_clip_norm))
        g_optimizer.step()

        f_optimizer.zero_grad()
        f_obj.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), float(args.gradient_clip_norm))
        f_optimizer.step()

        train_mse = float(structural_mse(g, train))
        val_mse = float(structural_mse(g, val))
        primary_val_mse = val_mse
        if val_client_id is not None:
            val_prediction = predictions_for_split(g, val)
            primary_val_mse, _, _ = equal_client_structural_mse(val_prediction, val.g, val_client_id)
        row_finite = finite_state_and_metrics(g, f, [train_mse, val_mse])
        if not row_finite:
            diverged = True
            finite_history = False
        mse_rows.append({
            "round": round_idx,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "primary_val_mse": primary_val_mse,
        })
        write_mse_csv(output_dir, mse_rows)

        if args.log_test_mse_by_round:
            test_mse = float(structural_mse(g, test))
            test_finite = math.isfinite(test_mse)
            test_rows.append({
                "round": round_idx,
                "test_mse": test_mse,
                "finite": test_finite,
                "diverged": diverged or not test_finite,
            })
            write_test_mse_by_round(str(output_dir), test_rows)
            if not test_finite:
                finite_history = False
                diverged = True

        if row_finite and primary_val_mse < best_validation_mse:
            best_validation_mse = primary_val_mse
            best_validation_round = round_idx
            best_state = capture_state(g, f)
            save_checkpoint(
                output_dir / "checkpoints" / "best_validation.pt",
                round_idx,
                "best_validation",
                best_state,
                {"train_mse": train_mse, "val_mse": val_mse},
                effective_config,
            )
        if diverged:
            break

    final_state = capture_state(g, f)
    final_train_mse = float(structural_mse(g, train))
    final_val_mse = float(structural_mse(g, val))
    final_test_mse = float(structural_mse(g, test))
    final_prediction_now = predictions_for_split(g, test)
    equal_client_final_validation_structural_mse = None
    sample_weighted_final_validation_structural_mse = None
    primary_final_val_mse = final_val_mse
    if val_client_id is not None:
        final_val_prediction = predictions_for_split(g, val)
        (
            equal_client_final_validation_structural_mse,
            sample_weighted_final_validation_structural_mse,
            _,
        ) = equal_client_structural_mse(final_val_prediction, val.g, val_client_id)
        primary_final_val_mse = equal_client_final_validation_structural_mse
    equal_client_final_test_structural_mse = None
    sample_weighted_final_test_structural_mse = None
    primary_final_test_mse = final_test_mse
    if test_client_id is not None:
        (
            equal_client_final_test_structural_mse,
            sample_weighted_final_test_structural_mse,
            _,
        ) = equal_client_structural_mse(final_prediction_now, test.g, test_client_id)
        primary_final_test_mse = equal_client_final_test_structural_mse
    save_checkpoint(
        output_dir / "checkpoints" / "final.pt",
        final_round,
        "final",
        final_state,
        {
            "train_mse": final_train_mse,
            "val_mse": final_val_mse,
            "test_mse": final_test_mse,
        },
        effective_config,
    )

    if best_state is None:
        best_state = copy.deepcopy(final_state)
        best_validation_round = final_round
        best_validation_mse = primary_final_val_mse
        save_checkpoint(
            output_dir / "checkpoints" / "best_validation.pt",
            final_round,
            "best_validation",
            best_state,
            {"train_mse": final_train_mse, "val_mse": final_val_mse},
            effective_config,
        )

    final_prediction = predictions_for_split(g, test)
    load_state(g, f, best_state)
    best_prediction = predictions_for_split(g, test)
    test_mse_at_best_validation = float(structural_mse_from_predictions(best_prediction, test.g))
    save_predictions_npz(str(output_dir), test, best_prediction, final_prediction, effective_config)

    # protocol_v1.md S6.2/metric_policy.md: centralized checkpoints must
    # reconstruct hospital membership for validation selection and test
    # reporting too, using the same equal-client primary metric as the
    # federated primary runs they are compared against.
    equal_client_test_mse_at_best_validation = None
    sample_weighted_test_mse_at_best_validation = None
    per_client_test_mse_at_best_validation = None
    primary_test_mse_at_best_validation = test_mse_at_best_validation
    equal_client_best_validation_structural_mse = None
    sample_weighted_validation_mse_at_best_validation = None
    per_client_test_moment_violation_at_best_validation = None
    equal_client_test_moment_violation_at_best_validation = None
    sample_weighted_test_moment_violation_at_best_validation = None
    per_client_metrics_artifact = None
    if test_client_id is not None:
        (
            equal_client_test_mse_at_best_validation,
            sample_weighted_test_mse_at_best_validation,
            per_client_test_mse_at_best_validation,
        ) = equal_client_structural_mse(best_prediction, test.g, test_client_id)
        primary_test_mse_at_best_validation = equal_client_test_mse_at_best_validation
        f.eval()
        with torch.no_grad():
            best_test_f_prediction = f(test.z)
        f.train()
        (
            equal_client_test_moment_violation_at_best_validation,
            sample_weighted_test_moment_violation_at_best_validation,
            per_client_test_moment_violation_at_best_validation,
        ) = equal_client_moment_violation(best_test_f_prediction, test.y, best_prediction, test_client_id)
        per_client_metrics_artifact = write_per_client_metrics(
            str(output_dir),
            per_client_test_mse_at_best_validation,
            per_client_test_moment_violation_at_best_validation,
        )
    if val_client_id is not None:
        best_val_prediction = predictions_for_split(g, val)
        (
            equal_client_best_validation_structural_mse,
            sample_weighted_validation_mse_at_best_validation,
            _,
        ) = equal_client_structural_mse(best_val_prediction, val.g, val_client_id)

    final_vs_best_validation_gap = primary_final_val_mse - best_validation_mse
    final_vs_best_test_gap = primary_final_test_mse - primary_test_mse_at_best_validation
    recent_primary_val_mse = [
        row["primary_val_mse"] for row in mse_rows[-50:] if row.get("primary_val_mse") is not None
    ]
    if recent_primary_val_mse:
        last_50_validation_mse_std = float(np.std(recent_primary_val_mse))
        last_50_validation_mse_range = float(max(recent_primary_val_mse) - min(recent_primary_val_mse))
    else:
        last_50_validation_mse_std = None
        last_50_validation_mse_range = None

    finite_history = finite_history and all(
        math.isfinite(float(row["train_mse"])) and math.isfinite(float(row["val_mse"]))
        for row in mse_rows
    )
    metrics = {
        "run_id": effective_config.get("run_id", ""),
        "method": effective_config.get("campaign_method_label", method),
        "seed": effective_config.get("random_seed", 0),
        "scenario_seed": effective_config.get("scenario_seed", 0),
        "optimizer_seed": effective_config.get("optimizer_seed", 0),
        "train_mse_final": final_train_mse,
        "val_mse_final": final_val_mse,
        "test_mse_final": final_test_mse,
        "final_train_mse": final_train_mse,
        "final_validation_mse": primary_final_val_mse,
        "final_test_mse": primary_final_test_mse,
        "best_validation_round": int(best_validation_round),
        "best_validation_mse": float(best_validation_mse),
        "test_mse_at_best_validation": primary_test_mse_at_best_validation,
        "selection_metric_source": "validation",
        "selection_metric": effective_config.get(
            "primary_selection_metric", "equal_client_validation_mse"
        ),
        "selection_source": effective_config.get("selection_source", "validation_only"),
        "selected_round": int(best_validation_round),
        "test_mse_reported_after_selection": True,
        "is_primary": False,
        "alignment_label": effective_config.get("alignment_label", ""),
        "test_mse_used_for_selection": False,
        "test_mse_logged_by_round": bool(args.log_test_mse_by_round),
        "diverged": bool(diverged),
        "finite_history": bool(finite_history),
        "completed_iterations": len(mse_rows),
        "runtime_seconds": time.time() - start_time,
        "objective_mode": objective_mode,
        "aggregation_weighting": "none",
        "scenario_checksum": effective_config.get("scenario_checksum", ""),
        "config_checksum": config_checksum(effective_config),
        "per_client_metrics_artifact": per_client_metrics_artifact,
        "curve_artifact": "mse_by_round.csv",
        "best_checkpoint_artifact": "checkpoints/best_validation.pt",
        "final_checkpoint_artifact": "checkpoints/final.pt",
        "effective_config_artifact": "effective_config.json",
        "equal_client_best_validation_structural_mse": equal_client_best_validation_structural_mse,
        "sample_weighted_validation_mse_at_best_validation": sample_weighted_validation_mse_at_best_validation,
        "equal_client_final_validation_structural_mse": equal_client_final_validation_structural_mse,
        "sample_weighted_final_validation_structural_mse": sample_weighted_final_validation_structural_mse,
        "equal_client_test_mse_at_best_validation": equal_client_test_mse_at_best_validation,
        "sample_weighted_test_mse_at_best_validation": sample_weighted_test_mse_at_best_validation,
        "equal_client_final_test_structural_mse": equal_client_final_test_structural_mse,
        "sample_weighted_final_test_structural_mse": sample_weighted_final_test_structural_mse,
        "equal_client_test_moment_violation_at_best_validation": equal_client_test_moment_violation_at_best_validation,
        "sample_weighted_test_moment_violation_at_best_validation": sample_weighted_test_moment_violation_at_best_validation,
        "equal_client_absolute_ate_error_at_best_validation": None,
        "sample_weighted_absolute_ate_error_at_best_validation": None,
        "equal_client_individual_effect_mae_at_best_validation": None,
        "sample_weighted_individual_effect_mae_at_best_validation": None,
        "final_vs_best_validation_gap": final_vs_best_validation_gap,
        "final_vs_best_test_gap": final_vs_best_test_gap,
        "last_50_validation_mse_std": last_50_validation_mse_std,
        "last_50_validation_mse_range": last_50_validation_mse_range,
    }
    write_metrics(str(output_dir), metrics)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help="One of abs, step, linear, sin, or eicu_semisynth (with --scenario-name)",
    )
    parser.add_argument(
        "--scenario-name", default=None,
        help="Required for --dataset eicu_semisynth, e.g. linear_seed0",
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None, help="0 means full-batch. Defaults: gda=0, sgda/oadam=256.")
    parser.add_argument("--g-lr", type=float, default=0.001)
    parser.add_argument("--f-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--hidden-widths",
        default="64,64",
        help="Comma-separated eICU hidden widths (Study A v2 uses 32,32).",
    )
    parser.add_argument(
        "--model-activation",
        choices=("relu", "leaky_relu", "tanh"),
        default="leaky_relu",
    )
    parser.add_argument("--data-dir", default=str(EXAMPLE_ROOT / "data"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-test-mse-by-round", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--objective-mode", choices=OBJECTIVE_MODE_CHOICES, default="legacy",
        help="paper_aligned uses the frozen-theta~/lambda=1/4 objective; required for eicu* datasets.",
    )
    parser.add_argument(
        "--scenario-seed", type=int, default=None,
        help="protocol_v1.md scenario_seed for provenance; defaults to --seed if omitted.",
    )
    parser.add_argument("--seed-pair-id", default=None)
    parser.add_argument("--campaign-role", default=None)
    parser.add_argument("--scenario-checksum", default=None)
    parser.add_argument("--protocol-version", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--g0", default=None)
    parser.add_argument("--alignment-label", default=None)
    parser.add_argument(
        "--primary-selection-metric", default="equal_client_validation_mse"
    )
    parser.add_argument("--selection-source", default="validation_only")
    parser.add_argument("--scenario-scope", default=None)
    parser.add_argument("--study-claim", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = run(args)
    print(f"centralized run complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
