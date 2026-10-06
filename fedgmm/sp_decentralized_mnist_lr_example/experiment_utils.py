import copy
import csv
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import nullcontext


ZOO_DATASETS = {"linear", "abs", "sin", "step", "zoo"}
# "uniform_clients" matches the paper's federated objective U = (1/N) sum_i U^i:
# every client participating in a round gets weight 1/K_t. "sample_size" is the
# legacy FedAvg-style weighting (weight_i = n_i / sum_j n_j) used by every result
# in the repo before this option existed; it is kept, not removed, so those
# results stay reproducible. Absence of the key defaults to "sample_size" for the
# same reason.
AGGREGATION_WEIGHTING_CHOICES = ("uniform_clients", "sample_size")
DEFAULT_AGGREGATION_WEIGHTING = "sample_size"
# "paper_aligned" uses a frozen previous-global-iterate theta~ and a fixed
# lambda=1/4 in the critic regularizer (see PaperAlignedMomentObjective).
# "legacy" is today's OptimalMomentObjective, completely unmodified: reuses
# the live theta for the regularizer and a tunable objective_lambda_1
# (default 0.1). Absence of the key defaults to "legacy" so every existing
# result stays reproducible.
OBJECTIVE_MODE_CHOICES = ("legacy", "paper_aligned")
DEFAULT_OBJECTIVE_MODE = "legacy"
SERVER_BUFFER_POLICY_CHOICES = ("direct_client_aggregate",)
DEFAULT_SERVER_BUFFER_POLICY = "direct_client_aggregate"
CLIENT_OPTIMIZER_CHOICES = (
    "sgd", "ogda", "fed_eg", "fed_eg_double", "fed_zo_eg",
)
PAPER_ALIGNED_LAMBDA = 0.25
PROFILE_ENV_VAR = "FEDGMM_PROFILE_RUNTIME"
PROFILE_ROOT_ENV_VAR = "FEDGMM_PROFILE_ROOT"
PROFILE_TELEMETRY_ENV_VAR = "FEDGMM_PROFILE_GPU_TELEMETRY"
PROFILE_BATCHES_ENV_VAR = "FEDGMM_PROFILE_BATCHES"
PROFILE_INTERVAL_ENV_VAR = "FEDGMM_PROFILE_GPU_INTERVAL_SECONDS"
PROFILE_SKIP_AUX_REG_ENV_VAR = "FEDGMM_PROFILE_SKIP_AUX_REG"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PROFILE_ROOT = os.path.join(
    REPO_ROOT,
    "results",
    "_profiling",
    "highdim_stochastic_gpu_util",
)
PREDICTION_OUTPUT_KEYS = ("best_validation_prediction", "final_prediction")
RUN_ARTIFACT_NAMES = {
    "effective_config.json",
    "mse_by_round.csv",
    "metrics.json",
    "predictions.npz",
    "test_mse_by_round.csv",
}
EFFECTIVE_CONFIG_FIELDS = (
    "dataset",
    "algorithm",
    "variant",
    "mode",
    "model",
    "federated_optimizer",
    "client_optimizer",
    "learning_rate",
    "g_learning_rate",
    "critic_multiplier",
    "f_learning_rate",
    "server_learning_rate",
    "server_buffer_policy",
    "eg_predictor_server_lr",
    "eg_corrector_server_lr",
    "eg_predictor_critic_server_lr",
    "eg_corrector_critic_server_lr",
    "resolved_predictor_g_server_lr",
    "resolved_predictor_f_server_lr",
    "resolved_corrector_g_server_lr",
    "resolved_corrector_f_server_lr",
    "zo_mu",
    "zo_num_directions",
    "stop_on_numerical_failure",
    "validation_deterioration_window",
    "validation_deterioration_min_rounds",
    "validation_deterioration_patience",
    "validation_deterioration_factor",
    "validation_deterioration_absolute_delta",
    "objective_lambda_1",
    "gradient_clip_norm",
    "weight_decay",
    "batch_size",
    "require_multibatch_stochastic",
    "max_local_steps_per_round",
    "epochs",
    "local_epochs",
    "comm_round",
    "client_num_in_total",
    "client_num_per_round",
    "partition_method",
    "partition_alpha",
    "data_cache_dir",
    "random_seed",
    "run_id",
    "output_dir",
    "using_gpu",
    "gpu_id",
    "frequency_of_the_test",
    "eval_freq",
    "print_freq",
    "simple_model_selection_epochs",
    "f_history_model_selection_epochs",
    "model_selection_batch_size",
    "model_selection_max_samples",
    "skip_model_selection",
    "skip_gmm_eval",
    "gmm_eval_proxy",
    "auxiliary_regression",
    "auxiliary_regression_epochs",
    "auxiliary_regression_state_device",
    "append_round_csv",
    "periodic_checkpoint_interval",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "log_test_mse_by_round",
    "test_mse_logged_by_round",
    "test_mse_used_for_selection",
    "selection_metric_source",
    "aggregation_weighting",
    "objective_mode",
    "input_dim_g",
    "input_dim_f",
    "hidden_widths",
    "model_activation",
    "enable_legacy_outputs",
    "enable_legacy_plot",
    "overwrite",
    "scenario_seed",
    "optimizer_seed",
    "init_state_override",
    "init_g_state_override",
    "init_f_state_override",
    "checkpoint_rounds",
    "local_bn_mode",
    "calibration_panel_size",
    "seed_pair_id",
    "campaign_role",
    "scenario_checksum",
    "protocol_version",
    "role",
    "scenario_name",
    "g0",
    "alignment_label",
    "primary_selection_metric",
    "selection_source",
    "scenario_scope",
    "study_claim",
    "compact_predictions_only",
)


def compute_client_weights(sample_counts, mode):
    """Per-participating-client aggregation weights, summing to 1.0.

    ``uniform_clients``: every client gets ``1/K`` regardless of its sample
    count, where ``K = len(sample_counts)`` is however many clients actually
    participated this round (so this is correct under partial participation
    too, not just full participation).

    ``sample_size``: ``weight_i = n_i / sum_j n_j`` — the legacy FedAvg
    weighting. Mathematically identical to the pre-existing inline computation
    it replaces, so results in this mode are bit-for-bit reproducible.
    """
    counts = [int(c) for c in sample_counts]
    if not counts:
        raise ValueError("compute_client_weights requires at least one client")
    if mode == "uniform_clients":
        k = len(counts)
        return [1.0 / k] * k
    if mode == "sample_size":
        total = sum(counts)
        if total <= 0:
            raise ValueError(
                "sample_size aggregation requires a positive total sample count, "
                f"got counts={counts}"
            )
        return [c / total for c in counts]
    raise ValueError(
        f"aggregation_weighting must be one of {AGGREGATION_WEIGHTING_CHOICES}, "
        f"got {mode!r}"
    )


def weighted_average_state_dicts(state_dicts, weights):
    """Aggregate same-keyed state mappings without averaging integer counters.

    This is the single function both FedGDA's direct aggregation and FedOGDA's
    pseudogradient/delta aggregation rely on (the delta is computed as
    ``weighted_average(...) - theta_t``), so fixing the weighting here fixes
    both optimizers identically rather than needing two separate patches.
    Floating and complex values use the requested weights. Nonfloating values
    use an elementwise maximum so counters retain their dtype and semantics.
    """
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"state_dicts and weights must have the same length, got "
            f"{len(state_dicts)} and {len(weights)}"
        )
    if not state_dicts:
        raise ValueError("weighted_average_state_dicts requires at least one entry")

    try:
        import torch
    except Exception:
        torch = None

    result = {}
    for key in state_dicts[0].keys():
        first = state_dicts[0][key]
        if torch is not None and torch.is_tensor(first) and not (
            torch.is_floating_point(first) or torch.is_complex(first)
        ):
            values = torch.stack([state_dict[key] for state_dict in state_dicts])
            if first.dtype == torch.bool:
                result[key] = values.any(dim=0)
            else:
                result[key] = values.amax(dim=0)
            continue
        if isinstance(first, (bool, int)):
            result[key] = max(state_dict[key] for state_dict in state_dicts)
            continue
        acc = None
        for state_dict, weight in zip(state_dicts, weights):
            term = state_dict[key] * weight
            acc = term if acc is None else acc + term
        result[key] = acc
    return result


def optional_path(value):
    """Normalize an optional path config value to a real path or None.

    The config writer serializes Python None as the STRING "None", so a key
    that is genuinely unset arrives as "None" rather than null. Treating that
    as a path made every run with an unset override fail at startup. Hand-
    written configs may also use "null" or an empty string.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("none", "null"):
        return None
    return text


def resolve_server_learning_rates(source):
    """Resolve the four server coefficients FedEG actually applies.

    Until now one coefficient per phase was shared by both players, so the
    critic's server step could not be varied independently of the structural
    model's -- which the critic-rate screen requires.

    The fallback chain makes backward compatibility a property of the resolver
    rather than something to hope for: with both critic keys absent,
    predictor_f == predictor_g and corrector_f == corrector_g, which is exactly
    the previous behaviour.

        predictor_g = eg_predictor_server_lr        or server_learning_rate
        predictor_f = eg_predictor_critic_server_lr or predictor_g
        corrector_g = eg_corrector_server_lr        or server_learning_rate
        corrector_f = eg_corrector_critic_server_lr or corrector_g

    Accepts either an args object or an effective-config dict so the launcher
    and the coordinator cannot drift apart.
    """
    def read(name):
        if isinstance(source, dict):
            return source.get(name)
        return getattr(source, name, None)

    base = read("server_learning_rate")
    base = 1.5 if base is None else float(base)
    predictor_g = read("eg_predictor_server_lr")
    predictor_g = base if predictor_g is None else float(predictor_g)
    corrector_g = read("eg_corrector_server_lr")
    corrector_g = base if corrector_g is None else float(corrector_g)
    predictor_f = read("eg_predictor_critic_server_lr")
    predictor_f = predictor_g if predictor_f is None else float(predictor_f)
    corrector_f = read("eg_corrector_critic_server_lr")
    corrector_f = corrector_g if corrector_f is None else float(corrector_f)
    return {"predictor_g": predictor_g, "predictor_f": predictor_f,
            "corrector_g": corrector_g, "corrector_f": corrector_f}


def apply_parameter_server_update(
    base_state,
    aggregated_state,
    parameter_keys,
    learning_rate,
    *,
    previous_parameter_delta=None,
    optimistic=False,
    delta_base_state=None,
    buffer_policy=DEFAULT_SERVER_BUFFER_POLICY,
):
    """Apply server optimizer math to parameters and direct aggregation to buffers.

    PyTorch ``state_dict`` mappings contain both trainable parameters and model
    buffers. BatchNorm running statistics and integer counters are state, but
    they are not optimizer variables and must never receive server learning-rate
    interpolation or optimistic extrapolation.
    """
    if buffer_policy not in SERVER_BUFFER_POLICY_CHOICES:
        raise ValueError(
            f"server buffer policy must be one of {SERVER_BUFFER_POLICY_CHOICES}, "
            f"got {buffer_policy!r}"
        )
    base_keys = set(base_state)
    aggregate_keys = set(aggregated_state)
    if base_keys != aggregate_keys:
        raise ValueError("base and aggregated state dictionaries must have identical keys")
    parameter_keys = set(parameter_keys)
    unknown_parameter_keys = parameter_keys - base_keys
    if unknown_parameter_keys:
        raise ValueError(
            f"parameter keys are absent from the state dictionary: {sorted(unknown_parameter_keys)}"
        )
    if delta_base_state is None:
        delta_base_state = base_state
    if set(delta_base_state) != base_keys:
        raise ValueError("delta-base and base state dictionaries must have identical keys")
    if optimistic and previous_parameter_delta is None:
        raise ValueError("optimistic update requires previous_parameter_delta")
    if previous_parameter_delta is not None and set(previous_parameter_delta) != parameter_keys:
        raise ValueError("previous_parameter_delta must contain exactly the trainable parameter keys")

    updated = {}
    parameter_delta = {}
    learning_rate = float(learning_rate)
    for key in base_state:
        if key not in parameter_keys:
            value = aggregated_state[key]
            updated[key] = value.clone() if hasattr(value, "clone") else copy.deepcopy(value)
            continue
        delta = aggregated_state[key] - delta_base_state[key]
        parameter_delta[key] = delta
        direction = delta
        if optimistic:
            direction = 2.0 * delta - previous_parameter_delta[key]
        updated[key] = base_state[key] + learning_rate * direction
    return updated, parameter_delta


def batchnorm_running_var_min(model, model_name="model"):
    """Return the minimum BatchNorm running variance and enforce its domain."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("BatchNorm validation requires torch") from exc

    minimum = None
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        running_var = module.running_var
        if running_var is None:
            continue
        qualified_name = f"{model_name}.{module_name}" if module_name else model_name
        if not bool(torch.isfinite(running_var).all().item()):
            message = f"{qualified_name}.running_var contains non-finite values"
            logging.error(message)
            raise FloatingPointError(message)
        current_minimum = float(running_var.min().detach().cpu())
        if current_minimum < 0.0:
            message = (
                f"{qualified_name}.running_var is negative: minimum={current_minimum}"
            )
            logging.error(message)
            raise FloatingPointError(message)
        minimum = current_minimum if minimum is None else min(minimum, current_minimum)
    return minimum


def check_eicu_aggregation_weighting(dataset, aggregation_weighting, campaign_role):
    """Raise unless an eICU run uses uniform_clients, with one named exception.

    protocol_v1.md S6.3/S6.4: primary eICU runs (confirmatory, tuning,
    centralized) must use uniform_clients so no hospital's sample count can
    dominate the global model. The required aggregation-weighting ablation
    role is the sole, explicit, auditable exception -- it exists precisely
    to measure sample_size against that primary policy, so it must opt in by
    name (``campaign_role == "aggregation_ablation"``) rather than merely by
    omitting aggregation_weighting.
    """
    is_eicu = str(dataset).startswith("eicu")
    ablation_authorized = (
        is_eicu
        and campaign_role == "aggregation_ablation"
        and aggregation_weighting == "sample_size"
    )
    if is_eicu and aggregation_weighting != "uniform_clients" and not ablation_authorized:
        raise ValueError(
            "eICU runs must set aggregation_weighting: uniform_clients to match "
            "the paper's federated objective U(theta,tau) = (1/N) sum_i U^i(theta,tau); "
            f"got aggregation_weighting={aggregation_weighting!r}. The legacy "
            "sample_size weighting lets large hospitals dominate the global model, "
            "which is exactly the failure mode this check exists to prevent. The only "
            "exception is campaign_role: aggregation_ablation, Study A's explicit, "
            "non-primary aggregation-weighting ablation role."
        )


def check_eicu_objective_mode(dataset, objective_mode):
    """Raise unless an eICU run uses the paper-aligned objective.

    protocol_v1.md S6.3: Study A exists to test the paper's actual FedDeepGMM
    formulation (frozen theta~, lambda=1/4), not the legacy objective. Applies
    identically to federated (FedAvgAPI) and centralized
    (run_centralized_lowdim.py) eICU runs, since both must be comparable under
    the same objective.
    """
    if str(dataset).startswith("eicu") and str(objective_mode) != "paper_aligned":
        raise ValueError(
            "eICU runs must set objective_mode: paper_aligned -- Study A exists to "
            "test the paper's actual FedDeepGMM formulation (frozen theta~, "
            f"lambda=1/4), not the legacy objective; got objective_mode={objective_mode!r}."
        )


def check_max_local_steps_participation(
    execution_mode, enable_multiprocessing, max_local_steps_per_round,
    client_num_per_round, client_num_in_total,
):
    """Raise unless max_local_steps_per_round's two load-bearing assumptions hold.

    train_gmm's local-batch rotation (fedml/ml/trainer/my_model_trainer_classification.py)
    tracks its offset as state on each client's persistent, per-client
    trainer instance -- one Client (and one deep-copied model_trainer) per
    client for the entire run, built once in FedAvgAPI._setup_clients. Two
    things this campaign always satisfies, but a future config change
    silently could not:

    1. SP execution. The multiprocess executor path re-materializes each
       task's batches fresh every round (_materialize_client_data) against
       a trainer object that is not the same long-lived per-client
       instance, so the rotation offset would not persist across rounds.
    2. Full participation. FedAvgAPI._run_primary_client_updates pairs
       ``zip(client_indexes, self.client_list)`` by *position*, and
       client_indices_for_round only returns the same
       ``range(client_num_in_total)`` order every round when
       client_num_per_round == client_num_in_total; a partial or reordered
       sample would silently apply one client's rotation offset to a
       different client's data.

    Both would silently reintroduce the "same first N batches forever" bug
    the rotation exists to fix, so this fails loudly instead.
    """
    if int(max_local_steps_per_round or 0) <= 0:
        return
    resolved_mode = str(execution_mode or "").strip().lower()
    if not resolved_mode:
        resolved_mode = "multi_gpu_processes" if bool(enable_multiprocessing) else "sp"
    if resolved_mode != "sp":
        raise ValueError(
            "max_local_steps_per_round > 0 requires client_execution_mode='sp' "
            "(its rotation state lives on the SP path's persistent per-client "
            "trainer instance; the multiprocess executor path does not carry "
            "that state across rounds)"
        )
    if int(client_num_per_round) != int(client_num_in_total):
        raise ValueError(
            "max_local_steps_per_round > 0 requires full participation "
            "(client_num_per_round == client_num_in_total); its rotation "
            "state is keyed by client_list position, which only maps onto a "
            "stable real client_idx every round under full participation"
        )


def default_critic_multiplier(dataset):
    multiplier = 20.0
    if dataset in ZOO_DATASETS:
        multiplier = 10.0
    if dataset in {
        "mnist_z",
        "femnist_z",
        "mnist_xz",
        "femnist_xz",
        "cifar10_z",
        "cifar10_xz",
        "cifar_xz",
    }:
        multiplier = 5.0
    elif dataset in {"mnist_x", "femnist_x", "cifar10_x", "cifar_x"}:
        multiplier = 3.0
    return multiplier


def get_effective_config(args):
    dataset = getattr(args, "dataset", "unknown")
    client_optimizer = getattr(args, "client_optimizer", "sgd")
    optimizer_name = str(client_optimizer).lower()
    if optimizer_name not in CLIENT_OPTIMIZER_CHOICES:
        raise ValueError(
            f"client_optimizer must be one of {CLIENT_OPTIMIZER_CHOICES}, "
            f"got {client_optimizer!r}"
        )
    algorithm = {
        "sgd": "fedgda",
        "ogda": "fedogda",
        "fed_eg": "fed_eg",
        "fed_eg_double": "fed_eg_double",
        "fed_zo_eg": "fed_zo_eg",
    }[optimizer_name]
    batch_size = getattr(args, "batch_size", 0)
    mode = "d" if batch_size is None or int(batch_size) <= 0 else "s"
    seed = int(getattr(args, "random_seed", 0))
    run_id = str(getattr(args, "run_id", "0"))
    output_dir = getattr(args, "output_dir", "results")
    variant = f"{algorithm}_{mode}"
    critic_multiplier = float(
        getattr(args, "critic_multiplier", default_critic_multiplier(dataset))
    )
    server_learning_rate = float(getattr(args, "server_learning_rate", 1.5))
    server_buffer_policy = str(
        getattr(args, "server_buffer_policy", DEFAULT_SERVER_BUFFER_POLICY)
    )
    if server_buffer_policy not in SERVER_BUFFER_POLICY_CHOICES:
        raise ValueError(
            f"server_buffer_policy must be one of {SERVER_BUFFER_POLICY_CHOICES}, "
            f"got {server_buffer_policy!r}"
        )
    eg_predictor_server_lr = getattr(args, "eg_predictor_server_lr", None)
    eg_corrector_server_lr = getattr(args, "eg_corrector_server_lr", None)
    if eg_predictor_server_lr is not None:
        eg_predictor_server_lr = float(eg_predictor_server_lr)
    if eg_corrector_server_lr is not None:
        eg_corrector_server_lr = float(eg_corrector_server_lr)
    eg_predictor_critic_server_lr = getattr(args, "eg_predictor_critic_server_lr", None)
    eg_corrector_critic_server_lr = getattr(args, "eg_corrector_critic_server_lr", None)
    _critic_rates = {}
    for _name, _value in (("eg_predictor_critic_server_lr", eg_predictor_critic_server_lr),
                          ("eg_corrector_critic_server_lr", eg_corrector_critic_server_lr)):
        if _value is None:
            _critic_rates[_name] = None
            continue
        _value = float(_value)
        # Zero is allowed and meaningful: the server discards the aggregated
        # critic parameter delta for that phase. Negative and non-finite are not.
        if not math.isfinite(_value) or _value < 0.0:
            raise ValueError(f"{_name} must be finite and non-negative, got {_value!r}")
        _critic_rates[_name] = _value
    eg_predictor_critic_server_lr = _critic_rates["eg_predictor_critic_server_lr"]
    eg_corrector_critic_server_lr = _critic_rates["eg_corrector_critic_server_lr"]
    zo_mu = float(getattr(args, "zo_mu", 1e-3))
    zo_num_directions = int(getattr(args, "zo_num_directions", 1))
    if optimizer_name == "fed_zo_eg":
        if zo_mu <= 0.0:
            raise ValueError("zo_mu must be positive")
        if zo_num_directions < 1:
            raise ValueError("zo_num_directions must be at least 1")
    objective_lambda_1 = float(getattr(args, "objective_lambda_1", 0.1))
    aggregation_weighting = str(
        getattr(args, "aggregation_weighting", DEFAULT_AGGREGATION_WEIGHTING)
    )
    objective_mode = str(getattr(args, "objective_mode", DEFAULT_OBJECTIVE_MODE))

    # Protocol v1 requires scenario_seed (controls DGP/split) and
    # optimizer_seed (controls init/minibatch/optimizer randomness) as
    # separate values bound by a seed_pair_id, never one conflated scalar.
    # Absent both, default each to random_seed so pre-existing configs that
    # only ever set random_seed keep their exact prior (conflated) behavior.
    scenario_seed = int(getattr(args, "scenario_seed", seed))
    optimizer_seed = int(getattr(args, "optimizer_seed", seed))
    _overrides = {}
    for _key in ("init_state_override", "init_g_state_override", "init_f_state_override"):
        _value = optional_path(getattr(args, _key, None))
        if _value is not None and not os.path.isfile(_value):
            raise ValueError(f"{_key} does not exist: {_value}")
        _overrides[_key] = _value
    init_state_override = _overrides["init_state_override"]
    init_g_state_override = _overrides["init_g_state_override"]
    init_f_state_override = _overrides["init_f_state_override"]
    local_bn_mode = str(getattr(args, "local_bn_mode", None) or "batch")
    if local_bn_mode not in ("batch", "frozen", "calibrated"):
        raise ValueError(
            "local_bn_mode must be 'batch', 'frozen' or 'calibrated', "
            f"got {local_bn_mode!r}")
    calibration_panel_size = int(getattr(args, "calibration_panel_size", 512) or 512)
    if local_bn_mode == "calibrated" and calibration_panel_size < 2:
        raise ValueError("calibration_panel_size must be at least 2")
    checkpoint_rounds = optional_path(getattr(args, "checkpoint_rounds", None))
    if checkpoint_rounds is not None:
        checkpoint_rounds = " ".join(
            str(int(part)) for part in
            str(checkpoint_rounds).replace(",", " ").split() if part != "")
    seed_pair_id = str(getattr(args, "seed_pair_id", ""))

    learning_rate = float(getattr(args, "learning_rate", 0.0))
    log_test_mse_by_round = bool(getattr(args, "log_test_mse_by_round", False))
    test_mse_used_for_selection = bool(getattr(args, "test_mse_used_for_selection", False))
    selection_metric_source = str(getattr(args, "selection_metric_source", "validation"))
    auxiliary_regression = bool(getattr(args, "auxiliary_regression", False))
    auxiliary_regression_epochs = int(
        getattr(args, "auxiliary_regression_epochs", int(getattr(args, "epochs", 0)) if auxiliary_regression else 0)
    )
    skip_model_selection = bool(getattr(args, "skip_model_selection", False))
    skip_gmm_eval = bool(getattr(args, "skip_gmm_eval", skip_model_selection))
    if skip_model_selection:
        skip_gmm_eval = True
    gmm_eval_proxy = "negative_val_mse" if skip_gmm_eval else "approx_psi"
    if not skip_model_selection:
        gmm_eval_proxy = str(getattr(args, "gmm_eval_proxy", gmm_eval_proxy))
    return json_safe({
        "dataset": dataset,
        "algorithm": algorithm,
        "variant": variant,
        "mode": "deterministic" if mode == "d" else "stochastic",
        "model": getattr(args, "model", ""),
        "federated_optimizer": getattr(args, "federated_optimizer", ""),
        "client_optimizer": client_optimizer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "require_multibatch_stochastic": bool(
            getattr(args, "require_multibatch_stochastic", False)
        ),
        # 0 (or unset) = unlimited = current behavior, unchanged. When > 0,
        # train_gmm's local loop stops after this many optimizer steps for
        # the round (summed across all local_epochs, not per epoch) -- see
        # fedavg_api.py's per-round client loop and
        # my_model_trainer_classification.py's train_gmm.
        "max_local_steps_per_round": int(
            getattr(args, "max_local_steps_per_round", 0)
        ),
        "epochs": int(getattr(args, "epochs", 0)),
        "local_epochs": int(getattr(args, "epochs", 0)),
        "comm_round": int(getattr(args, "comm_round", 0)),
        "client_num_in_total": int(getattr(args, "client_num_in_total", 0)),
        "client_num_per_round": int(getattr(args, "client_num_per_round", 0)),
        "frequency_of_the_test": int(getattr(args, "frequency_of_the_test", 1)),
        "eval_freq": int(getattr(args, "eval_freq", 1)),
        "print_freq": int(getattr(args, "print_freq", 1)),
        "simple_model_selection_epochs": int(getattr(args, "simple_model_selection_epochs", 100)),
        "f_history_model_selection_epochs": int(getattr(args, "f_history_model_selection_epochs", 60)),
        "model_selection_batch_size": int(getattr(args, "model_selection_batch_size", 200)),
        "model_selection_max_samples": int(getattr(args, "model_selection_max_samples", 0)),
        "skip_model_selection": skip_model_selection,
        "skip_gmm_eval": skip_gmm_eval,
        "gmm_eval_proxy": gmm_eval_proxy,
        "auxiliary_regression": auxiliary_regression,
        "auxiliary_regression_epochs": auxiliary_regression_epochs,
        "auxiliary_regression_state_device": str(getattr(args, "auxiliary_regression_state_device", "device")),
        "append_round_csv": bool(getattr(args, "append_round_csv", True)),
        "periodic_checkpoint_interval": int(getattr(args, "periodic_checkpoint_interval", 200)),
        "dataloader_num_workers": int(getattr(args, "dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(getattr(args, "dataloader_pin_memory", False)),
        "log_test_mse_by_round": log_test_mse_by_round,
        "test_mse_logged_by_round": log_test_mse_by_round,
        "test_mse_used_for_selection": test_mse_used_for_selection,
        "selection_metric_source": selection_metric_source,
        "g_learning_rate": learning_rate,
        "critic_multiplier": critic_multiplier,
        "f_learning_rate": critic_multiplier * learning_rate,
        "server_learning_rate": server_learning_rate,
        "server_buffer_policy": server_buffer_policy,
        "eg_predictor_server_lr": eg_predictor_server_lr,
        "eg_corrector_server_lr": eg_corrector_server_lr,
        "eg_predictor_critic_server_lr": eg_predictor_critic_server_lr,
        "eg_corrector_critic_server_lr": eg_corrector_critic_server_lr,
        # Derived, read-only. A run that applied server_learning_rate records
        # eg_predictor_server_lr: null, so the artifact never stated the number
        # actually used; a second independent rate would double that ambiguity.
        **{f"resolved_{_phase}_server_lr": _value
           for _phase, _value in resolve_server_learning_rates({
               "server_learning_rate": server_learning_rate,
               "eg_predictor_server_lr": eg_predictor_server_lr,
               "eg_corrector_server_lr": eg_corrector_server_lr,
               "eg_predictor_critic_server_lr": eg_predictor_critic_server_lr,
               "eg_corrector_critic_server_lr": eg_corrector_critic_server_lr,
           }).items()},
        "zo_mu": zo_mu,
        "zo_num_directions": zo_num_directions,
        "stop_on_numerical_failure": bool(getattr(args, "stop_on_numerical_failure", False)),
        "validation_deterioration_window": int(getattr(args, "validation_deterioration_window", 0)),
        "validation_deterioration_min_rounds": int(getattr(args, "validation_deterioration_min_rounds", 100)),
        "validation_deterioration_patience": int(getattr(args, "validation_deterioration_patience", 50)),
        "validation_deterioration_factor": float(getattr(args, "validation_deterioration_factor", 5.0)),
        "validation_deterioration_absolute_delta": float(getattr(args, "validation_deterioration_absolute_delta", 0.1)),
        "objective_lambda_1": objective_lambda_1,
        "aggregation_weighting": aggregation_weighting,
        "objective_mode": objective_mode,
        "input_dim_g": int(getattr(args, "input_dim_g", 0)),
        "input_dim_f": int(getattr(args, "input_dim_f", 0)),
        "hidden_widths": str(getattr(args, "hidden_widths", "64,64")),
        "model_activation": str(
            getattr(args, "model_activation", "leaky_relu")
        ),
        "random_seed": seed,
        "scenario_seed": scenario_seed,
        "optimizer_seed": optimizer_seed,
        "init_state_override": init_state_override,
        "init_g_state_override": init_g_state_override,
        "init_f_state_override": init_f_state_override,
        "checkpoint_rounds": checkpoint_rounds,
        "local_bn_mode": local_bn_mode,
        "calibration_panel_size": calibration_panel_size,
        "seed_pair_id": seed_pair_id,
        "campaign_role": str(getattr(args, "campaign_role", "")),
        "scenario_checksum": str(getattr(args, "scenario_checksum", "")),
        "protocol_version": str(getattr(args, "protocol_version", "")),
        "role": str(getattr(args, "role", "")),
        "scenario_name": str(getattr(args, "scenario_name", "")),
        "g0": str(getattr(args, "g0", "")),
        "alignment_label": str(getattr(args, "alignment_label", "")),
        "primary_selection_metric": str(
            getattr(args, "primary_selection_metric", "")
        ),
        "selection_source": str(getattr(args, "selection_source", "")),
        "scenario_scope": str(getattr(args, "scenario_scope", "")),
        "study_claim": str(getattr(args, "study_claim", "")),
        "run_id": run_id,
        "output_dir": output_dir,
        "partition_method": getattr(args, "partition_method", ""),
        "partition_alpha": float(getattr(args, "partition_alpha", 0.0)),
        "data_cache_dir": getattr(args, "data_cache_dir", ""),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "gradient_clip_norm": float(getattr(args, "gradient_clip_norm", 1.0)),
        "using_gpu": bool(getattr(args, "using_gpu", False)),
        "gpu_id": int(getattr(args, "gpu_id", 0)),
        "enable_legacy_outputs": bool(getattr(args, "enable_legacy_outputs", True)),
        "enable_legacy_plot": bool(getattr(args, "enable_legacy_plot", False)),
        "overwrite": bool(getattr(args, "overwrite", False)),
        # fedavg_api.py reads self.compact_predictions_only from
        # self.effective_config (this dict), never from args directly --
        # omitted here, every run would silently write the full ~10 GiB-
        # scale predictions.npz regardless of what the manifest/config
        # actually requested (found live, 2026-08-28: a real V4 signal run
        # had compact_predictions_only=true in its generated YAML but wrote
        # a full-size prediction file).
        "compact_predictions_only": bool(getattr(args, "compact_predictions_only", False)),
    })


def build_run_dir(output_dir, dataset, variant, seed, run_id):
    return os.path.join(str(output_dir), str(dataset), str(variant), f"seed_{int(seed)}", str(run_id))


def run_dir_from_config(config):
    return build_run_dir(
        config["output_dir"],
        config["dataset"],
        config["variant"],
        config["random_seed"],
        config["run_id"],
    )


def ensure_run_dirs(run_dir):
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)


def env_truthy(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


class _NoOpProfileSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ProfileSpan:
    def __init__(self, profiler, phase, round_idx=None, client_id=None, detail=""):
        self.profiler = profiler
        self.phase = phase
        self.round_idx = round_idx
        self.client_id = client_id
        self.detail = detail
        self.start_wall = None
        self.start_cpu = None

    def __enter__(self):
        self.profiler._synchronize_cuda()
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.profiler._synchronize_cuda()
        detail = self.detail
        if exc_type is not None:
            detail = f"{detail}; exception={exc_type.__name__}" if detail else f"exception={exc_type.__name__}"
        self.profiler.record(
            self.phase,
            time.perf_counter() - self.start_wall,
            process_cpu_seconds=time.process_time() - self.start_cpu,
            round_idx=self.round_idx,
            client_id=self.client_id,
            detail=detail,
        )
        return False


class RuntimeProfiler:
    """Low-overhead opt-in profiler for high-dimensional stochastic diagnostics."""

    RUNTIME_FIELDS = (
        "timestamp_epoch",
        "elapsed_since_profile_start",
        "phase",
        "round",
        "client_id",
        "duration_seconds",
        "process_cpu_seconds",
        "detail",
    )
    TELEMETRY_FIELDS = (
        "timestamp_epoch",
        "elapsed_since_profile_start",
        "gpu_index",
        "gpu_name",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "memory_used_mb",
        "memory_total_mb",
        "power_draw_w",
        "error",
    )
    TOP_LEVEL_PHASES = {
        "data_load",
        "model_create",
        "runner_init",
        "runner_run",
        "model_to_device",
        "global_tensor_to_device",
        "model_selection",
        "setup_clients",
        "training_total",
        "round_total",
        "finalization",
    }

    def __init__(self, enabled, profile_dir=None, effective_config=None):
        self.enabled = bool(enabled)
        self.profile_dir = os.path.abspath(profile_dir) if profile_dir else None
        self.effective_config = effective_config or {}
        self.runtime_path = os.path.join(self.profile_dir, "runtime_profile.csv") if self.profile_dir else None
        self.telemetry_path = os.path.join(self.profile_dir, "gpu_telemetry.csv") if self.profile_dir else None
        self.environment_path = os.path.join(self.profile_dir, "profile_environment.json") if self.profile_dir else None
        self.summary_path = os.path.join(self.profile_dir, "profile_summary.json") if self.profile_dir else None
        self.start_wall = time.time()
        self.start_perf = time.perf_counter()
        self.start_process_cpu = time.process_time()
        self._lock = threading.Lock()
        self._telemetry_stop = threading.Event()
        self._telemetry_thread = None
        self._telemetry_samples = []
        self._phase_totals = defaultdict(float)
        self._phase_cpu_totals = defaultdict(float)
        self._phase_counts = defaultdict(int)
        self._recorded_once = set()
        self.profile_batches = env_truthy(PROFILE_BATCHES_ENV_VAR, default=False)
        if self.enabled and self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)
            self._write_csv_header(self.runtime_path, self.RUNTIME_FIELDS)
            self._write_csv_header(self.telemetry_path, self.TELEMETRY_FIELDS)

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def disabled(cls):
        return cls(False)

    @classmethod
    def from_config(cls, effective_config):
        if not env_truthy(PROFILE_ENV_VAR, default=False):
            return cls.disabled()
        profile_root = os.path.abspath(os.environ.get(PROFILE_ROOT_ENV_VAR, DEFAULT_PROFILE_ROOT))
        profile_dir = build_run_dir(
            profile_root,
            effective_config.get("dataset", "unknown"),
            effective_config.get("variant", "unknown"),
            int(effective_config.get("random_seed", 0)),
            effective_config.get("run_id", "unknown"),
        )
        return cls(True, profile_dir=profile_dir, effective_config=dict(effective_config))

    def configure(self, effective_config, actual_run_dir=None):
        if not self.enabled:
            return self
        self.effective_config = dict(effective_config)
        self.actual_run_dir = os.path.abspath(str(actual_run_dir)) if actual_run_dir else None
        return self

    def _write_csv_header(self, path, fields):
        if not path or os.path.exists(path):
            return
        with open(path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    def _elapsed(self):
        return time.perf_counter() - self.start_perf

    def _synchronize_cuda(self):
        if not self.enabled:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    def span(self, phase, round_idx=None, client_id=None, detail=""):
        if not self.enabled:
            return _NoOpProfileSpan()
        return _ProfileSpan(self, phase, round_idx=round_idx, client_id=client_id, detail=detail)

    def record(self, phase, duration_seconds=0.0, process_cpu_seconds=0.0, round_idx=None, client_id=None, detail=""):
        if not self.enabled:
            return
        row = {
            "timestamp_epoch": time.time(),
            "elapsed_since_profile_start": self._elapsed(),
            "phase": phase,
            "round": "" if round_idx is None else int(round_idx),
            "client_id": "" if client_id is None else int(client_id),
            "duration_seconds": float(duration_seconds),
            "process_cpu_seconds": float(process_cpu_seconds),
            "detail": detail,
        }
        with self._lock:
            self._write_csv_header(self.runtime_path, self.RUNTIME_FIELDS)
            with open(self.runtime_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.RUNTIME_FIELDS)
                writer.writerow(row)
            self._phase_totals[phase] += float(duration_seconds)
            self._phase_cpu_totals[phase] += float(process_cpu_seconds)
            self._phase_counts[phase] += 1

    def record_once(self, key, phase, detail=""):
        if not self.enabled:
            return
        with self._lock:
            if key in self._recorded_once:
                return
            self._recorded_once.add(key)
        self.record(phase, duration_seconds=0.0, detail=detail)

    def record_environment(self, effective_config=None, actual_run_dir=None, device=None):
        if not self.enabled:
            return
        if effective_config is not None:
            self.effective_config = dict(effective_config)
        if actual_run_dir is not None:
            self.actual_run_dir = os.path.abspath(str(actual_run_dir))
        torch_info = {}
        try:
            import torch
            torch_info = {
                "torch_version": getattr(torch, "__version__", "unknown"),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else int(torch.cuda.device_count()),
            }
            if torch.cuda.is_available():
                current = torch.cuda.current_device()
                torch_info.update({
                    "cuda_current_device": int(current),
                    "cuda_current_device_name": torch.cuda.get_device_name(current),
                    "cuda_device_names": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ],
                })
        except Exception as exc:
            torch_info = {"torch_probe_error": repr(exc)}
        payload = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "profile_dir": self.profile_dir,
            "actual_run_dir": getattr(self, "actual_run_dir", None),
            "device": str(device) if device is not None else None,
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                "FEDGMM_PROFILE_RUNTIME": os.environ.get(PROFILE_ENV_VAR),
                "FEDGMM_PROFILE_ROOT": os.environ.get(PROFILE_ROOT_ENV_VAR),
                "FEDGMM_PROFILE_BATCHES": os.environ.get(PROFILE_BATCHES_ENV_VAR),
                "FEDGMM_PROFILE_SKIP_AUX_REG": os.environ.get(PROFILE_SKIP_AUX_REG_ENV_VAR),
            },
            "effective_config": json_safe(self.effective_config),
            "torch": torch_info,
        }
        with open(self.environment_path, "w") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def start_telemetry(self):
        if not self.enabled or not env_truthy(PROFILE_TELEMETRY_ENV_VAR, default=True):
            return
        if self._telemetry_thread is not None:
            return
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, name="fedgmm-gpu-telemetry", daemon=True)
        self._telemetry_thread.start()

    def _telemetry_loop(self):
        interval = _parse_float(os.environ.get(PROFILE_INTERVAL_ENV_VAR)) or 1.0
        interval = max(0.2, float(interval))
        while not self._telemetry_stop.is_set():
            self._sample_gpu_telemetry()
            self._telemetry_stop.wait(interval)

    def _sample_gpu_telemetry(self):
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=3, check=False)
            if completed.returncode != 0:
                self._write_telemetry_error(completed.stderr.strip() or completed.stdout.strip() or f"returncode={completed.returncode}")
                return
            for line in completed.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 7:
                    self._write_telemetry_error(f"unparseable nvidia-smi row: {line}")
                    continue
                row = {
                    "timestamp_epoch": time.time(),
                    "elapsed_since_profile_start": self._elapsed(),
                    "gpu_index": parts[0],
                    "gpu_name": parts[1],
                    "utilization_gpu_pct": _parse_float(parts[2]),
                    "utilization_memory_pct": _parse_float(parts[3]),
                    "memory_used_mb": _parse_float(parts[4]),
                    "memory_total_mb": _parse_float(parts[5]),
                    "power_draw_w": _parse_float(parts[6]),
                    "error": "",
                }
                self._append_telemetry(row)
        except Exception as exc:
            self._write_telemetry_error(repr(exc))

    def _write_telemetry_error(self, error):
        row = {
            "timestamp_epoch": time.time(),
            "elapsed_since_profile_start": self._elapsed(),
            "gpu_index": "",
            "gpu_name": "",
            "utilization_gpu_pct": "",
            "utilization_memory_pct": "",
            "memory_used_mb": "",
            "memory_total_mb": "",
            "power_draw_w": "",
            "error": error,
        }
        self._append_telemetry(row)

    def _append_telemetry(self, row):
        with self._lock:
            self._write_csv_header(self.telemetry_path, self.TELEMETRY_FIELDS)
            with open(self.telemetry_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.TELEMETRY_FIELDS)
                writer.writerow(row)
            self._telemetry_samples.append(row)

    def stop(self, extra=None):
        if not self.enabled:
            return
        self._telemetry_stop.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=5)
            self._telemetry_thread = None
        self.write_summary(extra=extra or {})

    def write_summary(self, extra=None):
        if not self.enabled:
            return
        wall = self._elapsed()
        phase_totals = dict(sorted(self._phase_totals.items()))
        phase_cpu_totals = dict(sorted(self._phase_cpu_totals.items()))
        process_cpu = time.process_time() - self.start_process_cpu
        top_phase_total = sum(
            total for phase, total in phase_totals.items()
            if phase in self.TOP_LEVEL_PHASES
        )
        gpu_util = [
            float(row["utilization_gpu_pct"])
            for row in self._telemetry_samples
            if row.get("utilization_gpu_pct") not in ("", None)
        ]
        gpu_memory = [
            float(row["memory_used_mb"])
            for row in self._telemetry_samples
            if row.get("memory_used_mb") not in ("", None)
        ]
        power = [
            float(row["power_draw_w"])
            for row in self._telemetry_samples
            if row.get("power_draw_w") not in ("", None)
        ]
        completed_rounds = int(self._phase_counts.get("round_total", 0) or 0)
        round_total = float(phase_totals.get("round_total", 0.0) or 0.0)
        summary = {
            "profile_dir": self.profile_dir,
            "actual_run_dir": getattr(self, "actual_run_dir", None),
            "wall_clock_profile_seconds": wall,
            "process_cpu_seconds_total": process_cpu,
            "process_cpu_utilization_pct": process_cpu / wall * 100 if wall > 0 else None,
            "completed_rounds": completed_rounds,
            "seconds_per_round_wall_clock": wall / completed_rounds if completed_rounds else None,
            "rounds_per_second_wall_clock": completed_rounds / wall if completed_rounds and wall > 0 else None,
            "seconds_per_round_training_loop": round_total / completed_rounds if completed_rounds and round_total > 0 else None,
            "rounds_per_second_training_loop": completed_rounds / round_total if completed_rounds and round_total > 0 else None,
            "phase_totals_seconds": phase_totals,
            "phase_process_cpu_seconds": phase_cpu_totals,
            "phase_counts": dict(sorted(self._phase_counts.items())),
            "top_phases_by_total_seconds": sorted(
                phase_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20],
            "top_level_explained_fraction_of_wall_clock": min(1.0, top_phase_total / wall) if wall > 0 else None,
            "gpu_telemetry_sample_count": len(self._telemetry_samples),
            "gpu_utilization_avg_pct": sum(gpu_util) / len(gpu_util) if gpu_util else None,
            "gpu_utilization_max_pct": max(gpu_util) if gpu_util else None,
            "gpu_memory_used_avg_mb": sum(gpu_memory) / len(gpu_memory) if gpu_memory else None,
            "gpu_memory_used_max_mb": max(gpu_memory) if gpu_memory else None,
            "gpu_power_avg_w": sum(power) / len(power) if power else None,
            "gpu_power_max_w": max(power) if power else None,
            "extra": json_safe(extra or {}),
        }
        with open(self.summary_path, "w") as handle:
            json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")


def exact_sample_count(indices_or_subset):
    return int(len(indices_or_subset))


def client_indices_for_round(random_seed, round_idx, client_num_in_total, client_num_per_round):
    if client_num_in_total == client_num_per_round:
        return list(range(client_num_in_total))
    num_clients = min(client_num_per_round, client_num_in_total)
    rng = random.Random(f"{int(random_seed)}:{int(round_idx)}")
    return rng.sample(range(client_num_in_total), num_clients)


def has_run_artifacts(run_dir):
    if not os.path.isdir(run_dir):
        return False
    for artifact_name in RUN_ARTIFACT_NAMES:
        if os.path.exists(os.path.join(run_dir, artifact_name)):
            return True
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    return os.path.isdir(checkpoint_dir) and bool(os.listdir(checkpoint_dir))


def prepare_run_dir(run_dir, overwrite=False, dry_run=False):
    if dry_run:
        return
    normalized = os.path.abspath(run_dir)
    protected = os.path.join(os.sep, "results", "_golden")
    if protected in normalized:
        raise ValueError(f"Refusing to overwrite protected golden output directory: {run_dir}")
    if has_run_artifacts(run_dir):
        if not overwrite:
            raise FileExistsError(
                f"Run directory already contains artifacts: {run_dir}. "
                "Use --overwrite or set overwrite: true to replace it."
            )
        shutil.rmtree(run_dir)


def _flatten_numbers(value):
    if value is None:
        return []
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            value = reshape(-1)
        except TypeError:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return [float(value)]


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    if isinstance(value, dict):
        return {str(key): json_safe(item_value) for key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item_value) for item_value in value]
    return str(value)


def format_no_valid_model_selection_error(context):
    context = context or {}
    fields = {
        "dataset": context.get("dataset", "unknown"),
        "seed": context.get("random_seed", "unknown"),
        "simple_model_selection_epochs": context.get("simple_model_selection_epochs", "unknown"),
        "f_history_model_selection_epochs": context.get("f_history_model_selection_epochs", "unknown"),
        "model_selection_batch_size": context.get("model_selection_batch_size", "unknown"),
        "learning_rate": context.get("learning_rate", "unknown"),
        "critic_multiplier": context.get("critic_multiplier", "unknown"),
        "f_learning_rate": context.get("f_learning_rate", "unknown"),
        "client_optimizer": context.get("client_optimizer", "unknown"),
        "best_score": context.get("best_score", "unknown"),
    }
    details = ", ".join(f"{key}={value}" for key, value in fields.items())
    return (
        "No valid model-selection candidate was selected before federated training started; "
        f"{details}; federated training never started"
    )


class ModelSelectionFailure(RuntimeError):
    """Raised instead of a bare RuntimeError when model selection cannot
    select a valid candidate before federated round 0. Carries structured
    diagnostics so main.py's top-level exception handler can write a
    pretraining_failure.json artifact; a plain exception without this type
    stays an unexplained process failure (closeout plan Phase 1 SS4.2)."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _sha256_file_or_none(path):
    if not path or not os.path.exists(path):
        return None
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pretraining_failure_artifact(
    run_dir,
    *,
    run_id,
    effective_config,
    per_epoch_diagnostics,
    best_score,
    terminal_reason,
    traceback_text,
    stdout_path=None,
    stderr_path=None,
    hash_bundle_id=None,
):
    """Writes pretraining_failure.json: the only artifact that may classify a
    pre-round-0 model-selection failure as terminal_pretraining_ineligible
    (see run_manifest.py's validate_artifacts). A bare nonzero return code
    with no such artifact must remain an unexplained process failure."""
    per_epoch_diagnostics = list(per_epoch_diagnostics or [])
    first_nonfinite_epoch = None
    for record in per_epoch_diagnostics:
        bad_components = [
            component for component in ("epsilon_dev_finite", "f_of_z_dev_finite")
            if not record.get(component, True)
        ]
        if bad_components:
            first_nonfinite_epoch = {
                "epoch": record.get("epoch"),
                "components": bad_components,
            }
            break
    try:
        best_score_value = float(best_score)
        if not math.isfinite(best_score_value):
            best_score_value = None
    except (TypeError, ValueError):
        best_score_value = None
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "effective_config_checksum": (
            config_checksum(effective_config) if effective_config else None
        ),
        "failure_phase": "model_selection",
        "federated_rounds_started": 0,
        "model_selection_epochs_attempted": len(per_epoch_diagnostics),
        "best_model_selection_score": best_score_value,
        "per_epoch_finite_status": per_epoch_diagnostics,
        "first_nonfinite_epoch": first_nonfinite_epoch,
        "terminal_reason": terminal_reason,
        "traceback": traceback_text,
        "stdout_sha256": _sha256_file_or_none(stdout_path),
        "stderr_sha256": _sha256_file_or_none(stderr_path),
        "hash_bundle_id": hash_bundle_id,
        # A path alone is a mutable label -- binding the bundle file's own
        # content hash here means a later swap of the file at that path
        # (not just a source-tree drift the bundle's own recorded hashes
        # would catch) is also detectable (closeout review hardening).
        # hash_bundle_id is conventionally REPO_ROOT-relative (every
        # launcher exports it that way, matching source_manifest/
        # final_result_dir elsewhere in this campaign) -- resolved against
        # this module's own REPO_ROOT rather than the process's current
        # working directory, since main.py chdir()s to its own file's
        # directory near the top of the script, well before this ever
        # runs. A bare CWD-relative lookup here would silently resolve
        # against the wrong directory and always miss.
        "hash_bundle_sha256": _sha256_file_or_none(
            hash_bundle_id if hash_bundle_id and os.path.isabs(hash_bundle_id)
            else (os.path.join(REPO_ROOT, hash_bundle_id) if hash_bundle_id else hash_bundle_id)
        ),
    }
    path = os.path.join(run_dir, "pretraining_failure.json")
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def structural_mse_from_predictions(prediction, true_g):
    pred_values = _flatten_numbers(prediction)
    true_values = _flatten_numbers(true_g)
    if len(pred_values) != len(true_values):
        raise ValueError(
            f"Prediction/target length mismatch: {len(pred_values)} != {len(true_values)}"
        )
    if not pred_values:
        raise ValueError("Cannot compute structural MSE on an empty split")
    total = 0.0
    for pred, true in zip(pred_values, true_values):
        total += (pred - true) ** 2
    return total / len(pred_values)


def moment_violation(f_values, y_values, g_values):
    """Held-out moment-restriction violation: ``|| E[f(Z,X) * (Y - g(D,X))] ||^2``.

    Real eICU data has no ground-truth ``g0``, so this is the only recovery
    diagnostic available for Study B; Study A tracks it alongside structural
    MSE (which it can compute, since g0 is known there) as a second, g0-free
    checkpoint criterion. Squaring the mean removes any dependence on which
    epsilon sign convention (``Y-g`` vs ``g-Y``) a caller's objective uses, so
    this is safe to compute the same way regardless of ``objective_mode``.
    """
    f_flat = _flatten_numbers(f_values)
    y_flat = _flatten_numbers(y_values)
    g_flat = _flatten_numbers(g_values)
    if not (len(f_flat) == len(y_flat) == len(g_flat)):
        raise ValueError(
            f"f/y/g length mismatch: {len(f_flat)}, {len(y_flat)}, {len(g_flat)}"
        )
    if not f_flat:
        raise ValueError("Cannot compute moment violation on an empty split")
    moment = sum(f * (y - g) for f, y, g in zip(f_flat, y_flat, g_flat)) / len(f_flat)
    return moment ** 2


def _client_id_array(client_id):
    import numpy as np

    if client_id is None:
        return None
    return np.asarray(_to_numpy(client_id)).reshape(-1)


def per_client_equal_and_sample_weighted(per_row_values, client_id):
    """Group a per-row scalar metric by client and report both aggregates.

    Per ``metric_policy.md``: ``Q_EC = mean_i(q_i)`` (every hospital equal
    weight), ``Q_SW = sum_i(n_i*q_i)/sum_i(n_i)`` (every row equal weight,
    which for a plain per-row mean is bit-equivalent to pooling all rows —
    this is what every legacy (non-eICU) caller already computed, so
    ``Q_SW`` is safe to treat as a drop-in relabeling of the old pooled
    value). Every client present in ``client_id`` gets exactly one entry in
    the equal-client mean regardless of row count; none are silently
    dropped, matching the policy's "denominator and client IDs must be
    recorded" rule.
    """
    import numpy as np

    values = np.asarray(_to_numpy(per_row_values)).reshape(-1)
    clients = _client_id_array(client_id)
    if clients is None or len(clients) != len(values):
        raise ValueError("client_id must be provided and match per-row metric length")
    per_client = {}
    for client in np.unique(clients):
        mask = clients == client
        per_client[int(client)] = {
            "value": float(values[mask].mean()),
            "n": int(mask.sum()),
        }
    equal_client = sum(entry["value"] for entry in per_client.values()) / len(per_client)
    total_n = sum(entry["n"] for entry in per_client.values())
    sample_weighted = sum(entry["value"] * entry["n"] for entry in per_client.values()) / total_n
    return equal_client, sample_weighted, per_client


def equal_client_structural_mse(prediction, true_g, client_id):
    """Equal-client / sample-weighted structural MSE, grouped by client_id.

    Returns ``(equal_client_mse, sample_weighted_mse, per_client)`` where
    ``per_client`` maps client id -> ``{"value": mse_i, "n": n_i}``.
    """
    import numpy as np

    pred = np.asarray(_to_numpy(prediction)).reshape(-1)
    true = np.asarray(_to_numpy(true_g)).reshape(-1)
    if len(pred) != len(true):
        raise ValueError(
            f"Prediction/target length mismatch: {len(pred)} != {len(true)}"
        )
    squared_error = (pred - true) ** 2
    return per_client_equal_and_sample_weighted(squared_error, client_id)


def equal_client_moment_violation(f_values, y_values, g_values, client_id):
    """Equal-client / sample-weighted held-out moment violation.

    Per ``metric_policy.md``, the client moment is averaged *within* each
    client first (``mu_i = mean_{S_i} f(Z,W)(Y-g(D,W))``), then squared, and
    only then aggregated across clients — squaring after the per-client mean
    (not per-row) is what distinguishes this from a plain pooled
    ``moment_violation`` and is why it cannot be derived from
    ``per_client_equal_and_sample_weighted`` directly.
    """
    import numpy as np

    f_arr = np.asarray(_to_numpy(f_values)).reshape(-1)
    y_arr = np.asarray(_to_numpy(y_values)).reshape(-1)
    g_arr = np.asarray(_to_numpy(g_values)).reshape(-1)
    if not (len(f_arr) == len(y_arr) == len(g_arr)):
        raise ValueError(
            f"f/y/g length mismatch: {len(f_arr)}, {len(y_arr)}, {len(g_arr)}"
        )
    clients = _client_id_array(client_id)
    if clients is None or len(clients) != len(f_arr):
        raise ValueError("client_id must be provided and match per-row metric length")
    residual_moment = f_arr * (y_arr - g_arr)
    per_client = {}
    for client in np.unique(clients):
        mask = clients == client
        mu_i = float(residual_moment[mask].mean())
        per_client[int(client)] = {"value": mu_i ** 2, "n": int(mask.sum())}
    equal_client = sum(entry["value"] for entry in per_client.values()) / len(per_client)
    total_n = sum(entry["n"] for entry in per_client.values())
    sample_weighted = sum(entry["value"] * entry["n"] for entry in per_client.values()) / total_n
    return equal_client, sample_weighted, per_client


def write_per_client_metrics(run_dir, per_client_mse, per_client_moment_violation):
    """``per_client_metrics.csv``: one row per client, MSE + moment violation.

    Accepted by the campaign validator's ``per_client_artifact_alternatives``.
    ATE/individual-effect columns are appended post-hoc by
    ``analyze_eicu_study_a_checkpoint.py``, which is the only place that
    knows how to construct D=0/1 counterfactual rows for a given scenario.
    """
    path = os.path.join(run_dir, "per_client_metrics.csv")
    client_ids = sorted(set(per_client_mse) | set(per_client_moment_violation))
    fieldnames = ["client_id", "n", "structural_mse", "moment_violation"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for client_id in client_ids:
            mse_entry = per_client_mse.get(client_id, {})
            moment_entry = per_client_moment_violation.get(client_id, {})
            writer.writerow({
                "client_id": client_id,
                "n": mse_entry.get("n", moment_entry.get("n", "")),
                "structural_mse": mse_entry.get("value", ""),
                "moment_violation": moment_entry.get("value", ""),
            })
    return path


def structural_mse(model, split, batch_size=1024):
    if not hasattr(split, "x") or not hasattr(split, "g"):
        raise ValueError("Split must expose x and ground-truth structural g")
    prediction = predictions_for_split(model, split, batch_size=batch_size)
    return structural_mse_from_predictions(prediction, split.g)


def predictions_for_split(model, split, batch_size=1024):
    if not hasattr(split, "x") or not hasattr(split, "g"):
        raise ValueError("Split must expose x and ground-truth structural g")
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if callable(eval_method):
        eval_method()
    try:
        import torch
        no_grad = torch.no_grad()
    except Exception:
        no_grad = nullcontext()
    shape = getattr(split.x, "shape", None)
    num_samples = int(shape[0]) if shape is not None else len(split.x)
    if batch_size is None or int(batch_size) <= 0:
        batch_size = num_samples
    predictions = []
    with no_grad:
        for start in range(0, num_samples, int(batch_size)):
            predictions.append(model(split.x[start : start + int(batch_size)]))
    if not predictions:
        raise ValueError("Cannot predict an empty split")
    try:
        import torch
        prediction = torch.cat(predictions, dim=0)
    except Exception:
        import numpy as np
        prediction = np.concatenate(predictions, axis=0)
    if callable(train_method) and was_training:
        train_method()
    return prediction


def state_is_finite(state_dict):
    try:
        import torch
    except Exception:
        torch = None
    for value in state_dict.values():
        if torch is not None and torch.is_tensor(value):
            if not bool(torch.isfinite(value).all().item()):
                return False
            continue
        for number in _flatten_numbers(value):
            if not math.isfinite(number):
                return False
    return True


def metric_values_are_finite(values):
    """Return whether every present numeric metric is finite."""
    try:
        return all(
            math.isfinite(number)
            for value in values
            if value is not None
            for number in _flatten_numbers(value)
        )
    except (TypeError, ValueError):
        return False


def config_checksum(config):
    """Deterministic sha256 over a JSON-safe config, for cross-artifact provenance."""
    import hashlib

    serialized = json.dumps(json_safe(config), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# protocol_v1.md campaign provenance fields. A prior Study A campaign ran all
# 105 confirmatory runs before anyone noticed the federated serializer had
# silently omitted these 8 fields from every effective_config.json; 60 runs
# then needed post-hoc repair via scripts/reconcile_eicu_study_a_artifacts.py
# (120 files archived). The root cause -- these fields not being threaded
# through -- is already fixed (see EFFECTIVE_CONFIG_FIELDS above,
# get_effective_config, and run_manifest.py's write_config), but nothing
# previously rejected an EMPTY value for them, so the same silent failure
# could recur from any future refactor. Keep this tuple's contents identical
# to scripts/reconcile_eicu_study_a_artifacts.py's PROVENANCE_FIELDS.
CAMPAIGN_PROVENANCE_FIELDS = (
    "role",
    "scenario_name",
    "g0",
    "alignment_label",
    "primary_selection_metric",
    "selection_source",
    "scenario_scope",
    "study_claim",
)


def write_effective_config(run_dir, config):
    """Write ``effective_config.json``, the run's serialized configuration.

    Campaign provenance guard: a run needs ``CAMPAIGN_PROVENANCE_FIELDS``
    populated exactly when it is an eICU Study A style campaign run --
    ``config["dataset"]`` starts with ``"eicu"`` (matching the exact gate
    already used by ``check_eicu_aggregation_weighting`` /
    ``check_eicu_objective_mode`` above for this same "Study A discipline
    applies" question) *and* ``config["protocol_version"]`` is a non-empty
    string.

    Note this is deliberately narrower than "protocol_version is set":
    ``protocol_version`` is a generic field used by many non-eICU campaign
    families in this repo (rerun_protocol_v1, highdim_coauthor_protocol_v1,
    sine_fedogda_tuning, curve_fitting_tuning, real-image ABS, GPU-util
    profiling, ...) that never populate ``CAMPAIGN_PROVENANCE_FIELDS`` and
    were never expected to -- gating on ``protocol_version`` alone would
    make every one of those families' still-resumable manifests (e.g. the
    FedOGDA-S tuning pilot) start raising on their
    very first future run, which is exactly the "must not break" case this
    guard is required to avoid. Every eICU-dataset manifest actually in this
    repo already populates all 8 fields, so this narrower gate loses no
    real coverage of the incident this guard exists to prevent.

    For eICU campaign runs, every field in ``CAMPAIGN_PROVENANCE_FIELDS``
    must be present and non-empty (``None`` and ``""`` both count as
    missing), or this raises ``ValueError`` naming exactly which fields are
    missing and which run directory hit the problem. Every other run --
    non-eICU, or eICU with an empty ``protocol_version`` -- is completely
    unaffected and keeps working exactly as today.
    """
    dataset = str(config.get("dataset") or "")
    protocol_version = str(config.get("protocol_version") or "").strip()
    if dataset.startswith("eicu") and protocol_version:
        missing = [
            field
            for field in CAMPAIGN_PROVENANCE_FIELDS
            if not str(config.get(field) or "").strip()
        ]
        if missing:
            raise ValueError(
                "write_effective_config: eICU campaign run (dataset="
                f"{dataset!r}, protocol_version={protocol_version!r}) for "
                f"run_dir={run_dir!r} is missing required non-empty "
                f"provenance field(s): {missing}. This guard exists because "
                "a prior Study A campaign silently dropped these fields for "
                "60/105 runs before anyone noticed (see "
                "scripts/reconcile_eicu_study_a_artifacts.py); check that "
                "both the manifest row and get_effective_config() set every "
                f"field in {list(CAMPAIGN_PROVENANCE_FIELDS)}."
            )
    ensure_run_dirs(run_dir)
    path = os.path.join(run_dir, "effective_config.json")
    with open(path, "w") as f:
        json.dump(json_safe(config), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_mse_by_round(run_dir, rows):
    path = os.path.join(run_dir, "mse_by_round.csv")
    fieldnames = [
        "round",
        "train_mse",
        "val_mse",
        "primary_val_mse",
        "equal_client_val_mse",
        "train_moment_violation",
        "val_moment_violation",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
        "g_bn_min_running_var",
        "f_bn_min_running_var",
        "finite",
        "diverged",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def append_mse_by_round(run_dir, row):
    path = os.path.join(run_dir, "mse_by_round.csv")
    fieldnames = [
        "round",
        "train_mse",
        "val_mse",
        "primary_val_mse",
        "equal_client_val_mse",
        "train_moment_violation",
        "val_moment_violation",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
        "g_bn_min_running_var",
        "f_bn_min_running_var",
        "finite",
        "diverged",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_aggregation_weights_by_round(run_dir, rows):
    path = os.path.join(run_dir, "aggregation_weights_by_round.csv")
    fieldnames = [
        "round",
        "aggregation_weighting",
        "k_participating",
        "weight_sum",
        "client_weights_json",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def append_aggregation_weights_by_round(run_dir, row):
    path = os.path.join(run_dir, "aggregation_weights_by_round.csv")
    fieldnames = [
        "round",
        "aggregation_weighting",
        "k_participating",
        "weight_sum",
        "client_weights_json",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_test_mse_by_round(run_dir, rows):
    path = os.path.join(run_dir, "test_mse_by_round.csv")
    fieldnames = [
        "round",
        "test_mse",
        "finite",
        "diverged",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def append_test_mse_by_round(run_dir, row):
    path = os.path.join(run_dir, "test_mse_by_round.csv")
    fieldnames = [
        "round",
        "test_mse",
        "finite",
        "diverged",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_metrics(run_dir, metrics):
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(json_safe(metrics), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_batching_summary(run_dir, rows):
    path = os.path.join(run_dir, "batching_summary.json")
    with open(path, "w") as f:
        json.dump(json_safe(rows), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _to_numpy(value):
    import numpy as np

    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return numpy_method()
    return np.asarray(value)


def save_predictions_npz(
    run_dir,
    test_split,
    best_prediction,
    final_prediction,
    metadata,
):
    import numpy as np

    x = _to_numpy(test_split.x)
    true_g = _to_numpy(test_split.g)
    best_pred = _to_numpy(best_prediction)
    final_pred = _to_numpy(final_prediction)
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        indices = np.argsort(x.reshape(x.shape[0], -1)[:, 0])
    else:
        # Images have many feature axes.  Flattening argsort results would
        # create one index per pixel rather than one per sample.
        indices = np.arange(x.shape[0])
    path = os.path.join(run_dir, "predictions.npz")
    np.savez(
        path,
        x=x[indices],
        true_g=true_g[indices],
        **{
            PREDICTION_OUTPUT_KEYS[0]: best_pred[indices],
            PREDICTION_OUTPUT_KEYS[1]: final_pred[indices],
        },
        algorithm=np.asarray(str(metadata.get("algorithm", ""))),
        variant=np.asarray(str(metadata.get("variant", ""))),
        seed=np.asarray(int(metadata.get("random_seed", 0))),
        run_id=np.asarray(str(metadata.get("run_id", ""))),
    )
    return path


COMPACT_PREDICTION_SCHEMA_VERSION = 1


def save_predictions_npz_compact(
    run_dir,
    test_split,
    best_prediction,
    final_prediction,
    metadata,
    filename="predictions_compact.npz",
):
    """Versioned compact prediction artifact (closeout plan Phase 1 SS4.4):
    sample IDs/coordinate, ground truth, best/final prediction, and run
    metadata -- deliberately omits the full test input tensor that makes
    save_predictions_npz's predictions.npz ~10 GiB across an image campaign.

    Default filename (predictions_compact.npz) is additive: it writes
    alongside an existing predictions.npz, e.g. for
    derive_compact_predictions_20260826.py deriving a compact copy of an
    already-completed run, keeping predictions.npz's schema unchanged for
    every downstream script that already reads it. fedavg_api.py instead
    passes filename="predictions.npz" for a run whose config opts fully into
    the compact schema (closeout plan Phase 1 SS4.4's requirement that
    future image runs not save the full test tensor at all) -- there is no
    separate full predictions.npz for such a run to be additive to."""
    import numpy as np

    x = _to_numpy(test_split.x)
    true_g = _to_numpy(test_split.g)
    best_pred = _to_numpy(best_prediction)
    final_pred = _to_numpy(final_prediction)
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        indices = np.argsort(x.reshape(x.shape[0], -1)[:, 0])
        sample_coordinate = x.reshape(x.shape[0], -1)[:, 0][indices]
    else:
        # Images have too many feature axes for a per-sample scalar
        # coordinate; a stable sample index is the compact identifier.
        indices = np.arange(x.shape[0])
        sample_coordinate = indices.astype(np.int64)
    path = os.path.join(run_dir, filename)
    np.savez(
        path,
        schema_version=np.asarray(COMPACT_PREDICTION_SCHEMA_VERSION),
        sample_id=indices.astype(np.int64),
        sample_coordinate=sample_coordinate,
        true_g=true_g[indices],
        **{
            PREDICTION_OUTPUT_KEYS[0]: best_pred[indices],
            PREDICTION_OUTPUT_KEYS[1]: final_pred[indices],
        },
        dataset=np.asarray(str(metadata.get("dataset", ""))),
        algorithm=np.asarray(str(metadata.get("algorithm", ""))),
        variant=np.asarray(str(metadata.get("variant", ""))),
        seed=np.asarray(int(metadata.get("random_seed", 0))),
        run_id=np.asarray(str(metadata.get("run_id", ""))),
    )
    return path


def verify_compact_predictions_numerically_equal(run_dir):
    """Loads both predictions.npz and predictions_compact.npz from run_dir
    and raises ValueError unless every retained field agrees exactly. Used to
    validate a derived compact copy before it is ever treated as a substitute
    for the original (closeout plan SS4.4) -- never deletes the original."""
    import numpy as np

    full_path = os.path.join(run_dir, "predictions.npz")
    compact_path = os.path.join(run_dir, "predictions_compact.npz")
    with np.load(full_path) as full, np.load(compact_path) as compact:
        # Both writers independently sort by the same rule (argsort(x) for
        # low-dim, identity for images) and store each field in that same
        # final order, so the two arrays already line up row-for-row --
        # sample_id records provenance (which original sample each row was),
        # it is not an index to re-apply to the already-sorted full array.
        for key in (*PREDICTION_OUTPUT_KEYS, "true_g"):
            full_values = full[key]
            compact_values = compact[key]
            if full_values.shape != compact_values.shape:
                raise ValueError(
                    f"{key} shape mismatch: full {full_values.shape} vs "
                    f"compact {compact_values.shape}"
                )
            if not np.array_equal(full_values, compact_values, equal_nan=True):
                raise ValueError(f"{key} values differ between full and compact artifacts")
    return {"run_dir": run_dir, "verified_fields": [*PREDICTION_OUTPUT_KEYS, "true_g"]}


def now_seconds():
    return time.time()


def expand_smoke_manifest(manifest):
    base = dict(manifest.get("base", {}))
    variants = manifest.get("variants", [])
    expanded = []
    for variant in variants:
        item = dict(base)
        item.update(variant)
        optimizer = item.get("client_optimizer", "sgd")
        batch_size = int(item.get("batch_size", 0))
        algorithm = {
            "sgd": "fedgda",
            "ogda": "fedogda",
            "fed_eg": "fed_eg",
            "fed_eg_double": "fed_eg_double",
            "fed_zo_eg": "fed_zo_eg",
        }[str(optimizer).lower()]
        suffix = "d" if batch_size <= 0 else "s"
        item["variant"] = variant.get("variant", f"{algorithm}_{suffix}")
        expanded.append(item)
    return expanded
