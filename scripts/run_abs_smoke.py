#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fedgmm.sp_decentralized_mnist_lr_example.experiment_utils import (
    build_run_dir,
    expand_smoke_manifest,
    prepare_run_dir,
)


DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "experiments", "abs_smoke_manifest.json")
DEFAULT_MAIN = os.path.join(
    REPO_ROOT,
    "fedgmm",
    "sp_decentralized_mnist_lr_example",
    "main.py",
)
VARIANT_MAPPING = {
    "fedgda_d": {"client_optimizer": "sgd", "mode": "deterministic", "batching": "full_batch"},
    "fedgda_s": {"client_optimizer": "sgd", "mode": "stochastic", "batching": "minibatch"},
    "fedogda_d": {"client_optimizer": "ogda", "mode": "deterministic", "batching": "full_batch"},
    "fedogda_s": {"client_optimizer": "ogda", "mode": "stochastic", "batching": "minibatch"},
    "fed_eg_d": {"client_optimizer": "fed_eg", "mode": "deterministic", "batching": "full_batch"},
    "fed_eg_s": {"client_optimizer": "fed_eg", "mode": "stochastic", "batching": "minibatch"},
    "fed_eg_double_d": {"client_optimizer": "fed_eg_double", "mode": "deterministic", "batching": "full_batch"},
    "fed_eg_double_s": {"client_optimizer": "fed_eg_double", "mode": "stochastic", "batching": "minibatch"},
    "fed_zo_eg_d": {"client_optimizer": "fed_zo_eg", "mode": "deterministic", "batching": "full_batch"},
    "fed_zo_eg_s": {"client_optimizer": "fed_zo_eg", "mode": "stochastic", "batching": "minibatch"},
}
FEDML_LOG_DIR_ENV = "FEDGMM_FEDML_LOG_DIR"
FEDML_TRACE_DIR_ENV = "FEDGMM_FEDML_TRACE_DIR"


def _yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_config(path, config):
    sections = {
        "common_args": {
            "training_type": "simulation",
            "using_mlops": False,
            "random_seed": config["random_seed"],
            "run_id": config["run_id"],
            "config_version": "release",
            "output_dir": config["output_dir"],
            "overwrite": config.get("overwrite", False),
        },
        "data_args": {
            "dataset": config["dataset"],
            "data_cache_dir": config["data_cache_dir"],
            "partition_method": config["partition_method"],
            "partition_alpha": config["partition_alpha"],
        },
        "model_args": {
            "model": config["model"],
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
            "eg_predictor_server_lr": config.get("eg_predictor_server_lr", config["server_learning_rate"]),
            "eg_corrector_server_lr": config.get("eg_corrector_server_lr", config["server_learning_rate"]),
            "zo_mu": config.get("zo_mu", 1e-3),
            "zo_num_directions": config.get("zo_num_directions", 1),
            "client_execution_mode": config.get("client_execution_mode", "sp"),
            "enable_multiprocessing": config.get("enable_multiprocessing", False),
            "multiprocessing_num_workers": config.get("multiprocessing_num_workers", 0),
            "multiprocessing_gpu_ids": config.get("multiprocessing_gpu_ids", ""),
            "multiprocessingsinglegpu_num_workers": config.get("multiprocessingsinglegpu_num_workers", 2),
            "multiprocessingsinglegpu_gpu_id": config.get("multiprocessingsinglegpu_gpu_id", config["gpu_id"]),
            "gradient_clip_norm": config["gradient_clip_norm"],
            "simple_model_selection_epochs": config.get("simple_model_selection_epochs", 100),
            "f_history_model_selection_epochs": config.get("f_history_model_selection_epochs", 60),
            "model_selection_batch_size": config.get("model_selection_batch_size", 200),
            "enable_legacy_outputs": config.get("enable_legacy_outputs", True),
            "require_multibatch_stochastic": config.get("require_multibatch_stochastic", False),
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
            "run_name": f"abs_{config['variant']}_seed_{config['random_seed']}_{config['run_id']}",
        },
    }
    with open(path, "w") as f:
        for section, values in sections.items():
            f.write(f"{section}:\n")
            for key, value in values.items():
                f.write(f"  {key}: {_yaml_scalar(value)}\n")
            f.write("\n")


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def command_for_variant(python_executable, main_path, config_path):
    return [python_executable, main_path, "--cf", config_path]


def filter_variants(variants, selected_variant=None):
    if selected_variant is None:
        return variants
    return [variant for variant in variants if variant["variant"] == selected_variant]


def assert_variant_mapping(config):
    variant = config["variant"]
    if variant not in VARIANT_MAPPING:
        raise ValueError(f"Unsupported smoke variant: {variant}")
    expected = VARIANT_MAPPING[variant]
    client_optimizer = str(config.get("client_optimizer", "")).lower()
    if client_optimizer != expected["client_optimizer"]:
        raise ValueError(
            f"{variant} must use client_optimizer={expected['client_optimizer']}, got {client_optimizer}"
        )
    batch_size = int(config.get("batch_size", 0))
    is_deterministic = batch_size <= 0
    expected_deterministic = expected["mode"] == "deterministic"
    if is_deterministic != expected_deterministic:
        raise ValueError(f"{variant} has inconsistent batch_size={batch_size}")
    require_multibatch = bool(config.get("require_multibatch_stochastic", False))
    if require_multibatch and expected["mode"] != "stochastic":
        raise ValueError(f"{variant} cannot require stochastic multibatch validation")
    if require_multibatch and batch_size <= 0:
        raise ValueError(f"{variant} requires a positive batch_size for multibatch validation")


def fedml_log_dir_for_variant(config):
    return os.path.join(
        REPO_ROOT,
        "logs",
        "fedml",
        str(config["variant"]),
        str(config["run_id"]),
    )


def fedml_trace_dir_for_variant(config):
    return os.path.join(
        REPO_ROOT,
        "logs",
        "fedml_trace",
        str(config["variant"]),
        str(config["run_id"]),
    )


def build_child_env(config, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    env[FEDML_LOG_DIR_ENV] = fedml_log_dir_for_variant(config)
    env[FEDML_TRACE_DIR_ENV] = fedml_trace_dir_for_variant(config)
    return env


def resolve_output_dir(output_dir):
    if os.path.isabs(str(output_dir)):
        return os.path.abspath(str(output_dir))
    return os.path.abspath(os.path.join(REPO_ROOT, str(output_dir)))


def main():
    parser = argparse.ArgumentParser(description="Run the four local ABS smoke variants.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--main", default=DEFAULT_MAIN)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_MAPPING),
        default=None,
        help="Run only one ABS smoke variant.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    variants = filter_variants(expand_smoke_manifest(manifest), args.variant)
    with tempfile.TemporaryDirectory(prefix="abs_smoke_configs_") as tmpdir:
        for variant in variants:
            if args.output_dir is not None:
                variant["output_dir"] = args.output_dir
            if args.run_id is not None:
                variant["run_id"] = args.run_id
            variant["output_dir"] = resolve_output_dir(variant["output_dir"])
            variant["overwrite"] = args.overwrite or bool(variant.get("overwrite", False))
            assert_variant_mapping(variant)
            config_path = os.path.join(tmpdir, f"{variant['variant']}.yaml")
            write_config(config_path, variant)
            run_dir = build_run_dir(
                variant["output_dir"],
                variant["dataset"],
                variant["variant"],
                variant["random_seed"],
                variant["run_id"],
            )
            fedml_log_dir = fedml_log_dir_for_variant(variant)
            fedml_trace_dir = fedml_trace_dir_for_variant(variant)
            cmd = command_for_variant(args.python, args.main, config_path)
            print(f"{variant['variant']} -> {run_dir}")
            print(f"{FEDML_LOG_DIR_ENV} -> {fedml_log_dir}")
            print(f"{FEDML_TRACE_DIR_ENV} -> {fedml_trace_dir}")
            print(" ".join(cmd))
            if args.dry_run:
                with open(config_path, "r") as config_file:
                    print(config_file.read().rstrip())
                continue
            prepare_run_dir(run_dir, overwrite=variant["overwrite"])
            os.makedirs(fedml_log_dir, exist_ok=True)
            os.makedirs(fedml_trace_dir, exist_ok=True)
            result = subprocess.run(cmd, cwd=REPO_ROOT, env=build_child_env(variant))
            if result.returncode != 0:
                raise SystemExit(f"{variant['variant']} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
