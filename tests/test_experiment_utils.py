import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

from fedgmm.sp_decentralized_mnist_lr_example.experiment_utils import (
    CAMPAIGN_PROVENANCE_FIELDS,
    EFFECTIVE_CONFIG_FIELDS,
    PREDICTION_OUTPUT_KEYS,
    append_mse_by_round,
    append_test_mse_by_round,
    build_run_dir,
    client_indices_for_round,
    config_checksum,
    equal_client_moment_violation,
    equal_client_structural_mse,
    exact_sample_count,
    expand_smoke_manifest,
    format_no_valid_model_selection_error,
    get_effective_config,
    metric_values_are_finite,
    ModelSelectionFailure,
    per_client_equal_and_sample_weighted,
    prepare_run_dir,
    save_predictions_npz,
    structural_mse,
    structural_mse_from_predictions,
    write_batching_summary,
    write_effective_config,
    write_mse_by_round,
    write_per_client_metrics,
    save_predictions_npz_compact,
    verify_compact_predictions_numerically_equal,
    write_pretraining_failure_artifact,
    write_test_mse_by_round,
)
from scripts.run_abs_smoke import (
    DEFAULT_MANIFEST,
    FEDML_LOG_DIR_ENV,
    FEDML_TRACE_DIR_ENV,
    REPO_ROOT,
    VARIANT_MAPPING,
    assert_variant_mapping,
    build_child_env,
    filter_variants,
    fedml_log_dir_for_variant,
    fedml_trace_dir_for_variant,
    resolve_output_dir,
)
from scripts.validate_smoke_run import ValidationError, validate_run


class LinearModel:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, x):
        return [2.0 * value for value in x]


def _npy_payload(values, shape):
    flat_values = [float(value) for value in values]
    header = {
        "descr": "<f8",
        "fortran_order": False,
        "shape": tuple(shape),
    }
    header_bytes = repr(header).encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return (
        b"\x93NUMPY"
        + bytes([1, 0])
        + struct.pack("<H", len(header_bytes))
        + header_bytes
        + struct.pack("<" + "d" * len(flat_values), *flat_values)
    )


def _write_npz(path, arrays):
    with zipfile.ZipFile(path, "w") as archive:
        for key, values in arrays.items():
            archive.writestr(f"{key}.npy", _npy_payload(values, (len(values), 1)))


def _write_fake_checkpoint(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "checkpoint/data.pkl",
            b"g_state_dict f_state_dict state g f",
        )


def _write_valid_run(
    root,
    variant="fedgda_d",
    batch_size=0,
    require_multibatch_stochastic=False,
    log_test_mse_by_round=False,
):
    run_dir = os.path.join(root, "abs", variant, "seed_0", "smoke")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "effective_config.json"), "w") as f:
        json.dump({
            "dataset": "abs",
            "variant": variant,
            "mode": "stochastic" if batch_size > 0 else "deterministic",
            "batch_size": batch_size,
            "require_multibatch_stochastic": require_multibatch_stochastic,
            "random_seed": 0,
            "run_id": "smoke",
            "comm_round": 2,
            "client_num_per_round": 2,
            "using_gpu": False,
            "log_test_mse_by_round": log_test_mse_by_round,
            "test_mse_logged_by_round": log_test_mse_by_round,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
        }, f)
    with open(os.path.join(run_dir, "mse_by_round.csv"), "w") as f:
        f.write("round,train_mse,val_mse,gmm_train_objective,gmm_val_objective,gmm_eval,finite,diverged\n")
        f.write("0,2.0,2.0,0.1,0.2,0.3,True,False\n")
        f.write("1,1.0,1.0,0.1,0.2,0.4,True,False\n")
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({
            "best_validation_round": 1,
            "best_validation_mse": 1.0,
            "test_mse_at_best_validation": 0.5,
            "final_validation_mse": 1.0,
            "final_test_mse": 0.5,
            "diverged": False,
            "test_mse_logged_by_round": log_test_mse_by_round,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
        }, f)
    if log_test_mse_by_round:
        write_test_mse_by_round(run_dir, [
            {"round": 0, "test_mse": 1.0, "finite": True, "diverged": False},
            {"round": 1, "test_mse": 0.5, "finite": True, "diverged": False},
        ])
    _write_npz(
        os.path.join(run_dir, "predictions.npz"),
        {
            "x": [0.0, 1.0],
            "true_g": [0.0, 1.0],
            "best_validation_prediction": [1.0, 1.0],
            "final_prediction": [1.0, 1.0],
        },
    )
    _write_fake_checkpoint(os.path.join(run_dir, "checkpoints", "best_validation.pt"))
    _write_fake_checkpoint(os.path.join(run_dir, "checkpoints", "final.pt"))
    return run_dir


class ExperimentUtilsTest(unittest.TestCase):
    def test_metric_values_are_finite_checks_every_present_number(self):
        self.assertTrue(metric_values_are_finite([1.0, None, [2.0, -3.0]]))
        self.assertFalse(metric_values_are_finite([1.0, float("nan")]))
        self.assertFalse(metric_values_are_finite([float("inf")]))
        self.assertFalse(metric_values_are_finite([float("-inf")]))

    def test_unique_output_directory_construction(self):
        path = build_run_dir("results", "abs", "fedgda_d", 0, "smoke")
        self.assertEqual(
            path,
            os.path.join("results", "abs", "fedgda_d", "seed_0", "smoke"),
        )

    def test_structural_mse_from_predictions(self):
        self.assertEqual(
            structural_mse_from_predictions([1.0, 3.0], [2.0, 1.0]),
            2.5,
        )

    def test_structural_mse_on_mock_split(self):
        split = SimpleNamespace(x=[1.0, 2.0, 3.0], g=[2.0, 5.0, 6.0])
        self.assertAlmostEqual(structural_mse(LinearModel(), split), 1.0 / 3.0)

    def test_save_predictions_preserves_image_sample_axis(self):
        import numpy as np

        x = np.arange(3 * 3 * 2 * 2, dtype=np.float32).reshape(3, 3, 2, 2)
        split = SimpleNamespace(x=x, g=np.asarray([[1.0], [2.0], [3.0]]))
        with tempfile.TemporaryDirectory() as run_dir:
            path = save_predictions_npz(
                run_dir,
                split,
                np.asarray([[1.1], [2.1], [3.1]]),
                np.asarray([[1.2], [2.2], [3.2]]),
                {"algorithm": "fedgda", "variant": "fedgda_s", "random_seed": 0},
            )
            with np.load(path) as saved:
                self.assertEqual(saved["x"].shape, (3, 3, 2, 2))
                np.testing.assert_array_equal(saved["x"], x)
                self.assertEqual(saved["true_g"].shape[0], 3)

    def test_config_defaults_and_overrides(self):
        args = SimpleNamespace(
            dataset="abs",
            client_optimizer="ogda",
            batch_size=0,
            epochs=1,
            comm_round=3,
            client_num_in_total=1000,
            client_num_per_round=2,
            frequency_of_the_test=1,
            learning_rate=0.001,
            random_seed=7,
            run_id="smoke",
            output_dir="local_results",
            partition_alpha=0.5,
            weight_decay=0.1,
            critic_multiplier=12,
            server_learning_rate=0.75,
            model="lr",
            federated_optimizer="FedAvg",
            partition_method="hetero",
            data_cache_dir="data",
            using_gpu=False,
            gpu_id=0,
            enable_legacy_outputs=False,
            overwrite=True,
        )
        config = get_effective_config(args)
        self.assertEqual(config["variant"], "fedogda_d")
        self.assertEqual(config["f_learning_rate"], 0.012)
        self.assertEqual(config["server_learning_rate"], 0.75)
        self.assertFalse(config["log_test_mse_by_round"])
        self.assertFalse(config["test_mse_logged_by_round"])
        self.assertFalse(config["test_mse_used_for_selection"])
        self.assertEqual(config["selection_metric_source"], "validation")
        for field in EFFECTIVE_CONFIG_FIELDS:
            self.assertIn(field, config)

    def test_compact_predictions_only_defaults_false_and_threads_through(self):
        # fedavg_api.py reads self.compact_predictions_only from
        # self.effective_config, never from args directly -- if this field
        # were dropped from get_effective_config()'s output, every run
        # would silently write the full ~10 GiB-scale predictions.npz
        # regardless of what the manifest/config actually requested (a real
        # bug caught live on a running V4 GPU job, 2026-08-28: the manifest
        # and generated YAML both correctly said compact_predictions_only=
        # true, but effective_config.json still came back with the key
        # missing entirely).
        base_args = dict(
            dataset="femnist_z", client_optimizer="sgd", batch_size=0, epochs=1,
            comm_round=3, client_num_in_total=10, client_num_per_round=10,
            frequency_of_the_test=1, learning_rate=0.01, random_seed=0,
            run_id="fixture", output_dir="local_results",
        )
        self.assertFalse(get_effective_config(SimpleNamespace(**base_args))["compact_predictions_only"])
        config = get_effective_config(SimpleNamespace(**base_args, compact_predictions_only=True))
        self.assertTrue(config["compact_predictions_only"])

    def test_test_mse_logging_config_metadata(self):
        config = get_effective_config(SimpleNamespace(
            dataset="sin",
            client_optimizer="ogda",
            batch_size=0,
            epochs=2,
            comm_round=3,
            client_num_in_total=1000,
            client_num_per_round=1000,
            frequency_of_the_test=1,
            learning_rate=0.001,
            random_seed=0,
            run_id="sine_logging_smoke",
            output_dir="local_results",
            partition_alpha=0.5,
            weight_decay=0.1,
            critic_multiplier=10,
            server_learning_rate=1.5,
            model="lr",
            federated_optimizer="FedAvg",
            partition_method="hetero",
            data_cache_dir="data",
            using_gpu=False,
            gpu_id=0,
            log_test_mse_by_round=True,
            test_mse_used_for_selection=False,
            selection_metric_source="validation",
        ))
        self.assertTrue(config["log_test_mse_by_round"])
        self.assertTrue(config["test_mse_logged_by_round"])
        self.assertFalse(config["test_mse_used_for_selection"])
        self.assertEqual(config["selection_metric_source"], "validation")

    def test_full_model_selection_defaults_are_preserved(self):
        config = get_effective_config(SimpleNamespace())
        self.assertEqual(config["simple_model_selection_epochs"], 100)
        self.assertEqual(config["f_history_model_selection_epochs"], 60)
        self.assertEqual(config["model_selection_batch_size"], 200)

    def test_smoke_model_selection_settings_are_explicit(self):
        with open(DEFAULT_MANIFEST, "r") as f:
            manifest = json.load(f)
        variants = expand_smoke_manifest(manifest)
        self.assertTrue(all(variant["simple_model_selection_epochs"] == 100 for variant in variants))
        self.assertTrue(all(variant["f_history_model_selection_epochs"] == 60 for variant in variants))
        self.assertTrue(all(variant["model_selection_batch_size"] == 200 for variant in variants))

    def test_exact_sample_count_with_partial_final_batch(self):
        assigned_indices = list(range(257))
        old_estimated_count = 3 * 128
        self.assertEqual(exact_sample_count(assigned_indices), 257)
        self.assertNotEqual(exact_sample_count(assigned_indices), old_estimated_count)

    def test_output_path_is_stable_after_cwd_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = resolve_output_dir("results")
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                run_dir = build_run_dir(output_dir, "abs", "fedgda_d", 0, "smoke")
            finally:
                os.chdir(original_cwd)
        self.assertEqual(
            run_dir,
            os.path.join(output_dir, "abs", "fedgda_d", "seed_0", "smoke"),
        )

    def test_client_sampling_reproducible_for_same_seed_and_round(self):
        first = client_indices_for_round(7, 2, 100, 10)
        second = client_indices_for_round(7, 2, 100, 10)
        self.assertEqual(first, second)

    def test_client_sampling_changes_with_seed(self):
        first = client_indices_for_round(7, 2, 100, 10)
        second = client_indices_for_round(8, 2, 100, 10)
        self.assertNotEqual(first, second)

    def test_existing_run_directory_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "results", "abs", "fedgda_d", "seed_0", "smoke")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                f.write("{}\n")
            with self.assertRaises(FileExistsError):
                prepare_run_dir(run_dir, overwrite=False)

    def test_overwrite_removes_existing_artifacts_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "results", "abs", "fedgda_d", "seed_0", "smoke")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                f.write("{}\n")
            prepare_run_dir(run_dir, overwrite=True)
            self.assertFalse(os.path.exists(run_dir))

    def test_smoke_manifest_expands_four_variants(self):
        manifest = {
            "base": {"dataset": "abs", "comm_round": 3},
            "variants": [
                {"client_optimizer": "sgd", "batch_size": 0},
                {"client_optimizer": "sgd", "batch_size": 4},
                {"client_optimizer": "ogda", "batch_size": 0},
                {"client_optimizer": "ogda", "batch_size": 4},
            ],
        }
        variants = expand_smoke_manifest(manifest)
        self.assertEqual(
            [variant["variant"] for variant in variants],
            ["fedgda_d", "fedgda_s", "fedogda_d", "fedogda_s"],
        )

    def test_variant_to_optimizer_mode_mapping(self):
        with open(DEFAULT_MANIFEST, "r") as f:
            manifest = json.load(f)
        for variant in expand_smoke_manifest(manifest):
            assert_variant_mapping(variant)
            expected = VARIANT_MAPPING[variant["variant"]]
            self.assertEqual(variant["client_optimizer"], expected["client_optimizer"])
            self.assertEqual(int(variant["batch_size"]) <= 0, expected["mode"] == "deterministic")

    def test_stochastic_smoke_manifest_uses_multibatch_guard(self):
        with open(DEFAULT_MANIFEST, "r") as f:
            manifest = json.load(f)
        variants = {variant["variant"]: variant for variant in expand_smoke_manifest(manifest)}
        self.assertEqual(variants["fedgda_s"]["batch_size"], 4)
        self.assertEqual(variants["fedogda_s"]["batch_size"], 4)
        self.assertTrue(variants["fedgda_s"]["require_multibatch_stochastic"])
        self.assertTrue(variants["fedogda_s"]["require_multibatch_stochastic"])
        self.assertFalse(variants["fedgda_d"].get("require_multibatch_stochastic", False))
        self.assertFalse(variants["fedogda_d"].get("require_multibatch_stochastic", False))

    def test_variant_mapping_rejects_inconsistent_config(self):
        bad = {"variant": "fedogda_s", "client_optimizer": "sgd", "batch_size": 4}
        with self.assertRaises(ValueError):
            assert_variant_mapping(bad)

    def test_variant_mapping_rejects_multibatch_guard_for_deterministic(self):
        bad = {
            "variant": "fedgda_d",
            "client_optimizer": "sgd",
            "batch_size": 0,
            "require_multibatch_stochastic": True,
        }
        with self.assertRaises(ValueError):
            assert_variant_mapping(bad)

    def test_abs_smoke_manifest_uses_two_epochs(self):
        with open(DEFAULT_MANIFEST, "r") as f:
            manifest = json.load(f)
        variants = expand_smoke_manifest(manifest)
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(variant["epochs"] == 2 for variant in variants))

    def test_variant_filtering_selects_one_manifest_entry(self):
        with open(DEFAULT_MANIFEST, "r") as f:
            manifest = json.load(f)
        variants = expand_smoke_manifest(manifest)
        filtered = filter_variants(variants, "fedogda_s")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["variant"], "fedogda_s")

    def test_fedml_log_directory_is_inside_repository(self):
        config = {"variant": "fedgda_s", "run_id": "smoke"}
        path = fedml_log_dir_for_variant(config)
        self.assertTrue(os.path.abspath(path).startswith(os.path.abspath(REPO_ROOT)))
        self.assertIn(os.path.join("logs", "fedml", "fedgda_s", "smoke"), path)

    def test_fedml_log_directory_differs_by_variant(self):
        first = fedml_log_dir_for_variant({"variant": "fedgda_s", "run_id": "smoke"})
        second = fedml_log_dir_for_variant({"variant": "fedogda_s", "run_id": "smoke"})
        self.assertNotEqual(first, second)

    def test_fedml_trace_directory_is_inside_repository(self):
        config = {"variant": "fedgda_s", "run_id": "smoke"}
        path = fedml_trace_dir_for_variant(config)
        self.assertTrue(os.path.abspath(path).startswith(os.path.abspath(REPO_ROOT)))
        self.assertIn(os.path.join("logs", "fedml_trace", "fedgda_s", "smoke"), path)

    def test_fedml_trace_directory_differs_by_variant(self):
        first = fedml_trace_dir_for_variant({"variant": "fedgda_s", "run_id": "smoke"})
        second = fedml_trace_dir_for_variant({"variant": "fedogda_s", "run_id": "smoke"})
        self.assertNotEqual(first, second)

    def test_child_process_receives_fedml_log_env(self):
        config = {"variant": "fedgda_s", "run_id": "smoke"}
        env = build_child_env(config, base_env={})
        self.assertEqual(env[FEDML_LOG_DIR_ENV], fedml_log_dir_for_variant(config))

    def test_child_process_receives_fedml_trace_env(self):
        config = {"variant": "fedgda_s", "run_id": "smoke"}
        env = build_child_env(config, base_env={})
        self.assertEqual(env[FEDML_TRACE_DIR_ENV], fedml_trace_dir_for_variant(config))

    def test_default_fedml_log_path_unchanged_without_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.pop(FEDML_LOG_DIR_ENV, None)
            env["HOME"] = tmpdir
            env["PYTHONPATH"] = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from fedml.computing.scheduler.slave.client_constants import ClientConstants; "
                        "from fedml.computing.scheduler.master.server_constants import ServerConstants; "
                        "print(ClientConstants.get_log_file_dir()); "
                        "print(ServerConstants.get_log_file_dir())"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected_client = os.path.join(tmpdir, ".fedml", "fedml-client", "fedml", "logs")
            expected_server = os.path.join(tmpdir, ".fedml", "fedml-server", "fedml", "logs")
            self.assertEqual(result.stdout.strip().splitlines(), [expected_client, expected_server])

    def test_default_fedml_trace_path_unchanged_without_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env.pop(FEDML_TRACE_DIR_ENV, None)
            env["HOME"] = tmpdir
            env["PYTHONPATH"] = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from fedml.runner import FedMLRunner; print(FedMLRunner.get_trace_dir())",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), os.path.join(tmpdir, ".fedml", "fedml_trace"))

    def test_fedml_runner_uses_trace_override_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override = os.path.join(tmpdir, "repo", "logs", "fedml_trace", "variant", "smoke")
            env = dict(os.environ)
            env[FEDML_TRACE_DIR_ENV] = override
            env["PYTHONPATH"] = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from fedml.runner import FedMLRunner; print(FedMLRunner.get_trace_dir())",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), override)

    def test_fedml_constants_use_override_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override = os.path.join(tmpdir, "repo", "logs", "fedml", "variant", "smoke")
            env = dict(os.environ)
            env[FEDML_LOG_DIR_ENV] = override
            env["PYTHONPATH"] = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from fedml.computing.scheduler.slave.client_constants import ClientConstants; "
                        "from fedml.computing.scheduler.master.server_constants import ServerConstants; "
                        "print(ClientConstants.get_log_file_dir()); "
                        "print(ServerConstants.get_log_file_dir())"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip().splitlines(), [override, override])

    def test_dry_run_creates_no_training_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "scripts", "run_abs_smoke.py"),
                    "--variant",
                    "fedgda_s",
                    "--dry-run",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(FEDML_LOG_DIR_ENV, result.stdout)
            self.assertIn(FEDML_TRACE_DIR_ENV, result.stdout)
            self.assertIn("batch_size: 4", result.stdout)
            self.assertIn("require_multibatch_stochastic: true", result.stdout)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "abs", "fedgda_s", "seed_0", "smoke")))

    def test_prediction_output_keys_include_best_and_final(self):
        self.assertIn("best_validation_prediction", PREDICTION_OUTPUT_KEYS)
        self.assertIn("final_prediction", PREDICTION_OUTPUT_KEYS)
        self.assertNotEqual(PREDICTION_OUTPUT_KEYS[0], PREDICTION_OUTPUT_KEYS[1])

    def test_validator_succeeds_on_golden_smoke_run(self):
        result = validate_run(
            os.path.join("results", "_golden", "abs", "fedgda_d", "seed_0", "smoke"),
            dataset="abs",
            variant="fedgda_d",
            seed=0,
            run_id="smoke",
        )
        self.assertEqual(result["variant"], "fedgda_d")

    def test_validator_fails_on_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(tmpdir)
            os.remove(os.path.join(run_dir, "metrics.json"))
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_validator_fails_on_non_finite_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(tmpdir)
            with open(os.path.join(run_dir, "metrics.json"), "r") as f:
                metrics = json.load(f)
            metrics["final_test_mse"] = math.inf
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_validator_fails_on_inconsistent_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(tmpdir)
            with open(os.path.join(run_dir, "metrics.json"), "r") as f:
                metrics = json.load(f)
            metrics["best_validation_round"] = 0
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_validator_accepts_test_mse_by_round_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(tmpdir, log_test_mse_by_round=True)
            result = validate_run(run_dir, dataset="abs", variant="fedgda_d", seed=0, run_id="smoke")
            self.assertEqual(result["variant"], "fedgda_d")

    def test_validator_requires_test_mse_by_round_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(tmpdir, log_test_mse_by_round=True)
            os.remove(os.path.join(run_dir, "test_mse_by_round.csv"))
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_validator_rejects_test_mse_selection_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(tmpdir, log_test_mse_by_round=True)
            with open(os.path.join(run_dir, "metrics.json"), "r") as f:
                metrics = json.load(f)
            metrics["test_mse_used_for_selection"] = True
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_validator_accepts_stochastic_multibatch_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(
                tmpdir,
                variant="fedgda_s",
                batch_size=4,
                require_multibatch_stochastic=True,
            )
            write_batching_summary(run_dir, [
                {
                    "variant": "fedgda_s",
                    "round": 0,
                    "client_id": 598,
                    "sample_count": 15,
                    "configured_batch_size": 4,
                    "num_batches": 4,
                    "first_actual_batch_size": 4,
                },
                {
                    "variant": "fedgda_s",
                    "round": 0,
                    "client_id": 691,
                    "sample_count": 7,
                    "configured_batch_size": 4,
                    "num_batches": 2,
                    "first_actual_batch_size": 4,
                },
                {
                    "variant": "fedgda_s",
                    "round": 1,
                    "client_id": 193,
                    "sample_count": 12,
                    "configured_batch_size": 4,
                    "num_batches": 3,
                    "first_actual_batch_size": 4,
                },
                {
                    "variant": "fedgda_s",
                    "round": 1,
                    "client_id": 218,
                    "sample_count": 54,
                    "configured_batch_size": 4,
                    "num_batches": 14,
                    "first_actual_batch_size": 4,
                },
            ])
            result = validate_run(run_dir, dataset="abs", variant="fedgda_s", seed=0, run_id="smoke")
            self.assertEqual(result["variant"], "fedgda_s")

    def test_validator_requires_batching_summary_for_stochastic_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(
                tmpdir,
                variant="fedgda_s",
                batch_size=4,
                require_multibatch_stochastic=True,
            )
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_validator_rejects_single_batch_stochastic_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_valid_run(
                tmpdir,
                variant="fedgda_s",
                batch_size=4,
                require_multibatch_stochastic=True,
            )
            rows = [
                {
                    "variant": "fedgda_s",
                    "round": 0,
                    "client_id": 598,
                    "sample_count": 15,
                    "configured_batch_size": 4,
                    "num_batches": 4,
                    "first_actual_batch_size": 4,
                },
                {
                    "variant": "fedgda_s",
                    "round": 0,
                    "client_id": 691,
                    "sample_count": 4,
                    "configured_batch_size": 4,
                    "num_batches": 1,
                    "first_actual_batch_size": 4,
                },
                {
                    "variant": "fedgda_s",
                    "round": 1,
                    "client_id": 193,
                    "sample_count": 12,
                    "configured_batch_size": 4,
                    "num_batches": 3,
                    "first_actual_batch_size": 4,
                },
                {
                    "variant": "fedgda_s",
                    "round": 1,
                    "client_id": 218,
                    "sample_count": 54,
                    "configured_batch_size": 4,
                    "num_batches": 14,
                    "first_actual_batch_size": 4,
                },
            ]
            write_batching_summary(run_dir, rows)
            with self.assertRaises(ValidationError):
                validate_run(run_dir)

    def test_cuda_visible_does_not_override_using_gpu_false(self):
        old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            config = get_effective_config(SimpleNamespace(using_gpu=False))
        finally:
            if old_value is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_value
        self.assertFalse(config["using_gpu"])

    def test_golden_outputs_are_protected_from_overwrite(self):
        with self.assertRaises(ValueError):
            prepare_run_dir(
                os.path.abspath(os.path.join("results", "_golden", "abs", "fedgda_d", "seed_0", "smoke")),
                overwrite=True,
            )

    def test_no_valid_model_selection_error_contains_context(self):
        message = format_no_valid_model_selection_error({
            "dataset": "abs",
            "random_seed": 0,
            "simple_model_selection_epochs": 10,
            "f_history_model_selection_epochs": 10,
            "model_selection_batch_size": 200,
            "learning_rate": 0.001,
            "critic_multiplier": 10.0,
            "f_learning_rate": 0.01,
            "client_optimizer": "sgd",
            "best_score": float("-inf"),
        })
        self.assertIn("dataset=abs", message)
        self.assertIn("seed=0", message)
        self.assertIn("simple_model_selection_epochs=10", message)
        self.assertIn("f_history_model_selection_epochs=10", message)
        self.assertIn("model_selection_batch_size=200", message)
        self.assertIn("learning_rate=0.001", message)
        self.assertIn("f_learning_rate=0.01", message)
        self.assertIn("best_score=-inf", message)
        self.assertIn("federated training never started", message)

    def test_append_round_csv_matches_full_rewrite_format(self):
        rows = [
            {
                "round": 0,
                "train_mse": 1.0,
                "val_mse": 1.1,
                "gmm_train_objective": 0.2,
                "gmm_val_objective": 0.3,
                "gmm_eval": 0.4,
                "finite": True,
                "diverged": False,
            },
            {
                "round": 1,
                "train_mse": 0.9,
                "val_mse": 1.0,
                "gmm_train_objective": 0.1,
                "gmm_val_objective": 0.2,
                "gmm_eval": 0.5,
                "finite": True,
                "diverged": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            append_dir = os.path.join(tmpdir, "append")
            rewrite_dir = os.path.join(tmpdir, "rewrite")
            os.makedirs(append_dir)
            os.makedirs(rewrite_dir)
            for row in rows:
                append_mse_by_round(append_dir, row)
            write_mse_by_round(rewrite_dir, rows)
            with open(os.path.join(append_dir, "mse_by_round.csv"), "r") as append_file:
                append_text = append_file.read()
            with open(os.path.join(rewrite_dir, "mse_by_round.csv"), "r") as rewrite_file:
                rewrite_text = rewrite_file.read()
            self.assertEqual(append_text, rewrite_text)

    def test_append_test_mse_csv_matches_full_rewrite_format(self):
        rows = [
            {"round": 0, "test_mse": 1.2, "finite": True, "diverged": False},
            {"round": 1, "test_mse": 1.0, "finite": True, "diverged": False},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            append_dir = os.path.join(tmpdir, "append")
            rewrite_dir = os.path.join(tmpdir, "rewrite")
            os.makedirs(append_dir)
            os.makedirs(rewrite_dir)
            for row in rows:
                append_test_mse_by_round(append_dir, row)
            write_test_mse_by_round(rewrite_dir, rows)
            with open(os.path.join(append_dir, "test_mse_by_round.csv"), "r") as append_file:
                append_text = append_file.read()
            with open(os.path.join(rewrite_dir, "test_mse_by_round.csv"), "r") as rewrite_file:
                rewrite_text = rewrite_file.read()
            self.assertEqual(append_text, rewrite_text)


class EqualClientMetricsTest(unittest.TestCase):
    def test_structural_mse_equal_client_vs_sample_weighted(self):
        # client 0 holds 4 rows, client 1 holds 2: equal-client and
        # sample-weighted must diverge, by hand:
        #   client0 sq-errors [0,1,4,9] -> mean 3.5, n=4
        #   client1 sq-errors [16,25]   -> mean 20.5, n=2
        #   equal_client = (3.5+20.5)/2 = 12.0
        #   sample_weighted = (4*3.5+2*20.5)/6 = 55/6
        prediction = [1, 2, 3, 4, 5, 6]
        true_g = [1, 1, 1, 1, 1, 1]
        client_id = [0, 0, 0, 0, 1, 1]
        equal_client, sample_weighted, per_client = equal_client_structural_mse(
            prediction, true_g, client_id
        )
        self.assertAlmostEqual(equal_client, 12.0)
        self.assertAlmostEqual(sample_weighted, 55.0 / 6.0)
        self.assertEqual(per_client[0]["n"], 4)
        self.assertEqual(per_client[1]["n"], 2)
        self.assertAlmostEqual(per_client[0]["value"], 3.5)
        self.assertAlmostEqual(per_client[1]["value"], 20.5)

    def test_structural_mse_matches_pooled_when_clients_balanced(self):
        # Equal client sizes: sample-weighted must equal the plain pooled
        # mean (they are mathematically identical in that case), and
        # equal-client need not.
        prediction = [1, 2, 3, 4, 5, 6]
        true_g = [1, 1, 1, 1, 1, 1]
        client_id = [0, 0, 0, 1, 1, 1]
        pooled = structural_mse_from_predictions(prediction, true_g)
        _, sample_weighted, _ = equal_client_structural_mse(prediction, true_g, client_id)
        self.assertAlmostEqual(sample_weighted, pooled)

    def test_structural_mse_rejects_missing_client_id(self):
        with self.assertRaises(ValueError):
            equal_client_structural_mse([1, 2], [1, 1], None)

    def test_moment_violation_equal_client_vs_sample_weighted(self):
        # f*(y-g) = [2,2,2,2,3,3] with client0=rows[0:4] (n=4, mean=2,
        # squared=4) and client1=rows[4:6] (n=2, mean=3, squared=9).
        #   equal_client = (4+9)/2 = 6.5
        #   sample_weighted = (4*4+2*9)/6 = 34/6
        f_values = [2, 2, 2, 2, 3, 3]
        y_values = [1, 1, 1, 1, 1, 1]
        g_values = [0, 0, 0, 0, 0, 0]
        client_id = [0, 0, 0, 0, 1, 1]
        equal_client, sample_weighted, per_client = equal_client_moment_violation(
            f_values, y_values, g_values, client_id
        )
        self.assertAlmostEqual(equal_client, 6.5)
        self.assertAlmostEqual(sample_weighted, 34.0 / 6.0)
        self.assertAlmostEqual(per_client[0]["value"], 4.0)
        self.assertAlmostEqual(per_client[1]["value"], 9.0)

    def test_moment_violation_sign_cancellation_within_client(self):
        # Positive and negative residuals within the same client cancel in
        # the per-client mean before squaring -- this is exactly the
        # behavior metric_policy.md specifies (mu_i first, then square).
        f_values = [1, -1, 1, -1]
        y_values = [1, 1, 1, 1]
        g_values = [0, 0, 0, 0]
        client_id = [0, 0, 0, 0]
        equal_client, sample_weighted, per_client = equal_client_moment_violation(
            f_values, y_values, g_values, client_id
        )
        self.assertAlmostEqual(equal_client, 0.0)
        self.assertAlmostEqual(sample_weighted, 0.0)

    def test_per_client_helper_drops_no_clients(self):
        values = [1.0, 2.0, 3.0, 4.0]
        client_id = [5, 5, 7, 9]
        _, _, per_client = per_client_equal_and_sample_weighted(values, client_id)
        self.assertEqual(set(per_client.keys()), {5, 7, 9})

    def test_write_per_client_metrics_union_of_clients(self):
        per_client_mse = {0: {"value": 1.5, "n": 4}, 1: {"value": 2.5, "n": 2}}
        per_client_moment = {0: {"value": 0.1, "n": 4}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_per_client_metrics(tmpdir, per_client_mse, per_client_moment)
            with open(path, newline="") as f:
                rows = list(__import__("csv").DictReader(f))
        self.assertEqual(len(rows), 2)
        by_client = {row["client_id"]: row for row in rows}
        self.assertEqual(by_client["0"]["moment_violation"], "0.1")
        self.assertEqual(by_client["1"]["moment_violation"], "")

    def test_config_checksum_deterministic_and_sensitive(self):
        config_a = {"b": 1, "a": 2}
        config_b = {"a": 2, "b": 1}
        config_c = {"a": 2, "b": 2}
        self.assertEqual(config_checksum(config_a), config_checksum(config_b))
        self.assertNotEqual(config_checksum(config_a), config_checksum(config_c))


class SeedSeparationEffectiveConfigTest(unittest.TestCase):
    def test_scenario_and_optimizer_seed_default_to_random_seed(self):
        args = SimpleNamespace(random_seed=7)
        config = get_effective_config(args)
        self.assertEqual(config["scenario_seed"], 7)
        self.assertEqual(config["optimizer_seed"], 7)
        self.assertEqual(config["seed_pair_id"], "")

    def test_scenario_and_optimizer_seed_are_independently_settable(self):
        args = SimpleNamespace(random_seed=7, scenario_seed=101, optimizer_seed=1101, seed_pair_id="confirmatory_01")
        config = get_effective_config(args)
        self.assertEqual(config["scenario_seed"], 101)
        self.assertEqual(config["optimizer_seed"], 1101)
        self.assertEqual(config["seed_pair_id"], "confirmatory_01")
        self.assertIn("scenario_seed", EFFECTIVE_CONFIG_FIELDS)
        self.assertIn("optimizer_seed", EFFECTIVE_CONFIG_FIELDS)
        self.assertIn("seed_pair_id", EFFECTIVE_CONFIG_FIELDS)


def _full_campaign_config(run_id, **overrides):
    """A dataset='eicu_...' config with every provenance field populated."""
    config = {
        "dataset": "eicu_semisynth_offhours_v2_demo",
        "protocol_version": "eicu_study_a_v2_offhours",
        "run_id": run_id,
        "role": "confirmatory",
        "scenario_name": "linear_scenario_seed101",
        "g0": "linear",
        "alignment_label": "primary_extension",
        "primary_selection_metric": "equal_client_validation_mse",
        "selection_source": "validation_only",
        "scenario_scope": "demo",
        "study_claim": "semi_synthetic_benchmark_no_clinical_claim",
    }
    config.update(overrides)
    return config


class WriteEffectiveConfigProvenanceGuardTest(unittest.TestCase):
    def test_eicu_campaign_run_with_all_fields_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            path = write_effective_config(run_dir, _full_campaign_config("ok"))
            self.assertTrue(os.path.exists(path))
            with open(path) as handle:
                written = json.load(handle)
            self.assertEqual(written["role"], "confirmatory")

    def test_eicu_campaign_run_missing_one_field_raises_with_field_name_and_run_dir(self):
        config = _full_campaign_config("bad")
        config["alignment_label"] = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            with self.assertRaises(ValueError) as ctx:
                write_effective_config(run_dir, config)
            message = str(ctx.exception)
            self.assertIn("alignment_label", message)
            self.assertIn(run_dir, message)
            # File must not have been written on failure.
            self.assertFalse(os.path.exists(os.path.join(run_dir, "effective_config.json")))

    def test_eicu_campaign_run_missing_field_entirely_raises(self):
        config = _full_campaign_config("bad2")
        del config["scenario_name"]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            with self.assertRaises(ValueError) as ctx:
                write_effective_config(run_dir, config)
            self.assertIn("scenario_name", str(ctx.exception))

    def test_eicu_campaign_run_with_none_field_raises(self):
        config = _full_campaign_config("bad3")
        config["study_claim"] = None
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            with self.assertRaises(ValueError) as ctx:
                write_effective_config(run_dir, config)
            self.assertIn("study_claim", str(ctx.exception))

    def test_error_names_all_missing_fields_at_once(self):
        config = _full_campaign_config("bad4")
        config["g0"] = ""
        config["role"] = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                write_effective_config(os.path.join(tmpdir, "run"), config)
            message = str(ctx.exception)
            self.assertIn("g0", message)
            self.assertIn("role", message)

    def test_non_eicu_dataset_with_protocol_version_is_unaffected(self):
        # Many non-eICU campaign families in this repo (rerun_protocol_v1,
        # highdim_coauthor_protocol_v1, sine_fedogda_tuning, ...) set a
        # non-empty protocol_version but never populate
        # CAMPAIGN_PROVENANCE_FIELDS -- this guard must not start rejecting
        # their still-resumable manifests.
        config = {
            "dataset": "abs",
            "protocol_version": "rerun_protocol_v1",
            "run_id": "smoke",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            path = write_effective_config(run_dir, config)
            self.assertTrue(os.path.exists(path))

    def test_eicu_dataset_without_protocol_version_is_unaffected(self):
        # A non-campaign eICU debugging run (empty/absent protocol_version)
        # must keep working exactly as before.
        config = {"dataset": "eicu_semisynth", "run_id": "debug"}
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            path = write_effective_config(run_dir, config)
            self.assertTrue(os.path.exists(path))

    def test_config_missing_dataset_key_entirely_is_unaffected(self):
        config = {"protocol_version": "eicu_study_a_v2_offhours", "run_id": "no_dataset_key"}
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            path = write_effective_config(run_dir, config)
            self.assertTrue(os.path.exists(path))

    def test_campaign_provenance_fields_matches_reconcile_script(self):
        # scripts/reconcile_eicu_study_a_artifacts.py.PROVENANCE_FIELDS must
        # stay identical to this tuple, since both encode the same incident.
        import importlib.util

        script_path = os.path.join(
            REPO_ROOT, "scripts", "reconcile_eicu_study_a_artifacts.py"
        )
        spec = importlib.util.spec_from_file_location(
            "reconcile_eicu_study_a_artifacts", script_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(tuple(CAMPAIGN_PROVENANCE_FIELDS), tuple(module.PROVENANCE_FIELDS))


class CompactPredictionArtifactTest(unittest.TestCase):
    """save_predictions_npz_compact (closeout plan Phase 1 SS4.4): must drop
    the full test-input tensor (the ~10 GiB-across-the-campaign cost) while
    keeping every retained field numerically identical to the full writer's
    output, so a compact copy is never a silent approximation."""

    def test_compact_omits_x_but_matches_full_writer_on_retained_fields(self):
        import numpy as np

        x = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
        split = SimpleNamespace(x=x, g=np.asarray([[1.0], [2.0], [3.0], [4.0]]))
        best = np.asarray([[1.1], [2.1], [3.1], [4.1]])
        final = np.asarray([[1.2], [2.2], [3.2], [4.2]])
        metadata = {
            "dataset": "cifar10_x", "algorithm": "fedgda_d", "variant": "fedgda_d",
            "random_seed": 0, "run_id": "fixture_run",
        }
        with tempfile.TemporaryDirectory() as run_dir:
            save_predictions_npz(run_dir, split, best, final, metadata)
            compact_path = save_predictions_npz_compact(run_dir, split, best, final, metadata)
            with np.load(compact_path) as saved:
                self.assertNotIn("x", saved.files)
                self.assertEqual(saved["schema_version"].item(), 1)
                np.testing.assert_array_equal(saved["sample_id"], np.arange(4))
                np.testing.assert_array_equal(saved["true_g"], split.g)
                self.assertEqual(str(saved["dataset"]), "cifar10_x")
            result = verify_compact_predictions_numerically_equal(run_dir)
            self.assertEqual(set(result["verified_fields"]), {
                "best_validation_prediction", "final_prediction", "true_g",
            })

    def test_filename_override_lets_the_compact_schema_replace_predictions_npz(self):
        # fedavg_api.py's compact_predictions_only path (closeout plan Phase
        # 1 SS4.4) writes the compact schema AS predictions.npz -- there is
        # no separate full artifact for such a run to be additive to.
        import numpy as np

        x = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
        split = SimpleNamespace(x=x, g=np.asarray([[1.0], [2.0]]))
        best = np.asarray([[1.1], [2.1]])
        final = np.asarray([[1.2], [2.2]])
        metadata = {"dataset": "femnist_z", "random_seed": 0, "run_id": "fixture_run"}
        with tempfile.TemporaryDirectory() as run_dir:
            path = save_predictions_npz_compact(
                run_dir, split, best, final, metadata, filename="predictions.npz",
            )
            self.assertEqual(os.path.basename(path), "predictions.npz")
            with np.load(path) as saved:
                self.assertNotIn("x", saved.files)
                self.assertEqual(saved["schema_version"].item(), 1)

    def test_low_dim_compact_uses_sorted_scalar_coordinate(self):
        import numpy as np

        # Unsorted on purpose -- both writers must sort by x identically.
        x = np.asarray([3.0, 1.0, 2.0])
        split = SimpleNamespace(x=x, g=np.asarray([30.0, 10.0, 20.0]))
        best = np.asarray([31.0, 11.0, 21.0])
        final = np.asarray([32.0, 12.0, 22.0])
        metadata = {"dataset": "abs", "random_seed": 0, "run_id": "fixture"}
        with tempfile.TemporaryDirectory() as run_dir:
            save_predictions_npz(run_dir, split, best, final, metadata)
            compact_path = save_predictions_npz_compact(run_dir, split, best, final, metadata)
            with np.load(compact_path) as saved:
                np.testing.assert_array_equal(saved["sample_coordinate"], [1.0, 2.0, 3.0])
                np.testing.assert_array_equal(saved["true_g"], [10.0, 20.0, 30.0])
            verify_compact_predictions_numerically_equal(run_dir)  # must not raise

    def test_tampered_compact_copy_is_rejected(self):
        import numpy as np

        x = np.asarray([1.0, 2.0, 3.0])
        split = SimpleNamespace(x=x, g=np.asarray([1.0, 2.0, 3.0]))
        best = np.asarray([1.0, 2.0, 3.0])
        final = np.asarray([1.0, 2.0, 3.0])
        metadata = {"random_seed": 0, "run_id": "fixture"}
        with tempfile.TemporaryDirectory() as run_dir:
            save_predictions_npz(run_dir, split, best, final, metadata)
            compact_path = save_predictions_npz_compact(run_dir, split, best, final, metadata)
            payload = dict(np.load(compact_path))
            payload["true_g"] = payload["true_g"] + 1.0  # corrupt one field
            np.savez(compact_path, **payload)
            with self.assertRaisesRegex(ValueError, "true_g"):
                verify_compact_predictions_numerically_equal(run_dir)


class WritePretrainingFailureArtifactTest(unittest.TestCase):
    """write_pretraining_failure_artifact (closeout plan Phase 1 SS4.2): the
    only artifact that may classify a run as terminal_pretraining_ineligible,
    so its schema must be exactly what run_manifest.py's
    validate_pretraining_failure_artifact expects."""

    def test_writes_expected_schema_and_derives_first_nonfinite_epoch(self):
        diagnostics = [
            {"epoch": 0, "epsilon_dev_finite": True, "f_of_z_dev_finite": True},
            {"epoch": 20, "epsilon_dev_finite": False, "f_of_z_dev_finite": True},
            {"epoch": 40, "epsilon_dev_finite": False, "f_of_z_dev_finite": False},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir)
            path = write_pretraining_failure_artifact(
                run_dir,
                run_id="fixture_run",
                effective_config={"dataset": "femnist_z", "run_id": "fixture_run"},
                per_epoch_diagnostics=diagnostics,
                best_score=float("-inf"),
                terminal_reason="No valid model-selection candidate was selected",
                traceback_text="Traceback (most recent call last):\n  ...\nRuntimeError: x",
                stdout_path=None,
                stderr_path=None,
                hash_bundle_id="bundle-123",
            )
            self.assertTrue(os.path.exists(path))
            payload = json.loads(open(path).read())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["run_id"], "fixture_run")
        self.assertEqual(payload["failure_phase"], "model_selection")
        self.assertEqual(payload["federated_rounds_started"], 0)
        self.assertEqual(payload["model_selection_epochs_attempted"], 3)
        # best_score=-inf is not JSON-finite, so it must be normalized to null,
        # never written as a literal -Infinity token.
        self.assertIsNone(payload["best_model_selection_score"])
        self.assertEqual(payload["first_nonfinite_epoch"]["epoch"], 20)
        self.assertEqual(payload["first_nonfinite_epoch"]["components"], ["epsilon_dev_finite"])
        self.assertEqual(payload["hash_bundle_id"], "bundle-123")
        # "bundle-123" is not a real file relative to CWD -- the content
        # hash can't be computed and must come back None, not raise or
        # silently omit the key.
        self.assertIsNone(payload["hash_bundle_sha256"])
        self.assertIsNotNone(payload["effective_config_checksum"])
        self.assertEqual(
            payload["effective_config_checksum"],
            config_checksum({"dataset": "femnist_z", "run_id": "fixture_run"}),
        )

    def test_hash_bundle_sha256_computed_from_the_real_bundle_file(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir)
            bundle_path = os.path.join(tmpdir, "hash_bundle.json")
            with open(bundle_path, "w") as f:
                f.write('[{"path": "x", "sha256": "y"}]')
            path = write_pretraining_failure_artifact(
                run_dir, run_id="fixture_run", effective_config={"run_id": "fixture_run"},
                per_epoch_diagnostics=[], best_score=None,
                terminal_reason="reason", traceback_text="tb",
                hash_bundle_id=bundle_path,
            )
            payload = json.loads(open(path).read())
            with open(bundle_path, "rb") as f:
                expected = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(payload["hash_bundle_sha256"], expected)

    def test_hash_bundle_sha256_resolves_correctly_after_main_py_chdirs(self):
        # main.py chdir()s to its own file's directory near the top of the
        # script, well before its except-block calls this function -- a
        # REPO_ROOT-relative hash_bundle_id (the convention every launcher
        # uses) must still resolve correctly, not silently look in the
        # wrong directory and come back None. Reproduces the exact bug
        # found via a real GPU certification run (2026-08-28): every
        # produced pretraining_failure.json had hash_bundle_sha256=null.
        import hashlib
        import shutil
        from fedgmm.sp_decentralized_mnist_lr_example.experiment_utils import REPO_ROOT

        tmp_in_repo = tempfile.mkdtemp(dir=REPO_ROOT)
        self.addCleanup(shutil.rmtree, tmp_in_repo, ignore_errors=True)
        bundle_path = os.path.join(tmp_in_repo, "hash_bundle.json")
        with open(bundle_path, "w") as f:
            f.write('[{"path": "x", "sha256": "y"}]')
        relative_hash_bundle_id = os.path.relpath(bundle_path, REPO_ROOT)

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    run_dir = os.path.join(tmpdir, "run")
                    os.makedirs(run_dir)
                    path = write_pretraining_failure_artifact(
                        run_dir, run_id="fixture_run", effective_config={"run_id": "fixture_run"},
                        per_epoch_diagnostics=[], best_score=None,
                        terminal_reason="reason", traceback_text="tb",
                        hash_bundle_id=relative_hash_bundle_id,
                    )
                    payload = json.loads(open(path).read())
            finally:
                os.chdir(original_cwd)
        with open(bundle_path, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        self.assertIsNotNone(payload["hash_bundle_sha256"])
        self.assertEqual(payload["hash_bundle_sha256"], expected)

    def test_all_finite_epochs_report_no_first_nonfinite_epoch(self):
        diagnostics = [{"epoch": 0, "epsilon_dev_finite": True, "f_of_z_dev_finite": True}]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir)
            path = write_pretraining_failure_artifact(
                run_dir, run_id="fixture_run", effective_config={"run_id": "fixture_run"},
                per_epoch_diagnostics=diagnostics, best_score=1.5,
                terminal_reason="reason", traceback_text="tb",
            )
            payload = json.loads(open(path).read())
        self.assertIsNone(payload["first_nonfinite_epoch"])
        self.assertEqual(payload["best_model_selection_score"], 1.5)

    def test_stdout_stderr_hashes_recorded_when_paths_given(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = os.path.join(tmpdir, "stdout.log")
            with open(stdout_path, "w") as handle:
                handle.write("hello")
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir)
            path = write_pretraining_failure_artifact(
                run_dir, run_id="fixture_run", effective_config={"run_id": "fixture_run"},
                per_epoch_diagnostics=[], best_score=None,
                terminal_reason="reason", traceback_text="tb",
                stdout_path=stdout_path, stderr_path=os.path.join(tmpdir, "missing.log"),
            )
            payload = json.loads(open(path).read())
        self.assertIsNotNone(payload["stdout_sha256"])
        self.assertIsNone(payload["stderr_sha256"])  # missing file -> None, not an error


class ModelSelectionFailureExceptionTest(unittest.TestCase):
    def test_carries_diagnostics_and_message(self):
        exc = ModelSelectionFailure("bad candidate", {"per_epoch": [], "best_score": float("-inf")})
        self.assertEqual(str(exc), "bad candidate")
        self.assertEqual(exc.diagnostics["best_score"], float("-inf"))
        self.assertIsInstance(exc, RuntimeError)


if __name__ == "__main__":
    unittest.main()
