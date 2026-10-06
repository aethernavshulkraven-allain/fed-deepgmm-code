import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch  # noqa: E402
import run_abs_smoke  # noqa: E402
import run_manifest  # noqa: E402
from fedml.simulation.sp.fedavg import fedavg_api as api_module  # noqa: E402
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI  # noqa: E402
from fedml.simulation.sp.fedavg.multiprocess_client import (  # noqa: E402
    MultiprocessClientExecutor,
)


def synthetic_executor(result_side_effect, alive=True):
    executor = object.__new__(MultiprocessClientExecutor)
    executor._closed = False
    executor.worker_count = 1
    executor.next_task_id = 0
    executor.processes = [mock.Mock(is_alive=lambda: alive)]
    executor.task_queues = [mock.Mock()]
    executor.result_queue = mock.Mock()
    executor.result_queue.get.side_effect = result_side_effect
    return executor


class FailurePropagationTest(unittest.TestCase):
    def test_client_exception_is_propagated(self):
        executor = synthetic_executor([
            {"task_id": 0, "worker_id": 0, "payload": None,
             "error": "client traceback"}
        ])
        with self.assertRaisesRegex(RuntimeError, "client traceback"):
            executor.run([{}])

    def test_worker_startup_exception_is_propagated(self):
        executor = synthetic_executor([
            {"task_id": None, "worker_id": 0, "payload": None,
             "error": "startup traceback"}
        ])
        with self.assertRaisesRegex(RuntimeError, "startup traceback"):
            executor.run([{}])

    def test_dead_worker_is_detected_after_timeout(self):
        executor = synthetic_executor([queue.Empty], alive=False)
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            executor.run([{}])

    def test_run_after_close_is_rejected(self):
        executor = synthetic_executor([])
        executor._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executor.run([{}])

    def test_close_is_idempotent_and_terminates_stuck_worker(self):
        executor = synthetic_executor([])
        process = mock.Mock()
        process.is_alive.side_effect = [True, False]
        executor.processes = [process]
        task_queue = mock.Mock()
        executor.task_queues = [task_queue]
        executor.result_queue = mock.Mock()

        executor.close()
        executor.close()

        task_queue.put_nowait.assert_called_once_with(None)
        process.terminate.assert_called_once_with()
        task_queue.close.assert_called_once_with()
        executor.result_queue.close.assert_called_once_with()


class CompatibilityRoutingTest(unittest.TestCase):
    def make_api(self, **overrides):
        values = {
            "client_execution_mode": None,
            "enable_multiprocessing": False,
            "multiprocessing_gpu_ids": "0,1,2",
            "multiprocessing_num_workers": 2,
            "client_num_per_round": 2,
        }
        values.update(overrides)
        api = object.__new__(FedAvgAPI)
        api.args = SimpleNamespace(**values)
        api.model_trainer = object()
        return api

    def test_legacy_enable_flag_selects_multi_gpu(self):
        api = self.make_api(enable_multiprocessing=True)
        captured = {}

        class Executor:
            def __init__(self, trainer, args, gpu_ids):
                captured["gpu_ids"] = gpu_ids

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4), \
             mock.patch.object(api_module, "MultiprocessClientExecutor", Executor):
            self.assertIsInstance(api._create_client_executor(), Executor)
        self.assertEqual(api.client_execution_mode, "multi_gpu_processes")
        self.assertEqual(captured["gpu_ids"], [0, 1])

    def test_no_cuda_falls_back_to_sp(self):
        api = self.make_api(enable_multiprocessing=True)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(api._create_client_executor())
        self.assertEqual(api.client_execution_mode, "sp")


class ConfigurationPropagationTest(unittest.TestCase):
    def test_abs_smoke_yaml_contains_same_gpu_mode(self):
        config = {
            "random_seed": 0, "run_id": "r", "output_dir": "/tmp/out",
            "dataset": "abs", "data_cache_dir": "data",
            "partition_method": "hetero", "partition_alpha": 0.5,
            "model": "lr", "federated_optimizer": "FedAvg",
            "client_id_list": "[]", "client_num_in_total": 2,
            "client_num_per_round": 2, "comm_round": 1, "epochs": 1,
            "batch_size": 0, "client_optimizer": "sgd",
            "learning_rate": 0.001, "weight_decay": 0.1,
            "critic_multiplier": 10.0, "server_learning_rate": 1.5,
            "gradient_clip_norm": 1.0, "frequency_of_the_test": 1,
            "using_gpu": True, "gpu_id": 0, "variant": "fedgda_d",
            "client_execution_mode": "multiprocessingsinglegpu",
            "multiprocessingsinglegpu_num_workers": 2,
            "multiprocessingsinglegpu_gpu_id": 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            run_abs_smoke.write_config(path, config)
            text = Path(path).read_text()
        self.assertIn('client_execution_mode: "multiprocessingsinglegpu"', text)
        self.assertIn("multiprocessingsinglegpu_num_workers: 2", text)
        self.assertIn("multiprocessingsinglegpu_gpu_id: 0", text)
        self.assertIn('backend: "sp"', text)

    def test_manifest_yaml_contains_multi_gpu_mode(self):
        config = {
            "client_execution_mode": "multi_gpu_processes",
            "enable_multiprocessing": False,
            "multiprocessing_num_workers": 2,
            "multiprocessing_gpu_ids": "0,1",
            "multiprocessingsinglegpu_num_workers": 2,
            "multiprocessingsinglegpu_gpu_id": 0,
        }
        # The writer has a broad required schema; patch a known-valid config
        # produced by the existing seed-field test helper instead of duplicating it.
        row = {
            "run_id": "r1", "dataset": "eicu_semisynth",
            "method": "fedgda_s", "seed": "1",
            "client_num_in_total": "2", "client_num_per_round": "2",
            "comm_round": "1", "epochs": "1", "batch_size": "2",
            "client_optimizer": "sgd", "learning_rate": "0.001",
            "weight_decay": "0.01", "partition_alpha": "0.0",
            **{key: str(value) for key, value in config.items()},
        }
        built = run_manifest.build_config(
            row,
            output_root=Path("/tmp/out"), gpu_id=0,
            default_learning_rate=None, default_weight_decay=None,
            override_comm_round=None, override_epochs=None,
            override_simple_model_selection_epochs=None,
            override_f_history_model_selection_epochs=None,
            override_model_selection_batch_size=None,
            override_model_selection_max_samples=None,
            override_skip_model_selection=None, override_skip_gmm_eval=None,
            override_auxiliary_regression=None,
            override_auxiliary_regression_epochs=None,
            override_append_round_csv=None,
            override_periodic_checkpoint_interval=None,
            override_dataloader_num_workers=None,
            override_dataloader_pin_memory=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            run_manifest.write_config(path, built)
            text = path.read_text()
        self.assertIn('client_execution_mode: "multi_gpu_processes"', text)
        self.assertIn("multiprocessing_num_workers: 2", text)
        self.assertIn('multiprocessing_gpu_ids: "0,1"', text)
        self.assertIn('backend: "sp"', text)


if __name__ == "__main__":
    unittest.main()
