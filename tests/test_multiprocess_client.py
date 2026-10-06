import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE_ROOT)

from fedml.simulation.sp.fedavg import fedavg_api as api_module  # noqa: E402
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI  # noqa: E402
from fedml.simulation.sp.fedavg.multiprocess_client import (  # noqa: E402
    MultiprocessClientExecutor,
    _configure_cuda_determinism,
    _release_worker_cuda_cache,
    _to_cpu,
    _to_device,
)


class MultiprocessingRoutingTest(unittest.TestCase):
    def make_api(self, **overrides):
        api = object.__new__(FedAvgAPI)
        values = {
            "client_execution_mode": "multi_gpu_processes",
            "enable_multiprocessing": False,
            "multiprocessing_gpu_ids": [0, 1, 2],
            "multiprocessing_num_workers": 2,
            "client_num_per_round": 3,
        }
        values.update(overrides)
        api.args = SimpleNamespace(**values)
        api.model_trainer = object()
        return api

    def test_multi_gpu_routes_and_caps_workers(self):
        api = self.make_api()
        captured = {}

        class Executor:
            def __init__(self, trainer, args, gpu_ids):
                captured["gpu_ids"] = gpu_ids

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4), \
             mock.patch.object(api_module, "MultiprocessClientExecutor", Executor):
            self.assertIsInstance(api._create_client_executor(), Executor)
        self.assertEqual(captured["gpu_ids"], [0, 1])

    def test_one_worker_falls_back_to_sp(self):
        api = self.make_api(multiprocessing_num_workers=1)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4):
            self.assertIsNone(api._create_client_executor())
        self.assertEqual(api.client_execution_mode, "sp")

    def test_invalid_gpu_is_rejected(self):
        api = self.make_api(multiprocessing_gpu_ids=[0, 4])
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4):
            with self.assertRaisesRegex(ValueError, "Invalid multiprocessing_gpu_ids"):
                api._create_client_executor()

    def test_unknown_explicit_mode_is_rejected(self):
        api = self.make_api(client_execution_mode="threads")
        with self.assertRaisesRegex(ValueError, "Unknown client_execution_mode"):
            api._create_client_executor()


class ExecutorInvariantTest(unittest.TestCase):
    def test_run_restores_input_order(self):
        executor = object.__new__(MultiprocessClientExecutor)
        executor._closed = False
        executor.worker_count = 2
        executor.next_task_id = 0
        executor.processes = [mock.Mock(is_alive=lambda: True) for _ in range(2)]
        executor.task_queues = [mock.Mock(), mock.Mock()]
        executor.result_queue = mock.Mock()
        executor.result_queue.get.side_effect = [
            {"task_id": 1, "worker_id": 1, "payload": "b", "error": None},
            {"task_id": 0, "worker_id": 0, "payload": "a", "error": None},
            {"task_id": 2, "worker_id": 1, "payload": "c", "error": None},
        ]
        self.assertEqual(executor.run([{}, {}, {}]), ["a", "b", "c"])

    def test_nested_cpu_results_are_detached_and_bit_exact(self):
        source = {"g": [torch.tensor([1.0], requires_grad=True)],
                  "f": (torch.tensor([2.0]),)}
        result = _to_device(_to_cpu(source), torch.device("cpu"))
        self.assertTrue(torch.equal(result["g"][0], source["g"][0]))
        self.assertTrue(torch.equal(result["f"][0], source["f"][0]))
        self.assertFalse(result["g"][0].requires_grad)

    def test_worker_determinism_settings(self):
        old_deterministic = torch.backends.cudnn.deterministic
        old_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            _configure_cuda_determinism()
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.deterministic = old_deterministic
            torch.backends.cudnn.benchmark = old_benchmark

    def test_cache_release_synchronizes_before_emptying(self):
        calls = []
        device = torch.device("cuda", 0)
        with mock.patch.object(torch.cuda, "synchronize", side_effect=lambda value: calls.append(("sync", value))), \
             mock.patch.object(torch.cuda, "empty_cache", side_effect=lambda: calls.append(("empty", None))):
            _release_worker_cuda_cache(device)
        self.assertEqual(calls, [("sync", device), ("empty", None)])




    def test_epoch_replay_preserves_each_materialized_epoch(self):
        from fedml.simulation.sp.fedavg.multiprocess_client import _EpochReplay

        replay = _EpochReplay([["epoch-0"], ["epoch-1"]])
        self.assertEqual(list(replay), ["epoch-0"])
        self.assertEqual(list(replay), ["epoch-1"])
        with self.assertRaisesRegex(RuntimeError, "more epochs"):
            list(replay)

    def test_materialization_preserves_distinct_loader_passes(self):
        class ChangingLoader:
            def __init__(self):
                self.pass_number = 0

            def __iter__(self):
                value = self.pass_number
                self.pass_number += 1
                return iter([(torch.tensor([value]),)])

        epochs = FedAvgAPI._materialize_client_data(ChangingLoader(), 3)
        self.assertEqual([epoch[0][0].item() for epoch in epochs], [0, 1, 2])
class SingleGPURoutingTest(unittest.TestCase):
    def make_api(self, **overrides):
        api = object.__new__(FedAvgAPI)
        values = {
            "client_execution_mode": "multiprocessingsinglegpu",
            "enable_multiprocessing": False,
            "multiprocessing_gpu_ids": None,
            "multiprocessing_num_workers": 0,
            "multiprocessingsinglegpu_gpu_id": 0,
            "multiprocessingsinglegpu_num_workers": 6,
            "client_num_per_round": 4,
        }
        values.update(overrides)
        api.args = SimpleNamespace(**values)
        api.device = torch.device("cuda", 0)
        api.model_trainer = object()
        return api

    def test_same_gpu_routes_and_caps_workers(self):
        api = self.make_api()
        captured = {}

        class Executor:
            def __init__(self, trainer, args, gpu_id, worker_count):
                captured.update(gpu_id=gpu_id, worker_count=worker_count)

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4), \
             mock.patch.object(api_module, "SingleGPUMultiprocessClientExecutor", Executor):
            self.assertIsInstance(api._create_client_executor(), Executor)
        self.assertEqual(captured, {"gpu_id": 0, "worker_count": 4})

    def test_same_gpu_rejects_coordinator_mismatch(self):
        api = self.make_api(multiprocessingsinglegpu_gpu_id=1)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4):
            with self.assertRaisesRegex(ValueError, "must match"):
                api._create_client_executor()

    def test_same_gpu_one_worker_falls_back_to_sp(self):
        api = self.make_api(multiprocessingsinglegpu_num_workers=1)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "device_count", return_value=4):
            self.assertIsNone(api._create_client_executor())
        self.assertEqual(api.client_execution_mode, "sp")
if __name__ == "__main__":
    unittest.main()
