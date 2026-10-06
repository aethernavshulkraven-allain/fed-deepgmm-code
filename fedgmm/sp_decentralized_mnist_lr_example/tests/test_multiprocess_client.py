from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from fedml.simulation.sp.fedavg import fedavg_api as fedavg_api_module
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI
from fedml.simulation.sp.fedavg.multiprocess_client import (
    SingleGPUMultiprocessClientExecutor,
    _configure_cuda_determinism,
    _release_worker_cuda_cache,
    _single_gpu_worker_ids,
    _to_cpu,
    _to_device,
)
from fedml.simulation.sp.fedavg.single_gpu_client import (
    SingleGPUClientExecutor,
    _iter_task_waves,
)


def test_single_gpu_task_waves_are_bounded_and_ordered():
    waves = list(_iter_task_waves(list("abcdefg"), 3))

    assert waves == [
        [(0, "a"), (1, "b"), (2, "c")],
        [(3, "d"), (4, "e"), (5, "f")],
        [(6, "g")],
    ]


def test_single_gpu_multiprocess_worker_ids_share_one_gpu():
    assert _single_gpu_worker_ids(3, 4) == [3, 3, 3, 3]
    assert SingleGPUMultiprocessClientExecutor.execution_mode == (
        "multiprocessingsinglegpu"
    )


def test_single_gpu_multiprocess_worker_ids_reject_empty_pool():
    with pytest.raises(ValueError, match="at least 1"):
        _single_gpu_worker_ids(0, 0)


def test_coordinator_routes_single_gpu_multiprocessing(monkeypatch):
    captured = {}

    class CapturingExecutor:
        execution_mode = "multiprocessingsinglegpu"

        def __init__(self, trainer, args, gpu_id, worker_count):
            captured.update(
                trainer=trainer,
                args=args,
                gpu_id=gpu_id,
                worker_count=worker_count,
            )

    api = object.__new__(FedAvgAPI)
    api.args = SimpleNamespace(
        client_execution_mode="multiprocessingsinglegpu",
        enable_multiprocessing=False,
        multiprocessingsinglegpu_gpu_id=0,
        multiprocessingsinglegpu_num_workers=6,
        client_num_per_round=4,
    )
    api.device = torch.device("cuda", 0)
    api.model_trainer = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        fedavg_api_module,
        "SingleGPUMultiprocessClientExecutor",
        CapturingExecutor,
    )

    executor = api._create_client_executor()

    assert isinstance(executor, CapturingExecutor)
    assert api.client_execution_mode == "multiprocessingsinglegpu"
    assert captured == {
        "trainer": api.model_trainer,
        "args": api.args,
        "gpu_id": 0,
        "worker_count": 4,
    }


def test_single_gpu_multiprocessing_rejects_coordinator_mismatch(monkeypatch):
    api = object.__new__(FedAvgAPI)
    api.args = SimpleNamespace(
        client_execution_mode="multiprocessingsinglegpu",
        enable_multiprocessing=False,
        multiprocessingsinglegpu_gpu_id=1,
        multiprocessingsinglegpu_num_workers=2,
        client_num_per_round=2,
    )
    api.device = torch.device("cuda", 0)
    api.model_trainer = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    with pytest.raises(ValueError, match="must match"):
        api._create_client_executor()


def test_single_gpu_executor_restores_task_order():
    tasks = [
        {"phase": "correction", "value": value}
        for value in "abcdefg"
    ]
    executor = object.__new__(SingleGPUClientExecutor)
    executor._closed = False
    executor.worker_count = 3
    executor.thread_pool = ThreadPoolExecutor(max_workers=3)
    executor._run_task = lambda slot_id, task: (slot_id, task["value"])
    try:
        results = executor.run(tasks)
    finally:
        executor.thread_pool.shutdown(wait=True)

    assert [result[1] for result in results] == list("abcdefg")
    assert [result[0] for result in results] == [0, 1, 2, 0, 1, 2, 0]


def test_single_gpu_primary_returns_only_gmm_results():
    tasks = [
        {"phase": "primary", "value": value}
        for value in "abcd"
    ]
    executor = object.__new__(SingleGPUClientExecutor)
    executor._closed = False
    executor.worker_count = 2
    executor.thread_pool = ThreadPoolExecutor(max_workers=2)
    executor._run_task = lambda slot_id, task: {
        "gmm": (slot_id, task["value"])
    }
    try:
        results = executor.run(tasks)
    finally:
        executor.thread_pool.shutdown(wait=True)

    assert [result["gmm"][1] for result in results] == list("abcd")
    assert all(set(result) == {"gmm"} for result in results)


def test_single_gpu_executor_rejects_run_after_close():
    executor = object.__new__(SingleGPUClientExecutor)
    executor._closed = True

    try:
        executor.run(["client"])
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("closed executor accepted a task")


def test_worker_configures_deterministic_cudnn():
    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        _configure_cuda_determinism()

        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark


def test_worker_releases_cuda_cache_after_client_task(monkeypatch):
    calls = []
    device = torch.device("cuda", 0)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda value: calls.append(("sync", value))
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append(("empty", None))
    )

    _release_worker_cuda_cache(device)

    assert calls == [("sync", device), ("empty", None)]


def test_nested_tensor_device_conversion_preserves_structure():
    value = {
        "weights": [torch.tensor([1.0]), (torch.tensor([2.0]),)],
        "metadata": "client",
    }

    cpu_value = _to_cpu(value)
    converted = _to_device(cpu_value, torch.device("cpu"))

    assert isinstance(converted, dict)
    assert isinstance(converted["weights"], list)
    assert isinstance(converted["weights"][1], tuple)
    assert converted["metadata"] == "client"
    assert converted["weights"][0].device.type == "cpu"
    assert converted["weights"][1][0].item() == 2.0


def test_to_cpu_detaches_worker_results():
    tensor = torch.tensor([3.0], requires_grad=True)

    result = _to_cpu(SimpleNamespace(value=tensor).value)

    assert result.device.type == "cpu"
    assert not result.requires_grad
