"""Concurrent client-local updates on CUDA streams within one process."""

import copy
import logging
from concurrent.futures import ThreadPoolExecutor

import torch

from .client import Client
from .multiprocess_client import (
    _configure_cuda_determinism,
    _to_cpu,
    _to_device,
)


def _iter_task_waves(tasks, slot_count):
    """Yield ordered waves containing at most one task per reusable slot."""
    if slot_count < 1:
        raise ValueError("slot_count must be at least 1")
    indexed_tasks = list(enumerate(tasks))
    for start in range(0, len(indexed_tasks), slot_count):
        yield indexed_tasks[start:start + slot_count]


class SingleGPUClientExecutor:
    """Run isolated client trainers concurrently on streams of one GPU."""

    execution_mode = "single_gpu_streams"

    def __init__(self, trainer, args, device, slot_count):
        if device.type != "cuda":
            raise ValueError("SingleGPUClientExecutor requires a CUDA device")
        if slot_count < 2:
            raise ValueError("SingleGPUClientExecutor requires at least two slots")

        self.args = copy.deepcopy(args)
        self.device = device
        self.worker_count = int(slot_count)
        self._closed = False
        _configure_cuda_determinism()

        with torch.cuda.device(self.device):
            self.streams = [
                torch.cuda.Stream(device=self.device)
                for _ in range(self.worker_count)
            ]
            self.trainers = []
            for _ in range(self.worker_count):
                slot_trainer = copy.deepcopy(trainer)
                slot_trainer.g.to(self.device)
                slot_trainer.f.to(self.device)
                slot_trainer.g_optimizer.state.clear()
                slot_trainer.f_optimizer.state.clear()
                self.trainers.append(slot_trainer)
            torch.cuda.synchronize(self.device)

        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="fedgmm-cuda-stream",
        )
        logging.info(
            "Started single-GPU client executor with %d streams on %s",
            self.worker_count,
            self.device,
        )

    def run(self, tasks):
        if self._closed:
            raise RuntimeError("SingleGPUClientExecutor is closed")
        if not tasks:
            return []

        results = [None] * len(tasks)
        for wave in _iter_task_waves(tasks, self.worker_count):
            futures = []
            for slot_id, (result_index, task) in enumerate(wave):
                future = self.thread_pool.submit(
                    self._run_task, slot_id, task
                )
                futures.append((result_index, future))
            for result_index, future in futures:
                results[result_index] = future.result()
        return results

    def _run_task(self, slot_id, task):
        trainer = self.trainers[slot_id]
        stream = self.streams[slot_id]
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            client = Client(
                task["client_idx"],
                _to_device(task["train_data"], self.device),
                None,
                task["sample_number"],
                self.args,
                self.device,
                trainer,
            )
            phase = task["phase"]
            if phase == "primary":
                gmm_weights = client.train(
                    task["g_global"], task["f_global"]
                )
                payload = {"gmm": _to_cpu(gmm_weights)}
            elif phase == "correction":
                reference_kwargs = (
                    {"g_reference": task["g_reference"]} if "g_reference" in task else {}
                )
                if task["use_zeroth_order"]:
                    weights = client.train_zo(
                        task["g_global"], task["f_global"], **reference_kwargs
                    )
                else:
                    weights = client.train(
                        task["g_global"], task["f_global"], **reference_kwargs
                    )
                payload = {"gmm": _to_cpu(weights)}
            else:
                raise ValueError(f"Unknown client phase: {phase}")
        stream.synchronize()
        return payload

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.thread_pool.shutdown(wait=True, cancel_futures=False)
        for stream in self.streams:
            stream.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
