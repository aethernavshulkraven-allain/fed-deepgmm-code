"""Persistent CUDA processes for independent federated client updates."""

import atexit
import copy
import logging
import queue
import traceback

import torch
import torch.multiprocessing as multiprocessing

from .client import Client


def _configure_cuda_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


def _move_trainer_to_cpu(trainer):
    trainer.g.cpu()
    trainer.f.cpu()
    if trainer.reg_model is not None:
        trainer.reg_model.cpu()
    trainer.g_optimizer.state.clear()
    trainer.f_optimizer.state.clear()
    return trainer


def _release_worker_cuda_cache(device):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()



class _EpochReplay:
    """Replay coordinator-materialized batches one local epoch at a time."""

    def __init__(self, epochs):
        self.epochs = epochs
        self.next_epoch = 0

    def __iter__(self):
        if self.next_epoch >= len(self.epochs):
            raise RuntimeError("Client training requested more epochs than serialized")
        batches = self.epochs[self.next_epoch]
        self.next_epoch += 1
        return iter(batches)
def _execute_worker_task(task, trainer, args, device):
    client = Client(
        task["client_idx"],
        _EpochReplay(_to_device(task["gmm_train_epochs"], device)),
        None,
        task["sample_number"],
        args,
        device,
        trainer,
    )
    try:
        phase = task["phase"]
        if phase == "primary":
            payload = {
                "gmm": _to_cpu(client.train(task["g_global"], task["f_global"]))
            }
            if task.get("reg_global") is not None:
                client.local_training_data = _EpochReplay(
                    _to_device(task["reg_train_epochs"], device)
                )
                payload["regression"] = _to_cpu(
                    client.train_reg(task["reg_global"])
                )
            return payload
        if phase == "correction":
            train_method = client.train_zo if task["use_zeroth_order"] else client.train
            reference_kwargs = (
                {"g_reference": task["g_reference"]} if "g_reference" in task else {}
            )
            return {
                "gmm": _to_cpu(train_method(
                    task["g_global"], task["f_global"], **reference_kwargs
                ))
            }
        raise ValueError(f"Unknown client phase: {phase}")
    finally:
        del client
        _release_worker_cuda_cache(device)


def _worker_loop(worker_id, gpu_id, trainer, args, task_queue, result_queue):
    try:
        torch.cuda.set_device(gpu_id)
        _configure_cuda_determinism()
        device = torch.device("cuda", gpu_id)
        trainer = _move_trainer_to_cpu(trainer)
        while True:
            task = task_queue.get()
            if task is None:
                return
            task_id = task["task_id"]
            try:
                payload = _execute_worker_task(task, trainer, args, device)
                result_queue.put({"task_id": task_id, "worker_id": worker_id,
                                  "payload": payload, "error": None})
            except Exception:
                result_queue.put({"task_id": task_id, "worker_id": worker_id,
                                  "payload": None, "error": traceback.format_exc()})
    except Exception:
        result_queue.put({"task_id": None, "worker_id": worker_id,
                          "payload": None, "error": traceback.format_exc()})


class MultiprocessClientExecutor:
    """Run client updates in persistent processes and preserve task order."""

    execution_mode = "multi_gpu_processes"

    def __init__(self, trainer, args, gpu_ids):
        self.context = multiprocessing.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.task_queues = []
        self.processes = []
        self.next_task_id = 0
        self._closed = False
        worker_args = copy.copy(args)
        if hasattr(worker_args, "_fedgmm_runtime_profiler"):
            delattr(worker_args, "_fedgmm_runtime_profiler")
        trainer_template = _move_trainer_to_cpu(copy.deepcopy(trainer))
        trainer_template.args = worker_args
        for worker_id, gpu_id in enumerate(gpu_ids):
            task_queue = self.context.Queue(maxsize=1)
            process = self.context.Process(
                target=_worker_loop,
                args=(worker_id, gpu_id, copy.deepcopy(trainer_template),
                      copy.deepcopy(worker_args), task_queue, self.result_queue),
                daemon=True,
            )
            process.start()
            self.task_queues.append(task_queue)
            self.processes.append(process)
        self.worker_count = len(self.processes)
        atexit.register(self.close)
        logging.info("Started %d federated client workers on logical GPUs %s with PIDs %s",
                     self.worker_count, gpu_ids,
                     [process.pid for process in self.processes])

    def run(self, tasks):
        if self._closed:
            raise RuntimeError("Cannot run tasks on a closed client executor")
        if not tasks:
            return []
        pending = list(enumerate(tasks))
        active = {}
        results = [None] * len(tasks)
        for worker_id in range(min(self.worker_count, len(pending))):
            index, task = pending.pop(0)
            self._submit(worker_id, index, task, active)
        completed = 0
        while completed < len(tasks):
            try:
                result = self.result_queue.get(timeout=30)
            except queue.Empty:
                dead = [i for i, process in enumerate(self.processes)
                        if not process.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"Federated client workers exited unexpectedly: {dead}"
                    )
                continue
            if result["task_id"] is None:
                raise RuntimeError(
                    f"Client worker {result['worker_id']} failed during startup:\n"
                    f"{result['error']}"
                )
            result_index, worker_id = active.pop(result["task_id"])
            if result["error"] is not None:
                raise RuntimeError(
                    f"Client worker {worker_id} failed:\n{result['error']}"
                )
            results[result_index] = result["payload"]
            completed += 1
            if pending:
                index, task = pending.pop(0)
                self._submit(worker_id, index, task, active)
        return results

    def _submit(self, worker_id, result_index, task, active):
        task = dict(task)
        task_id = self.next_task_id
        self.next_task_id += 1
        task["task_id"] = task_id
        active[task_id] = (result_index, worker_id)
        self.task_queues[worker_id].put(task)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for task_queue in self.task_queues:
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for task_queue in self.task_queues:
            task_queue.close()
        self.result_queue.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()


def _single_gpu_worker_ids(gpu_id, worker_count):
    worker_count = int(worker_count)
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    return [int(gpu_id)] * worker_count


class SingleGPUMultiprocessClientExecutor(MultiprocessClientExecutor):
    execution_mode = "multiprocessingsinglegpu"

    def __init__(self, trainer, args, gpu_id, worker_count):
        self.gpu_id = int(gpu_id)
        super().__init__(trainer, args,
                         _single_gpu_worker_ids(self.gpu_id, worker_count))
