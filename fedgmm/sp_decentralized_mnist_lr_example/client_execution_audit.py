"""Opt-in observational input/order audit; never iterates a live loader.

Hash cached batches before training without changing tensors or consuming RNG.
The normal entry point is unaffected unless FEDGMM_CLIENT_EXECUTION_AUDIT=1.
"""

import functools
import hashlib
import json
from pathlib import Path
import time

import torch


def batch_digest(value):
    digest = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), list(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}:".encode())
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"none:")
        else:
            raise TypeError(f"Unsupported audit batch component: {type(item)}")

    update(value)
    return digest.hexdigest()


def audit_batch_layout(local_data):
    output = {}
    for client_id in sorted(local_data):
        batches = local_data[client_id]
        if not isinstance(batches, (list, tuple)):
            raise TypeError("Execution audit requires already materialized batches")
        output[str(client_id)] = [batch_digest(batch) for batch in batches]
    return output


def install_execution_audit():
    from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI

    if getattr(FedAvgAPI, "_execution_audit_installed", False):
        return
    FedAvgAPI._execution_audit_installed = True
    original_train = FedAvgAPI.train

    @functools.wraps(original_train)
    def audited_train(self):
        started = time.perf_counter()
        layout = audit_batch_layout(self.train_data_local_dict)
        root = Path(self.run_dir)
        with (root / "execution_batch_layout.json").open("x") as handle:
            json.dump(layout, handle, sort_keys=True)
        with (root / "execution_audit_metadata.json").open("x") as handle:
            json.dump({
                "layout_hash_seconds": time.perf_counter() - started,
                "actual_client_execution_mode": self.client_execution_mode,
                "worker_count": getattr(self.client_executor, "worker_count", 0),
                "worker_pids": [p.pid for p in getattr(self.client_executor, "processes", [])],
                "scope": "ordered cached input batches and coordinator phase order",
            }, handle, sort_keys=True)
        return original_train(self)

    FedAvgAPI.train = audited_train

    def wrap_phase(name, phase):
        original = getattr(FedAvgAPI, name)

        @functools.wraps(original)
        def audited_phase(self, client_indexes, *args, **kwargs):
            with (Path(self.run_dir) / "execution_order.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "round": int(self.args._failure_round), "phase": phase,
                    "client_ids": [int(index) for index in client_indexes],
                }, sort_keys=True) + "\n")
            return original(self, client_indexes, *args, **kwargs)

        setattr(FedAvgAPI, name, audited_phase)

    wrap_phase("_run_primary_client_updates", "primary")
    wrap_phase("_run_correction_client_updates", "correction")
