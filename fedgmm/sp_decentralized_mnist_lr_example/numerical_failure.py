"""Opt-in numerical stop evidence for staged experiment campaigns."""

import hashlib
import json
import os
from pathlib import Path


class NumericalFailure(FloatingPointError):
    """A measured non-finite training quantity, not a process/infra failure."""


def ensure_finite(values, enabled, phase):
    if not enabled:
        return
    import torch

    checks = [torch.isfinite(value.detach()).all() for value in values
              if isinstance(value, torch.Tensor)]
    if checks and not bool(torch.stack(checks).all().item()):
        raise NumericalFailure(f"Non-finite tensor in {phase}")


def clip_gradients(parameters, max_norm, enabled=False):
    import torch

    try:
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=enabled
        )
    except RuntimeError as exc:
        if enabled and "non-finite" in str(exc):
            raise NumericalFailure(f"Non-finite gradient norm: {exc}") from exc
        raise


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_failure(run_dir, args, reason, traceback_text):
    from experiment_utils import config_checksum

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "effective_config.json").read_text())
    evidence = {}
    paths = [run_dir / "effective_config.json", run_dir / "mse_by_round.csv"]
    paths.extend(sorted((run_dir / "checkpoints").glob("*.pt")))
    for name in ("FEDGMM_JOB_STDOUT_LOG", "FEDGMM_JOB_STDERR_LOG"):
        if os.environ.get(name):
            paths.append(Path(os.environ[name]))
    for path in paths:
        if path.is_file() and path.resolve().is_relative_to(run_dir.resolve()):
            evidence[str(path.relative_to(run_dir))] = file_sha256(path)
    payload = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "terminal_numerical_failure",
        "round": int(getattr(args, "_failure_round", -1)),
        "phase": str(getattr(args, "_failure_phase", "initialization")),
        "reason": str(reason),
        "traceback": traceback_text,
        "effective_config_checksum": config_checksum(config),
        "evidence_sha256": evidence,
    }
    target = run_dir / "training_failure.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return payload
