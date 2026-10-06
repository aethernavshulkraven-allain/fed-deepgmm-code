#!/usr/bin/env python3
"""Derive predictions_compact.npz from an existing predictions.npz.

Post-hoc derivation for already-completed runs (closeout plan Phase 1 SS4.4):
reads the full predictions.npz array already on disk, writes the compact
schema alongside it, and numerically verifies the two agree. Never deletes
or modifies the original predictions.npz. Does not touch checkpoints,
metrics, or round curves -- run-dir contents outside predictions*.npz are
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FEDGMM_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(FEDGMM_ROOT))
from experiment_utils import (  # noqa: E402
    COMPACT_PREDICTION_SCHEMA_VERSION,
    PREDICTION_OUTPUT_KEYS,
)


def derive(run_dir: Path) -> Path:
    full_path = run_dir / "predictions.npz"
    config_path = run_dir / "effective_config.json"
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    metadata = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
        metadata = {
            "dataset": config.get("dataset", ""),
            "algorithm": config.get("federated_optimizer", config.get("variant", "")),
            "variant": config.get("variant", ""),
            "random_seed": config.get("random_seed", 0),
            "run_id": config.get("run_id", ""),
        }
    with np.load(full_path) as full:
        x = full["x"]
        true_g = full["true_g"]
        best_pred = full[PREDICTION_OUTPUT_KEYS[0]]
        final_pred = full[PREDICTION_OUTPUT_KEYS[1]]
        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            indices = np.argsort(x.reshape(x.shape[0], -1)[:, 0])
            sample_coordinate = x.reshape(x.shape[0], -1)[:, 0][indices]
        else:
            indices = np.arange(x.shape[0])
            sample_coordinate = indices.astype(np.int64)
        compact_path = run_dir / "predictions_compact.npz"
        np.savez(
            compact_path,
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
    return compact_path


def verify(run_dir: Path) -> dict:
    from experiment_utils import verify_compact_predictions_numerically_equal

    return verify_compact_predictions_numerically_equal(str(run_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Write predictions_compact.npz without the numerical-equality check.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    try:
        compact_path = derive(run_dir)
        result = {"compact_path": str(compact_path)}
        if not args.skip_verify:
            result["verification"] = verify(run_dir)
    except (OSError, KeyError, ValueError) as exc:
        print(f"COMPACT DERIVATION FAILED: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
