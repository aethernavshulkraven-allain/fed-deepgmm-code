"""Tests for scripts/validate_real_image_abs_runs.py's predictions.npz check.

Closeout review finding: save_predictions_npz_compact()'s schema
(experiment_utils.py) deliberately omits the full test input tensor "x" to
avoid its ~10 GiB-scale write across an image campaign, but this validator
unconditionally required "x" -- so any real run using the compact schema
would be reported as invalid for missing a field it was never supposed to
have."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_real_image_abs_runs as validator  # noqa: E402


def _write_run(run_dir: Path, *, run_id: str, dataset: str, method: str, seed: int,
                compact: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_validation.pt").write_bytes(b"fake")
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"fake")
    (run_dir / "effective_config.json").write_text(json.dumps({
        "run_id": run_id, "dataset": dataset, "variant": method, "random_seed": seed,
        "selection_metric_source": "validation", "test_mse_used_for_selection": False,
        "compact_predictions_only": compact,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "best_validation_mse": 0.1, "best_validation_round": 1, "final_validation_mse": 0.1,
        "final_test_mse": 0.2, "test_mse_at_best_validation": 0.2, "runtime_seconds": 10.0,
        "selection_metric_source": "validation", "test_mse_used_for_selection": False,
    }))
    (run_dir / "mse_by_round.csv").write_text(
        "round,train_mse,val_mse\n0,0.1,0.1\n"
    )
    predictions = {
        "true_g": np.zeros(4), "best_validation_prediction": np.zeros(4),
        "final_prediction": np.zeros(4),
    }
    if not compact:
        predictions["x"] = np.zeros(4)
    np.savez(run_dir / "predictions.npz", **predictions)


class ValidateRealImageAbsRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _row(self, run_dir: Path, **overrides) -> dict:
        row = {
            "run_id": "r1", "dataset": "cifar10_x", "method": "fedgda_d", "seed": "0",
            "final_result_dir": str(run_dir.relative_to(REPO_ROOT)),
        }
        row.update(overrides)
        return row

    def test_full_schema_run_with_x_present_is_valid(self):
        run_dir = self.tmp / "full"
        _write_run(run_dir, run_id="r1", dataset="cifar10_x", method="fedgda_d", seed=0, compact=False)
        errors = validator.validate_run(self._row(run_dir))
        self.assertEqual(errors, [])

    def test_compact_schema_run_without_x_is_valid(self):
        run_dir = self.tmp / "compact"
        _write_run(run_dir, run_id="r1", dataset="cifar10_x", method="fedgda_d", seed=0, compact=True)
        errors = validator.validate_run(self._row(run_dir))
        self.assertEqual(errors, [])

    def test_full_schema_run_missing_x_is_still_rejected(self):
        # A run that does NOT opt into the compact schema but is somehow
        # missing x is a real defect, not the expected compact shape.
        run_dir = self.tmp / "broken"
        _write_run(run_dir, run_id="r1", dataset="cifar10_x", method="fedgda_d", seed=0, compact=False)
        np.savez(
            run_dir / "predictions.npz",
            true_g=np.zeros(4), best_validation_prediction=np.zeros(4),
            final_prediction=np.zeros(4),
        )
        errors = validator.validate_run(self._row(run_dir))
        self.assertTrue(any("missing x" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
