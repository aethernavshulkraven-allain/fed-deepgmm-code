"""Tests for the --allow-repair gate on scripts/reconcile_eicu_study_a_artifacts.py.

This file is self-contained (does not import from
test_reconcile_eicu_study_a_artifacts.py) so it has no dependency on that
file's fixtures while another agent may be editing nearby files concurrently.

Covers:
  * the script refuses to modify anything when invoked without
    --allow-repair (and without --dry-run), and explains why;
  * --dry-run keeps working without --allow-repair (existing behavior is
    preserved -- a preview never modifies anything);
  * --allow-repair permits the existing repair behavior to proceed;
  * the pre-existing identity/checksum assertions in _assert_identity still
    fire even once --allow-repair is passed -- the new flag is an additional
    gate, not a replacement for those guards.
"""

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconcile_eicu_study_a_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconcile_eicu_artifacts_repair_gate", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReconcileAllowRepairGateTest(unittest.TestCase):
    def _fixture(self, root: Path, *, scenario_checksum_mismatch: bool = False):
        result_path = Path("eicu_semisynth/fedgda_s/seed_1101/run_1")
        run_dir = root / "results" / result_path
        run_dir.mkdir(parents=True)
        checksum = "b" * 64
        row = {
            "run_id": "run_1",
            "training_scope": "federated",
            "dataset": "eicu_semisynth",
            "method": "fedgda_s",
            "seed": "1101",
            "scenario_seed": "101",
            "scenario_checksum": checksum,
            "result_path": str(result_path),
            "role": "confirmatory",
            "scenario_name": "linear_scenario_seed101",
            "g0": "linear",
            "alignment_label": "primary_extension",
            "primary_selection_metric": "equal_client_validation_mse",
            "selection_source": "validation_only",
            "scenario_scope": "demo",
            "study_claim": "extension_no_published_target",
        }
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

        config_checksum = "mismatched-checksum" if scenario_checksum_mismatch else checksum
        config = {
            "run_id": "run_1",
            "dataset": "eicu_semisynth",
            "variant": "fedgda_s",
            "random_seed": 1101,
            "scenario_checksum": config_checksum,
        }
        metrics = {
            "run_id": "run_1",
            "method": "fedgda_s",
            "scenario_seed": 101,
            "scenario_checksum": checksum,
            "alignment_label": "",
            "selection_metric": "equal_client_validation_mse",
            "selection_source": "validation_only",
            "is_primary": True,
            "config_checksum": "old",
        }
        (run_dir / "effective_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        return manifest, run_dir, row

    def _run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = MODULE.main(argv)
        return code, out.getvalue()

    def test_refuses_without_allow_repair_and_leaves_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run_dir, _row = self._fixture(root)
            config_path = run_dir / "effective_config.json"
            metrics_path = run_dir / "metrics.json"
            config_before = config_path.read_bytes()
            metrics_before = metrics_path.read_bytes()
            backup = root / "backup"

            code, out = self._run_main(
                [
                    "--manifest",
                    str(manifest),
                    "--results-root",
                    str(root / "results"),
                    "--backup-root",
                    str(backup),
                ]
            )

            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertIn("error", payload)
            self.assertIn("--allow-repair", payload["error"])
            self.assertIn("write-time guard", payload["error"])
            self.assertEqual(config_path.read_bytes(), config_before)
            self.assertEqual(metrics_path.read_bytes(), metrics_before)
            self.assertFalse(backup.exists())

    def test_dry_run_without_allow_repair_still_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run_dir, _row = self._fixture(root)
            config_path = run_dir / "effective_config.json"
            metrics_path = run_dir / "metrics.json"
            config_before = config_path.read_bytes()
            metrics_before = metrics_path.read_bytes()
            backup = root / "backup"

            code, out = self._run_main(
                [
                    "--manifest",
                    str(manifest),
                    "--results-root",
                    str(root / "results"),
                    "--backup-root",
                    str(backup),
                    "--dry-run",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(out)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["runs_changed"], 1)
            self.assertEqual(summary["artifacts_archived"], 0)
            self.assertEqual(config_path.read_bytes(), config_before)
            self.assertEqual(metrics_path.read_bytes(), metrics_before)
            self.assertFalse(backup.exists())

    def test_allow_repair_permits_the_existing_repair_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run_dir, row = self._fixture(root)
            backup = root / "backup"

            code, out = self._run_main(
                [
                    "--manifest",
                    str(manifest),
                    "--results-root",
                    str(root / "results"),
                    "--backup-root",
                    str(backup),
                    "--allow-repair",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(out)
            self.assertEqual(summary["runs_changed"], 1)
            self.assertEqual(summary["artifacts_archived"], 2)

            config = json.loads(
                (run_dir / "effective_config.json").read_text(encoding="utf-8")
            )
            for field in MODULE.PROVENANCE_FIELDS:
                self.assertEqual(config[field], row[field])
            archived = backup / row["result_path"]
            self.assertTrue((archived / "effective_config.json").is_file())
            self.assertTrue((archived / "metrics.json").is_file())

    def test_identity_checksum_guard_still_fires_with_allow_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run_dir, _row = self._fixture(
                root, scenario_checksum_mismatch=True
            )
            backup = root / "backup"

            code, out = self._run_main(
                [
                    "--manifest",
                    str(manifest),
                    "--results-root",
                    str(root / "results"),
                    "--backup-root",
                    str(backup),
                    "--allow-repair",
                ]
            )

            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertIn("scenario checksum mismatch", payload["error"])
            self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
