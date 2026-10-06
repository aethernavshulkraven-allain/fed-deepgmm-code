import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconcile_eicu_study_a_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("reconcile_eicu_artifacts", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReconcileEicuStudyAArtifactsTest(unittest.TestCase):
    def _fixture(self, root: Path):
        result_path = Path("eicu_semisynth/fedgda_s/seed_1101/run_1")
        run_dir = root / "results" / result_path
        run_dir.mkdir(parents=True)
        checksum = "a" * 64
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
        config = {
            "run_id": "run_1",
            "dataset": "eicu_semisynth",
            "variant": "fedgda_s",
            "random_seed": 1101,
            "scenario_checksum": checksum,
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

    def test_reconciles_and_archives_original_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run_dir, row = self._fixture(root)
            backup = root / "backup"
            summary = MODULE.reconcile_campaign(
                manifest, root / "results", backup
            )
            self.assertEqual(summary["runs_changed"], 1)
            self.assertEqual(summary["artifacts_archived"], 2)

            config = json.loads(
                (run_dir / "effective_config.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            for field in MODULE.PROVENANCE_FIELDS:
                self.assertEqual(config[field], row[field])
            expected_checksum = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self.assertEqual(metrics["config_checksum"], expected_checksum)
            self.assertEqual(metrics["alignment_label"], "primary_extension")
            archived = backup / row["result_path"]
            self.assertTrue((archived / "effective_config.json").is_file())
            self.assertTrue((archived / "metrics.json").is_file())

    def test_refuses_identity_mismatch_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, run_dir, _ = self._fixture(root)
            config_path = run_dir / "effective_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["run_id"] = "wrong"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaises(MODULE.ReconciliationError):
                MODULE.reconcile_campaign(
                    manifest, root / "results", root / "backup"
                )
            self.assertFalse((root / "backup").exists())


if __name__ == "__main__":
    unittest.main()
