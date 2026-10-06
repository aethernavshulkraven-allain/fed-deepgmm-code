import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_real_image_abs_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_real_image_abs_manifest", SCRIPT_PATH)
manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = manifest
SPEC.loader.exec_module(manifest)


class RealImageAbsManifestTest(unittest.TestCase):
    def test_matrix_is_six_scenarios_one_alpha_five_seeds_four_methods(self):
        rows = manifest.generate_rows()
        self.assertEqual(len(rows), 120)
        self.assertEqual({row["training_scope"] for row in rows}, {"federated"})
        self.assertEqual({row["alpha"] for row in rows}, {0.5})
        self.assertEqual({row["partition_alpha"] for row in rows}, {0.5})
        self.assertEqual({row["dataset"] for row in rows}, set(manifest.DATASETS))
        self.assertEqual({row["seed"] for row in rows}, set(manifest.SEEDS))
        self.assertEqual({row["method"] for row in rows}, set(manifest.FEDERATED_METHODS))
        self.assertEqual(len({row["run_id"] for row in rows}), 120)

        by_dataset = Counter(row["dataset"] for row in rows)
        self.assertEqual(set(by_dataset.values()), {20})

        by_method = Counter(row["method"] for row in rows)
        self.assertEqual(set(by_method.values()), {30})

    def test_deterministic_and_stochastic_settings_match_rerun_protocol_shape(self):
        rows = manifest.generate_rows()
        deterministic = [row for row in rows if row["method"].endswith("_d")]
        stochastic = [row for row in rows if row["method"].endswith("_s")]
        self.assertTrue(all(row["client_num_per_round"] == 1000 for row in deterministic))
        self.assertTrue(all(row["batch_size"] == 0 for row in deterministic))
        self.assertTrue(all(row["preflight_required"] is True for row in deterministic))
        self.assertTrue(all(row["client_num_per_round"] == 10 for row in stochastic))
        self.assertTrue(all(row["batch_size"] == 256 for row in stochastic))
        self.assertTrue(all(row["preflight_required"] is False for row in stochastic))


if __name__ == "__main__":
    unittest.main()
