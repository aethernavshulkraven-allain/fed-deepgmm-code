import unittest

from scripts.analyze_real_image_abs_tuning import select
from scripts.materialize_real_image_abs_final_manifest import materialize


class RealImageAbsPipelineTest(unittest.TestCase):
    def test_validation_selection_produces_one_config_per_dataset_method(self):
        rows = []
        for dataset_index in range(6):
            for method_index in range(4):
                for candidate in range(4):
                    rows.append({
                        "run_id": f"run-{dataset_index}-{method_index}-{candidate}",
                        "dataset": f"dataset-{dataset_index}",
                        "method": f"method-{method_index}",
                        "seed": 0,
                        "learning_rate": 0.001 * (candidate + 1),
                        "weight_decay": 0.01,
                        "critic_multiplier": 10.0,
                        "best_validation_mse": float(candidate + 1),
                        "best_validation_round": 1,
                        "final_validation_mse": float(candidate + 1.1),
                        "last_50_val_mse_std": 0.1,
                        "final_vs_best_validation_gap": 0.1,
                        "diverged": False,
                        "result_dir": "unused",
                    })
        chosen = select(rows)
        self.assertEqual(len(chosen), 24)
        self.assertTrue(all(row["learning_rate"] == 0.001 for row in chosen))
        self.assertTrue(all("test_mse_at_best_validation" not in row for row in chosen))

    def test_validation_selection_excludes_diverged_candidate(self):
        rows = []
        for dataset_index in range(6):
            for method_index in range(4):
                for candidate in range(4):
                    rows.append({
                        "run_id": f"run-{dataset_index}-{method_index}-{candidate}",
                        "dataset": f"dataset-{dataset_index}",
                        "method": f"method-{method_index}",
                        "learning_rate": float(candidate + 1),
                        "weight_decay": 0.01,
                        "critic_multiplier": 10.0,
                        "best_validation_mse": float(candidate),
                        "last_50_val_mse_std": 0.1,
                        "final_vs_best_validation_gap": 0.1,
                        "diverged": candidate == 0,
                    })
        chosen = select(rows)
        self.assertTrue(all(row["learning_rate"] == 2.0 for row in chosen))

    def test_materialize_reuses_selection_for_all_three_seeds(self):
        selected = []
        base = []
        for dataset_index in range(6):
            for method_index in range(4):
                dataset = f"dataset-{dataset_index}"
                method = f"method-{method_index}"
                selected.append({
                    "dataset": dataset,
                    "method": method,
                    "run_id": f"tune-{dataset}-{method}",
                    "learning_rate": "0.003",
                    "weight_decay": "0.05",
                    "critic_multiplier": "10",
                })
                for seed in range(5):
                    base.append({
                        "run_id": f"final-{dataset}-{method}-{seed}",
                        "dataset": dataset,
                        "method": method,
                        "seed": str(seed),
                        "learning_rate": "",
                        "weight_decay": "",
                        "critic_multiplier": "10",
                        "learning_rate_status": "pending",
                        "implementation_status": "pending",
                        "preflight_required": "False",
                        "preflight_status": "not_required",
                        "notes": "fixed abs.",
                    })
        output = materialize(base, selected)
        self.assertEqual(len(output), 120)
        self.assertTrue(all(row["learning_rate"] == "0.003" for row in output))
        self.assertTrue(all(row["weight_decay"] == "0.05" for row in output))


if __name__ == "__main__":
    unittest.main()
