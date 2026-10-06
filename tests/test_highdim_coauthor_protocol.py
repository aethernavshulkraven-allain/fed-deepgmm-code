from __future__ import annotations

import csv
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_highdim_coauthor_protocol.py"
SPEC = importlib.util.spec_from_file_location("prepare_highdim_coauthor_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_highdim_coauthor_tuning.py"
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "analyze_highdim_coauthor_tuning", ANALYZE_SCRIPT
)
ANALYZE = importlib.util.module_from_spec(ANALYZE_SPEC)
assert ANALYZE_SPEC.loader is not None
ANALYZE_SPEC.loader.exec_module(ANALYZE)


class HighdimCoauthorProtocolTest(unittest.TestCase):
    def test_protocol_shape_and_fixed_response(self) -> None:
        fieldnames, base, historical = MODULE.source_indexes()
        self.assertIn("partition_alpha", fieldnames)
        for alpha in MODULE.ALPHAS:
            tuning = MODULE.make_tuning_rows(alpha, base, historical)
            final = MODULE.make_final_base_rows(alpha, base)
            self.assertEqual(len(tuning), 48)
            self.assertEqual(len(final), 120)
            self.assertEqual({float(row["partition_alpha"]) for row in tuning}, {alpha})
            self.assertEqual({float(row["partition_alpha"]) for row in final}, {alpha})
            self.assertEqual({int(row["comm_round"]) for row in tuning}, {150})
            self.assertEqual(
                {
                    int(row["comm_round"])
                    for row in final
                    if row["method"].endswith("_d")
                },
                {500},
            )
            self.assertEqual(
                {
                    int(row["comm_round"])
                    for row in final
                    if row["method"].endswith("_s")
                },
                {1500},
            )
            self.assertEqual({int(row["seed"]) for row in final}, {0, 1, 2, 3, 4})

    def test_weight_decay_is_not_a_tuning_dimension(self) -> None:
        _, base, historical = MODULE.source_indexes()
        rows = MODULE.make_tuning_rows(0.5, base, historical)
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault((row["dataset"], row["method"]), []).append(row)
        self.assertEqual(len(groups), 24)
        for (_, method), candidates in groups.items():
            self.assertEqual(len(candidates), 2)
            self.assertEqual(len({row["learning_rate"] for row in candidates}), 2)
            self.assertEqual(len({row["weight_decay"] for row in candidates}), 1)
            expected = 0.001 if method.endswith("_d") else 0.05
            self.assertEqual({float(row["weight_decay"]) for row in candidates}, {expected})

    def test_alpha0p5_paths_reuse_canonical_historical_runs(self) -> None:
        _, base, historical = MODULE.source_indexes()
        rows = MODULE.make_tuning_rows(0.5, base, historical)
        historical_ids = {row["run_id"] for row in historical.values()}
        self.assertTrue(all(row["run_id"] in historical_ids for row in rows))
        self.assertTrue(
            all(
                row["output_root"]
                == "results/rerun_protocol_v1_real_images_abs_alpha0p5_tuning"
                for row in rows
            )
        )

    def test_generated_manifests_match_builder(self) -> None:
        MODULE.main()
        for token in ("alpha0p1", "alpha0p5", "alpha1"):
            path = MODULE.PROTOCOL_DIR / token / "tuning_manifest.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 48)

    def test_selector_uses_validation_metrics_only(self) -> None:
        records = []
        for dataset_index in range(6):
            dataset = f"dataset_{dataset_index}"
            for method in ("fedgda_d", "fedgda_s", "fedogda_d", "fedogda_s"):
                for candidate_index, validation in enumerate((0.2, 0.1)):
                    records.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "learning_rate": 0.001 + candidate_index * 0.001,
                            "best_validation_mse": validation,
                            "last_50_val_mse_std": 0.0,
                            "final_vs_best_validation_gap": 0.0,
                            # A deliberately attractive Test value on the worse
                            # validation candidate must be irrelevant.
                            "test_mse_at_best_validation": 0.0 if candidate_index == 0 else 10.0,
                        }
                    )
        selected = ANALYZE.select(records)
        self.assertEqual(len(selected), 24)
        self.assertTrue(all(row["best_validation_mse"] == 0.1 for row in selected))


if __name__ == "__main__":
    unittest.main()
