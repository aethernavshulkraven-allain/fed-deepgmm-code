"""Tests for the Study A campaign orchestration scripts (P0 item #7):
manifest generation, tuning selection, and confirmatory aggregation.

Built around hand-written metrics.json fixtures rather than real training
runs, so the selection/aggregation *logic* is checked against known answers
independent of how long real runs take.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import prepare_eicu_study_a_manifest as manifest_gen  # noqa: E402
import select_eicu_study_a_tuning as tuning_select  # noqa: E402
import analyze_eicu_study_a_confirmatory as confirmatory  # noqa: E402


class ManifestGenerationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="study_a_manifest_")
        all_scenario_seeds = {p[1] for p in manifest_gen.TUNING_SEED_PAIRS} | {
            p[1] for p in manifest_gen.CONFIRMATORY_SEED_PAIRS
        }
        for g0 in manifest_gen.G0_VARIANTS:
            for scenario_seed in all_scenario_seeds:
                path = os.path.join(self.tmpdir, f"{g0}_scenario_seed{scenario_seed}_metadata.json")
                with open(path, "w") as handle:
                    json.dump({"n_features_x": 43, "n_features_z": 43, "n_clients": 3}, handle)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tuning_manifest_has_36_rows(self):
        rows = manifest_gen.generate_tuning(self.tmpdir, "/tmp/out")
        self.assertEqual(len(rows), len(manifest_gen.G0_VARIANTS) * 2 * 3 * 2)  # g0 x methods x lr x server_lr

    def test_tuning_rows_use_the_frozen_tuning_01_seed_pair_by_default(self):
        rows = manifest_gen.generate_tuning(self.tmpdir, "/tmp/out")
        expected_seed_pair_id, expected_scenario_seed, expected_optimizer_seed = manifest_gen.TUNING_SEED_PAIRS[0]
        self.assertTrue(all(r["seed_pair_id"] == expected_seed_pair_id for r in rows))
        self.assertTrue(all(r["scenario_seed"] == expected_scenario_seed for r in rows))
        # row["seed"] is the optimizer/training seed (run_manifest.py's
        # random_seed/run-dir semantics), never conflated with scenario_seed.
        self.assertTrue(all(r["seed"] == expected_optimizer_seed for r in rows))
        self.assertTrue(all(r["optimizer_seed"] == expected_optimizer_seed for r in rows))

    def test_tuning_can_generate_additional_seed_pairs_for_shortlisted_candidates(self):
        rows = manifest_gen.generate_tuning(self.tmpdir, "/tmp/out", seed_pairs=manifest_gen.TUNING_SEED_PAIRS)
        self.assertEqual(len(rows), len(manifest_gen.G0_VARIANTS) * 2 * 3 * 2 * 3)
        self.assertEqual(
            {r["seed_pair_id"] for r in rows}, {p[0] for p in manifest_gen.TUNING_SEED_PAIRS}
        )

    def test_tuning_uses_skip_model_selection(self):
        rows = manifest_gen.generate_tuning(self.tmpdir, "/tmp/out")
        self.assertTrue(all(r["skip_model_selection"] == "true" for r in rows))

    def test_tuning_uses_paper_aligned_and_uniform(self):
        rows = manifest_gen.generate_tuning(self.tmpdir, "/tmp/out")
        self.assertTrue(all(r["objective_mode"] == "paper_aligned" for r in rows))
        self.assertTrue(all(r["aggregation_weighting"] == "uniform_clients" for r in rows))

    def test_run_ids_are_unique(self):
        rows = manifest_gen.generate_tuning(self.tmpdir, "/tmp/out")
        run_ids = [r["run_id"] for r in rows]
        self.assertEqual(len(run_ids), len(set(run_ids)))

    def _selected(self):
        # Must match select_eicu_study_a_tuning.py's key convention exactly
        # (it groups by the manifest row's full method column, e.g.
        # "fedgda_s") -- use manifest_gen.selection_key rather than
        # hand-rolling the format, so a future rename can't silently
        # decouple this fixture from the real convention again.
        return {
            manifest_gen.selection_key(g0, method): {"learning_rate": 0.001, "server_learning_rate": 1.5}
            for g0 in manifest_gen.G0_VARIANTS
            for method in manifest_gen.METHODS
        }

    def test_confirmatory_manifest_has_30_rows(self):
        rows = manifest_gen.generate_confirmatory(self.tmpdir, "/tmp/out", self._selected())
        self.assertEqual(len(rows), 3 * 5 * 2)  # g0 x seeds x methods

    def test_confirmatory_spans_all_five_frozen_seed_pairs(self):
        rows = manifest_gen.generate_confirmatory(self.tmpdir, "/tmp/out", self._selected())
        self.assertEqual(
            {r["seed_pair_id"] for r in rows}, {p[0] for p in manifest_gen.CONFIRMATORY_SEED_PAIRS}
        )
        self.assertEqual(
            {(r["scenario_seed"], r["optimizer_seed"]) for r in rows},
            {(p[1], p[2]) for p in manifest_gen.CONFIRMATORY_SEED_PAIRS},
        )

    def test_confirmatory_missing_selection_raises(self):
        with self.assertRaises(KeyError):
            manifest_gen.generate_confirmatory(self.tmpdir, "/tmp/out", {})

    def test_confirmatory_does_not_authorize_sample_size_ablation(self):
        rows = manifest_gen.generate_confirmatory(self.tmpdir, "/tmp/out", self._selected())
        self.assertTrue(all(r["campaign_role"] == "" for r in rows))

    def test_fixed_no_tuning_status_is_preserved(self):
        selected = self._selected()
        for value in selected.values():
            value["learning_rate_status"] = "preregistered_fixed_no_tuning"
            value["selection_note"] = "demo fixed defaults; no tuning performed"
        confirmatory = manifest_gen.generate_confirmatory(
            self.tmpdir, "/tmp/out", selected
        )
        ablation = manifest_gen.generate_ablation(self.tmpdir, "/tmp/out", selected)
        self.assertTrue(
            all(
                row["learning_rate_status"] == "preregistered_fixed_no_tuning"
                for row in confirmatory + ablation
            )
        )
        self.assertTrue(
            all("no tuning performed" in row["notes"] for row in confirmatory + ablation)
        )

    def test_ablation_has_30_rows_across_all_g0(self):
        """protocol_v1.md S6.4/decision_register.md D16: all three g0
        variants, not linear only -- 3 g0 x 5 seed pairs x 2 methods = 30.
        """
        rows = manifest_gen.generate_ablation(self.tmpdir, "/tmp/out", self._selected())
        self.assertEqual(len(rows), 3 * 5 * 2)

    def test_ablation_spans_all_g0_variants(self):
        rows = manifest_gen.generate_ablation(self.tmpdir, "/tmp/out", self._selected())
        self.assertEqual({r["g0"] for r in rows}, set(manifest_gen.G0_VARIANTS))

    def test_ablation_uses_sample_size_weighting(self):
        rows = manifest_gen.generate_ablation(self.tmpdir, "/tmp/out", self._selected())
        self.assertTrue(all(r["aggregation_weighting"] == "sample_size" for r in rows))

    def test_ablation_authorizes_sample_size_via_campaign_role(self):
        """Without this, FedAvgAPI's eICU guard (check_eicu_aggregation_weighting)
        would reject every ablation row's sample_size weighting.
        """
        rows = manifest_gen.generate_ablation(self.tmpdir, "/tmp/out", self._selected())
        self.assertTrue(all(r["campaign_role"] == "aggregation_ablation" for r in rows))

    def test_dims_are_read_per_scenario_not_hardcoded(self):
        """Different (g0, scenario_seed) scenarios can have different
        covariate widths (categorical levels vary with which rows land in a
        tiny train split); the manifest must read each one's own metadata,
        not assume one width.
        """
        _, first_scenario_seed, _ = manifest_gen.CONFIRMATORY_SEED_PAIRS[1]
        path = os.path.join(self.tmpdir, f"linear_scenario_seed{first_scenario_seed}_metadata.json")
        with open(path, "w") as handle:
            json.dump({"n_features_x": 41, "n_features_z": 41, "n_clients": 3}, handle)
        rows = manifest_gen.generate_confirmatory(self.tmpdir, "/tmp/out", self._selected())
        by_scenario_seed = {r["scenario_seed"]: r for r in rows if r["g0"] == "linear"}
        self.assertEqual(by_scenario_seed[first_scenario_seed]["input_dim_g"], 41)
        other_scenario_seed = manifest_gen.CONFIRMATORY_SEED_PAIRS[0][1]
        self.assertEqual(by_scenario_seed[other_scenario_seed]["input_dim_g"], 43)


class CentralizedBaselineManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="study_a_centralized_manifest_")
        for g0 in manifest_gen.G0_VARIANTS:
            for _, scenario_seed, _ in manifest_gen.CONFIRMATORY_SEED_PAIRS:
                path = os.path.join(self.tmpdir, f"{g0}_scenario_seed{scenario_seed}_metadata.json")
                with open(path, "w") as handle:
                    json.dump({"n_features_x": 43, "n_features_z": 43, "n_clients": 3}, handle)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_45_rows_across_all_g0_seed_pairs_and_methods(self):
        rows = manifest_gen.generate_centralized_baseline(self.tmpdir, "/tmp/out")
        self.assertEqual(len(rows), 3 * 5 * 3)

    def test_methods_are_canonical_suffixed_labels(self):
        rows = manifest_gen.generate_centralized_baseline(self.tmpdir, "/tmp/out")
        self.assertEqual({r["method"] for r in rows}, {"gda_d", "sgda_s", "oadam_s"})

    def test_role_and_aggregation_mode(self):
        rows = manifest_gen.generate_centralized_baseline(self.tmpdir, "/tmp/out")
        self.assertTrue(all(r["role"] == "centralized_baseline" for r in rows))
        self.assertTrue(all(r["aggregation_weighting"] == "none" for r in rows))
        self.assertTrue(all(r["training_scope"] == "centralized" for r in rows))

    def test_gda_d_is_deterministic_batch_size_zero(self):
        rows = manifest_gen.generate_centralized_baseline(self.tmpdir, "/tmp/out")
        gda_rows = [r for r in rows if r["method"] == "gda_d"]
        self.assertTrue(all(r["batch_size"] == 0 for r in gda_rows))

    def test_run_ids_are_unique(self):
        rows = manifest_gen.generate_centralized_baseline(self.tmpdir, "/tmp/out")
        run_ids = [r["run_id"] for r in rows]
        self.assertEqual(len(run_ids), len(set(run_ids)))


class CombinedManifestTest(unittest.TestCase):
    """protocol_v1.json required_matrix: 30 confirmatory + 45 centralized_baseline
    + 30 aggregation_ablation = 105 rows in one canonical manifest (Gate 6).
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="study_a_combined_manifest_")
        for g0 in manifest_gen.G0_VARIANTS:
            for _, scenario_seed, _ in manifest_gen.CONFIRMATORY_SEED_PAIRS:
                path = os.path.join(self.tmpdir, f"{g0}_scenario_seed{scenario_seed}_metadata.json")
                with open(path, "w") as handle:
                    json.dump({"n_features_x": 43, "n_features_z": 43, "n_clients": 3}, handle)
        self.selected_path = os.path.join(self.tmpdir, "selected.json")
        with open(self.selected_path, "w") as handle:
            json.dump(
                {
                    manifest_gen.selection_key(g0, method): {
                        "learning_rate": 0.001, "server_learning_rate": 1.5
                    }
                    for g0 in manifest_gen.G0_VARIANTS
                    for method in manifest_gen.METHODS
                },
                handle,
            )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_stage_produces_105_rows_with_correct_role_counts(self):
        out_csv = os.path.join(self.tmpdir, "manifest.csv")
        exit_code = manifest_gen.main([
            "--stage", "all",
            "--scenario-dir", self.tmpdir,
            "--out", out_csv,
            "--selected-hyperparameters", self.selected_path,
        ])
        self.assertEqual(exit_code, 0)
        with open(out_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 105)
        role_counts = Counter(row["role"] for row in rows)
        self.assertEqual(role_counts, {
            "confirmatory": 30, "centralized_baseline": 45, "aggregation_ablation": 30,
        })
        run_ids = [row["run_id"] for row in rows]
        self.assertEqual(len(run_ids), len(set(run_ids)))


class TuningSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="study_a_selection_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run(self, run_id, output_root, dataset, method, seed, metrics):
        run_dir = os.path.join(output_root, dataset, method, f"seed_{seed}", run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as handle:
            json.dump(metrics, handle)
        return {
            "run_id": run_id, "output_root": output_root, "dataset": dataset,
            "method": method, "seed": seed, "g0": "linear",
            "learning_rate": 0.001, "server_learning_rate": 1.5,
        }

    def test_picks_lowest_validation_mse_among_non_diverged(self):
        root = self.tmpdir
        rows = [
            self._make_run("a", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 5.0,
                "best_moment_violation": 0.1, "final_validation_mse": 5.0,
            }),
            self._make_run("b", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 2.0,
                "best_moment_violation": 0.2, "final_validation_mse": 2.0,
            }),
        ]
        for r in rows:
            r["learning_rate"] = 0.002 if r["run_id"] == "b" else 0.001
        selected, report = tuning_select.select_candidates(rows)
        self.assertEqual(selected["linear:fedgda_s"]["run_id"], "b")

    def test_diverged_candidates_are_excluded(self):
        root = self.tmpdir
        rows = [
            self._make_run("a", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": True, "best_validation_mse": 0.001,
                "best_moment_violation": 0.001, "final_validation_mse": 0.001,
            }),
            self._make_run("b", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 9.0,
                "best_moment_violation": 0.5, "final_validation_mse": 9.0,
            }),
        ]
        selected, _ = tuning_select.select_candidates(rows)
        # 'a' has the better MSE but is diverged, so 'b' must win despite worse MSE.
        self.assertEqual(selected["linear:fedgda_s"]["run_id"], "b")

    def test_moment_violation_breaks_ties_on_validation_mse(self):
        root = self.tmpdir
        rows = [
            self._make_run("a", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 3.0,
                "best_moment_violation": 0.9, "final_validation_mse": 3.0,
            }),
            self._make_run("b", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 3.0,
                "best_moment_violation": 0.1, "final_validation_mse": 3.0,
            }),
        ]
        selected, _ = tuning_select.select_candidates(rows)
        self.assertEqual(selected["linear:fedgda_s"]["run_id"], "b")

    def test_all_diverged_leaves_no_selection(self):
        root = self.tmpdir
        rows = [
            self._make_run("a", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": True, "best_validation_mse": 1.0,
                "best_moment_violation": 1.0, "final_validation_mse": 1.0,
            }),
        ]
        selected, report = tuning_select.select_candidates(rows)
        self.assertNotIn("linear:fedgda_s", selected)
        self.assertIsNone(report["linear:fedgda_s"]["selected"])

    def test_missing_metrics_file_is_handled_not_crashed(self):
        row = {
            "run_id": "missing", "output_root": self.tmpdir, "dataset": "eicu_semisynth",
            "method": "fedgda_s", "seed": 0, "g0": "linear",
            "learning_rate": 0.001, "server_learning_rate": 1.5,
        }
        selected, report = tuning_select.select_candidates([row])
        self.assertNotIn("linear:fedgda_s", selected)
        self.assertEqual(report["linear:fedgda_s"]["n_missing"], 1)

    def test_at_checkpoint_moment_violation_overrides_best_ever_when_present(self):
        """protocol_v1.md S8.1: tie-break on moment violation *at the
        validation-selected checkpoint*, not the best moment violation seen
        at any round -- these two rows have identical best_validation_mse and
        identical (misleading) best_moment_violation, but the run with the
        worse at-checkpoint moment violation must lose.
        """
        root = self.tmpdir
        rows = [
            self._make_run("a", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 3.0,
                "best_moment_violation": 0.05, "final_validation_mse": 3.0,
                "equal_client_validation_moment_violation_at_best_validation": 0.9,
            }),
            self._make_run("b", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 3.0,
                "best_moment_violation": 0.05, "final_validation_mse": 3.0,
                "equal_client_validation_moment_violation_at_best_validation": 0.1,
            }),
        ]
        selected, _ = tuning_select.select_candidates(rows)
        self.assertEqual(selected["linear:fedgda_s"]["run_id"], "b")

    def test_final_vs_best_gap_overrides_raw_final_mse_when_present(self):
        """Second tie-break: final-minus-best validation gap, not raw final
        validation MSE -- run 'a' has a lower raw final MSE but a much worse
        degradation from its own best checkpoint, and must lose.
        """
        root = self.tmpdir
        rows = [
            self._make_run("a", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 3.0,
                "best_moment_violation": 0.1, "final_validation_mse": 3.1,
                "equal_client_validation_moment_violation_at_best_validation": 0.1,
                "final_vs_best_validation_gap": 5.0,
            }),
            self._make_run("b", root, "eicu_semisynth", "fedgda_s", 0, {
                "diverged": False, "best_validation_mse": 3.0,
                "best_moment_violation": 0.1, "final_validation_mse": 9.0,
                "equal_client_validation_moment_violation_at_best_validation": 0.1,
                "final_vs_best_validation_gap": 0.5,
            }),
        ]
        selected, _ = tuning_select.select_candidates(rows)
        self.assertEqual(selected["linear:fedgda_s"]["run_id"], "b")


class ConfirmatoryAggregationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="study_a_confirmatory_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run(self, method, seed, test_mse_best, test_mse_final, diverged=False):
        run_id = f"linear_{method}_seed{seed}"
        run_dir = os.path.join(self.tmpdir, "eicu_semisynth", method, f"seed_{seed}", run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as handle:
            json.dump(
                {
                    "diverged": diverged,
                    "test_mse_at_best_validation": test_mse_best,
                    "final_test_mse": test_mse_final,
                    "best_validation_mse": 1.0,
                    "final_validation_mse": 1.0,
                    "best_moment_violation": 0.1,
                    "final_moment_violation": 0.1,
                },
                handle,
            )
        return {
            "run_id": run_id, "output_root": self.tmpdir, "dataset": "eicu_semisynth",
            "method": method, "seed": seed, "g0": "linear",
        }

    def test_degradation_is_final_minus_best(self):
        rows = [self._make_run("fedgda_s", s, test_mse_best=2.0, test_mse_final=5.0) for s in range(5)]
        records = confirmatory.collect(rows)
        summary = confirmatory.summarize_group(records)
        self.assertAlmostEqual(summary["final_vs_best_degradation"]["mean"], 3.0)

    def test_diverged_runs_excluded_from_aggregates(self):
        rows = [
            self._make_run("fedgda_s", 0, test_mse_best=2.0, test_mse_final=2.0),
            self._make_run("fedgda_s", 1, test_mse_best=999.0, test_mse_final=999.0, diverged=True),
        ]
        records = confirmatory.collect(rows)
        summary = confirmatory.summarize_group(records)
        self.assertEqual(summary["n_diverged"], 1)
        self.assertEqual(summary["test_mse_at_best_validation"]["values"], [2.0])

    def test_paired_differences_are_ogda_minus_gda(self):
        gda_rows = [self._make_run("fedgda_s", s, test_mse_best=5.0, test_mse_final=5.0) for s in range(3)]
        ogda_rows = [self._make_run("fedogda_s", s, test_mse_best=3.0, test_mse_final=3.0) for s in range(3)]
        gda_records = confirmatory.collect(gda_rows)
        ogda_records = confirmatory.collect(ogda_rows)
        pw = confirmatory.paired_differences(gda_records, ogda_records)
        self.assertEqual(pw["differences"], [-2.0, -2.0, -2.0])
        self.assertEqual(pw["n_ogda_better"], 3)
        self.assertEqual(pw["n_gda_better"], 0)

    def test_paired_differences_only_use_shared_clean_seeds(self):
        gda_rows = [self._make_run("fedgda_s", s, test_mse_best=5.0, test_mse_final=5.0) for s in (0, 1)]
        ogda_rows = [
            self._make_run("fedogda_s", 0, test_mse_best=1.0, test_mse_final=1.0),
            self._make_run("fedogda_s", 1, test_mse_best=1.0, test_mse_final=1.0, diverged=True),
        ]
        gda_records = confirmatory.collect(gda_rows)
        ogda_records = confirmatory.collect(ogda_rows)
        pw = confirmatory.paired_differences(gda_records, ogda_records)
        self.assertEqual(pw["seeds"], [0])  # seed 1 excluded: ogda diverged there


class CrossScriptKeyConventionTest(unittest.TestCase):
    """Regression test for a real bug found while wiring these scripts
    together: select_eicu_study_a_tuning.py groups by the manifest row's full
    ``method`` column (e.g. "fedgda_s"), but prepare_eicu_study_a_manifest.py
    loops over bare method names ("fedgda"). The two must produce identical
    keys, checked here using each script's own key-construction logic rather
    than a hand-written dict on either side (which is exactly how the original
    mismatch went unnoticed).
    """

    def test_selection_output_keys_match_manifest_lookup_keys(self):
        rows = [
            {
                "run_id": "r", "output_root": "/tmp/out", "dataset": "eicu_semisynth",
                "method": f"{method}_s", "seed": 0, "g0": g0,
                "learning_rate": 0.001, "server_learning_rate": 1.5,
            }
            for g0 in manifest_gen.G0_VARIANTS
            for method in manifest_gen.METHODS
        ]
        # No metrics.json on disk for any of these -> nothing gets selected,
        # but select_candidates still reports every group's key, which is all
        # this test needs.
        _, report = tuning_select.select_candidates(rows)

        expected_keys = {
            manifest_gen.selection_key(g0, method)
            for g0 in manifest_gen.G0_VARIANTS
            for method in manifest_gen.METHODS
        }
        self.assertEqual(set(report.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
