"""Tests for the Study A centralized baseline runner's job matrix (Gate 4).

protocol_v1.md S6.2 requires 3 g0 x 5 confirmatory seed pairs x 3 methods
(gda_d, sgda_s, oadam_s) = 45 rows -- previously this runner only covered
gda/oadam (30 rows) on an ad hoc seed convention. These tests check the job
matrix and generated command line, not real training (that is exercised in
tests/test_run_centralized_lowdim_eicu_gate4.py).
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import run_eicu_centralized_baselines as runner  # noqa: E402


class JobMatrixTest(unittest.TestCase):
    def test_45_jobs_total(self):
        jobs = [
            (g0, seed_pair_id, scenario_seed, optimizer_seed, method)
            for g0 in runner.G0_VARIANTS
            for seed_pair_id, scenario_seed, optimizer_seed in runner.CONFIRMATORY_SEED_PAIRS
            for method in runner.METHODS
        ]
        self.assertEqual(len(jobs), 3 * 5 * 3)

    def test_methods_include_sgda(self):
        self.assertEqual(set(runner.METHODS), {"gda", "sgda", "oadam"})

    def test_seed_pairs_match_frozen_confirmatory_protocol(self):
        self.assertEqual(
            runner.CONFIRMATORY_SEED_PAIRS,
            (
                ("confirmatory_01", 101, 1101),
                ("confirmatory_02", 102, 1102),
                ("confirmatory_03", 103, 1103),
                ("confirmatory_04", 104, 1104),
                ("confirmatory_05", 105, 1105),
            ),
        )


class BuildCommandTest(unittest.TestCase):
    def _args(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            iterations=500, g_lr=0.001, f_lr=0.01, weight_decay=0.01, overwrite=False,
        )

    def test_command_uses_scenario_seed_and_optimizer_seed_separately(self):
        cmd, run_dir = runner.build_command(
            "python", "linear", "confirmatory_01", 101, 1101, "gda",
            runner.DEFAULT_SCENARIO_DIR, "/tmp/out", self._args(),
        )
        self.assertIn("--scenario-name", cmd)
        self.assertEqual(cmd[cmd.index("--scenario-name") + 1], "linear_scenario_seed101")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "1101")
        self.assertEqual(cmd[cmd.index("--scenario-seed") + 1], "101")
        self.assertEqual(cmd[cmd.index("--seed-pair-id") + 1], "confirmatory_01")
        self.assertEqual(cmd[cmd.index("--objective-mode") + 1], "paper_aligned")

    def test_gda_is_deterministic_batch_size_zero(self):
        cmd, _ = runner.build_command(
            "python", "linear", "confirmatory_01", 101, 1101, "gda",
            runner.DEFAULT_SCENARIO_DIR, "/tmp/out", self._args(),
        )
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "0")

    def test_sgda_and_oadam_are_stochastic(self):
        for method in ("sgda", "oadam"):
            cmd, _ = runner.build_command(
                "python", "linear", "confirmatory_01", 101, 1101, method,
                runner.DEFAULT_SCENARIO_DIR, "/tmp/out", self._args(),
            )
            self.assertEqual(cmd[cmd.index("--batch-size") + 1], "256")

    def test_run_dir_keyed_by_seed_pair_id_not_bare_seed(self):
        cmd, run_dir = runner.build_command(
            "python", "linear", "confirmatory_01", 101, 1101, "gda",
            runner.DEFAULT_SCENARIO_DIR, "/tmp/out", self._args(),
        )
        self.assertIn("confirmatory_01", run_dir)
        self.assertIn("seed_1101", run_dir)
        self.assertIn("gda_d", run_dir)
        self.assertEqual(
            cmd[cmd.index("--run-id") + 1],
            "centralized_linear_gda_d_confirmatory_01",
        )


if __name__ == "__main__":
    unittest.main()
