"""Two launcher traps that silently corrupted campaigns, now refused.

1. run_stochastic_eg_campaign.make_row applied ``row.update(candidate)`` and
   THEN a dict literal hardcoding ``critic_multiplier: 10``, so a per-dataset
   multiplier placed in the candidate was silently discarded. A whole starter
   batch ran at cm=10 because of this, and later campaigns worked around it by
   re-asserting the value afterwards.

2. run_manifest routes every row under a single --output-root and never reads
   the per-row ``output_root`` column that campaign builders write. When the
   two disagree -- e.g. a campaign encoding separate arms in the column -- all
   arms land in one directory. Routing per row would change where existing
   manifests resume to, so the disagreement is refused instead.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import run_manifest  # noqa: E402
import run_stochastic_eg_campaign as campaign  # noqa: E402


class CriticMultiplierIsNotClobberedTest(unittest.TestCase):
    def _row(self, candidate):
        return campaign.make_row(
            Path(REPO_ROOT) / "experiments" / "tmp_campaign",
            {"dataset": "femnist_x", "method": "fed_eg_s"},
            "screen", "femnist_x", 0.5, "fed_eg_s", 31, candidate, 150,
        )

    def test_candidate_value_survives(self):
        row = self._row({"candidate": 0, "critic_multiplier": 1, "learning_rate": 0.003})
        self.assertEqual(row["critic_multiplier"], 1)

    def test_default_is_unchanged_when_candidate_is_silent(self):
        row = self._row({"candidate": 0, "learning_rate": 0.003})
        self.assertEqual(row["critic_multiplier"], 10)


class DifferingOutputRootsAreRefusedTest(unittest.TestCase):
    """Rows naming the SAME root are a relocated campaign and must still run;
    rows naming DIFFERENT roots are per-row separation that silently collapses."""

    def _rows(self, *roots):
        return [
            {"run_id": f"r{i}", "dataset": "femnist_x", "method": "fed_eg_s",
             "seed": "31", "output_root": root}
            for i, root in enumerate(roots)
        ]

    def test_differing_roots_raise(self):
        rows = self._rows(
            str(Path(REPO_ROOT) / "results" / "arm_a"),
            str(Path(REPO_ROOT) / "results" / "arm_b"),
        )
        with self.assertRaises(run_manifest.ManifestLaunchError) as ctx:
            run_manifest.assert_single_output_root(rows)
        self.assertIn("different output_root values", str(ctx.exception))

    def test_same_root_is_allowed_even_if_it_differs_from_output_root(self):
        root = str(Path(REPO_ROOT) / "results" / "built_here")
        run_manifest.assert_single_output_root(self._rows(root, root))

    def test_absent_column_is_allowed(self):
        rows = self._rows("", "")
        for row in rows:
            del row["output_root"]
        run_manifest.assert_single_output_root(rows)

    def test_routing_is_unchanged(self):
        root = Path(REPO_ROOT) / "results" / "arm_a"
        row = self._rows(str(Path(REPO_ROOT) / "elsewhere"))[0]
        self.assertEqual(run_manifest._run_dir(root, row).name, "r0")


class FilteredLaunchFromMixedManifestTest(unittest.TestCase):
    """Selecting ONE arm out of a mixed-root manifest is legitimate.

    The guard originally ran on every manifest row, before --only and --limit,
    so a two-arm manifest was rejected even when the caller selected exactly
    one arm -- and a blocked or excluded row could veto a run it took no part
    in. It now validates the rows actually being launched.
    """

    def _mixed_rows(self):
        return [
            {"run_id": "a1", "dataset": "femnist_x", "method": "fed_eg_s", "seed": "31",
             "arm": "baseline_matched",
             "output_root": str(Path(REPO_ROOT) / "results" / "arm_a")},
            {"run_id": "b1", "dataset": "femnist_x", "method": "fed_eg_s", "seed": "31",
             "arm": "tuned_control",
             "output_root": str(Path(REPO_ROOT) / "results" / "arm_b")},
        ]

    def test_selecting_one_arm_is_allowed(self):
        rows = self._mixed_rows()
        selected = [r for r in rows if r["arm"] == "baseline_matched"]
        run_manifest.assert_single_output_root(selected)

    def test_launching_both_arms_together_is_still_refused(self):
        with self.assertRaises(run_manifest.ManifestLaunchError):
            run_manifest.assert_single_output_root(self._mixed_rows())

    def test_limit_is_applied_before_validation(self):
        """--limit narrowing to a single arm must also be accepted."""
        run_manifest.assert_single_output_root(self._mixed_rows()[:1])


if __name__ == "__main__":
    unittest.main()
