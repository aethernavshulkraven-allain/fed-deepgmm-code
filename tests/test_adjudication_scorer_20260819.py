"""Synthetic-data tests for scripts/score_highdim_adjudication_20260819.py,
covering the six scenarios required before the adjudication/confirmation
scorer can be trusted with real 500-round runs:

1. three finite, clearly separated candidates (no tie)
2. a pairwise tie (two of three candidates tie, one does not)
3. a three-way tie (all three candidates tie)
4. one candidate excluded by a single non-finite/diverged seed
5. only one eligible candidate (promoted without ranking)
6. no eligible candidate (cell requires retuning)

Every SeedResult below is deliberately fully specified (not defaulted) so
each test is self-contained and readable without cross-referencing the
scorer's dataclass defaults.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from score_highdim_adjudication_20260819 import (  # noqa: E402
    Candidate,
    SeedResult,
    score_cell,
)
import score_highdim_adjudication_20260819 as scorer_module  # noqa: E402


def ok_seed(seed: int, psi: float, mse: float) -> SeedResult:
    return SeedResult(seed=seed, psi_last50_mean=psi, mse_last50_mean=mse,
                       diverged=False, artifacts_complete=True, finite=True)


def bad_seed(seed: int, *, diverged: bool = False, missing: bool = False, nonfinite: bool = False) -> SeedResult:
    status = "incomplete" if missing else "terminal_ineligible"
    return SeedResult(
        seed=seed,
        psi_last50_mean=None if nonfinite else 0.0,
        mse_last50_mean=None if nonfinite else 0.0,
        diverged=diverged,
        artifacts_complete=not missing,
        finite=not nonfinite,
        status=status,
    )


class ThreeSeparatedCandidatesTest(unittest.TestCase):
    def test_top_candidate_wins_outright(self):
        a = Candidate("A", "psi_rank1", (ok_seed(0, 10, 0.3), ok_seed(1, 10, 0.3), ok_seed(2, 10, 0.3)))
        b = Candidate("B", "psi_rank2", (ok_seed(0, 5, 0.3), ok_seed(1, 5, 0.3), ok_seed(2, 5, 0.3)))
        c = Candidate("C", "mse_winner", (ok_seed(0, 1, 0.3), ok_seed(1, 1, 0.3), ok_seed(2, 1, 0.3)))
        outcome = score_cell([a, b, c])
        self.assertEqual(outcome.outcome, "promoted")
        self.assertEqual(outcome.winner.candidate_id, "A")
        self.assertEqual([c.candidate_id for c in outcome.tie_set], ["A"])


class PairwiseTieTest(unittest.TestCase):
    def test_two_of_three_tie_resolved_by_mse(self):
        # A and B's per-seed Psi differences flip sign across seeds -> tie.
        # C is clearly separated from A -> not part of the tie set.
        a = Candidate("A", "psi_rank1", (ok_seed(0, 5.0, 0.30), ok_seed(1, 4.0, 0.30), ok_seed(2, 6.0, 0.30)))
        b = Candidate("B", "psi_rank2", (ok_seed(0, 4.9, 0.20), ok_seed(1, 5.9, 0.20), ok_seed(2, 3.9, 0.20)))
        c = Candidate("C", "mse_winner", (ok_seed(0, 1.0, 0.35), ok_seed(1, 1.0, 0.35), ok_seed(2, 1.0, 0.35)))
        outcome = score_cell([a, b, c])
        self.assertEqual(outcome.outcome, "tie_resolved_by_mse")
        self.assertEqual({t.candidate_id for t in outcome.tie_set}, {"A", "B"})
        self.assertEqual(outcome.winner.candidate_id, "B")  # lower median MSE (0.20 < 0.30)


class ThreeWayTieTest(unittest.TestCase):
    def test_all_three_tie_resolved_by_mse(self):
        a = Candidate("A", "psi_rank1", (ok_seed(0, 5.0, 0.30), ok_seed(1, 4.0, 0.30), ok_seed(2, 6.0, 0.30)))
        b = Candidate("B", "psi_rank2", (ok_seed(0, 4.9, 0.25), ok_seed(1, 5.9, 0.25), ok_seed(2, 3.9, 0.25)))
        c = Candidate("C", "mse_winner", (ok_seed(0, 4.8, 0.35), ok_seed(1, 3.8, 0.35), ok_seed(2, 5.8, 0.35)))
        outcome = score_cell([a, b, c])
        self.assertEqual(outcome.outcome, "tie_resolved_by_mse")
        self.assertEqual({t.candidate_id for t in outcome.tie_set}, {"A", "B", "C"})
        self.assertEqual(outcome.winner.candidate_id, "B")  # lowest median MSE among all three


class OneNonFiniteSeedExcludesCandidateTest(unittest.TestCase):
    def test_diverged_seed_excludes_candidate_even_with_two_good_seeds(self):
        a = Candidate("A", "psi_rank1", (ok_seed(0, 10, 0.3), ok_seed(1, 10, 0.3), ok_seed(2, 10, 0.3)))
        b = Candidate("B", "psi_rank2", (ok_seed(0, 20, 0.1), ok_seed(1, 20, 0.1), bad_seed(2, diverged=True)))
        c = Candidate("C", "mse_winner", (ok_seed(0, 1, 0.3), ok_seed(1, 1, 0.3), ok_seed(2, 1, 0.3)))
        outcome = score_cell([a, b, c])
        self.assertEqual(outcome.outcome, "promoted")
        self.assertEqual(outcome.winner.candidate_id, "A")
        self.assertEqual({e.candidate_id for e in outcome.excluded}, {"B"})
        # B's two good seeds (Psi=20, best of all three) must NOT be averaged
        # or otherwise given credit -- confirms no two-seed median is computed.
        self.assertNotEqual(outcome.winner.candidate_id, "B")


class OnlyOneEligibleCandidateTest(unittest.TestCase):
    def test_missing_candidate_blocks_promotion_as_incomplete(self):
        a = Candidate("A", "psi_rank1", (ok_seed(0, 1, 0.9), ok_seed(1, 1, 0.9), ok_seed(2, 1, 0.9)))
        b = Candidate("B", "psi_rank2", (ok_seed(0, 99, 0.01), bad_seed(1, missing=True), ok_seed(2, 99, 0.01)))
        c = Candidate("C", "mse_winner", (bad_seed(0, nonfinite=True), ok_seed(1, 50, 0.05), ok_seed(2, 50, 0.05)))
        outcome = score_cell([a, b, c])
        self.assertEqual(outcome.outcome, "incomplete")
        self.assertIsNone(outcome.winner)
        self.assertEqual({e.candidate_id for e in outcome.excluded}, {"B", "C"})

    def test_single_eligible_candidate_promoted_when_others_are_terminal(self):
        a = Candidate("A", "psi_rank1", (ok_seed(0, 1, 0.9), ok_seed(1, 1, 0.9), ok_seed(2, 1, 0.9)))
        b = Candidate("B", "psi_rank2", (ok_seed(0, 99, 0.01), bad_seed(1, diverged=True), ok_seed(2, 99, 0.01)))
        outcome = score_cell([a, b])
        self.assertEqual(outcome.outcome, "promoted")
        self.assertEqual(outcome.winner.candidate_id, "A")
        self.assertEqual(outcome.detail, "only one eligible candidate, no ranking needed")


class NoEligibleCandidateTest(unittest.TestCase):
    def test_all_candidates_fail_requires_retuning(self):
        a = Candidate("A", "psi_rank1", (ok_seed(0, 10, 0.1), ok_seed(1, 10, 0.1), bad_seed(2, diverged=True)))
        b = Candidate("B", "psi_rank2", (bad_seed(0, diverged=True), ok_seed(1, 5, 0.2), ok_seed(2, 5, 0.2)))
        c = Candidate("C", "mse_winner", (bad_seed(0, diverged=True), bad_seed(1, diverged=True), bad_seed(2, diverged=True)))
        outcome = score_cell([a, b, c])
        self.assertEqual(outcome.outcome, "retune_required")
        self.assertIsNone(outcome.winner)
        self.assertEqual(len(outcome.eligible), 0)
        self.assertEqual(len(outcome.excluded), 3)


class ExactMseTieTest(unittest.TestCase):
    def test_exact_mse_tie_has_no_order_based_winner(self):
        a = Candidate("A", "first", (ok_seed(0, 5.0, 0.2), ok_seed(1, 4.0, 0.2), ok_seed(2, 6.0, 0.2)))
        b = Candidate("B", "second", (ok_seed(0, 4.9, 0.2), ok_seed(1, 5.9, 0.2), ok_seed(2, 3.9, 0.2)))
        outcome = score_cell([a, b])
        self.assertEqual(outcome.outcome, "mse_tie_unresolved")
        self.assertIsNone(outcome.winner)


class CliStageBarrierTest(unittest.TestCase):
    def test_retune_outcome_is_written_but_blocks_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "results.json"
            results = {
                "fixture/cell": {
                    "outcome": "retune_required",
                    "winner": None,
                    "detail": "no eligible candidate",
                }
            }
            argv = ["score", "--cells", "signal", "--out", str(out_path)]
            with mock.patch.object(scorer_module, "score_manifest", return_value=results):
                with mock.patch.object(sys, "argv", argv):
                    status = scorer_module.main()
            self.assertEqual(status, 3)
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
