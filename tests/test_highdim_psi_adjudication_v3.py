"""Static contract checks for the post-buffer-fix v3 campaign packet."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
)


class HighdimPsiAdjudicationV3Tests(unittest.TestCase):
    def test_every_candidate_seed_is_a_fresh_policy_pinned_run(self) -> None:
        total = 0
        all_ids = set()
        for cells, expected_count in (("signal", 66), ("x", 33)):
            summary = json.loads(
                (CAMPAIGN_DIR / f"adjudication_{cells}_summary.json").read_text()
            )
            with (CAMPAIGN_DIR / f"adjudication_{cells}_manifest.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary["new_runs"], expected_count)
            self.assertEqual(len(rows), expected_count)
            self.assertFalse(summary["model_state_reuse"])
            self.assertTrue(all(
                not candidate["reused_from_finals"]
                for cell in summary["plan"]
                for candidate in cell["candidates"]
            ))
            self.assertTrue(all(
                row["server_buffer_policy"] == "direct_client_aggregate"
                for row in rows
            ))
            self.assertTrue(all(
                row["scientific_status"] == "superseded_pre_fix_selected_shortlist"
                for row in rows
            ))
            self.assertTrue(all(row["comm_round"] == "500" for row in rows))
            ids = {row["run_id"] for row in rows}
            self.assertEqual(len(ids), expected_count)
            self.assertFalse(ids & all_ids)
            all_ids.update(ids)
            total += len(rows)
        self.assertEqual(total, 99)

    def test_flagged_configuration_is_the_clean_required_preflight(self) -> None:
        with (CAMPAIGN_DIR / "bn_buffer_diagnostic_manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["dataset"], "femnist_z")
        self.assertEqual(row["method"], "fedogda_d")
        self.assertEqual(row["seed"], "1")
        self.assertEqual(row["learning_rate"], "0.001")
        self.assertEqual(row["critic_multiplier"], "10")
        self.assertEqual(row["comm_round"], "120")
        self.assertEqual(row["server_buffer_policy"], "direct_client_aggregate")


if __name__ == "__main__":
    unittest.main()
