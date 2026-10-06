"""Tests for scripts/report_eicu_study_a_status.py.

The reporter's whole point is to derive campaign phase from artifacts rather
than from prose, so these tests build small synthetic fixture trees (never
touching the real experiment directories) and check that every phase
transition in PHASE_ORDER is reached by the right artifact shape -- including
future states (tuning in progress, a materialized final manifest, an
analyzed campaign) that do not exist in the real tree yet.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import report_eicu_study_a_status as reporter  # noqa: E402
import validate_eicu_study_a_campaign as validator  # noqa: E402


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_run(output_root: Path, result_path: str, complete: bool, diverged: bool = False) -> None:
    run_dir = output_root / result_path
    run_dir.mkdir(parents=True, exist_ok=True)
    if complete:
        write_json(run_dir / "metrics.json", {"diverged": diverged, "final_test_mse": 1.23})
    else:
        # A run that has started but not finished: per-round CSV present,
        # no metrics.json yet.
        (run_dir / "mse_by_round.csv").write_text("round,val_mse\n0,1.0\n", encoding="utf-8")


def manifest_row(run_id, role, output_root: Path, result_path: str) -> dict:
    return {
        "run_id": run_id,
        "role": role,
        "output_root": str(output_root),
        "result_path": result_path,
    }


class SplitNameAliasTests(unittest.TestCase):
    def test_canonical_and_alias_map_to_dev(self):
        self.assertEqual(validator.normalize_split_name("dev"), "dev")
        self.assertEqual(validator.normalize_split_name("Validation"), "dev")
        self.assertEqual(validator.normalize_split_name("VAL"), "dev")

    def test_train_and_test_are_stable(self):
        self.assertEqual(validator.normalize_split_name("train"), "train")
        self.assertEqual(validator.normalize_split_name("Test"), "test")

    def test_unknown_split_name_raises(self):
        with self.assertRaises(ValueError):
            validator.normalize_split_name("holdout")


class ManifestScanTests(unittest.TestCase):
    def test_missing_manifest_reports_not_exists(self):
        scan = reporter.scan_manifest_completion(Path("/nonexistent/manifest.csv"))
        self.assertFalse(scan["exists"])
        self.assertEqual(scan["total_planned"], 0)
        self.assertEqual(scan["total_completed"], 0)

    def test_partial_completion_and_diverged_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output_root = tmp / "results"
            manifest = tmp / "manifest.csv"
            rows = [
                manifest_row("r1", "tuning", output_root, "a/r1"),
                manifest_row("r2", "tuning", output_root, "a/r2"),
                manifest_row("r3", "tuning", output_root, "a/r3"),
            ]
            write_csv(manifest, rows)
            make_run(output_root, "a/r1", complete=True, diverged=False)
            make_run(output_root, "a/r2", complete=True, diverged=True)
            make_run(output_root, "a/r3", complete=False)

            scan = reporter.scan_manifest_completion(manifest)
            self.assertTrue(scan["exists"])
            self.assertEqual(scan["total_planned"], 3)
            self.assertEqual(scan["total_completed"], 2)
            self.assertEqual(scan["total_diverged"], 1)
            self.assertEqual(scan["by_role"]["tuning"]["incomplete_run_ids"], ["r3"])

    def test_roles_are_grouped_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output_root = tmp / "results"
            manifest = tmp / "manifest.csv"
            rows = [
                manifest_row("c1", "confirmatory", output_root, "c1"),
                manifest_row("b1", "centralized_baseline", output_root, "b1"),
                manifest_row("g1", "aggregation_ablation", output_root, "g1"),
            ]
            write_csv(manifest, rows)
            make_run(output_root, "c1", complete=True)
            make_run(output_root, "b1", complete=False)
            make_run(output_root, "g1", complete=True)

            scan = reporter.scan_manifest_completion(manifest)
            self.assertEqual(scan["by_role"]["confirmatory"]["completed"], 1)
            self.assertEqual(scan["by_role"]["centralized_baseline"]["completed"], 0)
            self.assertEqual(scan["by_role"]["aggregation_ablation"]["completed"], 1)


class ClientHeterogeneityTests(unittest.TestCase):
    def test_computes_min_median_max_and_exclusion_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cohort_csv = tmp / "cohort.csv"
            audit_csv = tmp / "client_eligibility_audit.csv"

            # Three eligible hospitals with 7, 10, and 13 rows.
            cohort_rows = []
            for hid, n in (("1", 7), ("2", 10), ("3", 13)):
                for _ in range(n):
                    cohort_rows.append({"hospitalid": hid})
            write_csv(cohort_csv, cohort_rows)

            audit_rows = [
                {"hospitalid": "1", "n_rows": "7", "eligible": "True", "exclusion_reasons": ""},
                {"hospitalid": "2", "n_rows": "10", "eligible": "True", "exclusion_reasons": ""},
                {"hospitalid": "3", "n_rows": "13", "eligible": "True", "exclusion_reasons": ""},
                {
                    "hospitalid": "4",
                    "n_rows": "3",
                    "eligible": "False",
                    "exclusion_reasons": "fewer_than_min_client_rows",
                },
                {
                    "hospitalid": "5",
                    "n_rows": "9",
                    "eligible": "False",
                    "exclusion_reasons": "fewer_than_min_client_rows;no_training_off_hours_variation",
                },
            ]
            write_csv(audit_csv, audit_rows)

            summary = reporter.compute_client_heterogeneity(cohort_csv, audit_csv)
            self.assertTrue(summary["available"])
            rc = summary["client_row_counts"]
            self.assertEqual(rc["min"], 7)
            self.assertEqual(rc["max"], 13)
            self.assertEqual(rc["median"], 10)
            self.assertEqual(rc["n_clients_under_10_rows"], 1)

            ef = summary["eligibility_funnel"]
            self.assertEqual(ef["n_candidate_hospitals"], 5)
            self.assertEqual(ef["n_eligible_hospitals"], 3)
            self.assertEqual(ef["n_excluded_hospitals"], 2)
            self.assertEqual(ef["exclusion_reason_occurrences"]["fewer_than_min_client_rows"], 2)
            self.assertEqual(ef["exclusion_reason_occurrences"]["no_training_off_hours_variation"], 1)

            self.assertTrue(summary["cross_check"]["consistent"])

    def test_matches_real_tree_figures(self):
        """Pin down the real Study A v2 demo artifact numbers this task's
        author claimed, so a future edit to the cohort regenerates a loud
        failure instead of silently drifting."""
        v2_dir = Path(REPO_ROOT) / "experiments" / "eicu_study_a_v2_offhours_demo_20260727"
        if not (v2_dir / "cohort.csv").is_file():
            self.skipTest("real v2 demo cohort artifacts not present in this checkout")
        summary = reporter.compute_client_heterogeneity(
            v2_dir / "cohort.csv", v2_dir / "client_eligibility_audit.csv"
        )
        rc = summary["client_row_counts"]
        self.assertEqual(rc["min"], 7)
        self.assertEqual(rc["max"], 26)
        self.assertEqual(rc["median"], 11)
        self.assertEqual(rc["n_clients_under_10_rows"], 14)
        ef = summary["eligibility_funnel"]
        self.assertEqual(ef["n_candidate_hospitals"], 186)
        self.assertEqual(ef["n_eligible_hospitals"], 179)
        self.assertEqual(ef["n_excluded_hospitals"], 7)


class V2FixtureCampaign:
    """Builds a minimal, self-contained v2-shaped campaign directory so phase
    transitions can be exercised without touching the real experiment tree."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.v2_dir = tmp / "eicu_study_a_v2_offhours_demo_fixture"
        self.results_root = tmp / "results"
        self.protocol_json = tmp / "protocol_v2.json"
        self.v2_dir.mkdir(parents=True, exist_ok=True)

        write_json(
            self.protocol_json,
            {
                "final": {
                    "primary_federated_runs": 2,
                    "centralized_runs": 2,
                    "aggregation_ablation_runs": 0,
                    "total_runs": 4,
                }
            },
        )

    def write_setup_artifacts(self):
        write_json(
            self.v2_dir / "cohort_metadata.json",
            {
                "n_rows": 40,
                "n_clients": 4,
                "off_hours_rate": 0.5,
                "split_sizes": {"train": 28, "dev": 6, "test": 6},
                "clients_per_split": {"train": 4, "dev": 4, "test": 4},
                "flow": [
                    {"step": "all ICU stays", "n_stays": 100},
                    {
                        "step": "eligible hospital clients before simulation",
                        "n_stays": 40,
                        "detail": "fixture",
                    },
                    {"step": "final Study A v2 cohort", "n_stays": 40},
                ],
            },
        )
        write_json(self.v2_dir / "setup_validation_summary.json", {"ok": True})

    def write_tuning_manifest(self, n=2):
        rows = []
        for i in range(n):
            rows.append(manifest_row(f"tune_{i}", "tuning", self.results_root, f"tuning/run_{i}"))
        write_csv(self.v2_dir / "tuning_manifest.csv", rows)
        return rows

    def complete_tuning_runs(self, indices):
        for i in indices:
            make_run(self.results_root, f"tuning/run_{i}", complete=True)

    def write_final_manifest(self):
        rows = [
            manifest_row("f1", "confirmatory", self.results_root, "final/f1"),
            manifest_row("f2", "confirmatory", self.results_root, "final/f2"),
            manifest_row("f3", "centralized_baseline", self.results_root, "final/f3"),
            manifest_row("f4", "centralized_baseline", self.results_root, "final/f4"),
        ]
        write_csv(self.v2_dir / "final_manifest.csv", rows)
        return rows

    def complete_final_runs(self, run_ids):
        mapping = {"f1": "final/f1", "f2": "final/f2", "f3": "final/f3", "f4": "final/f4"}
        for run_id in run_ids:
            make_run(self.results_root, mapping[run_id], complete=True)

    def write_analysis_ledger(self, passed=("f1", "f2", "f3", "f4")):
        write_json(
            self.v2_dir / "effect_metric_materialization.json",
            {
                "summary": {"passed": len(passed), "failed": 0, "missing": 0},
                "ledger": {"passed": list(passed), "failed": [], "missing": [], "dry_run": []},
            },
        )


class V2PhaseTransitionTests(unittest.TestCase):
    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_ctx.name)
        self.addCleanup(self._tmp_ctx.cleanup)
        self.campaign = V2FixtureCampaign(self.tmp)

    def _status(self):
        return reporter.derive_v2_status(self.campaign.v2_dir, self.campaign.protocol_json)

    def test_not_setup_when_cohort_metadata_absent(self):
        status = self._status()
        self.assertEqual(status["phase"], "not_setup")

    def test_setup_certified_with_zero_of_n_tuning_complete(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        status = self._status()
        self.assertEqual(status["phase"], "setup_certified")
        self.assertEqual(status["tuning"]["planned"], 4)
        self.assertEqual(status["tuning"]["completed"], 0)
        self.assertFalse(status["final"]["manifest_materialized"])

    def test_tuning_in_progress(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        self.campaign.complete_tuning_runs([0, 1])
        status = self._status()
        self.assertEqual(status["phase"], "tuning_in_progress")
        self.assertEqual(status["tuning"]["completed"], 2)
        self.assertEqual(status["tuning"]["planned"], 4)

    def test_tuning_complete_with_no_final_manifest_yet(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        self.campaign.complete_tuning_runs([0, 1, 2, 3])
        status = self._status()
        self.assertEqual(status["phase"], "tuning_complete")
        self.assertEqual(status["tuning"]["completed"], status["tuning"]["planned"])
        self.assertFalse(status["final"]["manifest_materialized"])
        # Planned final count must come from the design doc, not be hardcoded.
        self.assertEqual(status["final"]["design_total"], 4)

    def test_final_manifest_materialized_but_not_started(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        self.campaign.complete_tuning_runs([0, 1, 2, 3])
        self.campaign.write_final_manifest()
        status = self._status()
        self.assertEqual(status["phase"], "final_not_started")
        self.assertTrue(status["final"]["manifest_materialized"])
        self.assertEqual(status["final"]["completed"], 0)

    def test_final_in_progress(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        self.campaign.complete_tuning_runs([0, 1, 2, 3])
        self.campaign.write_final_manifest()
        self.campaign.complete_final_runs(["f1", "f3"])
        status = self._status()
        self.assertEqual(status["phase"], "final_in_progress")
        self.assertEqual(status["final"]["completed"], 2)
        self.assertEqual(status["final"]["planned"], 4)

    def test_final_complete_without_analysis_ledger(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        self.campaign.complete_tuning_runs([0, 1, 2, 3])
        self.campaign.write_final_manifest()
        self.campaign.complete_final_runs(["f1", "f2", "f3", "f4"])
        status = self._status()
        self.assertEqual(status["phase"], "final_complete")

    def test_analyzed_when_effect_metric_ledger_is_clean(self):
        self.campaign.write_setup_artifacts()
        self.campaign.write_tuning_manifest(n=4)
        self.campaign.complete_tuning_runs([0, 1, 2, 3])
        self.campaign.write_final_manifest()
        self.campaign.complete_final_runs(["f1", "f2", "f3", "f4"])
        self.campaign.write_analysis_ledger()
        status = self._status()
        self.assertEqual(status["phase"], "analyzed")

    def test_cohort_numbers_are_read_not_hardcoded(self):
        self.campaign.write_setup_artifacts()
        status = self._status()
        cn = status["cohort_numbers"]
        self.assertEqual(cn["n_rows"], 40)
        self.assertEqual(cn["n_clients"], 4)
        # dev is canonical and must survive round-trip through the alias map.
        self.assertEqual(cn["split_sizes_canonical_names"]["dev"], 6)


class V1RealTreeTests(unittest.TestCase):
    """The real v1 campaign is frozen and complete; this pins its reporter
    output down to the numbers stated in completion_record.json (105/105)
    and cross-checks them against an independent results/ scan."""

    def setUp(self):
        v1_cohort_dir = Path(reporter.DEFAULT_V1_COHORT_DIR)
        v1_campaign_dir = Path(reporter.DEFAULT_V1_CAMPAIGN_DIR)
        if not (v1_campaign_dir / "completion_record.json").is_file():
            self.skipTest("real v1 campaign artifacts not present in this checkout")
        self.status = reporter.derive_v1_status(v1_cohort_dir, v1_campaign_dir)

    def test_phase_is_final_complete(self):
        self.assertEqual(self.status["phase"], "final_complete")

    def test_independent_scan_matches_completion_record_claim(self):
        self.assertEqual(self.status["independent_results_scan"]["completed"], 105)
        self.assertEqual(self.status["independent_results_scan"]["planned"], 105)
        self.assertTrue(self.status["claim_matches_independent_scan"])

    def test_funnel_reaches_nine_rows_three_hospitals(self):
        funnel = self.status["funnel"]
        self.assertTrue(funnel["available"])
        self.assertEqual(funnel["clinical_gate_collapse"]["to_n_rows"], 201)
        self.assertEqual(funnel["instrument_variation_gate_collapse"]["to_n_rows"], 9)
        self.assertEqual(funnel["instrument_variation_gate_collapse"]["to_n_hospitals"], 3)
        self.assertEqual(
            sorted(funnel["instrument_variation_gate_collapse"]["eligible_hospital_ids"]),
            [184, 243, 407],
        )


class RenderMarkdownSmokeTest(unittest.TestCase):
    def test_render_markdown_does_not_crash_on_a_minimal_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            campaign = V2FixtureCampaign(tmp)
            campaign.write_setup_artifacts()
            v2_status = reporter.derive_v2_status(campaign.v2_dir, campaign.protocol_json)

            v1_cohort_dir = tmp / "v1_cohort"
            v1_campaign_dir = tmp / "v1_campaign"
            v1_cohort_dir.mkdir()
            v1_campaign_dir.mkdir()
            v1_status = reporter.derive_v1_status(v1_cohort_dir, v1_campaign_dir)
            self.assertEqual(v1_status["phase"], "not_setup")

            status = {
                "generated_at_utc": "2026-01-01T00:00:00Z",
                "phase_vocabulary": list(reporter.PHASE_ORDER),
                "studies": {"v1": v1_status, "v2": v2_status},
            }
            markdown = reporter.render_markdown(status)
            self.assertIn("eICU Study A status", markdown)
            self.assertIn("setup_certified", markdown)


if __name__ == "__main__":
    unittest.main()
