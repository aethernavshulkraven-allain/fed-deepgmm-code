"""Tests for the Stage-1 client feasibility audit.

The audit is a gate, so the property that matters most is that it *refuses* data
that cannot support the analysis — including the release currently on disk.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from audit_eicu_clients import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    decide_construction,
    group_audit,
    hospital_audit,
    main,
    probe_instrument,
)
from eicu_instrument import PREFERENCE_HOSPITAL, PREFERENCE_WARD  # noqa: E402


REGIONS = ("Midwest", "West", "South", "Northeast")
BED_CATEGORIES = ("<100", "100 - 249", "250 - 499", ">= 500")


def synthetic_cohort(n_hospitals=8, n_wards=3, per_ward=120, seed=0):
    """A cohort large enough to be ward-eligible, with real ward practice variation.

    Hospital characteristics span regions and bed categories so that the
    grouped-client fallback produces a realistic number of groups rather than one
    or two degenerate ones.
    """
    rng = np.random.default_rng(seed)
    rows = []
    stay = 0
    for h in range(n_hospitals):
        # Hospital-level practice style, so hospital preference varies inside a group.
        hospital_offset = 0.3 * ((h % 5) / 4.0)
        for w in range(n_wards):
            rate = min(0.15 + 0.2 * w + hospital_offset, 0.9)
            for _ in range(per_ward):
                rows.append(
                    {
                        "patientunitstayid": stay,
                        "patienthealthsystemstayid": stay,
                        "hospitalid": h,
                        "wardid": h * 100 + w,
                        "treatment": float(rng.random() < rate),
                        "outcome": float(rng.random() < 0.25),
                        "region": REGIONS[h % len(REGIONS)],
                        "teachingstatus": "t" if h % 2 else "f",
                        "numbedscategory": BED_CATEGORIES[(h // 2) % len(BED_CATEGORIES)],
                    }
                )
                stay += 1
    return pd.DataFrame(rows)


class HospitalAuditTest(unittest.TestCase):
    def test_counts_reconcile_with_the_cohort(self):
        cohort = synthetic_cohort()
        audit = hospital_audit(cohort, DEFAULT_THRESHOLDS)

        self.assertEqual(len(audit), cohort["hospitalid"].nunique())
        self.assertEqual(audit["n_patients"].sum(), len(cohort))
        self.assertEqual(audit["n_treated"].sum(), int(cohort["treatment"].sum()))
        self.assertEqual(audit["n_deaths"].sum(), int(cohort["outcome"].sum()))

    def test_treated_and_untreated_partition_each_client(self):
        audit = hospital_audit(synthetic_cohort(), DEFAULT_THRESHOLDS)
        np.testing.assert_array_equal(
            (audit["n_treated"] + audit["n_untreated"]).values,
            audit["n_patients"].values,
        )

    def test_large_hospitals_with_multiple_wards_are_eligible(self):
        audit = hospital_audit(synthetic_cohort(), DEFAULT_THRESHOLDS)
        self.assertTrue(audit["ward_eligible"].all())

    def test_single_ward_hospitals_are_not_ward_eligible(self):
        cohort = synthetic_cohort(n_wards=1, per_ward=400)
        audit = hospital_audit(cohort, DEFAULT_THRESHOLDS)
        self.assertFalse(audit["ward_eligible"].any())
        # ...but they can still contribute preference to a grouped client.
        self.assertTrue(audit["contributes_to_group"].all())

    def test_small_hospitals_are_not_eligible(self):
        cohort = synthetic_cohort(per_ward=5)
        audit = hospital_audit(cohort, DEFAULT_THRESHOLDS)
        self.assertFalse(audit["ward_eligible"].any())

    def test_hospital_with_no_treatment_variation_is_not_eligible(self):
        cohort = synthetic_cohort()
        cohort["treatment"] = 0.0
        audit = hospital_audit(cohort, DEFAULT_THRESHOLDS)
        self.assertFalse(audit["ward_eligible"].any())


class DecisionTest(unittest.TestCase):
    def _decide(self, cohort):
        hospitals = hospital_audit(cohort, DEFAULT_THRESHOLDS)
        groups, grouped = group_audit(cohort, DEFAULT_THRESHOLDS)
        ward = probe_instrument(cohort, PREFERENCE_WARD, "hospitalid")
        hosp = (
            probe_instrument(grouped, PREFERENCE_HOSPITAL, "client_group")
            if grouped is not None
            else {"available": False}
        )
        return decide_construction(hospitals, groups, ward, hosp, DEFAULT_THRESHOLDS)

    def test_picks_ward_when_hospitals_are_large_and_multi_ward(self):
        decision = self._decide(synthetic_cohort())
        self.assertEqual(decision["construction"], PREFERENCE_WARD)
        self.assertGreaterEqual(decision["n_ward_eligible_hospitals"], 5)

    def test_falls_back_to_grouped_when_hospitals_are_single_ward(self):
        # Many single-ward hospitals, each too small to be its own client, but
        # plenty of hospital-level preference variation inside each group.
        cohort = synthetic_cohort(n_hospitals=40, n_wards=1, per_ward=60)
        decision = self._decide(cohort)
        self.assertEqual(decision["construction"], "grouped")
        self.assertEqual(decision["n_ward_eligible_hospitals"], 0)
        self.assertGreaterEqual(decision["n_eligible_groups"], 5)

    def test_refuses_when_nothing_is_large_enough(self):
        cohort = synthetic_cohort(n_hospitals=3, n_wards=1, per_ward=4)
        decision = self._decide(cohort)
        self.assertEqual(decision["construction"], "insufficient_data")
        self.assertTrue(any("cannot support" in r for r in decision["reasons"]))

    def test_decision_records_the_frozen_thresholds(self):
        decision = self._decide(synthetic_cohort())
        self.assertEqual(decision["thresholds"], DEFAULT_THRESHOLDS)


class InstrumentProbeTest(unittest.TestCase):
    def test_ward_probe_finds_within_hospital_variation(self):
        probe = probe_instrument(synthetic_cohort(), PREFERENCE_WARD, "hospitalid")
        self.assertTrue(probe["available"])
        self.assertEqual(probe["n_clients_with_variation"], probe["n_clients"])

    def test_single_ward_hospitals_have_no_within_client_ward_variation(self):
        cohort = synthetic_cohort(n_hospitals=6, n_wards=1, per_ward=100)
        probe = probe_instrument(cohort, PREFERENCE_WARD, "hospitalid")
        self.assertEqual(probe["n_clients_with_variation"], 0)


class RealDemoReleaseTest(unittest.TestCase):
    """The audit must refuse the eICU demo currently on disk."""

    COHORT = os.path.join(REPO_ROOT, "experiments", "eicu_v1_demo", "cohort.csv")

    @unittest.skipUnless(
        os.path.exists(COHORT), "demo cohort not built; run prepare_eicu_cohort.py"
    )
    def test_demo_release_is_refused(self):
        cohort = pd.read_csv(self.COHORT)
        hospitals = hospital_audit(cohort, DEFAULT_THRESHOLDS)
        groups, grouped = group_audit(cohort, DEFAULT_THRESHOLDS)
        decision = decide_construction(
            hospitals,
            groups,
            probe_instrument(cohort, PREFERENCE_WARD, "hospitalid"),
            probe_instrument(grouped, PREFERENCE_HOSPITAL, "client_group"),
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(decision["construction"], "insufficient_data")
        self.assertEqual(decision["n_ward_eligible_hospitals"], 0)


class CliTest(unittest.TestCase):
    def test_writes_audit_artifacts(self):
        out = tempfile.mkdtemp(prefix="eicu_audit_")
        try:
            cohort_path = os.path.join(out, "cohort.csv")
            synthetic_cohort().to_csv(cohort_path, index=False)

            code = main(["--cohort", cohort_path, "--out", out])
            self.assertEqual(code, 0)

            for name in (
                "client_audit.csv",
                "client_audit.md",
                "construction_decision.json",
            ):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)

            with open(os.path.join(out, "construction_decision.json")) as handle:
                decision = json.load(handle)
            self.assertEqual(decision["construction"], PREFERENCE_WARD)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_thresholds_are_overridable_from_the_cli(self):
        out = tempfile.mkdtemp(prefix="eicu_audit_thr_")
        try:
            cohort_path = os.path.join(out, "cohort.csv")
            synthetic_cohort(per_ward=60).to_csv(cohort_path, index=False)

            main(["--cohort", cohort_path, "--out", out, "--min-patients", "1000"])
            with open(os.path.join(out, "construction_decision.json")) as handle:
                decision = json.load(handle)
            self.assertEqual(decision["thresholds"]["min_patients"], 1000)
            self.assertEqual(decision["n_ward_eligible_hospitals"], 0)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
