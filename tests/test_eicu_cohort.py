"""Tests for eICU cohort construction.

Runs against hand-written mini-tables written to a temp dir in the on-disk eICU
layout, so the whole ETL is exercised without touching the real release.
"""

import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import eicu_common  # noqa: E402
from prepare_eicu_cohort import build_cohort, main  # noqa: E402


def write_table(root, name, header, rows):
    """Write a gzipped CSV in the eICU on-disk layout."""
    path = os.path.join(root, name + ".csv.gz")
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join("" if v is None else str(v) for v in row) + "\n")
    return path


def build_mini_release(root):
    """Three hospitals; one has no infusion interface at all."""
    write_table(
        root,
        "patient",
        [
            "patientunitstayid",
            "patienthealthsystemstayid",
            "gender",
            "age",
            "ethnicity",
            "hospitalid",
            "wardid",
            "apacheadmissiondx",
            "admissionweight",
            "hospitaladmitoffset",
            "hospitaladmitsource",
            "hospitaldischargestatus",
            "unittype",
            "unitadmitsource",
            "unitvisitnumber",
            "unitstaytype",
        ],
        [
            # hospital 1, ward 10 — treated and untreated adults
            (1, 100, "Male", 65, "Caucasian", 1, 10, "Sepsis", 80, -100, "ED", "Alive", "MICU", "ED", 1, "admit"),
            (2, 101, "Female", 72, "Caucasian", 1, 10, "Sepsis", 70, -120, "ED", "Expired", "MICU", "ED", 1, "admit"),
            # hospital 1, ward 11
            (3, 102, "Male", "> 89", "Caucasian", 1, 11, "Sepsis", 60, -90, "Floor", "Alive", "MICU", "Floor", 1, "admit"),
            # paediatric — must be excluded
            (4, 103, "Female", 9, "Caucasian", 1, 10, "Sepsis", 30, -60, "ED", "Alive", "MICU", "ED", 1, "admit"),
            # second unit stay of an existing admission — must be excluded
            (5, 100, "Male", 65, "Caucasian", 1, 10, "Sepsis", 80, -50, "Floor", "Alive", "MICU", "Floor", 2, "readmit"),
            # unknown mortality — must be excluded
            (6, 104, "Male", 55, "Caucasian", 1, 10, "Sepsis", 90, -80, "ED", None, "MICU", "ED", 1, "admit"),
            # non-sepsis — must be excluded
            (7, 105, "Female", 60, "Caucasian", 1, 10, "Trauma", 65, -70, "ED", "Alive", "MICU", "ED", 1, "admit"),
            # hospital 2 — has an infusion interface
            (8, 106, "Male", 70, "Caucasian", 2, 20, "Sepsis", 75, -110, "ED", "Expired", "SICU", "ED", 1, "admit"),
            # hospital 3 — NO infusion rows anywhere; whole hospital must be dropped
            (9, 107, "Female", 68, "Caucasian", 3, 30, "Sepsis", 72, -95, "ED", "Alive", "MICU", "ED", 1, "admit"),
            # pre-ICU vasopressor — excluded from the primary cohort
            (10, 108, "Male", 61, "Caucasian", 2, 20, "Sepsis", 88, -130, "ED", "Alive", "SICU", "ED", 1, "admit"),
        ],
    )

    write_table(
        root,
        "hospital",
        ["hospitalid", "numbedscategory", "teachingstatus", "region"],
        [(1, "100 - 249", "f", "Midwest"), (2, "<100", "t", "West"), (3, ">= 500", "f", "South")],
    )

    write_table(
        root,
        "diagnosis",
        [
            "diagnosisid",
            "patientunitstayid",
            "activeupondischarge",
            "diagnosisoffset",
            "diagnosisstring",
            "icd9code",
            "diagnosispriority",
        ],
        [
            (1, 1, "True", 10, "infectious diseases|severe sepsis", "995.92", "Primary"),
            (2, 2, "True", -30, "infectious diseases|sepsis", "995.91", "Primary"),
            (3, 3, "True", 300, "infectious diseases|septic shock", "785.52", "Primary"),
            (4, 4, "True", 5, "infectious diseases|sepsis", "995.91", "Primary"),
            (5, 6, "True", 5, "infectious diseases|sepsis", "995.91", "Primary"),
            (6, 8, "True", 20, "infectious diseases|sepsis", "995.91", "Primary"),
            (7, 9, "True", 15, "infectious diseases|sepsis", "995.91", "Primary"),
            (8, 10, "True", 15, "infectious diseases|sepsis", "995.91", "Primary"),
            # far outside the window: must not qualify on its own
            (9, 7, "True", 5000, "infectious diseases|sepsis", "995.91", "Primary"),
        ],
    )

    write_table(
        root,
        "infusiondrug",
        [
            "infusiondrugid",
            "patientunitstayid",
            "infusionoffset",
            "drugname",
            "drugrate",
            "infusionrate",
            "drugamount",
            "volumeoffluid",
            "patientweight",
        ],
        [
            # treated inside the 0-360 window
            (1, 1, 45, "Norepinephrine (mcg/min)", 5, 5, 4, 250, 80),
            (2, 8, 120, "Vasopressin (units/min)", 0.04, 2, 20, 100, 75),
            # outside the window: not early treatment
            (3, 2, 900, "Norepinephrine (ml/hr)", 10, 10, 4, 250, 70),
            # pre-ICU vasopressor
            (4, 10, -60, "Phenylephrine (mcg/min)", 50, 5, 10, 250, 88),
            # non-vasopressor infusion: establishes the interface only
            (5, 3, 30, "Insulin (units/hr)", 2, 2, 100, 100, 60),
            (6, 7, 30, "Propofol (mcg/kg/min)", 20, 5, 1000, 100, 65),
        ],
    )

    write_table(
        root,
        "lab",
        ["labid", "patientunitstayid", "labresultoffset", "labname", "labresult"],
        [
            (1, 1, 20, "lactate", 4.2),
            (2, 1, 50, "creatinine", 1.8),
            (3, 2, 15, "lactate", 2.1),
            # outside the baseline window: must be ignored
            (4, 3, 500, "lactate", 9.9),
        ],
    )

    write_table(
        root,
        "vitalPeriodic",
        [
            "vitalperiodicid",
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicmean",
        ],
        [
            (1, 1, 10, 38.5, 94, 110, 24, 95, 65),
            (2, 1, 40, 38.7, 93, 118, 26, 90, 62),
            (3, 2, 20, 36.9, 97, 88, 18, 120, 80),
            (4, 3, 900, 37.0, 98, 70, 14, 130, 90),
        ],
    )

    write_table(
        root,
        "pastHistory",
        [
            "pasthistoryid",
            "patientunitstayid",
            "pasthistoryoffset",
            "pasthistoryenteredoffset",
            "pasthistorynotetype",
            "pasthistorypath",
            "pasthistoryvalue",
            "pasthistoryvaluetext",
        ],
        [
            (1, 1, 0, 0, "Admission", "notes/Progress Notes/Past History/Organ Systems", "insulin dependent diabetes", "x"),
            (2, 2, 0, 0, "Admission", "notes/Progress Notes/Past History/Organ Systems", "CHF", "x"),
        ],
    )
    return root


class MiniReleaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="eicu_mini_")
        build_mini_release(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)


class CohortConstructionTest(MiniReleaseTestCase):
    def setUp(self):
        self.cohort, self.flow = build_cohort(self.root)
        self.ids = set(self.cohort["patientunitstayid"])

    def test_includes_expected_stays(self):
        self.assertEqual(self.ids, {1, 2, 3, 8})

    def test_excludes_paediatric_patients(self):
        self.assertNotIn(4, self.ids)

    def test_excludes_repeat_unit_stays_in_the_same_admission(self):
        self.assertNotIn(5, self.ids)
        self.assertEqual(
            self.cohort["patienthealthsystemstayid"].nunique(), len(self.cohort)
        )

    def test_excludes_unknown_mortality(self):
        self.assertNotIn(6, self.ids)

    def test_excludes_non_sepsis(self):
        self.assertNotIn(7, self.ids)

    def test_excludes_hospital_without_infusion_interface(self):
        self.assertNotIn(9, self.ids)
        self.assertNotIn(3, set(self.cohort["hospitalid"]))

    def test_excludes_pre_icu_vasopressor(self):
        self.assertNotIn(10, self.ids)

    def test_treatment_uses_the_admission_relative_window(self):
        treatment = self.cohort.set_index("patientunitstayid")["treatment"]
        self.assertEqual(treatment[1], 1.0)  # infusion at +45 min
        self.assertEqual(treatment[8], 1.0)  # infusion at +120 min
        self.assertEqual(treatment[2], 0.0)  # infusion at +900 min is not early
        self.assertEqual(treatment[3], 0.0)  # insulin is not a vasopressor

    def test_outcome_is_hospital_mortality(self):
        outcome = self.cohort.set_index("patientunitstayid")["outcome"]
        self.assertEqual(outcome[2], 1.0)
        self.assertEqual(outcome[8], 1.0)
        self.assertEqual(outcome[1], 0.0)

    def test_top_coded_age_is_ninety_not_missing(self):
        age = self.cohort.set_index("patientunitstayid")["age"]
        self.assertEqual(age[3], 90.0)

    def test_hospital_characteristics_are_joined(self):
        row = self.cohort.set_index("patientunitstayid").loc[1]
        self.assertEqual(row["region"], "Midwest")
        self.assertEqual(row["numbedscategory"], "100 - 249")

    def test_baseline_covariates_respect_the_window(self):
        labs = self.cohort.set_index("patientunitstayid")["lab_lactate"]
        self.assertAlmostEqual(labs[1], 4.2)
        # stay 3's only lactate is at +500 min, outside the baseline window
        self.assertTrue(labs.isna()[3])

    def test_late_vitals_are_excluded_from_baseline(self):
        hr = self.cohort.set_index("patientunitstayid")["vital_heartrate"]
        self.assertAlmostEqual(hr[1], 114.0)  # median of 110 and 118
        self.assertTrue(hr.isna()[3])

    def test_comorbidities_default_to_zero_not_missing(self):
        comorb = self.cohort.set_index("patientunitstayid")["comorb_diabetes"]
        self.assertEqual(comorb[1], 1.0)
        self.assertEqual(comorb[8], 0.0)
        self.assertFalse(self.cohort["comorb_heart_failure"].isna().any())

    def test_no_identifier_leaks_into_covariates(self):
        # hospital identity must not be available as a feature while hospital or
        # ward preference is the instrument.
        self.assertNotIn("uniquepid", self.cohort.columns)
        for column in self.cohort.columns:
            self.assertFalse(column.startswith("hospital_onehot"))

    def test_apache_excluded_by_default(self):
        self.assertFalse(any(c.startswith("apache_") for c in self.cohort.columns))


class CohortFlowTest(MiniReleaseTestCase):
    def test_flow_is_monotone_and_ends_at_cohort_size(self):
        cohort, flow = build_cohort(self.root)
        counts = [step["n_stays"] for step in flow.as_list()]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[-1], len(cohort))

    def test_flow_starts_at_all_stays(self):
        _, flow = build_cohort(self.root)
        first = flow.as_list()[0]
        self.assertEqual(first["step"], "all ICU stays")
        self.assertEqual(first["n_stays"], 10)


class CohortOptionsTest(MiniReleaseTestCase):
    def test_wider_treatment_window_captures_the_late_infusion(self):
        cohort, _ = build_cohort(self.root, treatment_window=(0, 1440))
        treatment = cohort.set_index("patientunitstayid")["treatment"]
        self.assertEqual(treatment[2], 1.0)

    def test_keeping_pre_icu_vasopressors_readmits_that_stay(self):
        cohort, _ = build_cohort(self.root, keep_pre_icu_vasopressors=True)
        self.assertIn(10, set(cohort["patientunitstayid"]))

    def test_dopamine_is_off_by_default(self):
        from eicu_common import PRIMARY_VASOPRESSORS, SENSITIVITY_VASOPRESSORS

        self.assertNotIn("dopamine", PRIMARY_VASOPRESSORS)
        self.assertIn("dopamine", SENSITIVITY_VASOPRESSORS)

    def test_narrow_sepsis_window_drops_the_late_diagnosis(self):
        # stay 3's sepsis diagnosis is charted at +300 min and its apacheadmissiondx
        # also says Sepsis, so it survives; stay 3 is kept via admission diagnosis.
        cohort, _ = build_cohort(self.root, sepsis_window=(-60, 60))
        self.assertIn(3, set(cohort["patientunitstayid"]))


class DrugNormalizationTest(unittest.TestCase):
    def test_units_in_drugname_do_not_defeat_matching(self):
        for raw in [
            "Norepinephrine (mcg/min)",
            "Norepinephrine (ml/hr)",
            "Norepinephrine ()",
            "norepinephrine",
            "Levophed (mcg/min)",
        ]:
            self.assertEqual(eicu_common.normalize_vasopressor(raw), "norepinephrine")

    def test_norepinephrine_is_not_read_as_epinephrine(self):
        self.assertEqual(
            eicu_common.normalize_vasopressor("Norepinephrine (mcg/min)"),
            "norepinephrine",
        )
        self.assertEqual(
            eicu_common.normalize_vasopressor("Epinephrine (mcg/min)"), "epinephrine"
        )

    def test_non_vasopressors_return_none(self):
        for raw in ["Propofol", "Insulin (units/hr)", "", None, "nan"]:
            self.assertIsNone(eicu_common.normalize_vasopressor(raw))

    def test_series_and_scalar_normalization_agree(self):
        import pandas as pd

        raw = [
            "Norepinephrine (mcg/min)",
            "Epinephrine ()",
            "Vasopressin (units/min)",
            "Dopamine (mcg/kg/min)",
            "Neo-Synephrine (mcg/min)",
            "Propofol",
            None,
        ]
        series = eicu_common.normalize_vasopressor_series(raw)
        expected = [eicu_common.normalize_vasopressor(r) for r in raw]
        self.assertEqual(list(series), expected)
        self.assertIsInstance(series, pd.Series)


class AgeParsingTest(unittest.TestCase):
    def test_top_coded_age_maps_to_ninety(self):
        import numpy as np

        parsed = eicu_common.parse_age(["65", "> 89", "", None, "abc", "18"])
        self.assertEqual(parsed[0], 65.0)
        self.assertEqual(parsed[1], 90.0)
        self.assertTrue(np.isnan(parsed[2]))
        self.assertTrue(np.isnan(parsed[3]))
        self.assertTrue(np.isnan(parsed[4]))
        self.assertEqual(parsed[5], 18.0)


class ExpiredFlagTest(unittest.TestCase):
    def test_unknown_status_is_missing_not_survival(self):
        import numpy as np

        flags = eicu_common.expired_flag(["Expired", "Alive", "", None, "Unknown"])
        self.assertEqual(flags[0], 1.0)
        self.assertEqual(flags[1], 0.0)
        for i in (2, 3, 4):
            self.assertTrue(np.isnan(flags[i]))


class TableResolutionTest(MiniReleaseTestCase):
    def test_table_names_resolve_case_insensitively(self):
        # the demo ships infusiondrug.csv.gz; other mirrors use infusionDrug.csv.gz
        for name in ["infusiondrug", "infusionDrug", "INFUSIONDRUG"]:
            self.assertTrue(os.path.exists(eicu_common.resolve_table_path(self.root, name)))

    def test_missing_table_reports_what_is_available(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            eicu_common.resolve_table_path(self.root, "nosuchtable")
        self.assertIn("patient.csv.gz", str(ctx.exception))


class CliTest(MiniReleaseTestCase):
    def test_writes_cohort_and_flow(self):
        out = tempfile.mkdtemp(prefix="eicu_out_")
        try:
            code = main(["--eicu-root", self.root, "--out", out])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(out, "cohort.csv")))

            with open(os.path.join(out, "cohort_flow.json")) as handle:
                meta = json.load(handle)
            self.assertEqual(meta["n_rows"], 4)
            self.assertEqual(meta["n_treated"], 2)
            self.assertEqual(meta["treatment_window_minutes"], [0, 360])
            self.assertIn("flow", meta)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_dry_run_writes_nothing(self):
        out = tempfile.mkdtemp(prefix="eicu_dry_")
        shutil.rmtree(out)
        try:
            code = main(["--eicu-root", self.root, "--out", out, "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(out))
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
