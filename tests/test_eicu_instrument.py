"""Tests for the eICU preference instrument.

The properties asserted here are the ones that make the instrument admissible at
all: no own-treatment leakage, no validation/test leakage, shrinkage of tiny wards,
and the absence of within-client variation in the naive construction this module
replaces.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eicu_instrument import (  # noqa: E402
    DEFAULT_PRIOR_STRENGTH,
    PREFERENCE_HOSPITAL,
    PREFERENCE_WARD,
    PreferenceEstimator,
    assign_folds,
    build_instrument,
    instrument_variation,
    naive_leave_one_out_preference,
    structural_instrument_variation,
)


def make_frame(n_hospitals=4, n_wards=3, per_ward=25, seed=0):
    """Synthetic cohort with genuine ward-level practice variation."""
    rng = np.random.default_rng(seed)
    rows = []
    stay = 0
    for h in range(n_hospitals):
        for w in range(n_wards):
            # Ward practice style: rate varies systematically inside each hospital.
            rate = 0.15 + 0.25 * w
            for _ in range(per_ward):
                rows.append(
                    {
                        "patientunitstayid": stay,
                        "patienthealthsystemstayid": stay,
                        "hospitalid": h,
                        "wardid": h * 100 + w,
                        "treatment": float(rng.random() < rate),
                    }
                )
                stay += 1
    return pd.DataFrame(rows)


class AssignFoldsTest(unittest.TestCase):
    def test_every_row_assigned_within_range(self):
        frame = make_frame()
        folds = assign_folds(frame, "hospitalid", n_folds=5, seed=0)
        self.assertEqual(len(folds), len(frame))
        self.assertTrue(folds.between(0, 4).all())

    def test_folds_are_assigned_within_each_client(self):
        frame = make_frame()
        frame["fold"] = assign_folds(frame, "hospitalid", n_folds=5, seed=0)
        # Every hospital must see every fold, otherwise cross-fitting would drop
        # whole clients on some folds.
        per_hospital = frame.groupby("hospitalid")["fold"].nunique()
        self.assertTrue((per_hospital == 5).all())

    def test_same_admission_never_splits_across_folds(self):
        frame = make_frame()
        # Two unit stays sharing one hospital admission.
        frame.loc[frame.index[:2], "patienthealthsystemstayid"] = 999999
        frame["fold"] = assign_folds(
            frame, "hospitalid", n_folds=5, seed=3, unit_col="patienthealthsystemstayid"
        )
        shared = frame[frame["patienthealthsystemstayid"] == 999999]
        self.assertEqual(shared["fold"].nunique(), 1)

    def test_rejects_single_fold(self):
        with self.assertRaises(ValueError):
            assign_folds(make_frame(), "hospitalid", n_folds=1)


class CrossFittingTest(unittest.TestCase):
    def test_own_treatment_never_enters_own_instrument(self):
        """Flipping one patient's treatment must not move that patient's Z."""
        frame = make_frame()
        frame["fold"] = assign_folds(frame, "hospitalid", n_folds=5, seed=0)

        baseline = PreferenceEstimator(PREFERENCE_WARD).fit(frame).crossfit_transform(frame)

        flipped = frame.copy()
        target = 7
        flipped.loc[target, "treatment"] = 1.0 - flipped.loc[target, "treatment"]
        after = PreferenceEstimator(PREFERENCE_WARD).fit(flipped).crossfit_transform(flipped)

        self.assertAlmostEqual(baseline.loc[target], after.loc[target], places=12)

    def test_flipping_a_patient_moves_other_folds(self):
        """Sanity check on the previous test: the instrument is not simply constant."""
        frame = make_frame()
        frame["fold"] = assign_folds(frame, "hospitalid", n_folds=5, seed=0)

        baseline = PreferenceEstimator(PREFERENCE_WARD).fit(frame).crossfit_transform(frame)
        flipped = frame.copy()
        target = 7
        flipped.loc[target, "treatment"] = 1.0 - flipped.loc[target, "treatment"]
        after = PreferenceEstimator(PREFERENCE_WARD).fit(flipped).crossfit_transform(flipped)

        same_ward = frame["wardid"] == frame.loc[target, "wardid"]
        other_fold = frame["fold"] != frame.loc[target, "fold"]
        moved = (baseline[same_ward & other_fold] - after[same_ward & other_fold]).abs()
        self.assertGreater(moved.max(), 0.0)

    def test_crossfit_requires_fold_column(self):
        frame = make_frame()
        estimator = PreferenceEstimator(PREFERENCE_WARD).fit(frame)
        with self.assertRaises(RuntimeError):
            estimator.crossfit_transform(frame)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            PreferenceEstimator(PREFERENCE_WARD).transform(make_frame())

    def test_rejects_unknown_construction(self):
        with self.assertRaises(ValueError):
            PreferenceEstimator("by_astrological_sign")


class ShrinkageTest(unittest.TestCase):
    def test_tiny_ward_is_pulled_toward_hospital_rate(self):
        """A 1-patient all-treated ward must not report a preference of 1.0."""
        frame = pd.DataFrame(
            {
                "patientunitstayid": range(21),
                "patienthealthsystemstayid": range(21),
                "hospitalid": [0] * 21,
                "wardid": [0] * 20 + [1],
                "treatment": [0.0] * 20 + [1.0],
            }
        )
        z = PreferenceEstimator(PREFERENCE_WARD).fit(frame).transform(frame)
        tiny_ward_z = z.iloc[20]
        self.assertLess(tiny_ward_z, 0.5)
        self.assertGreater(tiny_ward_z, 0.0)

    def test_large_ward_stays_near_its_own_rate(self):
        frame = pd.DataFrame(
            {
                "patientunitstayid": range(400),
                "patienthealthsystemstayid": range(400),
                "hospitalid": [0] * 400,
                "wardid": [0] * 200 + [1] * 200,
                "treatment": [0.0] * 200 + [1.0] * 200,
            }
        )
        z = PreferenceEstimator(
            PREFERENCE_WARD, prior_strength=DEFAULT_PRIOR_STRENGTH
        ).fit(frame).transform(frame)
        self.assertGreater(z.iloc[300], 0.9)
        self.assertLess(z.iloc[0], 0.1)

    def test_stronger_prior_shrinks_further(self):
        frame = pd.DataFrame(
            {
                "patientunitstayid": range(30),
                "patienthealthsystemstayid": range(30),
                "hospitalid": [0] * 30,
                "wardid": [0] * 25 + [1] * 5,
                "treatment": [0.0] * 25 + [1.0] * 5,
            }
        )
        weak = PreferenceEstimator(PREFERENCE_WARD, prior_strength=1.0).fit(frame).transform(frame)
        strong = PreferenceEstimator(PREFERENCE_WARD, prior_strength=50.0).fit(frame).transform(frame)
        self.assertGreater(weak.iloc[27], strong.iloc[27])


class SplitLeakageTest(unittest.TestCase):
    def test_validation_rows_are_scored_from_training_only(self):
        """Changing a validation row's treatment must not change any instrument."""
        frame = make_frame()
        train = frame.iloc[:200].copy()
        val = frame.iloc[200:].copy()

        train_z, other_z, _ = build_instrument(
            train, others={"val": val}, construction=PREFERENCE_WARD, seed=0
        )

        val_flipped = val.copy()
        val_flipped["treatment"] = 1.0 - val_flipped["treatment"]
        train_z2, other_z2, _ = build_instrument(
            train, others={"val": val_flipped}, construction=PREFERENCE_WARD, seed=0
        )

        pd.testing.assert_series_equal(train_z, train_z2)
        pd.testing.assert_series_equal(other_z["val"], other_z2["val"])


class NaiveConstructionTest(unittest.TestCase):
    def test_leave_one_out_hospital_preference_has_two_values_per_hospital(self):
        """The construction this module replaces: Z is affine in the patient's own D."""
        frame = make_frame(n_hospitals=2, n_wards=1, per_ward=50)
        z = naive_leave_one_out_preference(frame, ["hospitalid"])
        for _, rows in frame.assign(z=z).groupby("hospitalid"):
            self.assertLessEqual(rows["z"].round(12).nunique(), 2)

    def test_leave_one_out_is_recoverable_from_own_treatment(self):
        frame = make_frame(n_hospitals=1, n_wards=1, per_ward=40)
        z = naive_leave_one_out_preference(frame, ["hospitalid"])
        n = len(frame)
        total = frame["treatment"].sum()
        implied = (total - z * (n - 1)).round(9)
        np.testing.assert_allclose(implied.values, frame["treatment"].values, atol=1e-9)

    def test_crossfitted_ward_preference_has_real_within_client_variation(self):
        frame = make_frame()
        z, _, _ = build_instrument(frame, construction=PREFERENCE_WARD, seed=0)
        probe = frame.assign(z=z.values)
        within = instrument_variation(probe, "hospitalid", "z")
        self.assertTrue((within > 0.01).all())


class FoldNoiseTest(unittest.TestCase):
    """Cross-fitting creates patient-level spread in Z that is not identifying.

    A hospital with one ward has no ward practice variation, but different folds
    hold out different patients and so produce slightly different preference
    estimates. Counting that as instrument variation would let the pipeline
    certify pure estimation noise as a usable instrument.
    """

    def setUp(self):
        frame = make_frame(n_hospitals=5, n_wards=1, per_ward=80)
        z, _, _ = build_instrument(frame, construction=PREFERENCE_WARD, seed=0)
        self.probe = frame.assign(z=z.values)

    def test_raw_variation_is_nonzero_for_single_ward_hospitals(self):
        raw = instrument_variation(self.probe, "hospitalid", "z")
        self.assertGreater(raw.max(), 0.0)

    def test_structural_variation_is_zero_for_single_ward_hospitals(self):
        structural = structural_instrument_variation(
            self.probe, "hospitalid", "wardid", "z"
        )
        np.testing.assert_allclose(structural.values, 0.0, atol=1e-12)

    def test_structural_variation_is_nonzero_when_wards_differ(self):
        frame = make_frame(n_hospitals=4, n_wards=3, per_ward=40)
        z, _, _ = build_instrument(frame, construction=PREFERENCE_WARD, seed=0)
        structural = structural_instrument_variation(
            frame.assign(z=z.values), "hospitalid", "wardid", "z"
        )
        self.assertTrue((structural > 0.01).all())


class HospitalConstructionTest(unittest.TestCase):
    def test_hospital_preference_varies_within_a_group_client(self):
        frame = make_frame(n_hospitals=6, n_wards=1, per_ward=40)
        frame["client_group"] = frame["hospitalid"] % 2
        z, _, _ = build_instrument(
            frame,
            construction=PREFERENCE_HOSPITAL,
            client_col="client_group",
            seed=0,
        )
        probe = frame.assign(z=z.values)
        within = instrument_variation(probe, "client_group", "z")
        self.assertTrue((within > 0.0).all())


if __name__ == "__main__":
    unittest.main()
