"""Tests for the eICU semi-synthetic study (Study A) and natural client partitioning.

Two things are being protected here:

* the simulated causal layer really is confounded and really is identified by the
  instrument, otherwise the benchmark would not test what it claims to test; and
* no information crosses a split boundary -- splits are keyed on hospital admission,
  standardisation uses training rows only, and the instrument never sees a patient's
  own treatment.
"""

import os
import sys
import types
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example"))

from eicu_iv_diagnostics import ols, two_stage_least_squares  # noqa: E402
from prepare_eicu_semisynth import (  # noqa: E402
    ELIGIBILITY_SEED,
    G0_CHOICES,
    G0_DISPLAY_LABEL,
    build_covariates,
    certify_simulated_first_stage,
    file_checksum,
    filter_clients_by_real_z_variation,
    generate,
    infer_scenario_scope,
    make_g0,
    split_by_admission,
    write_scenario,
)


def synthetic_cohort(n_hospitals=12, n_wards=3, per_ward=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    stay = 0
    for h in range(n_hospitals):
        for w in range(n_wards):
            for _ in range(per_ward):
                rows.append(
                    {
                        "patientunitstayid": stay,
                        "patienthealthsystemstayid": stay,
                        "hospitalid": h,
                        "wardid": h * 100 + w,
                        "treatment": float(rng.random() < 0.2 + 0.2 * w),
                        "outcome": float(rng.random() < 0.3),
                        "age": rng.normal(65, 12),
                        "admissionweight": rng.normal(80, 15),
                        "lab_lactate": rng.normal(2.5, 1.0) if rng.random() > 0.2 else np.nan,
                        "comorb_diabetes": float(rng.random() < 0.3),
                        "region": ["Midwest", "West", "South", "Northeast"][h % 4],
                        "teachingstatus": "t" if h % 2 else "f",
                        "numbedscategory": "100 - 249",
                        "gender": "Male" if rng.random() < 0.5 else "Female",
                        "ethnicity": "Caucasian",
                        "unittype": "Med-Surg ICU",
                        "unitstaytype": "admit",
                        "hospitaladmitsource": "Emergency Department",
                    }
                )
                stay += 1
    return pd.DataFrame(rows)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort()
        self.assignment = split_by_admission(
            self.cohort, "hospitalid", np.random.default_rng(0)
        )

    def test_every_row_assigned(self):
        self.assertEqual(set(self.assignment.unique()) <= {"train", "dev", "test"}, True)
        self.assertEqual(len(self.assignment), len(self.cohort))

    def test_admissions_never_span_splits(self):
        frame = self.cohort.assign(split=self.assignment)
        per_admission = frame.groupby("patienthealthsystemstayid")["split"].nunique()
        self.assertTrue((per_admission == 1).all())

    def test_splits_are_within_client(self):
        """Every reasonably sized hospital contributes to all three splits."""
        frame = self.cohort.assign(split=self.assignment)
        per_hospital = frame.groupby("hospitalid")["split"].nunique()
        self.assertTrue((per_hospital == 3).all())

    def test_train_is_the_largest_split(self):
        counts = self.assignment.value_counts()
        self.assertGreater(counts["train"], counts["dev"])
        self.assertGreater(counts["train"], counts["test"])


class CovariateTest(unittest.TestCase):
    def test_standardisation_uses_training_rows_only(self):
        cohort = synthetic_cohort()
        rng = np.random.default_rng(0)
        assignment = split_by_admission(cohort, "hospitalid", rng)
        train_mask = assignment == "train"

        covariates, names = build_covariates(cohort, train_mask)
        age = covariates[:, names.index("age")]
        # Training rows are standardised to mean 0 / sd 1; the other splits are not,
        # which is exactly what "fit on train only" means.
        self.assertAlmostEqual(float(age[train_mask.to_numpy()].mean()), 0.0, places=8)
        self.assertAlmostEqual(float(age[train_mask.to_numpy()].std()), 1.0, places=8)

    def test_missingness_indicator_is_emitted(self):
        cohort = synthetic_cohort()
        assignment = split_by_admission(cohort, "hospitalid", np.random.default_rng(0))
        _, names = build_covariates(cohort, assignment == "train")
        self.assertIn("lab_lactate_missing", names)

    def test_no_identifier_becomes_a_feature(self):
        cohort = synthetic_cohort()
        assignment = split_by_admission(cohort, "hospitalid", np.random.default_rng(0))
        _, names = build_covariates(cohort, assignment == "train")
        for banned in ("hospitalid", "wardid", "patientunitstayid", "patienthealthsystemstayid"):
            self.assertNotIn(banned, names)
            self.assertFalse(any(n.startswith(banned + "=") for n in names))

    def test_covariates_are_finite(self):
        cohort = synthetic_cohort()
        assignment = split_by_admission(cohort, "hospitalid", np.random.default_rng(0))
        covariates, _ = build_covariates(cohort, assignment == "train")
        self.assertTrue(np.isfinite(covariates).all())


class StructuralFunctionTest(unittest.TestCase):
    def test_all_choices_construct(self):
        rng = np.random.default_rng(0)
        for kind in G0_CHOICES:
            g0, meta = make_g0(kind, 5, rng)
            out = g0(np.array([0.0, 1.0]), np.zeros((2, 5)))
            self.assertEqual(out.shape, (2,))
            self.assertEqual(meta["kind"], kind)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            make_g0("quadratic_surprise", 3, np.random.default_rng(0))

    def test_mlp_g0_is_frozen_across_seeds(self):
        a, _ = make_g0("mlp", 4, np.random.default_rng(1))
        b, _ = make_g0("mlp", 4, np.random.default_rng(999))
        x = np.random.default_rng(0).normal(size=(6, 4))
        np.testing.assert_allclose(a(np.ones(6), x), b(np.ones(6), x))


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort()
        self.splits, self.meta = generate(self.cohort, g0_kind="linear", seed=0)

    def test_shapes_are_consistent(self):
        for name, arrays in self.splits.items():
            n = arrays["y"].shape[0]
            self.assertEqual(arrays["x"].shape[0], n)
            self.assertEqual(arrays["z"].shape[0], n)
            self.assertEqual(arrays["g"].shape, (n, 1))
            self.assertEqual(arrays["client_id"].shape, (n,))
        # x = [D, X] and z = [Z, X] share the covariate block, so widths match.
        self.assertEqual(self.meta["n_features_x"], self.meta["n_features_z"])

    def test_treatment_column_is_binary(self):
        d = self.splits["train"]["x"][:, 0]
        self.assertTrue(set(np.unique(d)) <= {0.0, 1.0})

    def test_true_ate_matches_linear_coefficient(self):
        self.assertAlmostEqual(self.meta["true_ate"], 1.0, places=8)

    def test_all_rows_are_accounted_for(self):
        total = sum(v["y"].shape[0] for v in self.splits.values())
        self.assertEqual(total, len(self.cohort))

    def test_confounding_biases_naive_regression(self):
        """If OLS were unbiased the benchmark would not need an instrument."""
        train = self.splits["train"]
        d = train["x"][:, 0]
        y = train["y"].ravel()
        naive = ols(d, y)["coef"][1]
        self.assertGreater(abs(naive - 1.0), 0.2)

    def test_instrument_recovers_the_effect_where_ols_fails(self):
        train = self.splits["train"]
        d = train["x"][:, 0]
        z = train["z"][:, 0]
        x = train["x"][:, 1:]
        y = train["y"].ravel()
        iv = two_stage_least_squares(z, d, covariates=x, outcome=y)["effect"]
        naive = ols(np.column_stack([d, x]), y)["coef"][1]
        self.assertLess(abs(iv - 1.0), abs(naive - 1.0))

    def test_client_heterogeneity_produces_varying_treatment_rates(self):
        train = self.splits["train"]
        frame = pd.DataFrame({"c": train["client_id"], "d": train["x"][:, 0]})
        rates = frame.groupby("c")["d"].mean()
        self.assertGreater(float(rates.std()), 0.05)

    def test_seeds_change_the_draw_but_not_the_shape(self):
        other, meta = generate(self.cohort, g0_kind="linear", seed=1)
        self.assertEqual(meta["n_features_x"], self.meta["n_features_x"])
        self.assertFalse(
            np.allclose(other["train"]["y"][: min(20, len(other["train"]["y"]))],
                        self.splits["train"]["y"][: min(20, len(self.splits["train"]["y"]))])
        )


class NaturalPartitionTest(unittest.TestCase):
    """The loader path that replaces the Dirichlet draw for eICU."""

    @staticmethod
    def _split(client_ids, n_features=3):
        import torch

        n = len(client_ids)
        rng = np.random.default_rng(0)
        make = lambda d: torch.as_tensor(rng.normal(size=(n, d))).double()
        split = types.SimpleNamespace(
            x=make(n_features),
            z=make(n_features),
            y=make(1),
            g=make(1),
            w=make(n_features),
            client_id=np.asarray(client_ids),
        )
        return split

    def _args(self, **kw):
        base = dict(client_num_in_total=3, client_num_per_round=3, batch_size=4)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_partitions_by_client_id(self):
        from fedml.data.MNIST.data_loader import load_data_natural

        ids = [0, 0, 1, 1, 2, 2]
        args = self._args()
        train_local, _, _, counts, _, _ = load_data_natural(
            args, self._split(ids), self._split(ids), self._split(ids)
        )
        self.assertEqual(args.client_num_in_total, 3)
        self.assertEqual(sorted(counts.values()), [2, 2, 2])

    def test_client_sizes_follow_the_data_not_partition_alpha(self):
        from fedml.data.MNIST.data_loader import load_data_natural

        ids = [0] * 10 + [1] * 2 + [2] * 5
        args = self._args(partition_alpha=0.1)
        _, _, _, counts, _, _ = load_data_natural(
            args, self._split(ids), self._split(ids), self._split(ids)
        )
        self.assertEqual(sorted(counts.values()), [2, 5, 10])

    def test_clients_missing_from_a_split_are_dropped(self):
        from fedml.data.MNIST.data_loader import load_data_natural

        train_ids = [0, 0, 1, 1, 2, 2]
        eval_ids = [0, 0, 1, 1]  # client 2 has no dev/test rows
        args = self._args()
        _, _, _, counts, _, _ = load_data_natural(
            args, self._split(train_ids), self._split(eval_ids), self._split(eval_ids)
        )
        self.assertEqual(args.client_num_in_total, 2)
        self.assertEqual(len(counts), 2)

    def test_raises_when_no_client_survives(self):
        from fedml.data.MNIST.data_loader import load_data_natural

        with self.assertRaises(ValueError):
            load_data_natural(
                self._args(),
                self._split([0, 0]),
                self._split([1, 1]),
                self._split([2, 2]),
            )

    def test_requires_client_id(self):
        from fedml.data.MNIST.data_loader import load_data_natural

        split = self._split([0, 0, 1, 1])
        bare = types.SimpleNamespace(
            x=split.x, z=split.z, y=split.y, g=split.g, w=split.w, client_id=None
        )
        with self.assertRaises(ValueError):
            load_data_natural(self._args(), bare, split, split)

    def test_client_num_per_round_is_clamped(self):
        from fedml.data.MNIST.data_loader import load_data_natural

        ids = [0, 0, 1, 1]
        args = self._args(client_num_in_total=2, client_num_per_round=50)
        load_data_natural(args, self._split(ids), self._split(ids), self._split(ids))
        self.assertLessEqual(args.client_num_per_round, args.client_num_in_total)


class ScenarioCertificationTest(unittest.TestCase):
    """P0 item #4: equal-client/sample-weighted ATE, per-client true effects,
    complete simulator coefficients, and a scenario checksum -- everything a
    post-hoc consumer needs without re-deriving g0 itself.
    """

    def setUp(self):
        self.cohort = synthetic_cohort()
        self.splits, self.meta = generate(self.cohort, g0_kind="interaction", seed=0)

    def test_counterfactual_arrays_are_stored_per_split(self):
        for name, arrays in self.splits.items():
            self.assertIn("g0_treated", arrays)
            self.assertIn("g0_control", arrays)
            self.assertIn("true_effect", arrays)
            np.testing.assert_allclose(
                arrays["g0_treated"] - arrays["g0_control"], arrays["true_effect"]
            )

    def test_sample_weighted_ate_matches_pooled_mean_of_true_effect(self):
        all_effects = np.concatenate(
            [v["true_effect"].ravel() for v in self.splits.values()]
        )
        self.assertAlmostEqual(
            self.meta["sample_weighted_true_ate"], float(all_effects.mean()), places=8
        )

    def test_equal_client_ate_is_the_mean_of_per_client_ates(self):
        values = list(self.meta["per_client_true_ate"].values())
        self.assertAlmostEqual(
            self.meta["equal_client_true_ate"], float(np.mean(values)), places=8
        )

    def test_per_client_true_ate_covers_every_hospital(self):
        hospital_ids = set(self.cohort["hospitalid"].unique())
        self.assertEqual(set(self.meta["per_client_true_ate"].keys()), hospital_ids)

    def test_equal_and_sample_weighted_ate_can_genuinely_differ(self):
        """For a treatment-effect-heterogeneous g0, hospital case-mix imbalance
        must be able to move the two aggregates apart -- otherwise storing both
        would be redundant. Needs a genuinely imbalanced cohort: the default
        fixture gives every hospital the same size, so equal-client and
        sample-weighted trivially coincide there.
        """
        imbalanced = synthetic_cohort(n_hospitals=6, n_wards=3, per_ward=5)
        rng = np.random.default_rng(1)
        extra_rows = imbalanced.sample(n=300, replace=True, random_state=1).copy()
        extra_rows["hospitalid"] = 0  # pile almost all extra volume onto one hospital
        stay = int(imbalanced["patientunitstayid"].max()) + 1
        extra_rows["patientunitstayid"] = range(stay, stay + len(extra_rows))
        extra_rows["patienthealthsystemstayid"] = extra_rows["patientunitstayid"]
        imbalanced = pd.concat([imbalanced, extra_rows], ignore_index=True)

        _, meta = generate(imbalanced, g0_kind="interaction", seed=0)
        self.assertNotAlmostEqual(
            meta["sample_weighted_true_ate"],
            meta["equal_client_true_ate"],
            places=3,
        )

    def test_linear_g0_gives_identical_equal_and_sample_weighted_ate(self):
        """Sanity check in the other direction: a constant treatment effect
        (linear g0) must make the two aggregates agree exactly.
        """
        _, meta = generate(self.cohort, g0_kind="linear", seed=0)
        self.assertAlmostEqual(
            meta["sample_weighted_true_ate"], meta["equal_client_true_ate"], places=10
        )

    def test_simulator_coefficients_are_complete(self):
        coeffs = self.meta["simulator_coefficients"]
        for key in ("beta_x_treat", "hospital_offsets", "instrument_z_mean", "instrument_z_std"):
            self.assertIn(key, coeffs)
        self.assertEqual(len(coeffs["beta_x_treat"]), self.meta["n_covariates"])
        self.assertEqual(len(coeffs["hospital_offsets"]), self.meta["n_clients"])

    def test_g0_meta_exposes_its_own_coefficients(self):
        _, g0_meta = make_g0("interaction", 5, np.random.default_rng(0))
        self.assertIn("beta_x", g0_meta)
        self.assertIn("gamma", g0_meta)
        self.assertEqual(len(g0_meta["beta_x"]), 5)
        self.assertEqual(len(g0_meta["gamma"]), 5)

    def test_client_code_to_hospital_is_a_bijection_onto_real_ids(self):
        mapping = self.meta["client_code_to_hospital"]
        hospital_ids = set(self.cohort["hospitalid"].unique())
        self.assertEqual(set(mapping.values()), hospital_ids)
        self.assertEqual(len(set(mapping.values())), len(mapping))  # no collisions


class ClientRelevanceFilterTest(unittest.TestCase):
    """P0 item #5: real-data structural-Z filter before simulation, then a
    per-client first-stage certification on the simulated treatment.
    """

    def test_hospitals_with_real_ward_variation_survive(self):
        cohort = synthetic_cohort(n_hospitals=8, n_wards=3, per_ward=40)
        kept, report = filter_clients_by_real_z_variation(cohort)
        self.assertEqual(report["n_hospitals_after"], 8)
        self.assertEqual(report["n_hospitals_dropped_for_no_z_variation"], 0)
        self.assertEqual(set(kept["hospitalid"].unique()), set(cohort["hospitalid"].unique()))

    def test_single_ward_hospitals_are_dropped(self):
        """No real ward practice variation to exploit -> not usable as a client."""
        cohort = synthetic_cohort(n_hospitals=6, n_wards=1, per_ward=60)
        kept, report = filter_clients_by_real_z_variation(cohort)
        self.assertEqual(report["n_hospitals_after"], 0)
        self.assertEqual(len(kept), 0)

    def test_generate_applies_the_filter_automatically(self):
        """Closes P0 item #3: scenario generation must not silently include
        clients with a degenerate real-data instrument.
        """
        cohort = synthetic_cohort(n_hospitals=6, n_wards=1, per_ward=60)
        with self.assertRaises(ValueError):
            generate(cohort, g0_kind="linear", seed=0)  # no client survives

    def test_generate_records_the_filter_report(self):
        cohort = synthetic_cohort()  # default fixture: real ward variation everywhere
        _, meta = generate(cohort, g0_kind="linear", seed=0)
        self.assertIn("client_filter_report", meta)
        self.assertEqual(meta["client_filter_report"]["n_hospitals_dropped_for_no_z_variation"], 0)

    def test_certify_simulated_first_stage_reports_every_client(self):
        cohort = synthetic_cohort(n_hospitals=6, n_wards=3, per_ward=40)
        rng = np.random.default_rng(0)
        n = len(cohort)
        z = rng.normal(size=n)
        treatment = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.0 * z))).astype("float64")
        client_category = cohort["hospitalid"].astype("category")
        client_codes = client_category.cat.codes.to_numpy()
        mapping = {i: int(h) for i, h in enumerate(client_category.cat.categories)}

        certification, summary = certify_simulated_first_stage(
            treatment, z, client_codes, mapping, np.ones(n, dtype=bool)
        )
        self.assertEqual(summary["n_clients_certified"], 6)
        for hospital_id in cohort["hospitalid"].unique():
            self.assertIn(hospital_id, certification)
            self.assertIn("partial_f", certification[hospital_id])

    def test_certify_simulated_first_stage_skips_tiny_clients(self):
        cohort = synthetic_cohort(n_hospitals=2, n_wards=1, per_ward=3)
        rng = np.random.default_rng(0)
        n = len(cohort)
        z = rng.normal(size=n)
        treatment = rng.random(n).round()
        client_category = cohort["hospitalid"].astype("category")
        client_codes = client_category.cat.codes.to_numpy()
        mapping = {i: int(h) for i, h in enumerate(client_category.cat.categories)}

        certification, summary = certify_simulated_first_stage(
            treatment, z, client_codes, mapping, np.ones(n, dtype=bool)
        )
        self.assertEqual(summary["n_clients_certified"], 0)  # 3 rows each, below the floor

    def test_certify_simulated_first_stage_excludes_test_rows(self):
        """protocol_v1.md S3.2/S8.1: per-client first-stage diagnostics must
        be computed from non-Test rows only. A client with 30 well-behaved
        non-test rows and 30 additional test-only rows must be certified
        identically whether or not those test rows are present in the
        arrays, because non_test_mask excludes them either way.
        """
        cohort = synthetic_cohort(n_hospitals=2, n_wards=3, per_ward=40)
        rng = np.random.default_rng(0)
        n = len(cohort)
        z = rng.normal(size=n)
        treatment = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.0 * z))).astype("float64")
        client_category = cohort["hospitalid"].astype("category")
        client_codes = client_category.cat.codes.to_numpy()
        mapping = {i: int(h) for i, h in enumerate(client_category.cat.categories)}

        # Mark the back half of each hospital's rows as test-only.
        non_test_mask = np.ones(n, dtype=bool)
        non_test_mask[n // 2 :] = False

        cert_excluding_test, summary_excluding = certify_simulated_first_stage(
            treatment, z, client_codes, mapping, non_test_mask
        )
        # Corrupt the excluded (test) rows with noise that would otherwise
        # wreck the first-stage fit -- if the mask were ignored, this would
        # change the certification; since it is honored, it must not.
        treatment_corrupted = treatment.copy()
        treatment_corrupted[~non_test_mask] = rng.random((~non_test_mask).sum()).round()
        cert_with_corrupted_test, summary_with_corrupted = certify_simulated_first_stage(
            treatment_corrupted, z, client_codes, mapping, non_test_mask
        )
        self.assertEqual(summary_excluding, summary_with_corrupted)
        for hospital_id in cert_excluding_test:
            self.assertAlmostEqual(
                cert_excluding_test[hospital_id]["partial_f"],
                cert_with_corrupted_test[hospital_id]["partial_f"],
            )
        self.assertEqual(summary_excluding["diagnostic_rows"], "non_test")


class EligibilityIsFrozenAcrossScenarioSeedsTest(unittest.TestCase):
    """protocol_v1.md S3.2: eligibility must be frozen before scenario
    generation -- the same cohort/g0 must produce the identical eligible
    client set regardless of scenario_seed, since cross-fitting fold
    assignment noise must not be allowed to silently change who is eligible.
    """

    def test_eligible_client_ids_identical_across_scenario_seeds(self):
        cohort = synthetic_cohort(n_hospitals=8, n_wards=3, per_ward=40)
        _, meta_a = generate(cohort, g0_kind="linear", seed=101)
        _, meta_b = generate(cohort, g0_kind="linear", seed=999)
        self.assertEqual(meta_a["eligible_client_ids"], meta_b["eligible_client_ids"])
        self.assertEqual(
            meta_a["client_filter_report"]["eligible_hospital_ids"],
            meta_b["client_filter_report"]["eligible_hospital_ids"],
        )

    def test_filter_uses_fixed_eligibility_seed_not_caller_seed(self):
        cohort = synthetic_cohort(n_hospitals=8, n_wards=3, per_ward=40)
        _, report_default = filter_clients_by_real_z_variation(cohort)
        _, report_explicit = filter_clients_by_real_z_variation(cohort, seed=ELIGIBILITY_SEED)
        self.assertEqual(report_default["eligibility_seed"], ELIGIBILITY_SEED)
        self.assertEqual(report_default["eligible_hospital_ids"], report_explicit["eligible_hospital_ids"])


class ScenarioMetadataProvenanceTest(unittest.TestCase):
    def test_metadata_records_dimensions_and_scope(self):
        cohort = synthetic_cohort()
        _, meta = generate(cohort, g0_kind="mlp", seed=0, scenario_scope="full_eicu")
        self.assertEqual(meta["input_dim"], meta["n_features_x"])
        self.assertEqual(meta["instrument_dim"], meta["n_features_z"])
        self.assertEqual(meta["outcome_dim"], 1)
        self.assertEqual(meta["scenario_scope"], "full_eicu")
        self.assertFalse(meta["is_demo"])
        self.assertEqual(meta["g0_display_label"], "frozen_random_mlp")
        self.assertEqual(meta["g0_display_label"], G0_DISPLAY_LABEL["mlp"])

    def test_demo_scope_defaults_true_for_is_demo(self):
        cohort = synthetic_cohort()
        _, meta = generate(cohort, g0_kind="linear", seed=0)  # default scenario_scope="demo"
        self.assertEqual(meta["scenario_scope"], "demo")
        self.assertTrue(meta["is_demo"])

    def test_eligible_client_provenance_is_recorded(self):
        cohort = synthetic_cohort()
        _, meta = generate(cohort, g0_kind="linear", seed=0)
        provenance = meta["eligible_client_provenance"]
        self.assertIn("real_z_filter", provenance)
        self.assertIn("simulated_first_stage", provenance)
        self.assertEqual(provenance["real_z_filter"]["eligibility_seed"], ELIGIBILITY_SEED)

    def test_display_label_mapping_covers_all_g0_choices(self):
        self.assertEqual(set(G0_DISPLAY_LABEL), set(G0_CHOICES))
        self.assertEqual(G0_DISPLAY_LABEL["linear"], "linear")
        self.assertEqual(G0_DISPLAY_LABEL["interaction"], "interaction")
        self.assertEqual(G0_DISPLAY_LABEL["mlp"], "frozen_random_mlp")


class InferScenarioScopeTest(unittest.TestCase):
    def test_demo_path_is_detected(self):
        self.assertEqual(infer_scenario_scope("experiments/eicu_v1_demo/cohort.csv"), "demo")
        self.assertEqual(infer_scenario_scope("/data/eICU-CRD-Demo/cohort.csv"), "demo")

    def test_non_demo_path_is_full_eicu(self):
        self.assertEqual(infer_scenario_scope("experiments/eicu_v1_full/cohort.csv"), "full_eicu")


class FileChecksumTest(unittest.TestCase):
    def test_checksum_is_deterministic_for_identical_content(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.npz")
            write_scenario({"train": {"x": np.array([[1.0, 2.0]])}}, path)
            self.assertEqual(file_checksum(path), file_checksum(path))

    def test_checksum_changes_when_content_changes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.npz")
            path_b = os.path.join(tmpdir, "b.npz")
            write_scenario({"train": {"x": np.array([[1.0, 2.0]])}}, path_a)
            write_scenario({"train": {"x": np.array([[9.0, 9.0]])}}, path_b)
            self.assertNotEqual(file_checksum(path_a), file_checksum(path_b))

    def test_checksum_is_prefixed_with_the_algorithm(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.npz")
            write_scenario({"train": {"x": np.array([[1.0]])}}, path)
            self.assertTrue(file_checksum(path).startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
