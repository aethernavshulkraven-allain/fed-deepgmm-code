#!/usr/bin/env python3
"""Generate and certify Study A v2 continuous-treatment off-hours scenarios.

The retained eICU cohort supplies hospital clients, a real patient-level
off-hours instrument, baseline covariates, and missingness.  Treatment and
outcome are simulated with a known structural response:

    U ~ N(0, 1)
    X = pi Z + beta'W + eta_h + rho_x U + eps_x
    Y = g0(X, W) + rho_y U + eps_y

Scenario rejection is campaign-wide for a seed: no hospital is removed using a
simulated first stage.  All acceptance diagnostics use Train+Dev only; Test is
never used to choose a DGP attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"

G0_CHOICES = ("linear", "interaction", "mlp")
G0_DISPLAY_LABEL = {
    "linear": "linear",
    "interaction": "interaction",
    "mlp": "frozen_random_mlp",
}
DEFAULT_SCENARIO_SEEDS = (11, 101, 102, 103, 104, 105)
CONTINUOUS_COVARIATES = (
    "age",
    "admissionweight",
    "vital_heartrate",
    "vital_map",
    "vital_respiration",
    "vital_sao2",
    "vital_temperature",
    "lab_creatinine",
    "lab_lactate",
    "lab_wbc",
    "lab_platelets",
    "lab_bilirubin",
    "lab_bicarbonate",
    "lab_bun",
    "lab_sodium",
    "lab_ph",
)
CATEGORICAL_COVARIATES = (
    "gender",
    "hospitaladmitsource",
)
FORBIDDEN_MODEL_COLUMNS = {
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "hospitalid",
    "wardid",
    "z_off_hours",
    "split",
}
CERTIFICATION_THRESHOLDS = {
    "minimum_global_partial_f": 10.0,
    "maximum_abs_corr_u_z": 0.10,
    "minimum_abs_u_coefficient_in_treatment": 0.20,
    "minimum_abs_u_coefficient_in_outcome_residual": 0.20,
    "minimum_naive_structural_coefficient_bias": 0.05,
    "maximum_true_moment_abs": 0.20,
}


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_covariates(
    cohort: pd.DataFrame, train_mask: np.ndarray
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Fit all imputation, scaling, and categorical levels on Train only."""
    blocks: list[np.ndarray] = []
    names: list[str] = []
    preprocessing: dict[str, Any] = {
        "fit_rows": "train_only",
        "continuous": {},
        "categorical": {},
        "dropped_columns": [],
    }

    for column in CONTINUOUS_COVARIATES:
        if column not in cohort:
            continue
        values = pd.to_numeric(cohort[column], errors="coerce").astype("float64")
        train_values = values.iloc[np.flatnonzero(train_mask)]
        observed_train = train_values.dropna()
        if observed_train.empty:
            preprocessing["dropped_columns"].append(
                {"column": column, "reason": "all_missing_in_train"}
            )
            continue
        median = float(observed_train.median())
        filled = values.fillna(median).to_numpy(dtype="float64")
        mean = float(filled[train_mask].mean())
        std = float(filled[train_mask].std())
        if not math.isfinite(std) or std <= 1e-8:
            std = 1.0
        blocks.append((filled - mean) / std)
        names.append(column)
        has_missing = bool(values.isna().any())
        if has_missing:
            blocks.append(values.isna().to_numpy(dtype="float64"))
            names.append(f"{column}_missing")
        preprocessing["continuous"][column] = {
            "median": median,
            "mean_after_imputation": mean,
            "std_after_imputation": std,
            "missing_indicator": has_missing,
        }

    for column in CATEGORICAL_COVARIATES:
        if column not in cohort:
            continue
        values = cohort[column].astype("string").fillna("__missing__")
        train_values = values.iloc[np.flatnonzero(train_mask)]
        categories = sorted(str(value) for value in train_values.unique())
        if not categories:
            preprocessing["dropped_columns"].append(
                {"column": column, "reason": "no_training_categories"}
            )
            continue
        reference = categories[0]
        for category in categories[1:]:
            blocks.append((values == category).to_numpy(dtype="float64"))
            names.append(f"{column}={category}")
        preprocessing["categorical"][column] = {
            "training_categories": categories,
            "reference_category": reference,
            "unknown_category_policy": "all_zero_reference_encoding",
        }

    if not blocks:
        raise ValueError("no usable Study A v2 covariates")
    if FORBIDDEN_MODEL_COLUMNS.intersection(names):
        raise ValueError("an identifier, client key, instrument, or split leaked into W")
    return np.column_stack(blocks), names, preprocessing


def _rng(seed: int, attempt: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([20260727, int(seed), int(attempt), int(stream)])
    )


def generate_common_dgp(
    w: np.ndarray,
    z: np.ndarray,
    client_codes: np.ndarray,
    *,
    scenario_seed: int,
    generation_attempt: int,
    instrument_strength: float,
    rho_x: float,
    treatment_noise: float,
    outcome_noise: float,
    client_heterogeneity: float,
) -> dict[str, np.ndarray]:
    n, n_covariates = w.shape
    treatment_rng = _rng(scenario_seed, generation_attempt, 1)
    noise_rng = _rng(scenario_seed, generation_attempt, 2)
    beta_w = treatment_rng.normal(scale=0.20, size=n_covariates)
    client_effects = treatment_rng.normal(
        scale=client_heterogeneity, size=int(client_codes.max()) + 1
    )
    u = treatment_rng.normal(size=n)
    eps_x = noise_rng.normal(scale=treatment_noise, size=n)
    eps_y = noise_rng.normal(scale=outcome_noise, size=n)
    treatment = (
        instrument_strength * z
        + w @ beta_w
        + client_effects[client_codes]
        + rho_x * u
        + eps_x
    )
    return {
        "treatment": treatment,
        "u": u,
        "eps_x": eps_x,
        "eps_y": eps_y,
        "beta_w_treatment": beta_w,
        "client_effects": client_effects,
    }


def make_g0(
    kind: str,
    n_covariates: int,
    *,
    scenario_seed: int,
    generation_attempt: int,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], dict[str, Any]]:
    stream = {"linear": 10, "interaction": 11, "mlp": 12}[kind]
    rng = _rng(scenario_seed, generation_attempt, stream)

    if kind == "linear":
        a = 1.0
        beta_w = rng.normal(scale=0.15, size=n_covariates)

        def g0(x: np.ndarray, w: np.ndarray) -> np.ndarray:
            return a * x + w @ beta_w

        return g0, {
            "kind": kind,
            "a": a,
            "beta_w": beta_w.tolist(),
        }

    if kind == "interaction":
        a = 1.0
        c = 0.5
        beta_w = rng.normal(scale=0.15, size=n_covariates)

        def g0(x: np.ndarray, w: np.ndarray) -> np.ndarray:
            return a * x + w @ beta_w + c * x * w[:, 0]

        return g0, {
            "kind": kind,
            "a": a,
            "c": c,
            "interaction_covariate_index": 0,
            "beta_w": beta_w.tolist(),
        }

    if kind == "mlp":
        width = 32
        input_dim = n_covariates + 1
        w1 = rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, width))
        b1 = rng.normal(scale=0.05, size=width)
        w2 = rng.normal(scale=1.0 / np.sqrt(width), size=width)
        b2 = float(rng.normal(scale=0.05))

        def g0(x: np.ndarray, w: np.ndarray) -> np.ndarray:
            inputs = np.column_stack([x, w])
            hidden = np.maximum(inputs @ w1 + b1, 0.0)
            return hidden @ w2 + b2

        return g0, {
            "kind": kind,
            "hidden_width": width,
            "activation": "relu",
            "scenario_seed": int(scenario_seed),
            "generation_attempt": int(generation_attempt),
            "w1": w1.tolist(),
            "b1": b1.tolist(),
            "w2": w2.tolist(),
            "b2": b2,
        }
    raise ValueError(f"unknown g0 kind {kind!r}")


def _fit(design: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(design, rcond=1e-10) @ outcome


def _partial_first_stage(
    treatment: np.ndarray, z: np.ndarray, w: np.ndarray
) -> dict[str, float]:
    base = np.column_stack([np.ones(len(z)), w])
    full = np.column_stack([base, z])
    restricted_residual = treatment - base @ _fit(base, treatment)
    full_beta = _fit(full, treatment)
    full_residual = treatment - full @ full_beta
    rss_restricted = float(restricted_residual @ restricted_residual)
    rss_full = float(full_residual @ full_residual)
    denominator_df = max(len(z) - full.shape[1], 1)
    denominator = max(rss_full / denominator_df, 1e-12)
    partial_f = max((rss_restricted - rss_full) / denominator, 0.0)
    return {
        "instrument_coefficient": float(full_beta[-1]),
        "partial_f": float(partial_f),
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def certify_candidate(
    cohort: pd.DataFrame,
    w: np.ndarray,
    common: dict[str, np.ndarray],
    g0: Callable[[np.ndarray, np.ndarray], np.ndarray],
    g0_metadata: dict[str, Any],
    *,
    rho_y: float,
) -> dict[str, Any]:
    """Return a preregistered non-Test acceptance report."""
    split = cohort["split"].to_numpy()
    non_test = split != "test"
    z_all = cohort["z_off_hours"].to_numpy(dtype="float64")
    z = z_all[non_test]
    w_nt = w[non_test]
    x = common["treatment"][non_test]
    u = common["u"][non_test]
    structural = g0(common["treatment"], w)
    outcome = structural + rho_y * common["u"] + common["eps_y"]
    y = outcome[non_test]
    g_true = structural[non_test]

    first_stage = _partial_first_stage(x, z, w_nt)
    x_design = np.column_stack([np.ones(len(x)), z, w_nt, u])
    u_coefficient_x = float(_fit(x_design, x)[-1])
    residual_y = y - g_true
    u_design = np.column_stack([np.ones(len(u)), u])
    u_coefficient_y = float(_fit(u_design, residual_y)[-1])

    moment_basis = np.column_stack([np.ones(len(z)), z, w_nt])
    moment_vector = np.mean(moment_basis * residual_y[:, None], axis=0)
    max_true_moment = float(np.max(np.abs(moment_vector)))

    per_client_z_variation = (
        cohort.groupby("hospitalid")["z_off_hours"].nunique() == 2
    )
    all_clients_have_z_variation = bool(per_client_z_variation.all())

    patient_leakage = int(
        (
            cohort.groupby("uniquepid")["split"].nunique()
            > 1
        ).sum()
    )
    stay_leakage = int(
        (
            cohort.groupby("patienthealthsystemstayid")["split"].nunique()
            > 1
        ).sum()
    )
    clients_per_split = {
        name: set(
            int(value)
            for value in cohort.loc[cohort["split"] == name, "hospitalid"].unique()
        )
        for name in ("train", "dev", "test")
    }
    same_clients_every_split = (
        clients_per_split["train"]
        == clients_per_split["dev"]
        == clients_per_split["test"]
    )

    diagnostics: dict[str, Any] = {
        "acceptance_rows": "train_plus_dev",
        "test_used_for_acceptance": False,
        "n_acceptance_rows": int(non_test.sum()),
        "global_first_stage": first_stage,
        "corr_u_z": _safe_corr(u, z),
        "u_coefficient_in_treatment_given_z_w": u_coefficient_x,
        "u_coefficient_in_outcome_residual": u_coefficient_y,
        "max_abs_empirical_moment_at_true_g0": max_true_moment,
        "all_clients_have_within_hospital_z_variation": all_clients_have_z_variation,
        "same_clients_in_train_dev_test": same_clients_every_split,
        "patient_split_leakage_count": patient_leakage,
        "hospital_stay_split_leakage_count": stay_leakage,
        "linear_diagnostics": None,
    }

    if g0_metadata["kind"] == "linear":
        base = np.column_stack([np.ones(len(x)), w_nt])
        naive_design = np.column_stack([np.ones(len(x)), x, w_nt])
        naive_x_coefficient = float(_fit(naive_design, y)[1])
        first_stage_design = np.column_stack([base, z])
        x_hat = first_stage_design @ _fit(first_stage_design, x)
        second_stage_design = np.column_stack([np.ones(len(x)), x_hat, w_nt])
        two_sls_x_coefficient = float(_fit(second_stage_design, y)[1])
        truth = float(g0_metadata["a"])
        naive_error = abs(naive_x_coefficient - truth)
        two_sls_error = abs(two_sls_x_coefficient - truth)
        diagnostics["linear_diagnostics"] = {
            "true_x_coefficient": truth,
            "naive_x_coefficient": naive_x_coefficient,
            "two_sls_x_coefficient": two_sls_x_coefficient,
            "naive_absolute_error": naive_error,
            "two_sls_absolute_error": two_sls_error,
            "two_sls_improves_over_naive": bool(two_sls_error < naive_error),
        }

    checks = {
        "client_z_variation": all_clients_have_z_variation,
        "global_first_stage": (
            first_stage["partial_f"]
            >= CERTIFICATION_THRESHOLDS["minimum_global_partial_f"]
        ),
        "u_independent_of_z": (
            abs(diagnostics["corr_u_z"])
            <= CERTIFICATION_THRESHOLDS["maximum_abs_corr_u_z"]
        ),
        "u_predicts_treatment": (
            abs(u_coefficient_x)
            >= CERTIFICATION_THRESHOLDS[
                "minimum_abs_u_coefficient_in_treatment"
            ]
        ),
        "u_predicts_outcome": (
            abs(u_coefficient_y)
            >= CERTIFICATION_THRESHOLDS[
                "minimum_abs_u_coefficient_in_outcome_residual"
            ]
        ),
        "true_moments_small": (
            max_true_moment
            <= CERTIFICATION_THRESHOLDS["maximum_true_moment_abs"]
        ),
        "no_split_leakage": (
            patient_leakage == 0
            and stay_leakage == 0
            and same_clients_every_split
        ),
    }
    if diagnostics["linear_diagnostics"] is not None:
        linear = diagnostics["linear_diagnostics"]
        checks["naive_regression_is_biased"] = (
            linear["naive_absolute_error"]
            >= CERTIFICATION_THRESHOLDS[
                "minimum_naive_structural_coefficient_bias"
            ]
        )
        checks["two_sls_improves_over_naive"] = bool(
            linear["two_sls_improves_over_naive"]
        )
    diagnostics["checks"] = checks
    diagnostics["accepted"] = bool(all(checks.values()))
    diagnostics["thresholds"] = CERTIFICATION_THRESHOLDS
    return diagnostics


def pack_splits(
    cohort: pd.DataFrame,
    w: np.ndarray,
    common: dict[str, np.ndarray],
    structural: np.ndarray,
    outcome: np.ndarray,
    g0_x1: np.ndarray,
    g0_x0: np.ndarray,
    client_codes: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    x_model = np.column_stack([common["treatment"], w])
    z_model = np.column_stack(
        [cohort["z_off_hours"].to_numpy(dtype="float64"), w]
    )
    true_effect = g0_x1 - g0_x0
    packed: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "dev", "test"):
        mask = (cohort["split"].to_numpy() == split_name)
        packed[split_name] = {
            "x": x_model[mask],
            "z": z_model[mask],
            "y": outcome[mask, None],
            "g": structural[mask, None],
            "w": x_model[mask],
            "client_id": client_codes[mask].astype("int64"),
            "g0_treated": g0_x1[mask, None],
            "g0_control": g0_x0[mask, None],
            "true_effect": true_effect[mask, None],
            "u": common["u"][mask, None],
            "observed_instrument": cohort.loc[mask, "z_off_hours"].to_numpy(
                dtype="float64"
            )[:, None],
        }
    return packed


def write_scenario(path: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    payload: dict[str, Any] = {"splits": np.asarray(list(splits))}
    for split_name, arrays in splits.items():
        for key, value in arrays.items():
            payload[f"{split_name}_{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def generate_seed(
    cohort: pd.DataFrame,
    *,
    scenario_seed: int,
    output_dir: Path,
    scenario_scope: str,
    max_attempts: int,
    instrument_strength: float,
    rho_x: float,
    rho_y: float,
    treatment_noise: float,
    outcome_noise: float,
    client_heterogeneity: float,
    cohort_path: Path,
    frozen_client_list_path: Path | None,
) -> list[dict[str, Any]]:
    train_mask = cohort["split"].to_numpy() == "train"
    w, covariate_names, preprocessing = build_covariates(cohort, train_mask)
    hospitals = sorted(int(value) for value in cohort["hospitalid"].unique())
    hospital_to_code = {hospital: code for code, hospital in enumerate(hospitals)}
    client_codes = cohort["hospitalid"].map(hospital_to_code).to_numpy(dtype="int64")
    z = cohort["z_off_hours"].to_numpy(dtype="float64")

    accepted_attempt = None
    accepted_common = None
    linear_g0 = None
    linear_meta = None
    linear_certification = None
    attempt_reports = []
    for attempt in range(max_attempts):
        common = generate_common_dgp(
            w,
            z,
            client_codes,
            scenario_seed=scenario_seed,
            generation_attempt=attempt,
            instrument_strength=instrument_strength,
            rho_x=rho_x,
            treatment_noise=treatment_noise,
            outcome_noise=outcome_noise,
            client_heterogeneity=client_heterogeneity,
        )
        g0, g0_meta = make_g0(
            "linear",
            w.shape[1],
            scenario_seed=scenario_seed,
            generation_attempt=attempt,
        )
        certification = certify_candidate(
            cohort, w, common, g0, g0_meta, rho_y=rho_y
        )
        attempt_reports.append(
            {
                "attempt": attempt,
                "accepted": certification["accepted"],
                "checks": certification["checks"],
            }
        )
        if certification["accepted"]:
            accepted_attempt = attempt
            accepted_common = common
            linear_g0 = g0
            linear_meta = g0_meta
            linear_certification = certification
            break

    if accepted_attempt is None or accepted_common is None:
        raise RuntimeError(
            f"scenario_seed={scenario_seed} failed all {max_attempts} "
            f"preregistered global attempts: {attempt_reports}"
        )

    metadata_records = []
    for kind in G0_CHOICES:
        if kind == "linear":
            g0 = linear_g0
            g0_meta = linear_meta
            certification = linear_certification
        else:
            g0, g0_meta = make_g0(
                kind,
                w.shape[1],
                scenario_seed=scenario_seed,
                generation_attempt=accepted_attempt,
            )
            certification = certify_candidate(
                cohort, w, accepted_common, g0, g0_meta, rho_y=rho_y
            )
            if not certification["accepted"]:
                raise RuntimeError(
                    f"accepted common DGP failed {kind} certification for "
                    f"scenario_seed={scenario_seed}: {certification['checks']}"
                )

        treatment = accepted_common["treatment"]
        structural = g0(treatment, w)
        outcome = structural + rho_y * accepted_common["u"] + accepted_common["eps_y"]
        g0_x1 = g0(np.ones(len(cohort)), w)
        g0_x0 = g0(np.zeros(len(cohort)), w)
        true_effect = g0_x1 - g0_x0
        splits = pack_splits(
            cohort,
            w,
            accepted_common,
            structural,
            outcome,
            g0_x1,
            g0_x0,
            client_codes,
        )
        scenario_name = f"{kind}_scenario_seed{scenario_seed}"
        scenario_path = output_dir / f"{scenario_name}.npz"
        write_scenario(scenario_path, splits)
        checksum = sha256_file(scenario_path)

        per_client_true_ate = {
            str(hospital): float(true_effect[client_codes == code].mean())
            for hospital, code in hospital_to_code.items()
        }
        metadata: dict[str, Any] = {
            "protocol_version": "eicu_study_a_v2_offhours",
            "study_label": "Study A v2",
            "scenario_name": scenario_name,
            "scenario_seed": int(scenario_seed),
            "generation_attempt": int(accepted_attempt),
            "generation_attempt_policy": {
                "max_attempts": int(max_attempts),
                "selection_rows": "train_plus_dev",
                "test_used": False,
                "rejects_entire_scenario_not_individual_clients": True,
                "attempt_reports": attempt_reports,
            },
            "scenario_scope": scenario_scope,
            "is_demo": scenario_scope == "demo",
            "scenario_path": scenario_path.name,
            "artifact_path": scenario_path.name,
            "scenario_checksum_sha256": checksum,
            "scenario_checksum": checksum,
            "cohort_path": str(cohort_path.resolve()),
            "cohort_sha256": sha256_file(cohort_path),
            "frozen_client_list_path": (
                str(frozen_client_list_path.resolve())
                if frozen_client_list_path is not None
                else None
            ),
            "frozen_client_list_sha256": (
                sha256_file(frozen_client_list_path)
                if frozen_client_list_path is not None
                else None
            ),
            "client_id_column": "hospitalid",
            "wardid_used_as_client": False,
            "eligible_client_ids": hospitals,
            "eligible_client_provenance": {
                "source": "frozen Study A v2 cohort",
                "simulated_first_stage_used_for_client_filtering": False,
                "n_clients": len(hospitals),
            },
            "n_clients": len(hospitals),
            "n_unique_hospitalid": int(cohort["hospitalid"].nunique()),
            "client_code_to_hospital": {
                str(code): hospital for hospital, code in hospital_to_code.items()
            },
            "n_total": int(len(cohort)),
            "input_dim": int(w.shape[1] + 1),
            "instrument_dim": int(w.shape[1] + 1),
            "outcome_dim": 1,
            "n_features_x": int(w.shape[1] + 1),
            "n_features_z": int(w.shape[1] + 1),
            "n_covariates": int(w.shape[1]),
            "covariate_names": covariate_names,
            "forbidden_model_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
            "preprocessing": preprocessing,
            "instrument": {
                "name": "off_hours_admission",
                "definition": "hospitaladmittime24 outside [07:00, 19:00)",
                "within_hospital_patient_level": True,
                "weekend_instrument_included": False,
            },
            "treatment_type": "continuous",
            "g0": g0_meta,
            "g0_display_label": G0_DISPLAY_LABEL[kind],
            "simulator_coefficients": {
                "instrument_strength_pi": float(instrument_strength),
                "beta_w_treatment": accepted_common[
                    "beta_w_treatment"
                ].tolist(),
                "rho_x": float(rho_x),
                "rho_y": float(rho_y),
                "treatment_noise_sd": float(treatment_noise),
                "outcome_noise_sd": float(outcome_noise),
                "client_heterogeneity_sd": float(client_heterogeneity),
                "hospital_effects_eta": {
                    str(hospital): float(
                        accepted_common["client_effects"][code]
                    )
                    for hospital, code in hospital_to_code.items()
                },
            },
            "effect_contrast": {
                "definition": "g0(X=1,W)-g0(X=0,W)",
                "continuous_treatment": True,
            },
            "sample_weighted_true_ate": float(true_effect.mean()),
            "equal_client_true_ate": float(
                np.mean(list(per_client_true_ate.values()))
            ),
            "per_client_true_ate": per_client_true_ate,
            "split_sizes": {
                name: int(len(arrays["y"])) for name, arrays in splits.items()
            },
            "clients_per_split": {
                name: int(len(np.unique(arrays["client_id"])))
                for name, arrays in splits.items()
            },
            "certification": certification,
            "certification_passed": bool(certification["accepted"]),
            "stored_truth_arrays": [
                "g",
                "g0_treated",
                "g0_control",
                "true_effect",
                "u",
            ],
        }
        metadata_path = output_dir / f"{scenario_name}_metadata.json"
        with metadata_path.open("w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        metadata_records.append(metadata)
    return metadata_records


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one scenario seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("scenario seeds must be unique")
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort",
        default=str(
            REPO_ROOT
            / "experiments"
            / "eicu_study_a_v2_offhours_demo_20260727"
            / "cohort.csv"
        ),
    )
    parser.add_argument(
        "--frozen-client-list",
        default=None,
        help="Defaults to frozen_client_list.json beside --cohort when present.",
    )
    parser.add_argument(
        "--out",
        default=str(EXAMPLE_ROOT / "data" / "eicu_semisynth_offhours_v2_demo"),
    )
    parser.add_argument(
        "--scenario-seeds",
        type=parse_seed_list,
        default=DEFAULT_SCENARIO_SEEDS,
    )
    parser.add_argument(
        "--scenario-scope", choices=("demo", "full_eicu"), default=None
    )
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--instrument-strength", type=float, default=2.0)
    parser.add_argument("--rho-x", type=float, default=1.0)
    parser.add_argument("--rho-y", type=float, default=1.0)
    parser.add_argument("--treatment-noise", type=float, default=0.5)
    parser.add_argument("--outcome-noise", type=float, default=0.5)
    parser.add_argument("--client-heterogeneity", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    cohort_path = Path(args.cohort).resolve()
    cohort = pd.read_csv(cohort_path)
    required = {
        "hospitalid",
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "z_off_hours",
        "split",
    }
    missing = required - set(cohort)
    if missing:
        raise ValueError(f"cohort is missing required columns: {sorted(missing)}")
    if set(cohort["split"]) != {"train", "dev", "test"}:
        raise ValueError("cohort split must contain exactly train/dev/test")

    frozen_client_list_path = (
        Path(args.frozen_client_list).resolve()
        if args.frozen_client_list
        else cohort_path.with_name("frozen_client_list.json")
    )
    if not frozen_client_list_path.exists():
        frozen_client_list_path = None
    scenario_scope = args.scenario_scope or (
        "demo" if "demo" in str(cohort_path).lower() else "full_eicu"
    )
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for scenario_seed in args.scenario_seeds:
        seed_records = generate_seed(
            cohort,
            scenario_seed=scenario_seed,
            output_dir=output_dir,
            scenario_scope=scenario_scope,
            max_attempts=args.max_attempts,
            instrument_strength=args.instrument_strength,
            rho_x=args.rho_x,
            rho_y=args.rho_y,
            treatment_noise=args.treatment_noise,
            outcome_noise=args.outcome_noise,
            client_heterogeneity=args.client_heterogeneity,
            cohort_path=cohort_path,
            frozen_client_list_path=frozen_client_list_path,
        )
        records.extend(seed_records)
        attempt = seed_records[0]["generation_attempt"]
        first_stage = seed_records[0]["certification"]["global_first_stage"][
            "partial_f"
        ]
        print(
            f"scenario_seed={scenario_seed} attempt={attempt} "
            f"partial_f={first_stage:.2f} variants={len(seed_records)}"
        )

    client_lists = {tuple(record["eligible_client_ids"]) for record in records}
    if len(client_lists) != 1:
        raise RuntimeError("generated scenarios do not share one frozen client list")
    if not all(record["certification_passed"] for record in records):
        raise RuntimeError("a written scenario lacks certification")

    summary = {
        "protocol_version": "eicu_study_a_v2_offhours",
        "scenario_scope": scenario_scope,
        "scenario_seeds": list(args.scenario_seeds),
        "g0_variants": list(G0_CHOICES),
        "n_scenarios": len(records),
        "n_clients": len(next(iter(client_lists))),
        "identical_client_list_across_scenarios": True,
        "all_scenarios_certified": True,
        "test_used_for_scenario_acceptance": False,
        "certification_thresholds": CERTIFICATION_THRESHOLDS,
        "scenario_checksums": {
            record["scenario_name"]: record["scenario_checksum_sha256"]
            for record in records
        },
    }
    summary_path = output_dir / "scenario_campaign_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {len(records)} scenarios under {output_dir}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
