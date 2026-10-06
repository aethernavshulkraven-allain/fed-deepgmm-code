"""Study A: semi-synthetic eICU benchmark.

Keeps everything real that carries the federated difficulty — hospital partitions,
covariate distributions, missingness pattern, ward structure, client size imbalance,
and the candidate instrument — and simulates only the causal layer:

    U  ~ N(0, 1)                                    unobserved confounder
    D  ~ Bernoulli(sigma(a*Z + b'X + c*U + xi_h))   treatment
    Y  = g0(D, X) + rho*U + eps                     outcome

Because ``g0`` is known, structural MSE against the truth is computable — which is
the metric the rest of the repo is built around and the one thing a real eICU
analysis can never provide. That is what makes this a paper result rather than an
anecdote, and it is why Study A does not need the ground-truth-free selection path.

``U`` enters both ``D`` and ``Y``, so OLS is biased and an instrument is genuinely
required; ``xi_h`` is a per-hospital treatment-propensity offset, so clients differ
in treatment prevalence the way real hospitals do.

Tensor packing follows the existing scenario contract so no objective or optimizer
changes are needed:

    x = [D, X]  ->  g(x) is g_theta(D, X)
    z = [Z, X]  ->  f(z) is f_tau(Z, X)

Usage:
    python scripts/prepare_eicu_semisynth.py --cohort experiments/eicu_v1_demo/cohort.csv \\
        --g0 linear --scenario-seed 0
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eicu_common import (  # noqa: E402
    CATEGORICAL_COVARIATES,
    continuous_covariate_columns,
)
from eicu_instrument import (  # noqa: E402
    PREFERENCE_WARD,
    build_instrument,
    structural_instrument_variation,
)
from eicu_iv_diagnostics import first_stage_diagnostics  # noqa: E402

EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")

G0_CHOICES = ("linear", "interaction", "mlp")
# protocol_v1.md S4.1: "The displayed label frozen_random_mlp maps to
# implementation label mlp only if metadata records that mapping."
G0_DISPLAY_LABEL = {"linear": "linear", "interaction": "interaction", "mlp": "frozen_random_mlp"}

DEFAULT_SPLIT = (0.6, 0.2, 0.2)


# ---------------------------------------------------------------------------
# Covariate matrix
# ---------------------------------------------------------------------------


def build_covariates(cohort, train_mask):
    """Numeric X with train-only standardisation and missingness indicators.

    Standardisation statistics come from training rows only. The repo's existing
    loaders standardise before partitioning, which leaks validation and test
    information; doing it here on the training mask keeps the semi-synthetic study
    honest about what a federated estimator could actually know.
    """
    import pandas as pd

    blocks, names = [], []

    for col in continuous_covariate_columns(cohort):
        values = cohort[col].astype("float64")
        missing = values.isna()
        if missing.all():
            continue
        median = values[train_mask].median()
        if not np.isfinite(median):
            median = 0.0
        filled = values.fillna(median).to_numpy()

        mean = filled[train_mask.to_numpy()].mean()
        std = filled[train_mask.to_numpy()].std()
        std = std if std > 1e-8 else 1.0
        blocks.append((filled - mean) / std)
        names.append(col)

        if missing.any():
            # Missingness in eICU partly encodes which interfaces a hospital
            # connected, so it is signal, not just absence.
            blocks.append(missing.astype("float64").to_numpy())
            names.append(col + "_missing")

    for col in CATEGORICAL_COVARIATES:
        if col not in cohort.columns:
            continue
        values = cohort[col].astype("string").fillna("__missing__")
        # Drop-first one-hot, categories fixed by the training rows.
        categories = sorted(values[train_mask].unique())[1:]
        for category in categories:
            blocks.append((values == category).astype("float64").to_numpy())
            names.append(f"{col}={category}")

    for col in sorted(c for c in cohort.columns if c.startswith("comorb_")):
        blocks.append(cohort[col].astype("float64").fillna(0.0).to_numpy())
        names.append(col)

    if not blocks:
        raise ValueError("no usable covariates found in the cohort")
    return np.column_stack(blocks), names


# ---------------------------------------------------------------------------
# Structural functions
# ---------------------------------------------------------------------------


def make_g0(kind, n_features, rng):
    """Return ``g0(D, X)`` plus a JSON-serialisable description of it."""
    if kind == "linear":
        beta_d = 1.0
        beta_x = rng.normal(scale=0.3, size=n_features)

        def g0(d, x):
            return beta_d * d + x @ beta_x

        return g0, {"kind": kind, "beta_d": beta_d, "beta_x": beta_x.tolist()}

    if kind == "interaction":
        beta_d = 1.0
        beta_x = rng.normal(scale=0.3, size=n_features)
        gamma = rng.normal(scale=0.3, size=n_features)

        def g0(d, x):
            # Treatment effect varies with X, so the ATE is not a single coefficient.
            return beta_d * d + d * np.tanh(x @ gamma) + x @ beta_x

        return g0, {
            "kind": kind,
            "beta_d": beta_d,
            "beta_x": beta_x.tolist(),
            "gamma": gamma.tolist(),
        }

    if kind == "mlp":
        # Frozen random MLP, identical across seeds for a given width so that
        # different optimisation seeds target the same function.
        fixed = np.random.default_rng(20260725)
        w1 = fixed.normal(scale=1.0 / np.sqrt(n_features + 1), size=(n_features + 1, 32))
        b1 = fixed.normal(scale=0.1, size=32)
        w2 = fixed.normal(scale=1.0 / np.sqrt(32), size=32)

        def g0(d, x):
            h = np.tanh(np.column_stack([d, x]) @ w1 + b1)
            return h @ w2

        return g0, {"kind": kind, "hidden": 32, "frozen_seed": 20260725}

    raise ValueError(f"unknown --g0 {kind!r}; choose from {G0_CHOICES}")


# ---------------------------------------------------------------------------
# Client relevance: pre-simulation filter, post-simulation certification
# ---------------------------------------------------------------------------

# Below this, a client's instrument is indistinguishable from cross-fitting
# noise (see eicu_instrument.structural_instrument_variation) -- matches the
# threshold audit_eicu_clients.py uses for the same reason.
MIN_STRUCTURAL_Z_SD = 0.01

# protocol_v1.md S3.2 ("Eligibility must be frozen before scenario
# generation"): which hospitals pass the real-Z structural-variation filter
# must be a fixed, cohort-level decision, not something that can drift with
# whichever scenario_seed a caller happens to pass. Passing scenario_seed
# straight through to build_instrument's cross-fitting fold assignment would
# let fold noise silently change the eligible client set between otherwise
# identical confirmatory seed pairs. A dedicated, fixed seed used only for
# this determination decouples "who is eligible" from "which DGP draw this
# replicate uses."
ELIGIBILITY_SEED = 20260101


def filter_clients_by_real_z_variation(
    cohort, client_col="hospitalid", min_structural_sd=MIN_STRUCTURAL_Z_SD, seed=ELIGIBILITY_SEED
):
    """Drop hospitals with no genuine real-data ward-preference variation.

    Answers "does the *real* ward-preference Z vary structurally within this
    hospital?" -- using the cohort's real `treatment` column, before anything
    is simulated. This is the same check audit_eicu_clients.py performs for
    reporting; here it is enforced, not just reported, because a hospital
    whose instrument is constant contributes a simulated D that Z cannot
    possibly predict, regardless of instrument_strength.

    ``seed`` defaults to the frozen ``ELIGIBILITY_SEED``, not a caller's
    scenario_seed -- see the module-level comment above.
    """
    z, _, _ = build_instrument(cohort, construction=PREFERENCE_WARD, client_col=client_col, seed=seed)
    probe = cohort.assign(_z=z.values)
    structural_sd = structural_instrument_variation(probe, client_col, "wardid", "_z")
    eligible = structural_sd[structural_sd > min_structural_sd].index
    kept = cohort[cohort[client_col].isin(eligible)].reset_index(drop=True)
    dropped = int(cohort[client_col].nunique() - len(eligible))
    return kept, {
        "min_structural_z_sd": min_structural_sd,
        "eligibility_seed": int(seed),
        "n_hospitals_before": int(cohort[client_col].nunique()),
        "n_hospitals_after": int(len(eligible)),
        "n_hospitals_dropped_for_no_z_variation": dropped,
        "eligible_hospital_ids": sorted(int(h) for h in eligible),
    }


def certify_simulated_first_stage(
    treatment, z_scaled, client_codes, client_code_to_hospital, non_test_mask
):
    """Per-client first-stage strength on the *simulated* treatment,
    restricted to non-Test rows.

    Answers the second relevance question: "does Z predict the newly
    simulated D within each hospital?" A real-data F-statistic computed
    before D exists (filter_clients_by_real_z_variation, above) is necessary
    but not the final word -- this checks the thing that is actually used
    for training. protocol_v1.md S3.2/S8.1: any per-client relevance
    diagnostic that could influence configuration or eligibility must be
    computed only from non-Test rows (train + dev here), never Test, even
    though it is currently recorded as informational rather than an
    enforced gate.
    """
    certification = {}
    for code in np.unique(client_codes):
        mask = (client_codes == code) & non_test_mask
        if mask.sum() < 5:  # too few rows for a meaningful within-client regression
            continue
        diag = first_stage_diagnostics(z_scaled[mask], treatment[mask])
        certification[client_code_to_hospital[int(code)]] = {
            "n": int(mask.sum()),
            "instrument_coef": diag["instrument_coef"],
            "partial_f": diag["partial_f"],
            "weak_instrument_warning": diag["weak_instrument_warning"],
        }
    n_weak = sum(1 for v in certification.values() if v["weak_instrument_warning"])
    weak_hospital_ids = sorted(
        hospital_id for hospital_id, v in certification.items() if v["weak_instrument_warning"]
    )
    return certification, {
        "n_clients_certified": len(certification),
        "n_clients_weak_simulated_first_stage": n_weak,
        "weak_simulated_first_stage_hospital_ids": weak_hospital_ids,
        "diagnostic_rows": "non_test",
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def split_by_admission(cohort, client_col, rng, fractions=DEFAULT_SPLIT):
    """Within-client 60/20/20 split, keyed on hospital admission.

    Splitting inside each client (rather than globally) is what makes a client's
    train and validation rows the same subpopulation. The repo's Dirichlet loader
    partitions the three splits independently, which is harmless for i.i.d.
    synthetic data and wrong for hospitals.
    """
    import pandas as pd

    assignment = pd.Series("train", index=cohort.index, dtype=object)
    unit_col = (
        "patienthealthsystemstayid"
        if "patienthealthsystemstayid" in cohort.columns
        else None
    )

    for _, rows in cohort.groupby(client_col, sort=True):
        units = (
            rows[unit_col].unique()
            if unit_col
            else np.asarray(rows.index)
        )
        order = rng.permutation(len(units))
        n_train = max(int(round(fractions[0] * len(units))), 1)
        n_dev = int(round(fractions[1] * len(units)))
        # Guarantee every client contributes to every split when it is big enough,
        # so no client silently vanishes from validation.
        if len(units) >= 3:
            n_dev = max(n_dev, 1)
            n_train = min(n_train, len(units) - 2)

        chosen = {}
        for rank, idx in enumerate(order):
            unit = units[idx]
            if rank < n_train:
                chosen[unit] = "train"
            elif rank < n_train + n_dev:
                chosen[unit] = "dev"
            else:
                chosen[unit] = "test"

        key = rows[unit_col] if unit_col else pd.Series(rows.index, index=rows.index)
        assignment.loc[rows.index] = key.map(chosen).values

    return assignment


def generate(
    cohort,
    g0_kind="linear",
    seed=0,
    instrument_strength=2.0,
    confounding=1.0,
    rho=1.0,
    noise=0.5,
    client_heterogeneity=0.5,
    client_col="hospitalid",
    min_structural_z_sd=MIN_STRUCTURAL_Z_SD,
    scenario_scope="demo",
):
    import pandas as pd

    rng = np.random.default_rng(seed)

    cohort, client_filter_report = filter_clients_by_real_z_variation(
        cohort, client_col=client_col, min_structural_sd=min_structural_z_sd, seed=ELIGIBILITY_SEED
    )

    assignment = split_by_admission(cohort, client_col, rng)
    train_mask = assignment == "train"

    covariates, names = build_covariates(cohort, train_mask)
    n, n_features = covariates.shape

    # Instrument is fitted on training rows only and cross-fitted within them.
    train_frame = cohort[train_mask].copy()
    others = {
        "dev": cohort[assignment == "dev"].copy(),
        "test": cohort[assignment == "test"].copy(),
    }
    train_z, other_z, _ = build_instrument(
        train_frame,
        others=others,
        construction=PREFERENCE_WARD,
        client_col=client_col,
        seed=seed,
    )
    instrument = pd.Series(np.nan, index=cohort.index, dtype="float64")
    instrument.loc[train_frame.index] = train_z.values
    for name, frame in others.items():
        if len(frame):
            instrument.loc[frame.index] = other_z[name].values
    instrument = instrument.fillna(instrument[train_mask].mean()).to_numpy()

    # Centre the instrument on the training split so its scale does not smuggle
    # information from dev/test into the design.
    z_mean = instrument[train_mask.to_numpy()].mean()
    z_std = instrument[train_mask.to_numpy()].std()
    z_std = z_std if z_std > 1e-8 else 1.0
    z_scaled = (instrument - z_mean) / z_std

    # Per-hospital propensity offset -> genuine treatment-prevalence heterogeneity.
    client_category = cohort[client_col].astype("category")
    client_codes = client_category.cat.codes.to_numpy()
    client_code_to_hospital = {
        int(code): int(hospital_id)
        for code, hospital_id in enumerate(client_category.cat.categories)
    }
    offsets = rng.normal(scale=client_heterogeneity, size=client_codes.max() + 1)
    xi = offsets[client_codes]

    confounder = rng.normal(size=n)
    beta_x_treat = rng.normal(scale=0.2, size=n_features)
    logit = (
        instrument_strength * z_scaled
        + covariates @ beta_x_treat
        + confounding * confounder
        + xi
    )
    treatment = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype("float64")

    # Second relevance checkpoint: does Z predict the treatment that was
    # actually simulated, within each hospital? (The first checkpoint, above,
    # already confirmed real ward-preference variation before D existed.)
    # Restricted to non-Test rows -- see certify_simulated_first_stage's docstring.
    non_test_mask = (assignment != "test").to_numpy()
    first_stage_certification, first_stage_summary = certify_simulated_first_stage(
        treatment, z_scaled, client_codes, client_code_to_hospital, non_test_mask
    )

    g0, g0_meta = make_g0(g0_kind, n_features, rng)
    structural = g0(treatment, covariates)
    outcome = structural + rho * confounder + rng.normal(scale=noise, size=n)

    # Counterfactual pair, per sample -- stored (not just summarised), so a
    # post-hoc consumer can recompute any ATE aggregate directly from the
    # scenario file without needing to re-derive g0 itself (frozen-MLP g0 in
    # particular is not something a downstream script should reconstruct from
    # scratch, since it depends on the exact fixed random draw in make_g0).
    g0_treated_all = g0(np.ones(n), covariates)
    g0_control_all = g0(np.zeros(n), covariates)
    true_effect_all = g0_treated_all - g0_control_all

    x = np.column_stack([treatment, covariates])
    z = np.column_stack([z_scaled, covariates])

    splits = {}
    for split_name in ("train", "dev", "test"):
        mask = (assignment == split_name).to_numpy()
        splits[split_name] = {
            "x": x[mask],
            "z": z[mask],
            "y": outcome[mask].reshape(-1, 1),
            "g": structural[mask].reshape(-1, 1),
            "w": x[mask],
            "client_id": client_codes[mask].astype("int64"),
            "g0_treated": g0_treated_all[mask].reshape(-1, 1),
            "g0_control": g0_control_all[mask].reshape(-1, 1),
            "true_effect": true_effect_all[mask].reshape(-1, 1),
        }

    # True effects, at three levels of aggregation. All three are the same
    # per-sample true_effect_all, aggregated differently -- computed once here
    # so every downstream script (post-hoc eval, tuning selection, reporting)
    # compares against one authoritative number rather than re-deriving it.
    sample_weighted_true_ate = float(true_effect_all.mean())
    per_client_true_ate = {}
    for code in np.unique(client_codes):
        client_mask = client_codes == code
        per_client_true_ate[client_code_to_hospital[int(code)]] = float(
            true_effect_all[client_mask].mean()
        )
    equal_client_true_ate = float(np.mean(list(per_client_true_ate.values())))

    metadata = {
        "g0": g0_meta,
        "g0_display_label": G0_DISPLAY_LABEL[g0_kind],
        # protocol_v1.md S7.1: this is the scenario/DGP seed. It never
        # controls model init/minibatch/optimizer randomness (this script
        # generates only the scenario NPZ; no training happens here) --
        # named scenario_seed, not the ambiguous "seed", so a downstream
        # consumer reading this metadata alongside a run's optimizer_seed
        # cannot mistake one for the other.
        "scenario_seed": seed,
        "n_total": int(n),
        "n_features_x": int(x.shape[1]),
        "n_features_z": int(z.shape[1]),
        # default_contract.json's canonical dimension field names, alongside
        # the n_features_x/z aliases above (kept for backward compatibility).
        "input_dim": int(x.shape[1]),
        "instrument_dim": int(z.shape[1]),
        "outcome_dim": int(outcome.reshape(-1, 1).shape[1]),
        "n_covariates": int(n_features),
        "n_clients": int(len(np.unique(client_codes))),
        "covariate_names": names,
        "client_code_to_hospital": client_code_to_hospital,
        # protocol_v1.md S3.2: eligibility is a frozen, cohort-level decision
        # (see ELIGIBILITY_SEED) -- this is the exact client set every
        # scenario_seed for this cohort/g0 combination will share.
        "eligible_client_ids": sorted(client_code_to_hospital.values()),
        "eligible_client_provenance": {
            "method": "structural_instrument_variation_then_simulated_first_stage",
            "real_z_filter": client_filter_report,
            "simulated_first_stage": first_stage_summary,
        },
        "scenario_scope": scenario_scope,
        "is_demo": scenario_scope == "demo",
        "instrument_strength": instrument_strength,
        "confounding": confounding,
        "rho": rho,
        "noise": noise,
        "client_heterogeneity": client_heterogeneity,
        "simulator_coefficients": {
            "beta_x_treat": beta_x_treat.tolist(),
            "hospital_offsets": {
                client_code_to_hospital[code]: float(offset)
                for code, offset in enumerate(offsets)
            },
            "instrument_z_mean": float(z_mean),
            "instrument_z_std": float(z_std),
        },
        "sample_weighted_true_ate": sample_weighted_true_ate,
        "equal_client_true_ate": equal_client_true_ate,
        "per_client_true_ate": per_client_true_ate,
        "client_filter_report": client_filter_report,
        "first_stage_certification_summary": first_stage_summary,
        "per_client_first_stage_certification": first_stage_certification,
        "treatment_rate": float(treatment.mean()),
        "true_ate": sample_weighted_true_ate,  # alias, kept for backward compatibility
        "split_sizes": {k: int(v["y"].shape[0]) for k, v in splits.items()},
        "clients_per_split": {
            k: int(len(np.unique(v["client_id"]))) for k, v in splits.items()
        },
    }
    return splits, metadata


def write_scenario(splits, path):
    payload = {"splits": list(splits)}
    for name, arrays in splits.items():
        for key, value in arrays.items():
            payload[f"{name}_{key}"] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **payload)


def file_checksum(path, algorithm="sha256"):
    """Hash of the written scenario file's exact bytes.

    Lets a downstream consumer (tuning, confirmatory runs, the post-hoc eval
    script) assert it is reading the exact scenario that was certified, not a
    regenerated one that happens to share a filename -- regeneration is not
    guaranteed byte-identical across numpy/platform versions even with the
    same seed.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return f"{algorithm}:{hasher.hexdigest()}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort",
        default=os.path.join(REPO_ROOT, "experiments", "eicu_v1_demo", "cohort.csv"),
    )
    parser.add_argument("--g0", choices=G0_CHOICES, default="linear")
    parser.add_argument(
        "--scenario-seed", "--seed", dest="scenario_seed", type=int, default=0,
        help="DGP/split seed (protocol_v1.md scenario_seed). --seed is a deprecated alias.",
    )
    parser.add_argument("--instrument-strength", type=float, default=2.0)
    parser.add_argument("--confounding", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--client-heterogeneity", type=float, default=0.5)
    parser.add_argument(
        "--out",
        default=None,
        help="target .npz (default data/eicu_semisynth/<g0>_scenario_seed<scenario_seed>.npz)",
    )
    parser.add_argument(
        "--scenario-scope", choices=("demo", "full_eicu"), default=None,
        help=(
            "protocol_v1.md: full-eICU only for final/reportable results, demo=smoke "
            "only. Defaults to inferring from --cohort's path (contains 'demo')."
        ),
    )
    return parser.parse_args(argv)


def infer_scenario_scope(cohort_path):
    return "demo" if "demo" in str(cohort_path).lower() else "full_eicu"


def main(argv=None):
    import pandas as pd

    args = parse_args(argv)
    cohort = pd.read_csv(args.cohort).reset_index(drop=True)
    scenario_scope = args.scenario_scope or infer_scenario_scope(args.cohort)

    splits, metadata = generate(
        cohort,
        g0_kind=args.g0,
        seed=args.scenario_seed,
        instrument_strength=args.instrument_strength,
        confounding=args.confounding,
        rho=args.rho,
        noise=args.noise,
        client_heterogeneity=args.client_heterogeneity,
        scenario_scope=scenario_scope,
    )

    out = args.out or os.path.join(
        EXAMPLE_ROOT, "data", "eicu_semisynth", f"{args.g0}_scenario_seed{args.scenario_seed}.npz"
    )
    out = os.path.abspath(out)
    write_scenario(splits, out)

    metadata["scenario_path"] = out
    checksum = file_checksum(out)
    # Keep the algorithm-qualified checksum for human-facing provenance while
    # exposing the canonical 64-hex SHA-256 value under the algorithm-specific
    # field consumed by manifests and the campaign validator.
    metadata["scenario_checksum_sha256"] = checksum.split(":", 1)[1]
    metadata["scenario_checksum"] = checksum
    metadata["cohort"] = os.path.abspath(args.cohort)
    metadata["cohort_checksum"] = file_checksum(args.cohort) if os.path.exists(args.cohort) else None
    meta_path = os.path.splitext(out)[0] + "_metadata.json"
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({k: v for k, v in metadata.items() if k != "covariate_names"},
                     indent=2, sort_keys=True))
    print(f"\nwrote {out}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
