"""Instrument diagnostics and linear IV baselines, in numpy.

``statsmodels`` and ``linearmodels`` are absent from the ``fedgmm`` env and the repo
convention is dependency-light (the analysis scripts use stdlib ``csv``), so 2SLS,
HC1 robust standard errors and the partial-F statistic are implemented here directly.
Everything is exercised against a simulated design with a known coefficient in
``tests/test_eicu_iv_diagnostics.py``.

No neural estimator repairs an invalid instrument, so these checks are meant to run
*before* any effect estimate is reported.
"""

import numpy as np


def _add_intercept(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return np.hstack([np.ones((matrix.shape[0], 1)), matrix])


def _lstsq(design, target):
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef


def ols(design, target, add_intercept=True):
    """OLS with HC1 robust standard errors."""
    target = np.asarray(target, dtype=float).ravel()
    full = _add_intercept(design) if add_intercept else np.asarray(design, dtype=float)

    coef = _lstsq(full, target)
    resid = target - full @ coef

    n, k = full.shape
    xtx_inv = np.linalg.pinv(full.T @ full)
    # HC1: finite-sample correction n / (n - k).
    meat = full.T @ (full * (resid ** 2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    total = ((target - target.mean()) ** 2).sum()
    r_squared = 1.0 - (resid ** 2).sum() / total if total > 0 else float("nan")

    return {
        "coef": coef,
        "stderr": stderr,
        "cov": cov,
        "resid": resid,
        "n": int(n),
        "k": int(k),
        "r_squared": float(r_squared),
    }


def two_stage_least_squares(instrument, treatment, covariates=None, outcome=None):
    """2SLS for a single endogenous regressor.

    Returns the treatment-effect estimate with a standard error computed from the
    *structural* residual (using the fitted treatment, not the first-stage fit),
    which is what makes it a valid IV standard error rather than an OLS one.
    """
    instrument = np.asarray(instrument, dtype=float)
    if instrument.ndim == 1:
        instrument = instrument.reshape(-1, 1)
    treatment = np.asarray(treatment, dtype=float).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()

    if covariates is None:
        controls = np.empty((len(treatment), 0))
    else:
        controls = np.asarray(covariates, dtype=float)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)

    # Stage 1: treatment on instrument(s) + controls.
    first_design = _add_intercept(np.hstack([instrument, controls]))
    first_coef = _lstsq(first_design, treatment)
    treatment_hat = first_design @ first_coef

    # Stage 2: outcome on fitted treatment + controls.
    second_design = _add_intercept(np.hstack([treatment_hat.reshape(-1, 1), controls]))
    coef = _lstsq(second_design, outcome)

    # Structural residual uses observed treatment, not the first-stage fit.
    structural_design = _add_intercept(np.hstack([treatment.reshape(-1, 1), controls]))
    resid = outcome - structural_design @ coef

    n, k = second_design.shape
    xtx_inv = np.linalg.pinv(second_design.T @ second_design)
    meat = second_design.T @ (second_design * (resid ** 2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    return {
        "effect": float(coef[1]),
        "effect_stderr": float(stderr[1]),
        "coef": coef,
        "stderr": stderr,
        "n": int(n),
        "first_stage_coef": first_coef,
    }


def first_stage_diagnostics(instrument, treatment, covariates=None):
    """Relevance of the instrument: partial F, partial R-squared, coefficient.

    The partial F compares the controls-only fit to the fit with the instrument
    added. F < 10 is reported as a weak-instrument *warning*, not a threshold with
    any theoretical status.
    """
    instrument = np.asarray(instrument, dtype=float)
    if instrument.ndim == 1:
        instrument = instrument.reshape(-1, 1)
    treatment = np.asarray(treatment, dtype=float).ravel()

    if covariates is None:
        controls = np.empty((len(treatment), 0))
    else:
        controls = np.asarray(covariates, dtype=float)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)

    restricted = ols(controls, treatment) if controls.size else ols(
        np.empty((len(treatment), 0)), treatment
    )
    unrestricted = ols(np.hstack([instrument, controls]), treatment)

    rss_r = float((restricted["resid"] ** 2).sum())
    rss_u = float((unrestricted["resid"] ** 2).sum())
    q = instrument.shape[1]
    df_resid = max(unrestricted["n"] - unrestricted["k"], 1)

    partial_f = ((rss_r - rss_u) / q) / (rss_u / df_resid) if rss_u > 0 else float("inf")
    partial_r2 = (rss_r - rss_u) / rss_r if rss_r > 0 else float("nan")

    coef_index = 1  # after the intercept
    return {
        "partial_f": float(partial_f),
        "partial_r2": float(partial_r2),
        "instrument_coef": float(unrestricted["coef"][coef_index]),
        "instrument_stderr": float(unrestricted["stderr"][coef_index]),
        "n": int(unrestricted["n"]),
        "weak_instrument_warning": bool(partial_f < 10.0),
    }


def standardized_mean_differences(covariates, instrument, names=None, quantile=0.2):
    """SMD of each covariate between the top and bottom instrument quantile.

    Large imbalance is evidence that the instrument encodes patient composition or
    ward specialisation rather than practice style, which is the exclusion-restriction
    threat this study cannot rule out by design.
    """
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)
    instrument = np.asarray(instrument, dtype=float).ravel()

    low_cut = np.quantile(instrument, quantile)
    high_cut = np.quantile(instrument, 1.0 - quantile)
    low = instrument <= low_cut
    high = instrument >= high_cut

    if names is None:
        names = [f"x{i}" for i in range(covariates.shape[1])]

    out = []
    for j, name in enumerate(names):
        col = covariates[:, j]
        lo, hi = col[low], col[high]
        if len(lo) == 0 or len(hi) == 0:
            out.append({"covariate": name, "smd": float("nan")})
            continue
        pooled = np.sqrt((np.var(lo, ddof=1) + np.var(hi, ddof=1)) / 2.0)
        smd = (hi.mean() - lo.mean()) / pooled if pooled > 0 else 0.0
        out.append(
            {
                "covariate": name,
                "mean_low": float(lo.mean()),
                "mean_high": float(hi.mean()),
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
            }
        )
    return sorted(out, key=lambda r: -(r.get("abs_smd") or 0.0))


def overlap_by_quantile(instrument, treatment, n_bins=5):
    """Treatment probability by instrument quantile — the visual first-stage check."""
    instrument = np.asarray(instrument, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()

    edges = np.quantile(instrument, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return []

    bins = np.clip(np.digitize(instrument, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": int(b),
                "z_low": float(edges[b]),
                "z_high": float(edges[b + 1]),
                "n": int(mask.sum()),
                "treatment_rate": float(treatment[mask].mean()),
            }
        )
    return rows
