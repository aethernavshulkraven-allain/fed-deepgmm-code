"""Cross-fitted, shrinkage-adjusted treatment-preference instruments for eICU.

Why not leave-one-out hospital preference
-----------------------------------------
With client = hospital ``h``, the leave-one-out preference

    Z_hi = (S_h - D_hi) / (n_h - 1),   S_h = sum_j D_hj

is an affine function of the patient's own treatment: ``D_hi = S_h - (n_h - 1) Z_hi``.
It therefore takes at most two values inside a hospital, carries no within-client
variation, and is not an external source of variation at all. Since FedDeepGMM
aggregates *client-local* moment conditions, an instrument with no within-client
variation cannot identify anything locally.

What this module builds instead
-------------------------------
``PREFERENCE_WARD`` (primary): clients stay hospitals; the instrument is the
cross-fitted early-treatment rate of the patient's **ward**, so preference varies
across wards inside one hospital client.

``PREFERENCE_HOSPITAL`` (fallback): used when most hospitals have a single usable
ward. Clients become groups of hospitals (region x teaching status x bed category)
and the instrument is cross-fitted **hospital** preference, which again varies
within a client — at the cost of a client no longer being one physical hospital.

Both share three guarantees, enforced by tests:

1. A patient's own treatment never enters their own instrument value (cross-fitting
   removes the whole fold, which is strictly stronger than leave-one-out).
2. Preference is estimated from **training rows only**. Validation and test rows are
   scored with the full training split, never with their own outcomes.
3. Small groups are shrunk toward their parent rate by a Beta-Binomial prior, so a
   two-patient ward cannot report a preference of exactly 0 or 1.
"""

import numpy as np
import pandas as pd

PREFERENCE_WARD = "ward"
PREFERENCE_HOSPITAL = "hospital"

DEFAULT_N_FOLDS = 5
# Prior strength in pseudo-patients. 10 keeps a 50-patient ward close to its own
# rate while pulling a 3-patient ward most of the way to the hospital rate.
DEFAULT_PRIOR_STRENGTH = 10.0

GROUP_COLUMNS = {
    PREFERENCE_WARD: ["hospitalid", "wardid"],
    PREFERENCE_HOSPITAL: ["hospitalid"],
}
# Level the shrinkage prior is centred on.
PARENT_COLUMNS = {
    PREFERENCE_WARD: ["hospitalid"],
    PREFERENCE_HOSPITAL: [],
}


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


def assign_folds(frame, client_col, n_folds=DEFAULT_N_FOLDS, seed=0, unit_col=None):
    """Assign a cross-fitting fold to every row, independently within each client.

    Folds are assigned at the level of ``unit_col`` (default: the frame index) so
    that all rows belonging to one hospital admission land in the same fold.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    rng = np.random.default_rng(seed)
    folds = pd.Series(-1, index=frame.index, dtype="int64")

    for _, rows in frame.groupby(client_col, sort=True):
        if unit_col is None:
            units = pd.Series(rows.index.values, index=rows.index)
        else:
            units = rows[unit_col]

        unique_units = pd.unique(units)
        shuffled = rng.permutation(len(unique_units))
        unit_fold = dict(zip(unique_units, shuffled % n_folds))
        folds.loc[rows.index] = units.map(unit_fold).astype("int64").values

    if (folds < 0).any():
        raise AssertionError("fold assignment left rows unassigned")
    return folds


# ---------------------------------------------------------------------------
# Preference estimation
# ---------------------------------------------------------------------------


def _sums_and_counts(frame, group_cols, treatment_col):
    grouped = frame.groupby(group_cols, sort=False)[treatment_col]
    return grouped.sum(), grouped.size()


def _shrunk_rate(treated, total, prior_mean, prior_strength):
    """Beta-Binomial posterior mean with prior centred on ``prior_mean``."""
    alpha = prior_strength * prior_mean
    beta = prior_strength * (1.0 - prior_mean)
    return (alpha + treated) / (alpha + beta + total)


class PreferenceEstimator(object):
    """Preference table fitted on training rows, reusable for val/test scoring.

    ``fit`` stores per-group and per-parent treated counts from the training split.
    ``transform`` scores unseen rows against the full training split.
    ``crossfit_transform`` scores the training rows themselves, holding out the
    row's own fold so no training row ever sees its own treatment.
    """

    def __init__(
        self,
        construction=PREFERENCE_WARD,
        prior_strength=DEFAULT_PRIOR_STRENGTH,
        treatment_col="treatment",
        fold_col="fold",
    ):
        if construction not in GROUP_COLUMNS:
            raise ValueError(
                f"construction must be one of {sorted(GROUP_COLUMNS)}, got {construction!r}"
            )
        self.construction = construction
        self.prior_strength = float(prior_strength)
        self.treatment_col = treatment_col
        self.fold_col = fold_col

        self.group_cols = list(GROUP_COLUMNS[construction])
        self.parent_cols = list(PARENT_COLUMNS[construction])

        self.global_rate_ = None
        self.group_treated_ = None
        self.group_total_ = None
        self.parent_treated_ = None
        self.parent_total_ = None
        self.fold_group_treated_ = None
        self.fold_group_total_ = None
        self.fold_parent_treated_ = None
        self.fold_parent_total_ = None

    # -- fitting ---------------------------------------------------------

    def fit(self, train):
        missing = [c for c in self.group_cols if c not in train.columns]
        if missing:
            raise KeyError(f"training frame is missing {missing}")

        self.global_rate_ = float(train[self.treatment_col].mean())
        self.group_treated_, self.group_total_ = _sums_and_counts(
            train, self.group_cols, self.treatment_col
        )
        if self.parent_cols:
            self.parent_treated_, self.parent_total_ = _sums_and_counts(
                train, self.parent_cols, self.treatment_col
            )

        if self.fold_col in train.columns:
            fold_keys = [self.fold_col] + self.group_cols
            self.fold_group_treated_, self.fold_group_total_ = _sums_and_counts(
                train, fold_keys, self.treatment_col
            )
            if self.parent_cols:
                parent_keys = [self.fold_col] + self.parent_cols
                self.fold_parent_treated_, self.fold_parent_total_ = _sums_and_counts(
                    train, parent_keys, self.treatment_col
                )
        return self

    # -- scoring ---------------------------------------------------------

    def _lookup(self, frame, table, keys):
        if table is None:
            return pd.Series(0.0, index=frame.index)
        index = pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(
            frame[keys[0]]
        )
        values = table.reindex(index).to_numpy(dtype="float64", na_value=0.0)
        return pd.Series(values, index=frame.index)

    def _parent_rate(self, frame, treated_table, total_table):
        """Prior mean for each row: parent-level rate, or the global rate."""
        if not self.parent_cols:
            return pd.Series(self.global_rate_, index=frame.index)
        treated = self._lookup(frame, treated_table, self.parent_cols)
        total = self._lookup(frame, total_table, self.parent_cols)
        # A parent with no training rows falls back to the global rate.
        rate = np.where(total > 0, treated / np.maximum(total, 1.0), self.global_rate_)
        return pd.Series(rate, index=frame.index)

    def transform(self, frame):
        """Score rows against the full training split (validation / test)."""
        self._require_fitted()
        treated = self._lookup(frame, self.group_treated_, self.group_cols)
        total = self._lookup(frame, self.group_total_, self.group_cols)
        prior_mean = self._parent_rate(frame, self.parent_treated_, self.parent_total_)
        return _shrunk_rate(treated, total, prior_mean, self.prior_strength)

    def crossfit_transform(self, train):
        """Score training rows, excluding each row's own fold."""
        self._require_fitted()
        if self.fold_group_treated_ is None:
            raise RuntimeError(
                f"fit() saw no {self.fold_col!r} column; cross-fitting is unavailable"
            )

        fold_keys = [self.fold_col] + self.group_cols
        in_fold_treated = self._lookup(train, self.fold_group_treated_, fold_keys)
        in_fold_total = self._lookup(train, self.fold_group_total_, fold_keys)

        treated = self._lookup(train, self.group_treated_, self.group_cols) - in_fold_treated
        total = self._lookup(train, self.group_total_, self.group_cols) - in_fold_total

        if self.parent_cols:
            parent_keys = [self.fold_col] + self.parent_cols
            parent_treated = self._lookup(
                train, self.parent_treated_, self.parent_cols
            ) - self._lookup(train, self.fold_parent_treated_, parent_keys)
            parent_total = self._lookup(
                train, self.parent_total_, self.parent_cols
            ) - self._lookup(train, self.fold_parent_total_, parent_keys)
            prior_mean = pd.Series(
                np.where(
                    parent_total > 0,
                    parent_treated / np.maximum(parent_total, 1.0),
                    self.global_rate_,
                ),
                index=train.index,
            )
        else:
            prior_mean = pd.Series(self.global_rate_, index=train.index)

        return _shrunk_rate(treated, total, prior_mean, self.prior_strength)

    def _require_fitted(self):
        if self.group_total_ is None:
            raise RuntimeError("PreferenceEstimator.fit() has not been called")


# ---------------------------------------------------------------------------
# Leakage ablation
# ---------------------------------------------------------------------------


def naive_leave_one_out_preference(
    frame, group_cols, treatment_col="treatment"
):
    """The construction this module exists to replace.

    Provided **only** so the ablation can show that leave-one-out preference inflates
    apparent first-stage strength. It must never be used for a reported estimate: at
    the hospital level it is an affine function of the patient's own treatment.
    """
    grouped = frame.groupby(group_cols, sort=False)[treatment_col]
    total = grouped.transform("size")
    treated = grouped.transform("sum")
    denom = (total - 1).clip(lower=1)
    return (treated - frame[treatment_col]) / denom


# ---------------------------------------------------------------------------
# Feasibility of each construction
# ---------------------------------------------------------------------------


def instrument_variation(frame, client_col, instrument_col):
    """Raw within-client standard deviation of the instrument, per client.

    Reported for completeness, but **not** the right feasibility criterion: see
    :func:`structural_instrument_variation`.
    """
    return frame.groupby(client_col)[instrument_col].std(ddof=0).fillna(0.0)


def structural_instrument_variation(
    frame, client_col, group_col, instrument_col
):
    """Between-group standard deviation of the instrument, within each client.

    Cross-fitting makes the raw within-client spread of Z misleading: even a
    hospital with a single ward shows non-zero patient-level variation, because
    different folds hold out different patients and therefore produce slightly
    different preference estimates. That variation is estimation noise, not
    practice style, and an instrument built on it would identify nothing.

    Collapsing to the group (ward, or hospital in the grouped-client fallback)
    mean first removes the fold noise, so a one-ward hospital correctly reports
    zero. This is the quantity the feasibility gate uses.
    """
    group_means = frame.groupby([client_col, group_col])[instrument_col].mean()
    return group_means.groupby(level=0).std(ddof=0).fillna(0.0)


def build_instrument(
    train,
    others=None,
    construction=PREFERENCE_WARD,
    client_col="hospitalid",
    treatment_col="treatment",
    unit_col="patienthealthsystemstayid",
    n_folds=DEFAULT_N_FOLDS,
    prior_strength=DEFAULT_PRIOR_STRENGTH,
    seed=0,
):
    """Fit on ``train`` and score every split.

    Returns ``(train_z, {name: z}, estimator)``. ``others`` maps a split name to a
    frame scored against the full training split.
    """
    train = train.copy()
    train["fold"] = assign_folds(
        train, client_col=client_col, n_folds=n_folds, seed=seed, unit_col=unit_col
    )

    estimator = PreferenceEstimator(
        construction=construction,
        prior_strength=prior_strength,
        treatment_col=treatment_col,
    ).fit(train)

    train_z = estimator.crossfit_transform(train)
    other_z = {
        name: estimator.transform(frame) for name, frame in (others or {}).items()
    }
    return train_z, other_z, estimator
