"""Shared helpers for the eICU federated-IV extension.

Release-agnostic: every entry point takes an ``--eicu-root`` pointing at a directory
holding the eICU-CRD tables as ``<table>.csv.gz``. The demo v2.0.1 tree and the full
v2.0 credentialed release both satisfy this, so the same code path serves pipeline
debugging and the reported analysis.

Table filenames differ in case between mirrors (``infusiondrug.csv.gz`` in the demo,
``infusionDrug.csv.gz`` elsewhere), so tables are always resolved case-insensitively.
"""

import csv
import gzip
import math
import os
import re
import statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMO_EICU_ROOT = os.path.join(
    REPO_ROOT, "physionet.org", "files", "eicu-crd-demo", "2.0.1"
)

# ---------------------------------------------------------------------------
# Table resolution / IO
# ---------------------------------------------------------------------------

_TABLE_SUFFIXES = (".csv.gz", ".csv")


def resolve_table_path(eicu_root, table):
    """Return the on-disk path for ``table``, matching filename case-insensitively.

    Raises FileNotFoundError listing what was actually present, because a silently
    missing table would otherwise show up much later as an empty cohort.
    """
    wanted = table.lower()
    try:
        entries = os.listdir(eicu_root)
    except OSError as exc:
        raise FileNotFoundError(f"eICU root not readable: {eicu_root}") from exc

    for entry in entries:
        stem = entry.lower()
        for suffix in _TABLE_SUFFIXES:
            if stem == wanted + suffix:
                return os.path.join(eicu_root, entry)

    available = sorted(
        e for e in entries if e.lower().endswith(_TABLE_SUFFIXES)
    )
    raise FileNotFoundError(
        f"table {table!r} not found under {eicu_root}. Available: {available}"
    )


def table_exists(eicu_root, table):
    try:
        resolve_table_path(eicu_root, table)
        return True
    except FileNotFoundError:
        return False


def read_table(eicu_root, table, usecols=None, dtype=None, nrows=None):
    """Read a whole eICU table into a DataFrame."""
    import pandas as pd

    return pd.read_csv(
        resolve_table_path(eicu_root, table),
        usecols=usecols,
        dtype=dtype,
        nrows=nrows,
        low_memory=False,
    )


def iter_table_chunks(eicu_root, table, usecols=None, chunksize=1_000_000):
    """Stream a table in chunks.

    Required for ``vitalPeriodic`` / ``nurseCharting``, which are a few hundred MB
    compressed in the full release and do not need to be materialised in full — the
    caller filters each chunk to the baseline window before concatenating.
    """
    import pandas as pd

    return pd.read_csv(
        resolve_table_path(eicu_root, table),
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )


def count_table_rows(eicu_root, table):
    """Row count without parsing, for cohort-flow provenance."""
    path = resolve_table_path(eicu_root, table)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


# ---------------------------------------------------------------------------
# Patient attributes
# ---------------------------------------------------------------------------

# eICU de-identifies ages above 89 as the literal string "> 89".
AGE_TOP_CODED_VALUE = 90.0


def parse_age(values):
    """Parse the eICU ``age`` column to float years.

    ``'> 89'`` maps to 90.0 (top-coded, not missing); blanks and unparseable
    strings map to NaN.
    """
    import numpy as np
    import pandas as pd

    series = pd.Series(values).astype("string")
    out = pd.Series(np.nan, index=series.index, dtype="float64")

    top_coded = series.str.contains(">", na=False)
    out[top_coded] = AGE_TOP_CODED_VALUE

    numeric = pd.to_numeric(series.where(~top_coded), errors="coerce").astype("float64")
    out[~top_coded] = numeric[~top_coded]
    return out


def expired_flag(values):
    """Binary in-hospital mortality from ``hospitaldischargestatus``.

    Returns NaN where status is absent — those rows are excluded rather than
    treated as survivors.
    """
    import numpy as np
    import pandas as pd

    series = pd.Series(values).astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[series == "expired"] = 1.0
    out[series == "alive"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Treatment: vasopressor infusions
# ---------------------------------------------------------------------------

# infusionDrug.drugname carries dose units inline, e.g. "Norepinephrine (mcg/min)",
# "Norepinephrine (ml/hr)", "Norepinephrine ()". Match on substrings only.
VASOPRESSOR_PATTERNS = {
    "norepinephrine": ("norepinephrine", "noradrenaline", "levophed"),
    "epinephrine": ("epinephrine", "adrenaline"),
    "vasopressin": ("vasopressin",),
    "phenylephrine": ("phenylephrine", "neosynephrine", "neo-synephrine"),
    "dopamine": ("dopamine",),
}

# Dopamine is held out of the primary definition: its clinical role (chronotropy at
# low dose) differs from the pure vasopressors, so it enters only as a sensitivity arm.
PRIMARY_VASOPRESSORS = (
    "norepinephrine",
    "epinephrine",
    "vasopressin",
    "phenylephrine",
)
SENSITIVITY_VASOPRESSORS = PRIMARY_VASOPRESSORS + ("dopamine",)

# "epinephrine" is a substring of "norepinephrine", so norepinephrine must be
# tested first and matching must stop at the first hit.
_VASOPRESSOR_ORDER = (
    "norepinephrine",
    "vasopressin",
    "phenylephrine",
    "dopamine",
    "epinephrine",
)


def normalize_vasopressor(drugname):
    """Map a raw ``drugname`` string to a canonical agent, or None."""
    if drugname is None:
        return None
    text = str(drugname).lower()
    if not text or text == "nan":
        return None
    for agent in _VASOPRESSOR_ORDER:
        for pattern in VASOPRESSOR_PATTERNS[agent]:
            if pattern in text:
                return agent
    return None


def normalize_vasopressor_series(values):
    """Vectorised :func:`normalize_vasopressor`.

    Unmatched rows are ``None`` (not NaN) so the result compares equal to the
    scalar version element-wise; a test pins the two implementations together.
    """
    import numpy as np
    import pandas as pd

    series = pd.Series(values).astype("string").fillna("").str.lower()
    out = pd.Series(
        np.full(len(series), None, dtype=object), index=series.index, dtype="object"
    )
    unassigned = pd.Series(True, index=series.index)
    for agent in _VASOPRESSOR_ORDER:
        if not unassigned.any():
            break
        pattern = "|".join(re.escape(p) for p in VASOPRESSOR_PATTERNS[agent])
        hit = unassigned & series.str.contains(pattern, na=False, regex=True)
        out[hit] = agent
        unassigned = unassigned & ~hit
    return out


# ---------------------------------------------------------------------------
# Cohort: sepsis at / near ICU admission
# ---------------------------------------------------------------------------

SEPSIS_TEXT_PATTERN = r"sepsis|septic|septicemia"

# ICD-9 sepsis / severe sepsis / septic shock / SIRS-with-infection.
SEPSIS_ICD9_PREFIXES = (
    "038",  # septicemia
    "995.91",  # sepsis
    "995.92",  # severe sepsis
    "785.52",  # septic shock
    "790.7",  # bacteremia
)

# diagnosisOffset is when a diagnosis was *entered*, not biological onset, so the
# window is deliberately symmetric around ICU admission and reported as a
# sensitivity parameter rather than treated as an onset time.
DEFAULT_SEPSIS_WINDOW_MINUTES = (-360, 360)

# Treatment window is relative to ICU admission, the database time origin.
DEFAULT_TREATMENT_WINDOW_MINUTES = (0, 360)

# Baseline physiology must predate treatment assignment.
DEFAULT_BASELINE_WINDOW_MINUTES = (0, 60)


def matches_sepsis_icd9(values):
    """Row-wise flag for any sepsis ICD-9 code.

    ``icd9code`` holds a comma-separated list, so the column is exploded, matched
    prefix-wise, and OR-reduced back to one flag per input row.
    """
    import pandas as pd

    series = pd.Series(values).astype("string").fillna("")
    exploded = series.str.split(",").explode().str.strip()
    is_sepsis = exploded.str.startswith(SEPSIS_ICD9_PREFIXES, na=False)
    return is_sepsis.groupby(level=0).any().reindex(series.index, fill_value=False)


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------

# lab.labname -> covariate name. Values are the eICU spellings, which include
# scaling suffixes ("WBC x 1000").
LAB_COVARIATES = {
    "creatinine": "lab_creatinine",
    "lactate": "lab_lactate",
    "WBC x 1000": "lab_wbc",
    "platelets x 1000": "lab_platelets",
    "total bilirubin": "lab_bilirubin",
    "bicarbonate": "lab_bicarbonate",
    "BUN": "lab_bun",
    "sodium": "lab_sodium",
    "pH": "lab_ph",
}

# vitalPeriodic column -> covariate name.
VITAL_PERIODIC_COVARIATES = {
    "heartrate": "vital_heartrate",
    "respiration": "vital_respiration",
    "sao2": "vital_sao2",
    "temperature": "vital_temperature",
    "systemicsystolic": "vital_sbp",
    "systemicmean": "vital_map",
}

# vitalAperiodic supplies cuff pressures when there is no arterial line.
VITAL_APERIODIC_COVARIATES = {
    "noninvasivesystolic": "vital_sbp",
    "noninvasivemean": "vital_map",
}

# pastHistory.pasthistoryvalue substrings -> comorbidity indicator.
COMORBIDITY_PATTERNS = {
    "comorb_malignancy": ("cancer", "malignan", "leukemia", "lymphoma", "metasta"),
    "comorb_ckd": ("renal failure", "renal insufficiency", "hemodialysis", "ckd"),
    "comorb_lung": ("copd", "asthma", "home oxygen", "pulmonary"),
    "comorb_heart_failure": ("chf", "heart failure", "cardiomyopathy"),
    "comorb_diabetes": ("diabetes", "insulin dependent"),
    "comorb_cirrhosis": ("cirrhosis", "hepatic failure", "varice"),
    "comorb_immunosuppression": ("immunosupp", "transplant", "steroid", "aids", "hiv"),
}

CATEGORICAL_COVARIATES = (
    "gender",
    "ethnicity",
    "hospitaladmitsource",
    "unittype",
    "unitstaytype",
    "numbedscategory",
    "teachingstatus",
    "region",
)

CONTINUOUS_COVARIATES = (
    "age",
    "admissionweight",
)

# Identifier columns that must never enter X. Hospital identity in particular would
# collide with a hospital/ward-preference instrument.
ID_COLUMNS = (
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "hospitalid",
    "wardid",
)


def continuous_covariate_columns(frame):
    """All continuous covariate columns present in a built cohort frame."""
    prefixes = ("lab_", "vital_")
    cols = [c for c in CONTINUOUS_COVARIATES if c in frame.columns]
    cols += [
        c
        for c in frame.columns
        if c.startswith(prefixes) and not c.endswith("_missing")
    ]
    return cols


# ---------------------------------------------------------------------------
# Final-iterate stability
# ---------------------------------------------------------------------------

DEFAULT_STABILITY_WINDOW = 50
DEFAULT_STABILITY_METRIC_COLUMN = "val_mse"


def _stability_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def final_iterate_stability(
    mse_by_round_csv,
    metric_column=DEFAULT_STABILITY_METRIC_COLUMN,
    window=DEFAULT_STABILITY_WINDOW,
):
    """Canonical final-iterate instability + divergence diagnostic for a run.

    This is the ONE place in the repo that should compute "last-50
    stability" going forward. It was previously reimplemented ad hoc, with
    differing return signatures, in ``scripts/analyze_fedogda_s_focused_v3.py``
    (line 147), ``scripts/analyze_optimistic_curve_screen_v1.py`` (line 148),
    ``scripts/analyze_fedogda_s_step_fast_v5.py`` (line 187), and
    ``scripts/analyze_fedogda_s_tuning_pilot.py`` (line 87). Those four
    callers are left untouched (they belong to other, already-analyzed
    campaigns; changing them risks their results) -- this function exists so
    every *new* analysis, starting with
    ``scripts/analyze_eicu_study_a_v2_confirmatory.py``, has one
    unambiguous, documented definition to import instead of copy-pasting a
    fifth variant.

    Window: the last ``min(window, n_rows)`` rows of ``mse_by_round_csv``, in
    file order. Rows are written/appended strictly in round order by
    ``experiment_utils.write_mse_by_round`` / ``append_mse_by_round``, so
    file order is round order; this function does not re-sort by ``round``.

    Statistic: population standard deviation (``statistics.pstdev``, i.e.
    divide-by-``n``, not divide-by-``n-1``) of ``metric_column`` over that
    window. This is the exact statistic all four prior implementations used
    (``pstdev_or_zero`` / ``statistics.pstdev`` with a 0.0 fallback for a
    single-row window) -- picking it is not inventing a new definition, it's
    codifying the one thing all four callers already agreed on. Also reports
    mean/min/max/range/coefficient-of-variation over the same window, since
    ``metric_policy.md``'s stability section wants range as well as std.

    ``metric_column`` defaults to ``"val_mse"`` to match the four priors
    exactly (their datasets have no ``client_id``, so pooled and
    equal-client validation MSE coincide). Study A v2's
    ``mse_by_round.csv`` additionally has ``primary_val_mse`` (the actual
    checkpoint selector: equal-client validation MSE when client_id is
    available, per ``fedavg_api.py``) and ``equal_client_val_mse`` columns;
    ``scripts/analyze_eicu_study_a_v2_confirmatory.py`` passes
    ``metric_column="primary_val_mse"`` explicitly, per
    ``metric_policy.md``'s "Stability metrics" section (equal-client
    validation structural MSE, not pooled).

    Divergence flag: ``True`` if ANY row across the FULL run history (not
    just the tail window) has ``diverged`` truthy or ``finite`` falsy --
    again matching all four priors exactly. A run that diverged early and
    then produced 50 finite-looking rows before a manual stop is still
    flagged diverged: divergence is a permanent taint on the run's numerical
    trustworthiness, not a momentary state that can be "recovered from" by
    looking away from it.

    Unlike the four priors (which call ``float()`` on every tail value and
    raise on the first non-finite one, so a single divergent run stops a
    whole batch analysis), this implementation skips non-finite / unparsable
    tail values when computing the window statistics and still returns a
    result -- ``diverged`` is already set from the full history regardless,
    so nothing about divergence detection is weakened; a confirmatory
    analyzer comparing 30 runs needs to be able to report "these seeds
    diverged" instead of crashing on the first one.

    Returns a dict (not a tuple -- the four priors also disagreed with each
    other on tuple arity/order, which is exactly the ambiguity this
    canonicalization removes) with keys:

    ``metric_column``, ``window_requested``, ``window_used``,
    ``n_rows_total``, ``n_tail_finite``, ``n_tail_nonfinite``, ``tail_mean``,
    ``tail_std``, ``tail_min``, ``tail_max``, ``tail_range``, ``tail_cv``
    (``tail_std / abs(tail_mean)``, ``None`` if the mean is ~0 or the window
    has no finite values), ``diverged``, ``first_diverged_round``.
    """
    with open(mse_by_round_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    n_rows_total = len(rows)
    if n_rows_total == 0:
        # No data to trust: conservatively flag diverged rather than silently
        # reporting "stable" for a run that produced nothing.
        return {
            "metric_column": metric_column,
            "window_requested": int(window),
            "window_used": 0,
            "n_rows_total": 0,
            "n_tail_finite": 0,
            "n_tail_nonfinite": 0,
            "tail_mean": None,
            "tail_std": None,
            "tail_min": None,
            "tail_max": None,
            "tail_range": None,
            "tail_cv": None,
            "diverged": True,
            "first_diverged_round": None,
        }

    diverged = False
    first_diverged_round = None
    for row in rows:
        row_diverged = _stability_to_bool(
            row.get("diverged", "false")
        ) or not _stability_to_bool(row.get("finite", "true"))
        if row_diverged and first_diverged_round is None:
            raw_round = row.get("round", "")
            try:
                first_diverged_round = int(raw_round)
            except (TypeError, ValueError):
                first_diverged_round = raw_round
        diverged = diverged or row_diverged

    window = int(window)
    tail_rows = rows[-window:] if window > 0 else rows
    window_used = len(tail_rows)

    tail_values = []
    n_nonfinite = 0
    for row in tail_rows:
        try:
            value = float(row.get(metric_column, ""))
        except (TypeError, ValueError):
            n_nonfinite += 1
            continue
        if not math.isfinite(value):
            n_nonfinite += 1
            continue
        tail_values.append(value)

    if tail_values:
        tail_mean = statistics.fmean(tail_values)
        tail_std = statistics.pstdev(tail_values) if len(tail_values) > 1 else 0.0
        tail_min = min(tail_values)
        tail_max = max(tail_values)
        tail_range = tail_max - tail_min
        tail_cv = tail_std / abs(tail_mean) if abs(tail_mean) > 1e-12 else None
    else:
        tail_mean = tail_std = tail_min = tail_max = tail_range = tail_cv = None

    return {
        "metric_column": metric_column,
        "window_requested": window,
        "window_used": window_used,
        "n_rows_total": n_rows_total,
        "n_tail_finite": len(tail_values),
        "n_tail_nonfinite": n_nonfinite,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_min": tail_min,
        "tail_max": tail_max,
        "tail_range": tail_range,
        "tail_cv": tail_cv,
        "diverged": diverged,
        "first_diverged_round": first_diverged_round,
    }
