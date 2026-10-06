"""Build the eICU sepsis cohort for the federated IV study.

Emits one row per included ICU stay with treatment, outcome, client/ward keys and
pre-treatment covariates, plus a cohort-flow record of every exclusion.

Design points that are not negotiable downstream:

* The treatment window is relative to **ICU admission**, the database time origin.
  ``diagnosisOffset`` records when a diagnosis was charted, not biological onset, so
  it is used only to place the stay in the sepsis cohort, never to time treatment.
* The cohort is **sepsis-at-admission**. Vasopressor-defined septic shock is not used,
  because selecting on it would condition cohort membership on the treatment.
* Covariates come from the first 60 minutes only. ``apacheApsVar`` holds APACHE-day
  worst values, which for a 6-hour treatment window are post-treatment; it is
  extracted only when ``--include-apache`` is passed, for the sensitivity arm.
* Absence of an infusion row is not evidence of no treatment until the hospital is
  shown to have a working infusion interface, so hospitals with no infusion records
  at all are dropped rather than counted as all-untreated.

Usage:
    python scripts/prepare_eicu_cohort.py --eicu-root <dir> --out <dir>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eicu_common import (  # noqa: E402
    CATEGORICAL_COVARIATES,
    COMORBIDITY_PATTERNS,
    DEFAULT_BASELINE_WINDOW_MINUTES,
    DEFAULT_SEPSIS_WINDOW_MINUTES,
    DEFAULT_TREATMENT_WINDOW_MINUTES,
    DEMO_EICU_ROOT,
    LAB_COVARIATES,
    PRIMARY_VASOPRESSORS,
    REPO_ROOT,
    SENSITIVITY_VASOPRESSORS,
    SEPSIS_TEXT_PATTERN,
    VITAL_APERIODIC_COVARIATES,
    VITAL_PERIODIC_COVARIATES,
    expired_flag,
    iter_table_chunks,
    matches_sepsis_icd9,
    normalize_vasopressor_series,
    parse_age,
    read_table,
    table_exists,
)


class CohortFlow(object):
    """Ordered record of how many stays survive each cohort step."""

    def __init__(self):
        self.steps = []

    def record(self, label, n_stays, detail=None):
        entry = {"step": label, "n_stays": int(n_stays)}
        if detail:
            entry["detail"] = detail
        self.steps.append(entry)
        return entry

    def as_list(self):
        return list(self.steps)

    def render(self):
        lines = []
        previous = None
        for entry in self.steps:
            n = entry["n_stays"]
            delta = "" if previous is None else f"  (-{previous - n})"
            lines.append(f"{entry['step']:<48s} {n:>8d}{delta}")
            previous = n
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------


def load_base_population(eicu_root, flow):
    """patient x hospital, restricted to adult first ICU stays with known mortality."""
    import pandas as pd

    patient = read_table(eicu_root, "patient")
    flow.record("all ICU stays", len(patient))

    patient["age_years"] = parse_age(patient["age"])
    patient = patient[patient["age_years"] >= 18]
    flow.record("adults (age >= 18)", len(patient))

    patient = patient[patient["unitvisitnumber"] == 1]
    flow.record("first unit visit (unitvisitnumber == 1)", len(patient))

    # A hospital admission can still contribute several unit stays with
    # unitvisitnumber == 1 (e.g. transfers). Keep the one closest to hospital
    # admission: hospitaladmitoffset is negative minutes from unit admit back to
    # hospital admit, so the largest value is the earliest ICU stay.
    before = len(patient)
    patient = (
        patient.sort_values(
            ["patienthealthsystemstayid", "hospitaladmitoffset"], ascending=[True, False]
        )
        .drop_duplicates("patienthealthsystemstayid", keep="first")
        .copy()
    )
    flow.record(
        "first ICU stay per hospital admission",
        len(patient),
        detail=f"dropped {before - len(patient)} repeat unit stays",
    )

    patient["died_in_hospital"] = expired_flag(patient["hospitaldischargestatus"])
    patient = patient[patient["died_in_hospital"].notna()]
    flow.record("known hospital discharge status", len(patient))

    patient = patient[patient["hospitalid"].notna() & patient["wardid"].notna()]
    flow.record("known hospital and ward id", len(patient))

    if table_exists(eicu_root, "hospital"):
        hospital = read_table(eicu_root, "hospital")
        patient = patient.merge(hospital, on="hospitalid", how="left")
    else:
        for col in ("numbedscategory", "teachingstatus", "region"):
            patient[col] = pd.NA

    return patient


def sepsis_stay_ids(eicu_root, window):
    """Stays with a sepsis diagnosis charted near ICU admission, or a sepsis admitDx."""
    lower, upper = window
    ids = set()

    diagnosis = read_table(
        eicu_root,
        "diagnosis",
        usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring", "icd9code"],
    )
    in_window = diagnosis["diagnosisoffset"].between(lower, upper)
    text_hit = diagnosis["diagnosisstring"].str.contains(
        SEPSIS_TEXT_PATTERN, case=False, na=False, regex=True
    )
    code_hit = matches_sepsis_icd9(diagnosis["icd9code"])
    ids.update(diagnosis.loc[in_window & (text_hit | code_hit), "patientunitstayid"])

    # admissionDx has no clinically meaningful offset to filter on: it is by
    # construction the reason for admission.
    if table_exists(eicu_root, "admissionDx"):
        admit = read_table(
            eicu_root, "admissionDx", usecols=["patientunitstayid", "admitdxpath"]
        )
        hit = admit["admitdxpath"].str.contains(
            SEPSIS_TEXT_PATTERN, case=False, na=False, regex=True
        )
        ids.update(admit.loc[hit, "patientunitstayid"])

    return ids


def apache_admission_sepsis(patient):
    return patient["apacheadmissiondx"].str.contains(
        SEPSIS_TEXT_PATTERN, case=False, na=False, regex=True
    )


def load_vasopressor_infusions(eicu_root, agents):
    """All vasopressor infusion rows, annotated with the canonical agent."""
    infusion = read_table(
        eicu_root,
        "infusiondrug",
        usecols=["patientunitstayid", "infusionoffset", "drugname"],
    )
    infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
    infusion = infusion[infusion["agent"].isin(agents)]
    return infusion


def hospitals_with_infusion_interface(eicu_root):
    """Hospitals contributing at least one infusion row of any kind.

    Without this check, a hospital that simply never connected its infusion pumps
    would enter the study as a hospital where nobody is ever treated.

    Coverage is judged over the hospital's *entire* patient population, not the
    sepsis cohort: a hospital with a working interface can easily have no cohort
    member on an infusion, and dropping it for that reason would select on treatment.
    """
    infusion = read_table(eicu_root, "infusiondrug", usecols=["patientunitstayid"])
    all_patients = read_table(
        eicu_root, "patient", usecols=["patientunitstayid", "hospitalid"]
    )
    covered = all_patients[
        all_patients["patientunitstayid"].isin(set(infusion["patientunitstayid"]))
    ]
    return set(covered["hospitalid"].dropna().unique())


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------


def baseline_labs(eicu_root, stay_ids, window):
    """First recorded value per lab within the baseline window."""
    import pandas as pd

    lower, upper = window
    wanted = set(LAB_COVARIATES)
    frames = []
    for chunk in iter_table_chunks(
        eicu_root,
        "lab",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
    ):
        chunk = chunk[
            chunk["patientunitstayid"].isin(stay_ids)
            & chunk["labname"].isin(wanted)
            & chunk["labresultoffset"].between(lower, upper)
        ]
        if len(chunk):
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["patientunitstayid"]).set_index("patientunitstayid")

    labs = pd.concat(frames, ignore_index=True)
    labs = labs.sort_values("labresultoffset").drop_duplicates(
        ["patientunitstayid", "labname"], keep="first"
    )
    wide = labs.pivot(
        index="patientunitstayid", columns="labname", values="labresult"
    )
    return wide.rename(columns=LAB_COVARIATES)


def _baseline_vitals_from(eicu_root, table, column_map, stay_ids, window):
    import pandas as pd

    lower, upper = window
    usecols = ["patientunitstayid", "observationoffset"] + list(column_map)
    frames = []
    for chunk in iter_table_chunks(eicu_root, table, usecols=usecols):
        chunk = chunk[
            chunk["patientunitstayid"].isin(stay_ids)
            & chunk["observationoffset"].between(lower, upper)
        ]
        if len(chunk):
            frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    vitals = pd.concat(frames, ignore_index=True)
    vitals = vitals.rename(columns=column_map)
    value_cols = sorted(set(column_map.values()))
    # Median over the window is robust to the monitor artefacts that dominate
    # single first readings in vitalPeriodic.
    return vitals.groupby("patientunitstayid")[value_cols].median()


def baseline_vitals(eicu_root, stay_ids, window):
    """Baseline vitals, preferring arterial readings and filling from cuff values."""
    import pandas as pd

    periodic = _baseline_vitals_from(
        eicu_root, "vitalPeriodic", VITAL_PERIODIC_COVARIATES, stay_ids, window
    )
    aperiodic = pd.DataFrame()
    if table_exists(eicu_root, "vitalAperiodic"):
        aperiodic = _baseline_vitals_from(
            eicu_root, "vitalAperiodic", VITAL_APERIODIC_COVARIATES, stay_ids, window
        )

    if periodic.empty:
        return aperiodic
    if aperiodic.empty:
        return periodic

    combined = periodic.reindex(periodic.index.union(aperiodic.index))
    for col in aperiodic.columns:
        if col in combined.columns:
            combined[col] = combined[col].fillna(aperiodic[col])
        else:
            combined[col] = aperiodic[col]
    return combined


def comorbidity_flags(eicu_root, stay_ids):
    """Binary comorbidity indicators from pastHistory."""
    import pandas as pd

    columns = sorted(COMORBIDITY_PATTERNS)
    if not table_exists(eicu_root, "pastHistory"):
        return pd.DataFrame(columns=columns)

    history = read_table(
        eicu_root,
        "pastHistory",
        usecols=["patientunitstayid", "pasthistoryvalue", "pasthistorypath"],
    )
    history = history[history["patientunitstayid"].isin(stay_ids)]
    if history.empty:
        return pd.DataFrame(columns=columns)

    text = (
        history["pasthistoryvalue"].fillna("").astype(str)
        + " "
        + history["pasthistorypath"].fillna("").astype(str)
    ).str.lower()

    out = pd.DataFrame({"patientunitstayid": history["patientunitstayid"].values})
    for name, patterns in COMORBIDITY_PATTERNS.items():
        out[name] = text.str.contains("|".join(patterns), regex=True, na=False).values
    return out.groupby("patientunitstayid").max().astype(float)


def baseline_apache(eicu_root, stay_ids):
    """APACHE APS variables — sensitivity arm only (post-treatment for a 6h window)."""
    import pandas as pd

    if not table_exists(eicu_root, "apacheApsVar"):
        return pd.DataFrame()
    apache = read_table(eicu_root, "apacheApsVar")
    apache = apache[apache["patientunitstayid"].isin(stay_ids)]
    drop = [c for c in ("apacheapsvarid",) if c in apache.columns]
    apache = apache.drop(columns=drop).set_index("patientunitstayid")
    apache = apache.replace(-1, pd.NA)  # eICU sentinel for "not measured"
    return apache.add_prefix("apache_")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_cohort(
    eicu_root,
    sepsis_window=DEFAULT_SEPSIS_WINDOW_MINUTES,
    treatment_window=DEFAULT_TREATMENT_WINDOW_MINUTES,
    baseline_window=DEFAULT_BASELINE_WINDOW_MINUTES,
    include_dopamine=False,
    include_apache=False,
    keep_pre_icu_vasopressors=False,
):
    import numpy as np
    import pandas as pd

    flow = CohortFlow()
    patient = load_base_population(eicu_root, flow)

    sepsis_ids = sepsis_stay_ids(eicu_root, sepsis_window)
    is_sepsis = patient["patientunitstayid"].isin(sepsis_ids) | apache_admission_sepsis(
        patient
    )
    patient = patient[is_sepsis]
    flow.record(
        "sepsis at / near ICU admission",
        len(patient),
        detail=f"diagnosisoffset window {sepsis_window} min, or sepsis admitDx",
    )

    covered = hospitals_with_infusion_interface(eicu_root)
    patient = patient[patient["hospitalid"].isin(covered)]
    flow.record("hospital has a working infusion interface", len(patient))

    agents = SENSITIVITY_VASOPRESSORS if include_dopamine else PRIMARY_VASOPRESSORS
    infusion = load_vasopressor_infusions(eicu_root, agents)
    infusion = infusion[infusion["patientunitstayid"].isin(patient["patientunitstayid"])]

    if not keep_pre_icu_vasopressors:
        pre_icu = set(
            infusion.loc[infusion["infusionoffset"] < 0, "patientunitstayid"].unique()
        )
        patient = patient[~patient["patientunitstayid"].isin(pre_icu)]
        infusion = infusion[
            infusion["patientunitstayid"].isin(patient["patientunitstayid"])
        ]
        flow.record(
            "no vasopressor before ICU admission",
            len(patient),
            detail=f"excluded {len(pre_icu)} stays with infusionoffset < 0",
        )

    lower, upper = treatment_window
    treated = set(
        infusion.loc[
            infusion["infusionoffset"].between(lower, upper), "patientunitstayid"
        ].unique()
    )

    cohort = patient.copy()
    cohort["treatment"] = (
        cohort["patientunitstayid"].isin(treated).astype(float)
    )
    cohort["outcome"] = cohort["died_in_hospital"].astype(float)
    cohort["age"] = cohort["age_years"]

    stay_ids = set(cohort["patientunitstayid"])
    for extra in (
        baseline_labs(eicu_root, stay_ids, baseline_window),
        baseline_vitals(eicu_root, stay_ids, baseline_window),
        comorbidity_flags(eicu_root, stay_ids),
    ):
        if len(extra):
            cohort = cohort.merge(
                extra, left_on="patientunitstayid", right_index=True, how="left"
            )

    for name in COMORBIDITY_PATTERNS:
        if name not in cohort.columns:
            cohort[name] = 0.0
        cohort[name] = cohort[name].fillna(0.0)

    if include_apache:
        apache = baseline_apache(eicu_root, stay_ids)
        if len(apache):
            cohort = cohort.merge(
                apache, left_on="patientunitstayid", right_index=True, how="left"
            )

    keep = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "wardid",
        "treatment",
        "outcome",
        "age",
        "admissionweight",
        "unitadmitsource",
    ]
    keep += [c for c in CATEGORICAL_COVARIATES if c in cohort.columns]
    keep += sorted(COMORBIDITY_PATTERNS)
    keep += [
        c
        for c in cohort.columns
        if c.startswith(("lab_", "vital_", "apache_")) and c not in keep
    ]
    keep = [c for c in dict.fromkeys(keep) if c in cohort.columns]
    cohort = cohort[keep].copy()

    cohort["hospitalid"] = cohort["hospitalid"].astype(np.int64)
    cohort["wardid"] = cohort["wardid"].astype(np.int64)

    flow.record("final cohort", len(cohort))
    return cohort, flow


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _window(text):
    parts = [int(p) for p in str(text).split(",")]
    if len(parts) != 2 or parts[0] > parts[1]:
        raise argparse.ArgumentTypeError(f"expected 'lower,upper', got {text!r}")
    return (parts[0], parts[1])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eicu-root", default=DEMO_EICU_ROOT)
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "experiments", "eicu_v1_demo"),
        help="output directory for cohort.csv and cohort_flow.json",
    )
    parser.add_argument("--sepsis-window", type=_window, default=DEFAULT_SEPSIS_WINDOW_MINUTES)
    parser.add_argument(
        "--treatment-window", type=_window, default=DEFAULT_TREATMENT_WINDOW_MINUTES
    )
    parser.add_argument(
        "--baseline-window", type=_window, default=DEFAULT_BASELINE_WINDOW_MINUTES
    )
    parser.add_argument("--include-dopamine", action="store_true")
    parser.add_argument(
        "--include-apache",
        action="store_true",
        help="extract APACHE APS variables (post-treatment; sensitivity arm only)",
    )
    parser.add_argument("--keep-pre-icu-vasopressors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print flow, write nothing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cohort, flow = build_cohort(
        args.eicu_root,
        sepsis_window=args.sepsis_window,
        treatment_window=args.treatment_window,
        baseline_window=args.baseline_window,
        include_dopamine=args.include_dopamine,
        include_apache=args.include_apache,
        keep_pre_icu_vasopressors=args.keep_pre_icu_vasopressors,
    )

    print("Cohort flow")
    print("-----------")
    print(flow.render())
    print()
    if len(cohort):
        print(f"hospitals            : {cohort['hospitalid'].nunique()}")
        print(f"wards                : {cohort['wardid'].nunique()}")
        print(f"treated (early vaso) : {int(cohort['treatment'].sum())}")
        print(f"deaths               : {int(cohort['outcome'].sum())}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    os.makedirs(args.out, exist_ok=True)
    cohort_path = os.path.join(args.out, "cohort.csv")
    cohort.to_csv(cohort_path, index=False)

    metadata = {
        "eicu_root": os.path.abspath(args.eicu_root),
        "sepsis_window_minutes": list(args.sepsis_window),
        "treatment_window_minutes": list(args.treatment_window),
        "baseline_window_minutes": list(args.baseline_window),
        "include_dopamine": bool(args.include_dopamine),
        "include_apache": bool(args.include_apache),
        "keep_pre_icu_vasopressors": bool(args.keep_pre_icu_vasopressors),
        "n_rows": int(len(cohort)),
        "n_hospitals": int(cohort["hospitalid"].nunique()) if len(cohort) else 0,
        "n_wards": int(cohort["wardid"].nunique()) if len(cohort) else 0,
        "n_treated": int(cohort["treatment"].sum()) if len(cohort) else 0,
        "n_deaths": int(cohort["outcome"].sum()) if len(cohort) else 0,
        "flow": flow.as_list(),
    }
    flow_path = os.path.join(args.out, "cohort_flow.json")
    with open(flow_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nwrote {cohort_path}")
    print(f"wrote {flow_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
