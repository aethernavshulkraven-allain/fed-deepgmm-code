#!/usr/bin/env python3
"""Build the versioned Study A v2 off-hours eICU cohort.

Study A v2 is deliberately independent of the archived v1 vasopressor/sepsis
cohort.  It keeps real hospital partitions, admission timing, baseline
covariates, and missingness, but does not require sepsis, mortality, observed
vasopressor use, an infusion interface, or ward-level preference variation.

The output contains one row per first ICU stay for a hospital admission.  A
patient's hospital admissions are assigned to one split as a group, and
hospitals without non-empty Train/Dev/Test splits or within-hospital off-hours
variation are removed before any treatment or outcome is simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eicu_common import DEMO_EICU_ROOT, parse_age, read_table  # noqa: E402
from prepare_eicu_cohort import (  # noqa: E402
    CohortFlow,
    baseline_labs,
    baseline_vitals,
)


SPLIT_FRACTIONS = {"train": 0.70, "dev": 0.15, "test": 0.15}
SPLIT_SEED = 20260727
MIN_CLIENT_ROWS = 7
ID_COLUMNS = (
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "hospitalid",
    "wardid",
)
CONTINUOUS_COLUMNS = (
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
CATEGORICAL_COLUMNS = (
    "gender",
    "hospitaladmitsource",
)


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time_minutes(values: pd.Series) -> pd.Series:
    """Parse eICU's HH:MM[:SS] admission clock into minutes after midnight."""
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            text.loc[missing], format="%H:%M", errors="coerce"
        )
    return parsed.dt.hour.astype("float64") * 60.0 + parsed.dt.minute.astype("float64")


def off_hours_from_minutes(minutes: pd.Series) -> pd.Series:
    """1 outside the half-open regular-hours interval [07:00, 19:00)."""
    return ((minutes < 7 * 60) | (minutes >= 19 * 60)).astype("int64")


def _first_icu_stay_per_admission(patient: pd.DataFrame) -> pd.DataFrame:
    """Match the established v1 definition without imposing v1 clinical gates."""
    rows = patient[patient["unitvisitnumber"] == 1].copy()
    return (
        rows.sort_values(
            ["patienthealthsystemstayid", "hospitaladmitoffset"],
            ascending=[True, False],
        )
        .drop_duplicates("patienthealthsystemstayid", keep="first")
        .copy()
    )


def load_population(eicu_root: str, flow: CohortFlow) -> pd.DataFrame:
    patient = read_table(eicu_root, "patient")
    flow.record("all ICU stays", len(patient))

    patient["age"] = parse_age(patient["age"])
    patient = patient[patient["age"] >= 18].copy()
    flow.record("adults (age >= 18)", len(patient))

    patient = _first_icu_stay_per_admission(patient)
    flow.record("first ICU stay per hospital admission", len(patient))

    required = (
        patient["hospitalid"].notna()
        & patient["patientunitstayid"].notna()
        & patient["patienthealthsystemstayid"].notna()
        & patient["uniquepid"].notna()
    )
    patient = patient[required].copy()
    flow.record("valid hospital, stay, and patient identifiers", len(patient))

    patient["admission_clock_minutes"] = parse_time_minutes(
        patient["hospitaladmittime24"]
    )
    patient = patient[patient["admission_clock_minutes"].notna()].copy()
    flow.record("valid hospital admission clock time", len(patient))
    patient["z_off_hours"] = off_hours_from_minutes(
        patient["admission_clock_minutes"]
    )

    # A patient represented at multiple hospitals would otherwise be owned by
    # more than one federated client and could cross client/split boundaries.
    patient_hospitals = patient.groupby("uniquepid")["hospitalid"].nunique()
    cross_hospital_patients = set(patient_hospitals[patient_hospitals > 1].index)
    patient = patient[~patient["uniquepid"].isin(cross_hospital_patients)].copy()
    flow.record(
        "patient belongs to one hospital client",
        len(patient),
        detail=f"excluded {len(cross_hospital_patients)} cross-hospital patients",
    )
    return patient


def _target_split_counts(
    n_rows: int, tie_order: tuple[str, ...] = ("train", "dev", "test")
) -> dict[str, int]:
    """Largest-remainder 70/15/15 targets with every split non-empty."""
    if n_rows < 3:
        raise ValueError("at least three rows are required for three splits")
    names = tuple(SPLIT_FRACTIONS)
    rank = {name: index for index, name in enumerate(tie_order)}
    raw = {name: SPLIT_FRACTIONS[name] * n_rows for name in names}
    counts = {name: max(int(np.floor(raw[name])), 1) for name in names}
    while sum(counts.values()) > n_rows:
        candidates = [name for name in names if counts[name] > 1]
        if not candidates:
            raise ValueError(f"cannot allocate {n_rows} rows across three splits")
        name = max(
            candidates,
            key=lambda key: (counts[key] - raw[key], -rank.get(key, 0)),
        )
        counts[name] -= 1
    while sum(counts.values()) < n_rows:
        name = max(
            names,
            key=lambda key: (raw[key] - counts[key], -rank.get(key, 0)),
        )
        counts[name] += 1
    return counts


def _hospital_rng(seed: int, hospital_id: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(hospital_id), 0xE1C0])
    )


def assign_patient_group_splits(
    cohort: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    client_col: str = "hospitalid",
    patient_col: str = "uniquepid",
) -> pd.Series:
    """Deterministically approximate 70/15/15 while keeping patients intact."""
    assignment = pd.Series("", index=cohort.index, dtype="object")

    for hospital_id, rows in cohort.groupby(client_col, sort=True):
        rng = _hospital_rng(seed, int(hospital_id))
        heldout_order = (
            ("dev", "test")
            if int(rng.integers(0, 2)) == 0
            else ("test", "dev")
        )
        tie_order = ("train",) + heldout_order
        targets = _target_split_counts(len(rows), tie_order=tie_order)
        groups: list[dict[str, Any]] = []
        for patient_id, patient_rows in rows.groupby(patient_col, sort=True):
            z_values = set(patient_rows["z_off_hours"].astype(int))
            groups.append(
                {
                    "patient_id": str(patient_id),
                    "indices": list(patient_rows.index),
                    "size": int(len(patient_rows)),
                    "z_values": z_values,
                    "jitter": float(rng.random()),
                }
            )
        groups.sort(key=lambda item: (-item["size"], item["jitter"], item["patient_id"]))
        counts = {name: 0 for name in SPLIT_FRACTIONS}
        chosen: dict[str, str] = {}

        # Seed Train with both real instrument levels when possible. This is a
        # real-data, pre-simulation eligibility operation, never a simulated
        # first-stage filter.
        for z_level in (0, 1):
            candidate = next(
                (
                    group
                    for group in groups
                    if group["patient_id"] not in chosen
                    and z_level in group["z_values"]
                ),
                None,
            )
            if candidate is not None:
                chosen[candidate["patient_id"]] = "train"
                counts["train"] += candidate["size"]

        # Ensure Dev and Test each receive a whole patient group.
        for split in ("dev", "test"):
            candidate = next(
                (
                    group
                    for group in reversed(groups)
                    if group["patient_id"] not in chosen
                ),
                None,
            )
            if candidate is not None:
                chosen[candidate["patient_id"]] = split
                counts[split] += candidate["size"]

        for group in groups:
            if group["patient_id"] in chosen:
                continue
            split = max(
                SPLIT_FRACTIONS,
                key=lambda name: (
                    (targets[name] - counts[name]) / max(targets[name], 1),
                    SPLIT_FRACTIONS[name],
                    -tie_order.index(name),
                ),
            )
            chosen[group["patient_id"]] = split
            counts[split] += group["size"]

        for group in groups:
            assignment.loc[group["indices"]] = chosen[group["patient_id"]]

    if (assignment == "").any():
        raise RuntimeError("internal error: not every cohort row received a split")
    return assignment


def client_eligibility(
    cohort: pd.DataFrame, *, min_client_rows: int = MIN_CLIENT_ROWS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only real-data, scenario-independent client gates."""
    records = []
    for hospital_id, rows in cohort.groupby("hospitalid", sort=True):
        split_counts = rows["split"].value_counts()
        z_counts = rows["z_off_hours"].value_counts()
        train_z = rows.loc[rows["split"] == "train", "z_off_hours"].nunique()
        reasons = []
        if len(rows) < min_client_rows:
            reasons.append("fewer_than_min_client_rows")
        if rows["uniquepid"].nunique() < 3:
            reasons.append("fewer_than_three_unique_patients")
        if any(int(split_counts.get(split, 0)) == 0 for split in SPLIT_FRACTIONS):
            reasons.append("missing_train_dev_or_test")
        if int(z_counts.get(0, 0)) == 0 or int(z_counts.get(1, 0)) == 0:
            reasons.append("no_within_hospital_off_hours_variation")
        if train_z < 2:
            reasons.append("no_training_off_hours_variation")
        records.append(
            {
                "hospitalid": int(hospital_id),
                "n_rows": int(len(rows)),
                "n_unique_patients": int(rows["uniquepid"].nunique()),
                "n_train": int(split_counts.get("train", 0)),
                "n_dev": int(split_counts.get("dev", 0)),
                "n_test": int(split_counts.get("test", 0)),
                "n_regular_hours": int(z_counts.get(0, 0)),
                "n_off_hours": int(z_counts.get(1, 0)),
                "train_z_varies": bool(train_z == 2),
                "eligible": not reasons,
                "exclusion_reasons": ";".join(reasons),
            }
        )
    audit = pd.DataFrame.from_records(records)
    eligible = set(audit.loc[audit["eligible"], "hospitalid"])
    kept = cohort[cohort["hospitalid"].isin(eligible)].copy()
    return kept, audit


def assert_no_leakage(cohort: pd.DataFrame) -> None:
    for column in ("patientunitstayid", "patienthealthsystemstayid", "uniquepid"):
        counts = cohort.groupby(column)["split"].nunique()
        if int((counts > 1).sum()):
            raise ValueError(f"{column} crosses Train/Dev/Test")
    per_client = cohort.groupby("hospitalid")["split"].nunique()
    if not bool((per_client == 3).all()):
        raise ValueError("a retained hospital is absent from at least one split")
    if cohort.groupby("hospitalid")["z_off_hours"].nunique().min() != 2:
        raise ValueError("a retained hospital lacks within-hospital Z variation")
    if cohort.loc[cohort["split"] == "train"].groupby("hospitalid")[
        "z_off_hours"
    ].nunique().min() != 2:
        raise ValueError("a retained hospital lacks Train-split Z variation")


def build_cohort(
    eicu_root: str,
    *,
    min_client_rows: int = MIN_CLIENT_ROWS,
    split_seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, CohortFlow, pd.DataFrame]:
    flow = CohortFlow()
    patient = load_population(eicu_root, flow)

    stay_ids = set(patient["patientunitstayid"])
    for extra in (
        baseline_labs(eicu_root, stay_ids, (0, 60)),
        baseline_vitals(eicu_root, stay_ids, (0, 60)),
    ):
        if len(extra):
            patient = patient.merge(
                extra, left_on="patientunitstayid", right_index=True, how="left"
            )

    # Keep wardid only as an audit field; it is never a client key or model input.
    keep = list(ID_COLUMNS) + [
        "hospitaladmittime24",
        "admission_clock_minutes",
        "z_off_hours",
    ]
    keep += list(CONTINUOUS_COLUMNS) + list(CATEGORICAL_COLUMNS)
    keep = [column for column in dict.fromkeys(keep) if column in patient.columns]
    cohort = patient[keep].copy()
    cohort["hospitalid"] = cohort["hospitalid"].astype("int64")
    if "wardid" in cohort:
        cohort["wardid"] = pd.to_numeric(cohort["wardid"], errors="coerce").astype(
            "Int64"
        )

    cohort["split"] = assign_patient_group_splits(cohort, seed=split_seed)
    cohort, audit = client_eligibility(
        cohort, min_client_rows=min_client_rows
    )
    flow.record(
        "eligible hospital clients before simulation",
        len(cohort),
        detail=(
            f"{cohort['hospitalid'].nunique()} hospitals; min rows "
            f"{min_client_rows}; real off-hours and split gates"
        ),
    )
    assert_no_leakage(cohort)
    cohort = cohort.sort_values(
        ["hospitalid", "split", "patienthealthsystemstayid", "patientunitstayid"]
    ).reset_index(drop=True)
    flow.record("final Study A v2 cohort", len(cohort))
    return cohort, flow, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eicu-root", default=DEMO_EICU_ROOT)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "experiments" / "eicu_study_a_v2_offhours_demo_20260727"
        ),
    )
    parser.add_argument("--min-client-rows", type=int, default=MIN_CLIENT_ROWS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--scenario-scope",
        choices=("demo", "full_eicu"),
        default=None,
        help="Defaults to demo when --eicu-root contains 'demo', otherwise full_eicu.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_client_rows < 3:
        raise ValueError("--min-client-rows must be at least 3")
    cohort, flow, audit = build_cohort(
        args.eicu_root,
        min_client_rows=args.min_client_rows,
        split_seed=args.split_seed,
    )
    scenario_scope = args.scenario_scope or (
        "demo" if "demo" in str(args.eicu_root).lower() else "full_eicu"
    )

    print("Cohort flow")
    print("-----------")
    print(flow.render())
    print(
        f"\nclients={cohort['hospitalid'].nunique()} rows={len(cohort)} "
        f"splits={cohort['split'].value_counts().sort_index().to_dict()} "
        f"off_hours_rate={cohort['z_off_hours'].mean():.4f}"
    )
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cohort_path = out / "cohort.csv"
    audit_path = out / "client_eligibility_audit.csv"
    client_path = out / "frozen_client_list.json"
    flow_path = out / "cohort_metadata.json"
    cohort.to_csv(cohort_path, index=False)
    audit.to_csv(audit_path, index=False)

    client_ids = sorted(int(value) for value in cohort["hospitalid"].unique())
    client_payload = {
        "client_id_column": "hospitalid",
        "wardid_used_as_client": False,
        "n_clients": len(client_ids),
        "n_unique_hospitalid": int(cohort["hospitalid"].nunique()),
        "client_ids": client_ids,
    }
    with client_path.open("w") as handle:
        json.dump(client_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    metadata = {
        "protocol_version": "eicu_study_a_v2_offhours",
        "study_label": "Study A v2",
        "scenario_scope": scenario_scope,
        "is_demo": scenario_scope == "demo",
        "eicu_root": str(Path(args.eicu_root).resolve()),
        "instrument": {
            "name": "off_hours_admission",
            "source_column": "hospitaladmittime24",
            "definition": "outside [07:00, 19:00)",
            "weekend_instrument_included": False,
            "weekend_reason": "eICU patient table has no reliable admission date or weekday",
        },
        "client_id_column": "hospitalid",
        "wardid_used_as_client": False,
        "split_fractions": SPLIT_FRACTIONS,
        "split_seed": int(args.split_seed),
        "split_group_columns": ["uniquepid", "patienthealthsystemstayid"],
        "min_client_rows": int(args.min_client_rows),
        "n_rows": int(len(cohort)),
        "n_clients": len(client_ids),
        "n_unique_hospitalid": int(cohort["hospitalid"].nunique()),
        "split_sizes": {
            name: int((cohort["split"] == name).sum())
            for name in SPLIT_FRACTIONS
        },
        "clients_per_split": {
            name: int(
                cohort.loc[cohort["split"] == name, "hospitalid"].nunique()
            )
            for name in SPLIT_FRACTIONS
        },
        "off_hours_rate": float(cohort["z_off_hours"].mean()),
        "continuous_covariate_candidates": [
            column for column in CONTINUOUS_COLUMNS if column in cohort
        ],
        "categorical_covariate_candidates": [
            column for column in CATEGORICAL_COLUMNS if column in cohort
        ],
        "identifier_columns_excluded_from_model": list(ID_COLUMNS),
        "flow": flow.as_list(),
    }
    metadata["cohort_sha256"] = sha256_file(cohort_path)
    metadata["frozen_client_list_sha256"] = sha256_file(client_path)
    with flow_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {cohort_path}")
    print(f"wrote {audit_path}")
    print(f"wrote {client_path}")
    print(f"wrote {flow_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
