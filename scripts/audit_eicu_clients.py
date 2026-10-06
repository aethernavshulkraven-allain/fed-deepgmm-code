"""Stage-1 feasibility audit for the eICU federated IV study.

This is a **gate**: no neural-network training happens until this reports a viable
construction. It answers one question — is there enough *within-client* instrument
variation to identify anything — and applies a pre-registered rule to choose between:

* ``ward``      : clients are hospitals, Z is cross-fitted ward preference (primary)
* ``grouped``   : clients are hospital groups, Z is cross-fitted hospital preference
* ``insufficient_data`` : neither construction is supportable

The eligibility thresholds are frozen here, before any treatment effect is computed,
so the choice cannot be made by looking at the answer.

Usage:
    python scripts/audit_eicu_clients.py --cohort experiments/eicu_v1_demo/cohort.csv
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eicu_common import REPO_ROOT  # noqa: E402
from eicu_instrument import (  # noqa: E402
    PREFERENCE_HOSPITAL,
    PREFERENCE_WARD,
    build_instrument,
    instrument_variation,
    structural_instrument_variation,
)

# The unit whose preference the instrument measures, per construction.
PREFERENCE_UNIT = {
    PREFERENCE_WARD: "wardid",
    PREFERENCE_HOSPITAL: "hospitalid",
}

# Pre-specified client eligibility. Frozen before any effect estimate is inspected;
# changing them after Stage 4 would be selecting the answer.
#
# min_deaths defaults to 0 (not enforced): this gate is Study A's (the outcome
# is simulated there, so a hospital's real death count says nothing about
# whether structural recovery is feasible). It only belongs on the real-outcome
# Study B gate -- pass --min-deaths 20 explicitly for that use.
DEFAULT_THRESHOLDS = {
    "min_patients": 200,
    "min_treated": 20,
    "min_untreated": 20,
    "min_deaths": 0,
    "min_wards": 2,
    "min_per_ward": 50,
}

# A client whose instrument barely moves contributes no local identifying variation.
MIN_INSTRUMENT_SD = 0.01

GROUPING_COLUMNS = ("region", "teachingstatus", "numbedscategory")


# ---------------------------------------------------------------------------
# Per-hospital audit table
# ---------------------------------------------------------------------------


def hospital_audit(cohort, thresholds):
    import pandas as pd

    rows = []
    for hospital_id, rows_h in cohort.groupby("hospitalid", sort=True):
        ward_sizes = rows_h.groupby("wardid").size()
        eligible_wards = ward_sizes[ward_sizes >= thresholds["min_per_ward"]]

        n = len(rows_h)
        treated = int(rows_h["treatment"].sum())
        deaths = int(rows_h["outcome"].sum())

        # Treatment must actually vary inside a ward for ward preference to be
        # anything other than a constant.
        ward_rates = rows_h.groupby("wardid")["treatment"].mean()

        record = {
            "hospitalid": int(hospital_id),
            "n_patients": n,
            "n_treated": treated,
            "n_untreated": n - treated,
            "n_deaths": deaths,
            "treatment_rate": float(rows_h["treatment"].mean()),
            "mortality_rate": float(rows_h["outcome"].mean()),
            "n_wards": int(ward_sizes.size),
            "n_eligible_wards": int(eligible_wards.size),
            "max_ward_size": int(ward_sizes.max()),
            "ward_treatment_rate_sd": float(ward_rates.std(ddof=0))
            if ward_rates.size > 1
            else 0.0,
        }
        for col in GROUPING_COLUMNS:
            if col in rows_h.columns:
                values = rows_h[col].dropna().unique()
                record[col] = values[0] if len(values) else None

        record["ward_eligible"] = bool(
            n >= thresholds["min_patients"]
            and treated >= thresholds["min_treated"]
            and (n - treated) >= thresholds["min_untreated"]
            and deaths >= thresholds["min_deaths"]
            and eligible_wards.size >= thresholds["min_wards"]
        )
        # Grouped-client mode drops the ward requirements; the hospital only needs
        # to be a usable *unit of preference*, not a usable client on its own.
        record["contributes_to_group"] = bool(
            n >= 1 and treated >= 1 and (n - treated) >= 1
        )
        rows.append(record)

    return pd.DataFrame(rows)


def group_audit(cohort, thresholds):
    """Audit for the fallback: hospitals grouped into clients."""
    import pandas as pd

    available = [c for c in GROUPING_COLUMNS if c in cohort.columns]
    if not available:
        return pd.DataFrame(), None

    grouped = cohort.copy()
    grouped["client_group"] = (
        grouped[available].astype("string").fillna("NA").agg(" | ".join, axis=1)
    )

    rows = []
    for group, rows_g in grouped.groupby("client_group", sort=True):
        n = len(rows_g)
        treated = int(rows_g["treatment"].sum())
        deaths = int(rows_g["outcome"].sum())
        hospital_rates = rows_g.groupby("hospitalid")["treatment"].mean()
        rows.append(
            {
                "client_group": group,
                "n_patients": n,
                "n_hospitals": int(rows_g["hospitalid"].nunique()),
                "n_treated": treated,
                "n_untreated": n - treated,
                "n_deaths": deaths,
                "treatment_rate": float(rows_g["treatment"].mean()),
                "mortality_rate": float(rows_g["outcome"].mean()),
                "hospital_treatment_rate_sd": float(hospital_rates.std(ddof=0))
                if hospital_rates.size > 1
                else 0.0,
                "eligible": bool(
                    n >= thresholds["min_patients"]
                    and treated >= thresholds["min_treated"]
                    and (n - treated) >= thresholds["min_untreated"]
                    and deaths >= thresholds["min_deaths"]
                    and rows_g["hospitalid"].nunique() >= 2
                ),
            }
        )
    return pd.DataFrame(rows), grouped


# ---------------------------------------------------------------------------
# Instrument variation probe
# ---------------------------------------------------------------------------


def probe_instrument(cohort, construction, client_col, seed=0):
    """Build the instrument on the whole cohort to measure within-client variation.

    This is a feasibility probe, not an estimation step: it never produces an effect,
    and Stage 2+ rebuilds the instrument from the training split only.
    """
    try:
        train_z, _, _ = build_instrument(
            cohort,
            construction=construction,
            client_col=client_col,
            seed=seed,
        )
    except (ValueError, KeyError, RuntimeError) as exc:
        return {"available": False, "reason": str(exc)}

    probe = cohort.copy()
    probe["z"] = train_z.values

    raw = instrument_variation(probe, client_col, "z")
    # Fold noise is removed by collapsing to the preference unit first; only this
    # counts as identifying variation.
    structural = structural_instrument_variation(
        probe, client_col, PREFERENCE_UNIT[construction], "z"
    )

    return {
        "available": True,
        "z_mean": float(probe["z"].mean()),
        "z_sd_overall": float(probe["z"].std(ddof=0)),
        "raw_within_client_sd_mean": float(raw.mean()) if len(raw) else 0.0,
        "within_client_sd_mean": float(structural.mean()) if len(structural) else 0.0,
        "within_client_sd_median": float(structural.median()) if len(structural) else 0.0,
        "n_clients_with_variation": int((structural > MIN_INSTRUMENT_SD).sum()),
        "n_clients": int(len(structural)),
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide_construction(hospitals, groups, ward_probe, hospital_probe, thresholds):
    """Apply the pre-registered rule and explain the outcome."""
    n_ward_eligible = int(hospitals["ward_eligible"].sum()) if len(hospitals) else 0
    n_group_eligible = int(groups["eligible"].sum()) if len(groups) else 0

    ward_varying = ward_probe.get("n_clients_with_variation", 0) if ward_probe.get(
        "available"
    ) else 0
    group_varying = hospital_probe.get("n_clients_with_variation", 0) if hospital_probe.get(
        "available"
    ) else 0

    reasons = []

    # Primary: enough hospitals that are usable clients *and* have ward variation.
    if n_ward_eligible >= 5 and ward_varying >= 5:
        decision = PREFERENCE_WARD
        reasons.append(
            f"{n_ward_eligible} hospitals meet client eligibility and "
            f"{ward_varying} have within-hospital ward-preference variation"
        )
    elif n_group_eligible >= 5 and group_varying >= 5:
        decision = "grouped"
        reasons.append(
            f"only {n_ward_eligible} hospitals are ward-eligible, but "
            f"{n_group_eligible} hospital groups are eligible with "
            f"{group_varying} showing within-group hospital-preference variation"
        )
    else:
        decision = "insufficient_data"
        reasons.append(
            f"ward-eligible hospitals: {n_ward_eligible} (need >= 5); "
            f"eligible hospital groups: {n_group_eligible} (need >= 5)"
        )
        reasons.append(
            "no construction has enough clients with within-client instrument "
            "variation; this release cannot support the federated IV analysis"
        )

    return {
        "construction": decision,
        "n_ward_eligible_hospitals": n_ward_eligible,
        "n_eligible_groups": n_group_eligible,
        "n_ward_clients_with_variation": ward_varying,
        "n_group_clients_with_variation": group_varying,
        "thresholds": dict(thresholds),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(cohort, hospitals, groups, decision, ward_probe, hospital_probe):
    lines = []
    add = lines.append

    add("# eICU federated IV — client feasibility audit\n")
    add(f"- cohort rows: **{len(cohort)}**")
    add(f"- hospitals: **{cohort['hospitalid'].nunique()}**")
    add(f"- wards: **{cohort['wardid'].nunique()}**")
    add(f"- treated (early vasopressor): **{int(cohort['treatment'].sum())}**")
    add(f"- in-hospital deaths: **{int(cohort['outcome'].sum())}**")
    add("")

    add("## Decision\n")
    add(f"**construction = `{decision['construction']}`**\n")
    for reason in decision["reasons"]:
        add(f"- {reason}")
    add("")
    add("Thresholds (pre-registered, frozen before any effect estimate):\n")
    add("| threshold | value |")
    add("|---|---|")
    for key, value in sorted(decision["thresholds"].items()):
        add(f"| `{key}` | {value} |")
    add("")

    add("## Per-hospital distribution\n")
    if len(hospitals):
        sizes = hospitals["n_patients"]
        add("| statistic | value |")
        add("|---|---|")
        add(f"| hospitals | {len(hospitals)} |")
        add(f"| patients per hospital (median) | {sizes.median():.0f} |")
        add(f"| patients per hospital (max) | {sizes.max():.0f} |")
        add(f"| hospitals with >1 ward | {(hospitals['n_wards'] > 1).sum()} |")
        add(f"| hospitals meeting client eligibility | {int(hospitals['ward_eligible'].sum())} |")
        add("")

    add("## Instrument variation probe\n")
    add(
        "`structural sd` is the between-unit (ward, or hospital in the grouped "
        "fallback) spread of the instrument within a client. `raw sd` additionally "
        "contains cross-fitting fold noise, which is not identifying variation and "
        "is excluded from the decision."
    )
    add("")
    add("| construction | clients | with structural variation | mean structural sd | mean raw sd |")
    add("|---|---|---|---|---|")
    for name, probe in (("ward", ward_probe), ("hospital", hospital_probe)):
        if probe.get("available"):
            add(
                f"| {name} | {probe['n_clients']} | "
                f"{probe['n_clients_with_variation']} | "
                f"{probe['within_client_sd_mean']:.4f} | "
                f"{probe['raw_within_client_sd_mean']:.4f} |"
            )
        else:
            add(f"| {name} | — | — | — | unavailable: {probe.get('reason', 'n/a')} |")
    add("")

    if decision["construction"] == "insufficient_data":
        add("## Consequence\n")
        add(
            "This release cannot back a reported estimate. It is usable only as a "
            "pipeline harness: the ETL, instrument construction, splitting and "
            "training code can be exercised end to end, but every number produced "
            "from it is a smoke-test artefact, not a result."
        )
        add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort",
        default=os.path.join(REPO_ROOT, "experiments", "eicu_v1_demo", "cohort.csv"),
    )
    parser.add_argument("--out", default=None, help="defaults to the cohort's directory")
    parser.add_argument("--seed", type=int, default=0)
    for key, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=value)
    return parser.parse_args(argv)


def main(argv=None):
    import pandas as pd

    args = parse_args(argv)
    thresholds = {key: getattr(args, key) for key in DEFAULT_THRESHOLDS}

    cohort = pd.read_csv(args.cohort)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.cohort))
    os.makedirs(out_dir, exist_ok=True)

    hospitals = hospital_audit(cohort, thresholds)
    groups, grouped_cohort = group_audit(cohort, thresholds)

    ward_probe = probe_instrument(cohort, PREFERENCE_WARD, "hospitalid", seed=args.seed)
    if grouped_cohort is not None and len(grouped_cohort):
        hospital_probe = probe_instrument(
            grouped_cohort, PREFERENCE_HOSPITAL, "client_group", seed=args.seed
        )
    else:
        hospital_probe = {"available": False, "reason": "no grouping columns in cohort"}

    decision = decide_construction(
        hospitals, groups, ward_probe, hospital_probe, thresholds
    )
    decision["ward_probe"] = ward_probe
    decision["hospital_probe"] = hospital_probe

    hospitals_path = os.path.join(out_dir, "client_audit.csv")
    hospitals.to_csv(hospitals_path, index=False)

    groups_path = os.path.join(out_dir, "group_audit.csv")
    if len(groups):
        groups.to_csv(groups_path, index=False)

    decision_path = os.path.join(out_dir, "construction_decision.json")
    with open(decision_path, "w") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = render_report(
        cohort, hospitals, groups, decision, ward_probe, hospital_probe
    )
    report_path = os.path.join(out_dir, "client_audit.md")
    with open(report_path, "w") as handle:
        handle.write(report + "\n")

    print(report)
    print()
    print(f"wrote {hospitals_path}")
    if len(groups):
        print(f"wrote {groups_path}")
    print(f"wrote {decision_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
