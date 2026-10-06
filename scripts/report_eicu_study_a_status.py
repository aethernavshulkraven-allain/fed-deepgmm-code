#!/usr/bin/env python3
"""Read-only status reporter for the Study A v1 and v2 eICU campaigns.

Anti-drift purpose
-------------------
Every number this script prints is derived from artifacts on disk -- manifest
CSVs, metadata JSON, and a filesystem scan of ``results/`` -- never typed in
by hand and never copied from a write-up. Prose drifts: a campaign correctly
described as "not run yet" when a document was written can be mid-run five
minutes later. This tool has no memory between invocations, so it cannot
drift; it reports whatever is true of the tree at the moment it runs.

In particular this script does NOT assume Study A v2's tuning campaign is
unstarted. It counts ``metrics.json`` files under the paths named by
``tuning_manifest.csv`` (and, once one is materialized, the final manifest)
and reports whatever that count is right now.

Read-only guarantee
--------------------
This script only ever ``stat``s and reads files under ``results/``. It never
launches training, never writes under ``results/``, and never modifies any of
the frozen/checksummed v1 or v2 setup artifacts (``cohort.csv``,
``cohort_metadata.json``, ``frozen_client_list.json``,
``setup_validation_summary.json``, ``tuning_manifest.*``,
``client_eligibility_audit.csv``, or anything under
``experiments/eicu_v1_demo/`` / ``experiments/eicu_study_a_demo_stage_a_v1_20260727/``).
It writes only new, derived files: ``status.json`` / ``status.md`` next to
the v2 setup artifacts, a v1 funnel extension alongside (not over)
``cohort_flow.json``, and a client-heterogeneity summary next to the v2
client audit.

Usage
-----
    python scripts/report_eicu_study_a_status.py
    python scripts/report_eicu_study_a_status.py --no-write   # print only

See ``--help`` for path overrides (used by the test suite to point the
reporter at synthetic fixture trees instead of the real experiment dirs).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Canonical split-name alias map lives in validate_eicu_study_a_campaign.py
# (task-4 fix) so there is exactly one source of truth; this reporter only
# consumes it.
import validate_eicu_study_a_campaign as split_names  # noqa: E402

DEFAULT_V2_DIR = REPO_ROOT / "experiments" / "eicu_study_a_v2_offhours_demo_20260727"
DEFAULT_V2_PROTOCOL_JSON = REPO_ROOT / "experiments" / "eicu_study_a_v2_offhours" / "protocol_v2.json"
DEFAULT_V1_COHORT_DIR = REPO_ROOT / "experiments" / "eicu_v1_demo"
DEFAULT_V1_CAMPAIGN_DIR = REPO_ROOT / "experiments" / "eicu_study_a_demo_stage_a_v1_20260727"

# Role -> the protocol_v2.json "final" block key that states its planned count.
FINAL_ROLE_DESIGN_KEYS = {
    "confirmatory": "primary_federated_runs",
    "centralized_baseline": "centralized_runs",
    "aggregation_ablation": "aggregation_ablation_runs",
}

PHASE_ORDER = (
    "not_setup",
    "setup_certified",
    "tuning_in_progress",
    "tuning_complete",
    "final_not_started",
    "final_in_progress",
    "final_complete",
    "analyzed",
)


# ---------------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """Read JSON, returning None if the file is absent.

    A malformed/partially-written file (e.g. a run still flushing its
    ``metrics.json`` while this reporter runs concurrently) is reported as
    absent rather than crashing the whole report.
    """
    if path is None or not Path(path).is_file():
        return None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Manifest -> results/ completion scan
# ---------------------------------------------------------------------------


def run_is_complete(output_root: str, result_path: str) -> bool:
    """A run is complete iff its result directory has a ``metrics.json``.

    Empirically (checked against both the frozen v1 campaign and the v2
    tuning runs in flight while this reporter was written) ``metrics.json``
    is written once, at the end of a run; a run that is queued, running, or
    was interrupted has at most partial per-round CSVs and checkpoints but no
    ``metrics.json``.
    """
    if not output_root or not result_path:
        return False
    root = Path(output_root)
    if not root.is_absolute():
        # Manifests generated by prepare_eicu_study_a_v2_manifest.py always
        # write an absolute output_root; a relative one only shows up from an
        # ad hoc/non-standard manifest, so resolve it against the repo root
        # rather than the reporter's current working directory.
        root = REPO_ROOT / root
    run_dir = root / result_path
    return (run_dir / "metrics.json").is_file()


def scan_manifest_completion(manifest_csv: Path) -> dict[str, Any]:
    """Derive planned-vs-completed counts per ``role`` from a manifest CSV.

    This is read-only: for every row it only checks whether
    ``<output_root>/<result_path>/metrics.json`` exists. It never launches
    anything and never writes anything.
    """
    manifest_csv = Path(manifest_csv) if manifest_csv is not None else None
    result: dict[str, Any] = {
        "manifest_path": str(manifest_csv) if manifest_csv else None,
        "exists": bool(manifest_csv and manifest_csv.is_file()),
        "total_planned": 0,
        "total_completed": 0,
        "total_diverged": 0,
        "by_role": {},
    }
    if not result["exists"]:
        return result

    rows = read_csv_rows(manifest_csv)
    by_role: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"planned": 0, "completed": 0, "diverged": 0, "incomplete_run_ids": []}
    )
    for row in rows:
        role = row.get("role") or row.get("campaign_role") or "unknown"
        entry = by_role[role]
        entry["planned"] += 1
        run_id = row.get("run_id", "")
        output_root = row.get("output_root", "")
        result_path = row.get("result_path", "")
        if run_is_complete(output_root, result_path):
            entry["completed"] += 1
            metrics = read_json(Path(output_root) / result_path / "metrics.json")
            if isinstance(metrics, dict) and metrics.get("diverged") is True:
                entry["diverged"] += 1
        else:
            entry["incomplete_run_ids"].append(run_id)

    result["by_role"] = dict(by_role)
    result["total_planned"] = sum(v["planned"] for v in by_role.values())
    result["total_completed"] = sum(v["completed"] for v in by_role.values())
    result["total_diverged"] = sum(v["diverged"] for v in by_role.values())
    return result


def find_final_manifest(v2_dir: Path) -> Path | None:
    """Locate a materialized final manifest, if one exists.

    ``RUNBOOK.md`` names it ``final_manifest.csv``; that exact name is
    checked first. As a fallback (in case a future campaign names it
    differently) any other ``*_manifest.csv`` in the directory whose ``role``
    column contains a non-tuning role is also accepted, so this function
    degrades gracefully rather than silently reporting "no final manifest"
    forever if the filename convention changes.
    """
    v2_dir = Path(v2_dir)
    candidate = v2_dir / "final_manifest.csv"
    if candidate.is_file():
        return candidate
    if not v2_dir.is_dir():
        return None
    for path in sorted(v2_dir.glob("*_manifest.csv")):
        if path.name == "tuning_manifest.csv":
            continue
        rows = read_csv_rows(path)
        roles = {row.get("role") for row in rows}
        if roles & set(FINAL_ROLE_DESIGN_KEYS):
            return path
    return None


def load_final_matrix_design(protocol_json_path: Path) -> dict[str, Any]:
    """Planned final-matrix counts, read from ``protocol_v2.json`` (structured
    data), never hardcoded in this script."""
    doc = read_json(protocol_json_path)
    if not isinstance(doc, dict):
        return {"available": False, "by_role": {}, "total": None}
    final_block = doc.get("final", {})
    by_role = {
        role: final_block.get(key)
        for role, key in FINAL_ROLE_DESIGN_KEYS.items()
        if key in final_block
    }
    return {
        "available": True,
        "by_role": by_role,
        "total": final_block.get("total_runs"),
    }


def check_analysis_ledger(v2_dir: Path) -> dict[str, Any]:
    """Look for the effect-metric materialization ledger RUNBOOK.md names
    (``effect_metric_materialization.json``). Presence with zero
    failed/missing entries is this reporter's signal for the ``analyzed``
    phase."""
    path = Path(v2_dir) / "effect_metric_materialization.json"
    doc = read_json(path)
    if not isinstance(doc, dict):
        return {"exists": False}
    ledger = doc.get("ledger", {})
    summary = doc.get("summary", {})
    return {
        "exists": True,
        "path": str(path),
        "passed": len(ledger.get("passed", [])) if ledger else summary.get("passed"),
        "failed": len(ledger.get("failed", [])) if ledger else summary.get("failed"),
        "missing": len(ledger.get("missing", [])) if ledger else summary.get("missing"),
    }


# ---------------------------------------------------------------------------
# Client-size heterogeneity (task 3)
# ---------------------------------------------------------------------------


def compute_client_heterogeneity(cohort_csv: Path, audit_csv: Path) -> dict[str, Any]:
    """Client-size and eligibility-drop statistics computed directly from
    ``cohort.csv`` and ``client_eligibility_audit.csv``.

    Every number here is measured, not asserted -- this function exists
    specifically to check claims like "179 hospital clients, 7-26 rows,
    median 11" against the artifacts rather than trust them.
    """
    cohort_rows = read_csv_rows(cohort_csv)
    audit_rows = read_csv_rows(audit_csv)

    out: dict[str, Any] = {
        "cohort_csv": str(cohort_csv) if cohort_csv else None,
        "audit_csv": str(audit_csv) if audit_csv else None,
        "available": bool(cohort_rows and audit_rows),
    }
    if not out["available"]:
        return out

    sizes: Counter[str] = Counter()
    for row in cohort_rows:
        hospital_id = row.get("hospitalid")
        if hospital_id is not None:
            sizes[hospital_id] += 1
    size_values = sorted(sizes.values())

    if size_values:
        out["client_row_counts"] = {
            "n_clients_in_cohort": len(size_values),
            "total_rows": sum(size_values),
            "min": size_values[0],
            "median": statistics.median(size_values),
            "mean": round(statistics.mean(size_values), 3),
            "max": size_values[-1],
            "n_clients_under_10_rows": sum(1 for v in size_values if v < 10),
        }

    eligible_flags = [row.get("eligible") for row in audit_rows]
    n_candidates = len(audit_rows)
    n_eligible = sum(1 for flag in eligible_flags if str(flag).strip().lower() == "true")
    n_excluded = n_candidates - n_eligible

    reason_counter: Counter[str] = Counter()
    excluded_hospitals: list[dict[str, Any]] = []
    for row in audit_rows:
        if str(row.get("eligible", "")).strip().lower() == "true":
            continue
        reasons = [r.strip() for r in (row.get("exclusion_reasons") or "").split(";") if r.strip()]
        for reason in reasons:
            reason_counter[reason] += 1
        excluded_hospitals.append(
            {
                "hospitalid": row.get("hospitalid"),
                "n_rows": row.get("n_rows"),
                "exclusion_reasons": reasons,
            }
        )

    out["eligibility_funnel"] = {
        "n_candidate_hospitals": n_candidates,
        "n_eligible_hospitals": n_eligible,
        "n_excluded_hospitals": n_excluded,
        "exclusion_reason_occurrences": dict(sorted(reason_counter.items())),
        "note": (
            "occurrences can exceed n_excluded_hospitals because a hospital "
            "may fail more than one gate"
        ),
        "excluded_hospitals": excluded_hospitals,
    }

    if size_values:
        consistent = len(size_values) == n_eligible
        out["cross_check"] = {
            "n_clients_in_cohort_csv": len(size_values),
            "n_eligible_in_audit_csv": n_eligible,
            "consistent": consistent,
        }

    return out


# ---------------------------------------------------------------------------
# Cohort funnel parity (task 2)
# ---------------------------------------------------------------------------


def build_v2_funnel(cohort_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cohort_metadata, dict):
        return {"available": False}
    flow = cohort_metadata.get("flow", [])
    stages = []
    for step in flow:
        category = "clinical_or_linkage_gates"
        if "eligible" in step.get("step", "").lower() or "final" in step.get("step", "").lower():
            category = "instrument_and_eligibility_gates"
        stages.append(
            {
                "stage_category": category,
                "step": step.get("step"),
                "n_rows": step.get("n_stays"),
                "detail": step.get("detail"),
            }
        )
    return {
        "available": True,
        "protocol_note": (
            "v2 applies no clinical requirement (no sepsis / mortality / "
            "vasopressor / infusion-interface gate) by protocol design, so "
            "there is no v1-style clinical-gate collapse; the only collapse "
            "is linkage (one hospital per patient) plus the instrument / "
            "min-rows / split-eligibility gate."
        ),
        "stages": stages,
        "final_n_rows": cohort_metadata.get("n_rows"),
        "final_n_clients": cohort_metadata.get("n_clients"),
    }


def build_v1_funnel(v1_cohort_dir: Path, v1_campaign_dir: Path) -> dict[str, Any]:
    """v1's complete funnel through to 9 rows / 3 hospitals.

    ``cohort_flow.json`` alone stops at "final cohort: 201" -- the clinical
    gate collapse. The decisive second collapse (201 rows / 89 hospitals ->
    9 rows / 3 hospitals) is scattered across ``construction_decision.json``
    (``ward_probe.n_clients_with_variation``), ``freeze_record.json``
    (``eligibility.expected_eligible_hospital_ids``), and the frozen scenario
    metadata under ``fedgmm/.../data/eicu_semisynth/*_metadata.json``
    (``split_sizes`` / ``clients_per_split`` for the final 9-row cohort).
    This function reads all of them and does not edit any of them.
    """
    cohort_flow = read_json(Path(v1_cohort_dir) / "cohort_flow.json")
    construction = read_json(Path(v1_cohort_dir) / "construction_decision.json")
    freeze_record = read_json(Path(v1_campaign_dir) / "freeze_record.json")

    out: dict[str, Any] = {"available": cohort_flow is not None}
    if cohort_flow is None:
        return out

    stages = []
    for step in cohort_flow.get("flow", []):
        stages.append(
            {
                "stage_category": "clinical_gates",
                "step": step.get("step"),
                "n_rows": step.get("n_stays"),
                "detail": step.get("detail"),
            }
        )
    clinical_final_n = cohort_flow.get("n_rows")
    clinical_final_hospitals = cohort_flow.get("n_hospitals")

    # Second collapse: instrument-variation / ward-preference eligibility gate.
    if isinstance(construction, dict):
        ward_probe = construction.get("ward_probe", {})
        stages.append(
            {
                "stage_category": "instrument_variation_gates",
                "step": "ward-construction within-client instrument-variation probe",
                "n_rows": clinical_final_n,
                "n_hospitals_evaluated": ward_probe.get("n_clients"),
                "n_hospitals_with_variation": ward_probe.get("n_clients_with_variation"),
                "detail": (
                    "minimum_structural_instrument_sd="
                    f"{(freeze_record or {}).get('eligibility', {}).get('minimum_structural_instrument_sd')}"
                ),
            }
        )

    eligible_ids = None
    if isinstance(freeze_record, dict):
        eligible_ids = freeze_record.get("eligibility", {}).get("expected_eligible_hospital_ids")

    # Final 9-row / 3-hospital cohort: read the authoritative split_sizes /
    # clients_per_split straight out of a frozen scenario metadata file
    # (identical field names to v2's cohort_metadata.json, which is what
    # makes the two funnels comparable).
    final_split = None
    scenario_meta_path = (
        REPO_ROOT
        / "fedgmm"
        / "sp_decentralized_mnist_lr_example"
        / "data"
        / "eicu_semisynth"
        / "linear_scenario_seed101_metadata.json"
    )
    scenario_meta = read_json(scenario_meta_path)
    if isinstance(scenario_meta, dict):
        final_split = {
            "split_sizes": scenario_meta.get("split_sizes"),
            "clients_per_split": scenario_meta.get("clients_per_split"),
            "n_clients": scenario_meta.get("n_clients"),
            "eligible_client_ids": scenario_meta.get("eligible_client_ids"),
            "source": str(scenario_meta_path),
        }
        stages.append(
            {
                "stage_category": "instrument_variation_gates",
                "step": "final Study A v1 cohort (post instrument-variation gate)",
                "n_rows": sum(final_split["split_sizes"].values())
                if final_split.get("split_sizes")
                else None,
                "n_hospitals": final_split.get("n_clients"),
                "detail": (
                    f"split_sizes={final_split['split_sizes']}, "
                    f"clients_per_split={final_split['clients_per_split']}"
                ),
            }
        )

    out.update(
        {
            "stages": stages,
            "clinical_gate_collapse": {
                "from_n_rows": cohort_flow.get("flow", [{}])[0].get("n_stays"),
                "to_n_rows": clinical_final_n,
                "to_n_hospitals": clinical_final_hospitals,
            },
            "instrument_variation_gate_collapse": {
                "from_n_rows": clinical_final_n,
                "from_n_hospitals": clinical_final_hospitals,
                "to_n_rows": final_split.get("split_sizes")
                and sum(final_split["split_sizes"].values()),
                "to_n_hospitals": len(eligible_ids) if eligible_ids else None,
                "eligible_hospital_ids": eligible_ids,
            },
            "final_split": final_split,
        }
    )
    return out


# ---------------------------------------------------------------------------
# v2 status
# ---------------------------------------------------------------------------


def derive_v2_status(v2_dir: Path, protocol_json_path: Path) -> dict[str, Any]:
    v2_dir = Path(v2_dir)
    cohort_metadata = read_json(v2_dir / "cohort_metadata.json")
    setup_summary = read_json(v2_dir / "setup_validation_summary.json")

    tuning_scan = scan_manifest_completion(v2_dir / "tuning_manifest.csv")
    final_manifest_path = find_final_manifest(v2_dir)
    final_scan = scan_manifest_completion(final_manifest_path)
    final_design = load_final_matrix_design(protocol_json_path)
    analysis_ledger = check_analysis_ledger(v2_dir)

    setup_ok = cohort_metadata is not None and setup_summary is not None

    # Phase determination -- see PHASE_ORDER for the full vocabulary.
    if not setup_ok:
        phase = "not_setup"
    elif final_scan["exists"]:
        if final_scan["total_completed"] == 0:
            phase = "final_not_started"
        elif final_scan["total_completed"] < final_scan["total_planned"]:
            phase = "final_in_progress"
        elif analysis_ledger.get("exists") and not analysis_ledger.get("failed") and not analysis_ledger.get("missing"):
            phase = "analyzed"
        else:
            phase = "final_complete"
    elif tuning_scan["total_completed"] == 0:
        phase = "setup_certified"
    elif tuning_scan["total_completed"] < tuning_scan["total_planned"]:
        phase = "tuning_in_progress"
    else:
        phase = "tuning_complete"

    cohort_numbers = {}
    if isinstance(cohort_metadata, dict):
        split_sizes = cohort_metadata.get("split_sizes", {})
        clients_per_split = cohort_metadata.get("clients_per_split", {})
        cohort_numbers = {
            "n_rows": cohort_metadata.get("n_rows"),
            "n_clients": cohort_metadata.get("n_clients"),
            "off_hours_rate": cohort_metadata.get("off_hours_rate"),
            "split_sizes": split_sizes,
            "split_sizes_canonical_names": {
                split_names.normalize_split_name(k): v for k, v in split_sizes.items()
            },
            "clients_per_split": clients_per_split,
            "middle_split_alias_note": (
                "cohort_metadata.json and prepare_eicu_study_a_v2_cohort.py "
                "key the middle split 'dev'; some write-ups (protocol_v2.md, "
                "RUNBOOK.md) call the identical split 'Validation'. Canonical "
                "name is 'dev' (see scripts/validate_eicu_study_a_campaign.py"
                ":normalize_split_name)."
            ),
        }

    heterogeneity = compute_client_heterogeneity(
        v2_dir / "cohort.csv", v2_dir / "client_eligibility_audit.csv"
    )

    return {
        "study": "Study A v2 (eicu_study_a_v2_offhours)",
        "phase": phase,
        "setup_certified": setup_ok,
        "cohort_numbers": cohort_numbers,
        "client_heterogeneity": heterogeneity,
        "tuning": {
            "manifest_path": tuning_scan["manifest_path"],
            "planned": tuning_scan["total_planned"],
            "completed": tuning_scan["total_completed"],
            "diverged": tuning_scan["total_diverged"],
            "by_role": tuning_scan["by_role"],
        },
        "final": {
            "manifest_materialized": final_scan["exists"],
            "manifest_path": final_scan["manifest_path"],
            "design_by_role": final_design["by_role"],
            "design_total": final_design["total"],
            "planned": final_scan["total_planned"] or final_design["total"],
            "completed": final_scan["total_completed"],
            "diverged": final_scan["total_diverged"],
            "by_role": final_scan["by_role"],
        },
        "analysis_ledger": analysis_ledger,
        "funnel": build_v2_funnel(cohort_metadata),
    }


# ---------------------------------------------------------------------------
# v1 status
# ---------------------------------------------------------------------------


def derive_v1_status(v1_cohort_dir: Path, v1_campaign_dir: Path) -> dict[str, Any]:
    v1_cohort_dir = Path(v1_cohort_dir)
    v1_campaign_dir = Path(v1_campaign_dir)

    completion_record = read_json(v1_campaign_dir / "completion_record.json")
    manifest_scan = scan_manifest_completion(v1_campaign_dir / "manifest.csv")

    setup_ok = completion_record is not None or manifest_scan["exists"]
    if not setup_ok:
        phase = "not_setup"
    elif manifest_scan["total_completed"] == 0:
        phase = "final_not_started"
    elif manifest_scan["total_completed"] < manifest_scan["total_planned"]:
        phase = "final_in_progress"
    else:
        phase = "final_complete"

    reconciled_matches_claim = None
    if isinstance(completion_record, dict):
        claimed_total = completion_record.get("execution", {}).get("total_completed")
        reconciled_matches_claim = claimed_total == manifest_scan["total_completed"]

    return {
        "study": "Study A v1 (archived pipeline validation)",
        "phase": phase,
        "completion_record_path": str(v1_campaign_dir / "completion_record.json"),
        "completion_record_claim": (
            completion_record.get("execution") if isinstance(completion_record, dict) else None
        ),
        "independent_results_scan": {
            "manifest_path": manifest_scan["manifest_path"],
            "planned": manifest_scan["total_planned"],
            "completed": manifest_scan["total_completed"],
            "diverged": manifest_scan["total_diverged"],
            "by_role": manifest_scan["by_role"],
        },
        "claim_matches_independent_scan": reconciled_matches_claim,
        "tuning": {
            "planned": 0,
            "completed": 0,
            "note": "tuning deliberately skipped for v1; preregistered fixed defaults were used",
        },
        "funnel": build_v1_funnel(v1_cohort_dir, v1_campaign_dir),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def render_markdown(status: dict[str, Any]) -> str:
    lines = []
    lines.append("# eICU Study A status")
    lines.append("")
    lines.append(f"Generated: {status['generated_at_utc']} (derived from artifacts, not hand-written)")
    lines.append("")
    lines.append("## Campaign phase and run counts")
    lines.append("")
    lines.append(
        "| study | phase | tuning (done/planned) | final (done/planned) | cohort rows | clients |"
    )
    lines.append("|---|---|---|---|---|---|")
    for key in ("v1", "v2"):
        s = status["studies"][key]
        if key == "v1":
            tuning = "n/a (skipped by design)"
            final = f"{s['independent_results_scan']['completed']}/{s['independent_results_scan']['planned']}"
            rows = s["funnel"].get("clinical_gate_collapse", {}).get("to_n_rows") if s["funnel"].get("available") else None
            iv = s["funnel"].get("instrument_variation_gate_collapse", {}) if s["funnel"].get("available") else {}
            final_rows = iv.get("to_n_rows")
            clients = iv.get("to_n_hospitals")
            rows_display = f"{final_rows} (after 2 collapses; {rows} after clinical gates only)"
        else:
            tuning = f"{s['tuning']['completed']}/{s['tuning']['planned']}"
            final_planned = s["final"]["planned"]
            final = f"{s['final']['completed']}/{_fmt(final_planned)}" + (
                "" if s["final"]["manifest_materialized"] else " (not materialized)"
            )
            rows_display = _fmt(s["cohort_numbers"].get("n_rows"))
            clients = s["cohort_numbers"].get("n_clients")
        lines.append(
            f"| {s['study']} | **{s['phase']}** | {tuning} | {final} | {rows_display} | {_fmt(clients)} |"
        )
    lines.append("")

    v2 = status["studies"]["v2"]
    lines.append("## Study A v2 cohort numbers (read from cohort_metadata.json)")
    lines.append("")
    cn = v2["cohort_numbers"]
    lines.append(f"- admissions: **{_fmt(cn.get('n_rows'))}**")
    lines.append(f"- hospital clients: **{_fmt(cn.get('n_clients'))}**")
    split = cn.get("split_sizes_canonical_names", {})
    lines.append(
        f"- split (train/dev/test): **{_fmt(split.get('train'))}/{_fmt(split.get('dev'))}/{_fmt(split.get('test'))}**"
        " (\"dev\" is canonical; some write-ups say \"Validation\" for the identical split)"
    )
    off_hours = cn.get("off_hours_rate")
    lines.append(f"- off-hours rate: **{off_hours:.4f}**" if isinstance(off_hours, (int, float)) else "- off-hours rate: -")
    lines.append("")

    het = v2.get("client_heterogeneity", {})
    if het.get("available"):
        rc = het.get("client_row_counts", {})
        ef = het.get("eligibility_funnel", {})
        lines.append("## Study A v2 client-size heterogeneity")
        lines.append("")
        lines.append(
            f"- rows per client: min **{_fmt(rc.get('min'))}**, median **{_fmt(rc.get('median'))}**, "
            f"max **{_fmt(rc.get('max'))}**; **{_fmt(rc.get('n_clients_under_10_rows'))}** clients have fewer than 10 rows"
        )
        lines.append(
            f"- candidate hospitals **{_fmt(ef.get('n_candidate_hospitals'))}** -> eligible "
            f"**{_fmt(ef.get('n_eligible_hospitals'))}** ({_fmt(ef.get('n_excluded_hospitals'))} dropped)"
        )
        for reason, count in ef.get("exclusion_reason_occurrences", {}).items():
            lines.append(f"  - `{reason}`: {count}")
        lines.append("")

    lines.append("## Cohort funnels (clinical gates vs instrument-variation gates)")
    lines.append("")
    v1 = status["studies"]["v1"]
    v1f = v1.get("funnel", {})
    if v1f.get("available"):
        cg = v1f["clinical_gate_collapse"]
        iv = v1f["instrument_variation_gate_collapse"]
        lines.append(
            f"- v1 clinical gates: {_fmt(cg.get('from_n_rows'))} -> **{_fmt(cg.get('to_n_rows'))}** rows "
            f"({_fmt(cg.get('to_n_hospitals'))} hospitals)"
        )
        lines.append(
            f"- v1 instrument-variation gates: {_fmt(iv.get('from_n_rows'))} rows / "
            f"{_fmt(iv.get('from_n_hospitals'))} hospitals -> **{_fmt(iv.get('to_n_rows'))} rows / "
            f"{_fmt(iv.get('to_n_hospitals'))} hospitals** ({_fmt(iv.get('eligible_hospital_ids'))})"
        )
    v2f = v2.get("funnel", {})
    if v2f.get("available"):
        lines.append(
            f"- v2: {v2f['stages'][0]['n_rows']} -> **{_fmt(v2f.get('final_n_rows'))} rows / "
            f"{_fmt(v2f.get('final_n_clients'))} clients**. {v2f.get('protocol_note')}"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `setup_validation_summary.json`'s `full_test_suite` counts are a frozen snapshot from "
        "campaign setup time, not the live suite size; do not compare them to a fresh test run."
    )
    lines.append(
        "- This report is a live read of the tree. If a tuning or final campaign is running "
        "concurrently, re-running this script will show a different, more complete count."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_status(
    v2_dir: Path,
    protocol_json_path: Path,
    v1_cohort_dir: Path,
    v1_campaign_dir: Path,
) -> dict[str, Any]:
    return {
        "generated_at_utc": _iso_now(),
        "phase_vocabulary": list(PHASE_ORDER),
        "studies": {
            "v1": derive_v1_status(v1_cohort_dir, v1_campaign_dir),
            "v2": derive_v2_status(v2_dir, protocol_json_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_V2_PROTOCOL_JSON)
    parser.add_argument("--v1-cohort-dir", type=Path, default=DEFAULT_V1_COHORT_DIR)
    parser.add_argument("--v1-campaign-dir", type=Path, default=DEFAULT_V1_CAMPAIGN_DIR)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Defaults to <v2-dir>/status.json",
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=None,
        help="Defaults to <v2-dir>/status.md",
    )
    parser.add_argument(
        "--out-v1-funnel-json",
        type=Path,
        default=None,
        help="Defaults to <v1-cohort-dir>/cohort_flow_full.json",
    )
    parser.add_argument(
        "--out-client-heterogeneity-json",
        type=Path,
        default=None,
        help="Defaults to <v2-dir>/client_heterogeneity_summary.json",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the markdown report only; do not write any derived artifact.",
    )
    args = parser.parse_args(argv)

    status = build_status(args.v2_dir, args.protocol_json, args.v1_cohort_dir, args.v1_campaign_dir)
    markdown = render_markdown(status)

    if not args.no_write:
        out_json = args.out_json or (args.v2_dir / "status.json")
        out_markdown = args.out_markdown or (args.v2_dir / "status.md")
        out_v1_funnel = args.out_v1_funnel_json or (args.v1_cohort_dir / "cohort_flow_full.json")
        out_heterogeneity = args.out_client_heterogeneity_json or (
            args.v2_dir / "client_heterogeneity_summary.json"
        )

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
            handle.write("\n")

        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text(markdown, encoding="utf-8")

        out_v1_funnel.parent.mkdir(parents=True, exist_ok=True)
        with out_v1_funnel.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at_utc": status["generated_at_utc"],
                    "description": (
                        "v1's complete funnel through the clinical-gate collapse "
                        "(2,111 -> 201) AND the instrument-variation-gate collapse "
                        "(201 -> 9 rows / 3 hospitals). Derived from cohort_flow.json, "
                        "construction_decision.json, freeze_record.json, and the frozen "
                        "v1 scenario metadata. Does not edit cohort_flow.json."
                    ),
                    **status["studies"]["v1"]["funnel"],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        out_heterogeneity.parent.mkdir(parents=True, exist_ok=True)
        with out_heterogeneity.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at_utc": status["generated_at_utc"],
                    **status["studies"]["v2"]["client_heterogeneity"],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
