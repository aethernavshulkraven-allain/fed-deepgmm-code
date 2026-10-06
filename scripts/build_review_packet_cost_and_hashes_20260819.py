#!/usr/bin/env python3
"""Review packet pieces 4b/5: recalculated scenario-specific cost (from
real measured 500-round runtimes in deterministic_finals_20260813, not the
flat 47.9h provisional estimate) and SHA-256 hashes of every frozen input,
for the two v2 adjudication manifests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALS_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
SCREEN_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
CAMPAIGN_V2_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def per_cell_measured_rate_h() -> dict:
    with (FINALS_DIR / "finals_manifest.csv").open(newline="") as handle:
        finals = list(csv.DictReader(handle))
    per_cell = defaultdict(list)
    for r in finals:
        d = Path(r["final_result_dir"])
        d = d if d.is_absolute() else REPO_ROOT / d
        mpath = d / "metrics.json"
        if mpath.exists():
            m = json.load(open(mpath))
            rs = m.get("runtime_seconds")
            if rs:
                per_cell[(r["dataset"], r["method"])].append(rs / 3600)
    return {k: sum(v) / len(v) for k, v in per_cell.items() if v}


def main() -> int:
    rate = per_cell_measured_rate_h()

    cost_breakdown = []
    total_h = 0.0
    for cells in ("x", "signal"):
        with (CAMPAIGN_V2_DIR / f"adjudication_{cells}_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        cell_counts = defaultdict(int)
        for r in rows:
            cell_counts[(r["dataset"], r["method"])] += 1
        for (ds, method), n in sorted(cell_counts.items()):
            r_h = rate.get((ds, method))
            if r_h is None:
                cost_breakdown.append({"manifest": cells, "dataset": ds, "method": method, "n_runs": n, "rate_h_per_run": None, "est_h": None, "note": "no measured rate available"})
                continue
            est = n * r_h
            total_h += est
            cost_breakdown.append({"manifest": cells, "dataset": ds, "method": method, "n_runs": n, "rate_h_per_run": round(r_h, 4), "est_h": round(est, 3)})

    cost_out = {
        "method": "scenario-specific measured 500-round runtime from deterministic_finals_20260813 (108 runs, mean per (dataset,method) cell), NOT the flat 47.9h provisional estimate from 2026-08-18",
        "total_estimated_gpu_hours": round(total_h, 2),
        "breakdown": cost_breakdown,
    }
    cost_path = CAMPAIGN_V2_DIR / "review_packet_4_recalculated_cost.json"
    with cost_path.open("w") as handle:
        json.dump(cost_out, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Immutable hashes of every frozen input to this review packet.
    frozen_inputs = [
        SCREEN_DIR / "BOUNDARY_RULE_AMENDMENT_20260818.md",
        SCREEN_DIR / "PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md",
        SCREEN_DIR / "INVALIDATION_NOTE_20260819.md",
        SCREEN_DIR / "psi_rescore_corrected_v2.json",
        SCREEN_DIR / "candidate_audit_ledger_20260819.json",
        SCREEN_DIR / "candidate_audit_ledger_20260819.csv",
        SCREEN_DIR / "screen_manifest.csv",
        SCREEN_DIR / "screen_expand_manifest.csv",
        SCREEN_DIR / "screen_expand2_manifest.csv",
        SCREEN_DIR / "screen_expand2_corrected_v1_manifest.csv",
        SCREEN_DIR / "screen_expand2_launcher_results.json",
        SCREEN_DIR / "screen_expand2_corrected_v1_launcher_results.json",
        FINALS_DIR / "RELABEL_20260818.md",
        FINALS_DIR / "frozen_winners.json",
        FINALS_DIR / "finals_manifest.csv",
        CAMPAIGN_V2_DIR / "adjudication_x_manifest.csv",
        CAMPAIGN_V2_DIR / "adjudication_signal_manifest.csv",
        CAMPAIGN_V2_DIR / "review_packet_provisional_vs_v2_diff.json",
        SCREEN_DIR / "review_packet_1_before_after_diff.json",
        CAMPAIGN_V2_DIR / "review_packet_3_reuse_ledger.json",
        CAMPAIGN_V2_DIR / "REVIEW_PACKET_20260819.md",
        REPO_ROOT / "scripts/run_manifest.py",
        REPO_ROOT / "scripts/prepare_highdim_deterministic_screen_expand2_corrected_v1_20260819.py",
        REPO_ROOT / "scripts/prepare_highdim_psi_adjudication_20260818.py",
        REPO_ROOT / "scripts/score_highdim_screen_corrected_v2_20260819.py",
        REPO_ROOT / "scripts/score_highdim_adjudication_20260819.py",
        REPO_ROOT / "scripts/launch_highdim_psi_adjudication_20260819_v2.sh",
        REPO_ROOT / "tests/test_adjudication_scorer_20260819.py",
        REPO_ROOT / "tests/test_adjudication_scorer_integration_20260819.py",
    ]
    hashes = {}
    for p in frozen_inputs:
        if p.exists():
            hashes[str(p.relative_to(REPO_ROOT))] = sha256_of(p)
        else:
            hashes[str(p)] = "MISSING"
    hash_path = SCREEN_DIR / "review_packet_5_immutable_hashes.json"
    with hash_path.open("w") as handle:
        json.dump(hashes, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Total recalculated cost for v2 adjudication manifests: {round(total_h, 2)} GPU-h")
    print(f"Written: {cost_path}")
    print(f"Written: {hash_path} ({len(hashes)} files hashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
