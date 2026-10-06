#!/usr/bin/env python3
"""Review packet piece 1: before/after candidate diff for all 12 cells, and
the three-way comparison (Psi rank-1, Psi rank-2, existing MSE winner)
required before the adjudication manifests are rebuilt.

before = psi_rescore.json (2026-08-18, screen+expansion-1 only, computed
         before the mislabeled-optimizer bug was found)
after  = psi_rescore_corrected_v2.json (2026-08-19, full corrected pool:
         screen + expansion-1 + valid expansion-2 fedgda_d + corrected
         expansion-2 fedogda_d)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
FINALS_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"


def cand_key(c):
    if c is None:
        return None
    return (c["lr"], c["cm"])


def main() -> int:
    old = json.load(open(CAMPAIGN_DIR / "psi_rescore.json"))
    new = json.load(open(CAMPAIGN_DIR / "psi_rescore_corrected_v2.json"))
    mse_winners = json.load(open(FINALS_DIR / "frozen_winners.json"))

    diff = []
    for key in sorted(old.keys()):
        ds, method = old[key]["dataset"], old[key]["method"]
        o = old[key]
        n = new[key]
        mse_w = mse_winners[f"{ds}/{method}"]
        changed_rank1 = cand_key(o["rank_1"]) != cand_key(n["rank_1"])
        changed_rank2 = cand_key(o["rank_2"]) != cand_key(n["rank_2"])
        expand2_survivor_in_top2 = False
        if n["rank_1"] and n["rank_1"]["run_id"].startswith("det_screen_expand2corr_"):
            expand2_survivor_in_top2 = True
        if n["rank_2"] and n["rank_2"]["run_id"].startswith("det_screen_expand2corr_"):
            expand2_survivor_in_top2 = True
        new_rank1_is_mse_winner = cand_key(n["rank_1"]) == (mse_w["lr"], mse_w["cm"])
        mse_winner_in_new_top2 = new_rank1_is_mse_winner or (n["rank_2"] and cand_key(n["rank_2"]) == (mse_w["lr"], mse_w["cm"]))
        diff.append({
            "cell": key,
            "dataset": ds,
            "method": method,
            "old_rank_1": o["rank_1"],
            "old_rank_2": o["rank_2"],
            "new_rank_1": n["rank_1"],
            "new_rank_2": n["rank_2"],
            "rank_1_changed_by_correction": changed_rank1,
            "rank_2_changed_by_correction": changed_rank2,
            "n_valid_candidates_old": o["n_valid_candidates"],
            "n_valid_candidates_new": n["n_valid_candidates"],
            "existing_mse_winner": {"lr": mse_w["lr"], "cm": mse_w["cm"], "run_id": mse_w.get("run_id")},
            "mse_winner_in_new_top2": mse_winner_in_new_top2,
            "corrected_expand2_candidate_in_new_top2": expand2_survivor_in_top2,
            "new_at_boundary": n["at_boundary"],
            "new_boundary_detail": n["boundary_detail"],
        })

    out = CAMPAIGN_DIR / "review_packet_1_before_after_diff.json"
    with out.open("w") as handle:
        json.dump(diff, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"{'cell':28s} {'rank1 changed?':>15s} {'rank2 changed?':>15s} {'mse_winner_in_top2':>20s} {'expand2_survivor_in_top2':>25s}")
    for d in diff:
        print(f"{d['dataset']+'/'+d['method']:28s} {str(d['rank_1_changed_by_correction']):>15s} "
              f"{str(d['rank_2_changed_by_correction']):>15s} {str(d['mse_winner_in_new_top2']):>20s} "
              f"{str(d['corrected_expand2_candidate_in_new_top2']):>25s}")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
