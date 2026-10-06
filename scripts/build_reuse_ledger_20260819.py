#!/usr/bin/env python3
"""Review packet piece 3: reuse ledger. For every candidate marked
reused_from_finals in the v2 adjudication summaries, verify the existing
500-round trajectory in deterministic_finals_20260813 actually matches
method, optimizer, alpha, seed, learning_rate, and critic_multiplier --
not just that (lr, cm) matched frozen_winners.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import METHOD_TO_OPTIMIZER  # noqa: E402
from validate_smoke_run import _load_json  # noqa: E402

CAMPAIGN_V2_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
FINALS_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
FINALS_RESULTS_ROOT = REPO_ROOT / "results/highdim_deterministic_finals_20260813"
SEEDS = (0, 1, 2)


def main() -> int:
    with (FINALS_DIR / "finals_manifest.csv").open(newline="") as handle:
        finals_rows = list(csv.DictReader(handle))
    # Finals covers alpha in {0.1, 0.5, 1.0}; the key MUST include alpha or
    # rows silently collide across alphas for the same (dataset, method, lr,
    # cm, seed) -- caught by this script's first run, which verified 36/36
    # "reused" trajectories against the wrong alpha (1.0, not 0.5).
    finals_index = {
        (r["dataset"], r["method"], float(r["learning_rate"]), float(r["critic_multiplier"]), int(r["seed"]), float(r["alpha"])): r
        for r in finals_rows
    }

    ledger = []
    for cells in ("x", "signal"):
        summary = json.load(open(CAMPAIGN_V2_DIR / f"adjudication_{cells}_summary.json"))
        for cell in summary["plan"]:
            ds, method = cell["dataset"], cell["method"]
            for cand in cell["candidates"]:
                entry_base = {
                    "cell": f"{ds}/{method}", "dataset": ds, "method": method,
                    "lr": cand["lr"], "cm": cand["cm"], "labels": cand["labels"],
                }
                if not cand["reused_from_finals"]:
                    ledger.append({**entry_base, "kind": "new_run", "verified": None, "detail": "generated fresh in adjudication manifest"})
                    continue
                for seed in SEEDS:
                    key = (ds, method, cand["lr"], cand["cm"], seed, 0.5)
                    row = finals_index.get(key)
                    entry = {**entry_base, "seed": seed, "kind": "reused_from_finals"}
                    if row is None:
                        entry.update({"verified": False, "detail": f"NO MATCHING FINALS ROW for {key}"})
                        ledger.append(entry)
                        continue
                    checks = {
                        "method_matches": row["method"] == method,
                        "optimizer_matches": row["client_optimizer"] == METHOD_TO_OPTIMIZER[method],
                        "alpha_matches": abs(float(row["alpha"]) - 0.5) < 1e-9,
                        "seed_matches": int(row["seed"]) == seed,
                        "lr_matches": abs(float(row["learning_rate"]) - cand["lr"]) < 1e-9,
                        "cm_matches": abs(float(row["critic_multiplier"]) - cand["cm"]) < 1e-9,
                        "comm_round_is_500": int(row["comm_round"]) == 500,
                    }
                    run_dir = Path(row["final_result_dir"]) if Path(row["final_result_dir"]).is_absolute() else REPO_ROOT / row["final_result_dir"]
                    artifacts_exist = (run_dir / "metrics.json").exists()
                    checks["artifacts_exist"] = artifacts_exist
                    if artifacts_exist:
                        metrics = _load_json(run_dir / "metrics.json")
                        checks["not_diverged"] = not bool(metrics.get("diverged", False))
                    all_ok = all(checks.values())
                    entry.update({
                        "verified": all_ok, "run_id": row["run_id"], "run_dir": str(run_dir),
                        "checks": checks,
                    })
                    ledger.append(entry)

    out = CAMPAIGN_V2_DIR / "review_packet_3_reuse_ledger.json"
    with out.open("w") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        handle.write("\n")

    reused = [e for e in ledger if e["kind"] == "reused_from_finals"]
    bad = [e for e in reused if not e["verified"]]
    new_runs = [e for e in ledger if e["kind"] == "new_run"]
    print(f"reuse ledger: {len(reused)} reused-trajectory checks ({len(reused)//3} candidates x 3 seeds), {len(bad)} FAILED verification")
    for e in bad:
        print("  FAIL:", e["cell"], e.get("run_id", "?"), e.get("detail", e.get("checks")))
    print(f"new-run candidates (not reused): {len(new_runs)}")
    print(f"\nWritten: {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
