#!/usr/bin/env python3
"""Corrected full-pool re-score (2026-08-19), superseding
psi_rescore.json (2026-08-18).

Builds one candidate audit ledger across every source that has ever
touched the deterministic screen boundary decision:

  - original screen (screen_manifest.csv, 72 rows)
  - expansion-1 (screen_expand_manifest.csv, 19 rows)
  - expansion-2, fedgda_d rows only (screen_expand2_manifest.csv, 7 of 17
    rows -- the other 10 were the mislabeled fedogda_d rows, quarantined,
    included in the ledger for the audit trail but never eligible)
  - expansion-2, corrected fedogda_d rows (screen_expand2_corrected_v1_manifest.csv,
    10 rows, replacing the quarantined ones)

Every candidate is recorded in the ledger with an explicit status and
reason -- eligible candidates feed the ranking; everything else is kept
for the audit trail with why it was excluded. Ranking logic (gmm_eval
descending, val_mse tiebreak within 1e-9, boundary flags) is unchanged
from score_highdim_screen_by_psi.py.

No test_mse field is ever read.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_smoke_run import _load_json  # noqa: E402
from score_highdim_screen_by_psi import structurally_valid, rank_cell, boundary_flags  # noqa: E402

CAMPAIGN_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
RESULTS_ROOT = REPO_ROOT / "results/highdim_deterministic_screen_20260813"
QUARANTINE_RESULTS_ROOT = CAMPAIGN_DIR / "QUARANTINE_20260819_mislabeled_fedogda_expand2/results"

SOURCES = [
    ("original_screen", CAMPAIGN_DIR / "screen_manifest.csv", RESULTS_ROOT, None),
    ("expansion_1", CAMPAIGN_DIR / "screen_expand_manifest.csv", RESULTS_ROOT, None),
    ("expansion_2_fedgda_d", CAMPAIGN_DIR / "screen_expand2_manifest.csv", RESULTS_ROOT, "fedgda_d"),
    ("expansion_2_mislabeled_fedogda_d_QUARANTINED", CAMPAIGN_DIR / "screen_expand2_manifest.csv", QUARANTINE_RESULTS_ROOT, "fedogda_d"),
    ("expansion_2_corrected_fedogda_d", CAMPAIGN_DIR / "screen_expand2_corrected_v1_manifest.csv", RESULTS_ROOT, None),
]


def load_metrics_if_present(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:  # noqa: BLE001
        return None


def build_ledger() -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    ledger: list[dict] = []
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for source_name, manifest_path, results_root, method_filter in SOURCES:
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if method_filter is not None and row["method"] != method_filter:
                continue
            run_dir = results_root / row["dataset"] / row["method"] / f"seed_{row['seed']}" / row["run_id"]
            entry = {
                "run_id": row["run_id"],
                "source": source_name,
                "dataset": row["dataset"],
                "method": row["method"],
                "lr": float(row["learning_rate"]),
                "cm": float(row["critic_multiplier"]),
                "run_dir": str(run_dir),
            }

            if source_name == "expansion_2_mislabeled_fedogda_d_QUARANTINED":
                metrics = load_metrics_if_present(run_dir)
                entry.update({
                    "status": "excluded",
                    "exclusion_category": "mislabeled_optimizer_bug",
                    "exclusion_reason": (
                        "reference_row() template-lookup bug: client_optimizer='sgd' "
                        "instead of 'ogda' for method=fedogda_d. Trained and completed "
                        "as FedGDA-D under a fedogda_d run_id. Quarantined, never "
                        "eligible for ranking. See INVALIDATION_NOTE_20260819.md."
                    ),
                    "gmm_eval": metrics.get("best_gmm_eval") if metrics else None,
                    "val_mse": metrics.get("best_validation_mse") if metrics else None,
                    "actually_ran_as": "fedgda_d (sgd)",
                })
                ledger.append(entry)
                continue

            ok, reason = structurally_valid(run_dir)
            if not ok:
                category = "structurally_invalid"
                if source_name == "expansion_2_fedgda_d":
                    category = "fedgda_d_pretraining_failure"
                elif source_name == "expansion_2_corrected_fedogda_d":
                    category = "corrected_fedogda_d_nonfinite_or_diverged"
                elif source_name in ("original_screen", "expansion_1"):
                    category = "original_screen_or_expansion1_exclusion"
                entry.update({
                    "status": "excluded",
                    "exclusion_category": category,
                    "exclusion_reason": reason,
                    "gmm_eval": None,
                    "val_mse": None,
                })
                ledger.append(entry)
                continue

            metrics = _load_json(run_dir / "metrics.json")
            entry.update({
                "status": "eligible",
                "exclusion_category": None,
                "exclusion_reason": None,
                "gmm_eval": float(metrics["best_gmm_eval"]),
                "val_mse": float(metrics["best_validation_mse"]),
            })
            ledger.append(entry)
            cells[(row["dataset"], row["method"])].append({
                "run_id": row["run_id"],
                "lr": entry["lr"],
                "cm": entry["cm"],
                "gmm_eval": entry["gmm_eval"],
                "val_mse": entry["val_mse"],
                "source": source_name,
            })

    return ledger, cells


def main() -> int:
    ledger, cells = build_ledger()

    output: dict[str, dict] = {}
    for (ds, method), candidates in sorted(cells.items()):
        if not candidates:
            output[f"{ds}|{method}"] = {
                "dataset": ds, "method": method, "rank_1": None, "rank_2": None,
                "n_valid_candidates": 0, "at_boundary": False, "boundary_detail": [],
            }
            continue
        ranked = rank_cell(candidates)
        top2 = ranked[:2]
        winner = top2[0]
        flags = boundary_flags(winner, candidates)
        output[f"{ds}|{method}"] = {
            "dataset": ds, "method": method,
            "rank_1": winner,
            "rank_2": top2[1] if len(top2) > 1 else None,
            "n_valid_candidates": len(candidates),
            "at_boundary": bool(flags),
            "boundary_detail": flags,
        }

    psi_out = CAMPAIGN_DIR / "psi_rescore_corrected_v2.json"
    with psi_out.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")

    ledger_out = CAMPAIGN_DIR / "candidate_audit_ledger_20260819.json"
    with ledger_out.open("w") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        handle.write("\n")

    ledger_csv_out = CAMPAIGN_DIR / "candidate_audit_ledger_20260819.csv"
    fieldnames = ["run_id", "source", "dataset", "method", "lr", "cm", "status",
                  "exclusion_category", "exclusion_reason", "gmm_eval", "val_mse",
                  "actually_ran_as", "run_dir"]
    with ledger_csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ledger:
            writer.writerow(row)

    n_eligible = sum(1 for r in ledger if r["status"] == "eligible")
    n_excluded = sum(1 for r in ledger if r["status"] == "excluded")
    by_category: dict[str, int] = defaultdict(int)
    for r in ledger:
        if r["status"] == "excluded":
            by_category[r["exclusion_category"]] += 1

    print(f"Ledger: {len(ledger)} total candidates, {n_eligible} eligible, {n_excluded} excluded")
    for cat, n in sorted(by_category.items()):
        print(f"  {cat}: {n}")
    print(f"\nWritten: {psi_out}")
    print(f"Written: {ledger_out}")
    print(f"Written: {ledger_csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
