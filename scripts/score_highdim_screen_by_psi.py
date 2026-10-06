#!/usr/bin/env python3
"""Re-score the deterministic screen by the ADOPTED plan's actual primary
metric -- validation approx_psi_eval ("gmm_eval") -- not validation MSE,
which is what the original single-winner selection used.

Per the decided plan (doe_review_and_revised_grid.md Part VI): "Selection:
validation Psi surrogate primary, validation MSE secondary." This reprocesses
existing metrics.json files only -- no new GPU runs.

For each (dataset, method) cell:
  1. Gather every candidate from the given manifest(s) that has a
     structurally valid result (see `structurally_valid` below).
  2. Rank by best_gmm_eval, descending (higher = better -- confirmed from
     fedavg_api.py's own `max(eval_history)` / `curr_eval > max_recent_eval`
     selection logic and approx_psi_eval's `if psi > max_eval` progress
     check).
  3. Report the top 2 (the "Rank" stage shape from the adopted plan), with
     best_validation_mse alongside as a secondary diagnostic/tiebreaker
     (used only if two candidates' gmm_eval values are equal within 1e-9).
  4. Never reads any test_mse field.
  5. Flags whether the rank-1 candidate sits at the tested grid's edge.

Usage:
  python scripts/score_highdim_screen_by_psi.py \\
      --manifest experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813/screen_manifest.csv \\
      --manifest experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813/screen_expand_manifest.csv \\
      --results-root results/highdim_deterministic_screen_20260813 \\
      --out experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813/psi_rescore.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_smoke_run import (  # noqa: E402
    ValidationError,
    _assert_checkpoints,
    _assert_finite_number,
    _assert_json_numbers_finite,
    _load_csv_rows,
    _load_json,
)


def structurally_valid(run_dir: Path) -> tuple[bool, str | None]:
    """Same core integrity checks as validate_smoke_run.validate_run
    (finite metrics, not diverged, CSV row count matches comm_round, CSV
    row-level finite/diverged flags, checkpoints carry real g/f state) --
    MINUS validate_run's predictions.npz cross-shape check, which
    false-positives on this campaign's image-x scenarios (x is
    (N,1,28,28), not scalar-shaped like the low-dim campaigns validate_run
    was built for). Checked directly against real screen output below
    before trusting this as a substitute.
    """
    required = [
        "effective_config.json", "metrics.json", "mse_by_round.csv",
        "predictions.npz", "checkpoints/best_validation.pt", "checkpoints/final.pt",
    ]
    missing = [f for f in required if not (run_dir / f).exists()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        config = _load_json(run_dir / "effective_config.json")
        metrics = _load_json(run_dir / "metrics.json")
        rows = _load_csv_rows(run_dir / "mse_by_round.csv")
        _assert_json_numbers_finite(metrics, "metrics")
        if bool(metrics.get("diverged", False)):
            return False, "diverged"
        comm_round = int(config["comm_round"])
        if len(rows) != comm_round:
            return False, f"csv row count {len(rows)} != comm_round {comm_round}"
        for row in rows:
            for field in ("val_mse", "gmm_eval"):
                _assert_finite_number(row[field], f"csv.{field}")
            if str(row.get("finite")) != "True" or str(row.get("diverged")) != "False":
                return False, "csv row flags indicate non-finite/diverged"
        _assert_checkpoints(str(run_dir))
    except ValidationError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def load_candidates(manifest_paths: list[Path], results_root: Path) -> dict:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    invalid_log = []
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        for row in manifest_rows:
            run_dir = results_root / row["dataset"] / row["method"] / f"seed_{row['seed']}" / row["run_id"]
            ok, reason = structurally_valid(run_dir)
            if not ok:
                invalid_log.append({"run_id": row["run_id"], "reason": reason})
                continue
            metrics = _load_json(run_dir / "metrics.json")
            cells[(row["dataset"], row["method"])].append({
                "run_id": row["run_id"],
                "lr": float(row["learning_rate"]),
                "cm": float(row["critic_multiplier"]),
                "gmm_eval": float(metrics["best_gmm_eval"]),
                "val_mse": float(metrics["best_validation_mse"]),
            })
    return cells, invalid_log


def rank_cell(candidates: list[dict]) -> list[dict]:
    # Primary: gmm_eval descending. Secondary/tiebreak: val_mse ascending,
    # only exercised if gmm_eval values are equal within float noise.
    return sorted(candidates, key=lambda c: (-round(c["gmm_eval"], 9), c["val_mse"]))


def boundary_flags(winner: dict, candidates: list[dict]) -> list[str]:
    all_lrs = sorted({c["lr"] for c in candidates})
    all_cms = sorted({c["cm"] for c in candidates})
    flags = []
    if winner["lr"] == max(all_lrs):
        flags.append(f"lr={winner['lr']:g} at tested max {all_lrs}")
    if winner["cm"] == max(all_cms):
        flags.append(f"cm={winner['cm']:g} at tested max {all_cms}")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_paths = [Path(p) for p in args.manifest]
    results_root = Path(args.results_root)
    cells, invalid_log = load_candidates(manifest_paths, results_root)

    print(f"{len(invalid_log)} candidates excluded as structurally invalid "
          f"(diverged / missing or inconsistent artifacts):")
    for entry in invalid_log:
        print(f"  {entry['run_id']}: {entry['reason']}")
    print()

    output: dict[str, dict] = {}
    print(f"{'scenario':12s} {'method':10s} {'rank':>4s} {'lr':>8s} {'cm':>6s} "
          f"{'gmm_eval':>12s} {'val_mse':>10s}  boundary?")
    for (ds, method), candidates in sorted(cells.items()):
        if not candidates:
            print(f"{ds:12s} {method:10s}  NO VALID CANDIDATES")
            continue
        ranked = rank_cell(candidates)
        top2 = ranked[:2]
        winner = top2[0]
        flags = boundary_flags(winner, candidates)
        for rank, cand in enumerate(top2, start=1):
            flag_str = ("; ".join(flags) if flags else "-") if rank == 1 else ""
            print(f"{ds:12s} {method:10s} {rank:>4d} {cand['lr']:>8g} {cand['cm']:>6g} "
                  f"{cand['gmm_eval']:>12.6f} {cand['val_mse']:>10.6f}  {flag_str}")
        output[f"{ds}|{method}"] = {
            "dataset": ds, "method": method,
            "rank_1": winner,
            "rank_2": top2[1] if len(top2) > 1 else None,
            "n_valid_candidates": len(candidates),
            "at_boundary": bool(flags),
            "boundary_detail": flags,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
