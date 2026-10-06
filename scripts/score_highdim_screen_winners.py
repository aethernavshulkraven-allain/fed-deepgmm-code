#!/usr/bin/env python3
"""Score a deterministic tuning screen: pick the validation-MSE winner per
(dataset, method) cell, and flag any winner sitting at the edge of the
tested learning_rate/critic_multiplier grid.

Usage:
  python scripts/score_highdim_screen_winners.py \\
      --manifest experiments/highdim_coauthor_protocol_v1/fedeg_screen/screen_manifest.csv \\
      --results-root results/fedeg_screen \\
      --out experiments/highdim_coauthor_protocol_v1/fedeg_screen/winners.json

If you also ran a boundary-expansion re-screen (extra candidates for cells
that hit the edge), pass its manifest too with a second --manifest -- the
script unions all given manifests before picking winners, so a cell's
winner is chosen across every candidate tried for it so far.

A winner is "at the grid boundary" for an axis if its value equals the max
tested for that (dataset, method) cell. This isn't automatically wrong --
check whether the trend into that edge is still improving meaningfully
(what mattered in the original FedGDA-D/FedOGDA-D screen) or was basically
flat. If it's a real trend, add one more rung on that axis and re-run this
script against the union of both manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_rows(manifest_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in manifest_paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def result_for_row(results_root: Path, row: dict[str, str]) -> dict | None:
    run_dir = results_root / row["dataset"] / row["method"] / f"seed_{row['seed']}" / row["run_id"]
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True, help="Repeatable; union all given manifests.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True, help="Where to write the winners JSON.")
    args = parser.parse_args()

    manifest_paths = [Path(p) for p in args.manifest]
    results_root = Path(args.results_root)

    rows = load_rows(manifest_paths)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    missing = 0
    for row in rows:
        metrics = result_for_row(results_root, row)
        candidate = {
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
            "run_id": row["run_id"],
        }
        if metrics is None:
            missing += 1
            candidate.update(diverged=None, val_mse=None)
        else:
            candidate.update(
                diverged=metrics.get("diverged"),
                val_mse=metrics.get("best_validation_mse"),
            )
        cells[(row["dataset"], row["method"])].append(candidate)

    if missing:
        print(f"WARNING: {missing} manifest rows have no metrics.json yet "
              f"(run not complete) -- scoring only against what's finished.\n")

    winners: dict[str, dict] = {}
    print(f"{'scenario':12s} {'method':16s} {'winner lr':>10s} {'winner cm':>10s} {'val_mse':>12s}  boundary?")
    for (dataset, method), candidates in sorted(cells.items()):
        valid = [c for c in candidates if c["diverged"] is False and c["val_mse"] is not None]
        if not valid:
            print(f"{dataset:12s} {method:16s}  NO SURVIVORS ({len(candidates)} candidates, all diverged/missing)")
            continue
        winner = min(valid, key=lambda c: c["val_mse"])
        all_lrs = sorted({c["lr"] for c in candidates})
        all_cms = sorted({c["cm"] for c in candidates})
        flags = []
        if winner["lr"] == max(all_lrs):
            flags.append(f"lr={winner['lr']:g} at tested max {all_lrs}")
        if winner["cm"] == max(all_cms):
            flags.append(f"cm={winner['cm']:g} at tested max {all_cms}")
        winners[f"{dataset}|{method}"] = {
            "dataset": dataset, "method": method,
            "lr": winner["lr"], "cm": winner["cm"], "val_mse": winner["val_mse"],
            "at_boundary": bool(flags),
        }
        print(f"{dataset:12s} {method:16s} {winner['lr']:>10g} {winner['cm']:>10g} "
              f"{winner['val_mse']:>12.6f}  {'; '.join(flags) if flags else '-'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(winners, handle, indent=2, sort_keys=True)
        handle.write("\n")

    n_boundary = sum(1 for w in winners.values() if w["at_boundary"])
    print(f"\n{len(winners)}/{len(cells)} cells resolved, {n_boundary} at a grid boundary.")
    if n_boundary:
        print("Review the boundary cells above before freezing for finals -- if the "
              "trend into that edge looks like it's still improving meaningfully (not "
              "flat/noisy), add one more grid rung on that axis and re-run this script "
              "with both manifests passed via --manifest.")
    print(f"Winners written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
