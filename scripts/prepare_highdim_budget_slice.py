"""Carve a GPU-budget-sized slice out of a high-dimensional final manifest.

The remaining high-dim co-author queue is far larger than a weekly GPU quota
(161 runs x ~0.53 GPU-h = ~85 GPU-h against a 48 GPU-h/week budget), so it cannot
be launched wholesale. Launching an arbitrary prefix instead produces half-finished
cells, which are not interpretable: a `(dataset, alpha)` cell is only usable once
**both** methods have all five seeds, since the whole point is the FedGDA-vs-FedOGDA
comparison at matched seeds.

This script therefore selects whole cells, cheapest-to-complete first:

1. cells already part-done, so the fewest runs finish an interpretable comparison
2. then untouched cells, in manifest order
3. never a cell that cannot be finished inside the stated budget

Runs already on disk are excluded from the cost estimate, and `run_manifest.py
--resume-skip-completed` skips them again at launch as a backstop.

Usage:
    python scripts/prepare_highdim_budget_slice.py \
        --manifest experiments/highdim_coauthor_protocol_v1/alpha0p5/final_manifest_stochastic.csv \
        --budget-hours 17 --out <path>.csv
"""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# Measured from completed runs in the matching result families
# (results/rerun_protocol_v1_real_images_abs_alpha{0p5,1}): median 0.53 h, max 0.77 h.
# Budgeting on the worst case avoids overrunning the weekly quota on a bad draw.
DEFAULT_HOURS_PER_RUN = 0.77

CELL_KEYS = ("dataset",)


def run_dir_for(row, output_root):
    return os.path.join(
        REPO_ROOT,
        output_root,
        row["dataset"],
        row["method"],
        f"seed_{int(row['seed'])}",
        row["run_id"],
    )


def is_complete(row, output_root):
    """A run counts as done when it has the metrics file the validator requires."""
    return os.path.exists(os.path.join(run_dir_for(row, output_root), "metrics.json"))


def group_cells(rows, output_root):
    cells = {}
    for row in rows:
        key = tuple(row[k] for k in CELL_KEYS)
        cell = cells.setdefault(key, {"rows": [], "done": 0})
        cell["rows"].append(row)
        if is_complete(row, output_root):
            cell["done"] += 1
    for key, cell in cells.items():
        cell["remaining"] = [r for r in cell["rows"] if not is_complete(r, output_root)]
        cell["n_remaining"] = len(cell["remaining"])
    return cells


def select_cells(cells, budget_hours, hours_per_run):
    """Cheapest-to-complete first; only whole cells that fit."""
    # Part-done cells first (fewest remaining), then untouched ones.
    order = sorted(
        cells.items(),
        key=lambda kv: (kv[1]["n_remaining"] == len(kv[1]["rows"]), kv[1]["n_remaining"]),
    )

    chosen, spent = [], 0.0
    skipped = []
    for key, cell in order:
        if cell["n_remaining"] == 0:
            continue
        cost = cell["n_remaining"] * hours_per_run
        if spent + cost <= budget_hours:
            chosen.append((key, cell))
            spent += cost
        else:
            skipped.append((key, cell, cost))
    return chosen, spent, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-hours", type=float, required=True)
    parser.add_argument(
        "--hours-per-run", type=float, default=DEFAULT_HOURS_PER_RUN
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with open(args.manifest, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    output_root = rows[0]["output_root"]
    cells = group_cells(rows, output_root)
    chosen, spent, skipped = select_cells(cells, args.budget_hours, args.hours_per_run)

    selected_rows = []
    for _, cell in chosen:
        selected_rows.extend(cell["remaining"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "source_manifest": os.path.relpath(args.manifest, REPO_ROOT),
        "output_manifest": os.path.relpath(args.out, REPO_ROOT),
        "output_root": output_root,
        "budget_hours": args.budget_hours,
        "hours_per_run_assumed": args.hours_per_run,
        "estimated_hours": round(spent, 2),
        "n_runs_selected": len(selected_rows),
        "cells_selected": [
            {
                "cell": "/".join(k),
                "already_done": c["done"],
                "to_run": c["n_remaining"],
            }
            for k, c in chosen
        ],
        "cells_skipped_over_budget": [
            {"cell": "/".join(k), "to_run": c["n_remaining"], "would_cost_h": round(cost, 2)}
            for k, c, cost in skipped
        ],
    }
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
