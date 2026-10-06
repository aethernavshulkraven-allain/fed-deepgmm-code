#!/usr/bin/env python3
"""Materialize Test effect metrics for every completed Study A v2 final run.

This runs only after validation-selected checkpoints exist. It evaluates the
predeclared continuous-treatment contrast g(1,W)-g(0,W), writes per-client and
aggregate artifacts inside each run directory, and never changes checkpoint
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "analyze_eicu_study_a_checkpoint.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = Path(args.manifest).resolve()
    results_root = Path(args.results_root).resolve()
    with manifest.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("role")
            in {"confirmatory", "centralized_baseline", "aggregation_ablation"}
        ]
    if len(rows) != 105:
        raise ValueError(f"expected 105 final rows, found {len(rows)}")

    ledger = {"passed": [], "failed": [], "missing": [], "dry_run": []}
    for row in rows:
        run_dir = results_root / row["result_path"]
        checkpoint = run_dir / "checkpoints" / "best_validation.pt"
        metadata = Path(row["scenario_metadata_path"])
        if not metadata.is_absolute():
            metadata = (manifest.parent / metadata).resolve()
        scenario = metadata.with_name(
            metadata.name.replace("_metadata.json", ".npz")
        )
        if not checkpoint.is_file() or not scenario.is_file():
            print(f"MISSING {row['run_id']}")
            ledger["missing"].append(row["run_id"])
            if not args.keep_going:
                break
            continue
        command = [
            args.python,
            str(EVALUATOR),
            "--checkpoint",
            str(checkpoint),
            "--scenario",
            str(scenario),
            "--split",
            "test",
            "--out",
            str(run_dir / "effect_metrics"),
        ]
        if args.dry_run:
            print(" ".join(command))
            ledger["dry_run"].append(row["run_id"])
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT)
        key = "passed" if completed.returncode == 0 else "failed"
        ledger[key].append(row["run_id"])
        print(f"{key.upper()} {row['run_id']}")
        if key == "failed" and not args.keep_going:
            break

    summary = {
        "total_manifest_rows": len(rows),
        **{key: len(value) for key, value in ledger.items()},
        "selection_changed": False,
        "effect_contrast": "g(1,W)-g(0,W)",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.ledger:
        path = Path(args.ledger).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump({"summary": summary, "ledger": ledger}, handle, indent=2)
            handle.write("\n")
    return 1 if ledger["failed"] or ledger["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
