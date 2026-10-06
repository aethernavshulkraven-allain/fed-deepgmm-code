#!/usr/bin/env python3
"""Run one deterministic or stochastic five-seed high-dimensional queue."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1"
ALPHA_DIRS = ("alpha0p5", "alpha0p1", "alpha1")


def filtered_manifest(alpha_dir: Path, regime: str) -> Path:
    source = alpha_dir / "final_manifest.csv"
    regime_source = alpha_dir / f"final_manifest_{regime}.csv"
    if not source.exists():
        if regime_source.exists():
            return regime_source
        raise FileNotFoundError(
            f"{source} is missing; tuning must complete and be selected first"
        )
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if row["method"].endswith("_d" if regime == "deterministic" else "_s")
        ]
    target = alpha_dir / f"final_manifest_{regime}.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def output_root(manifest: Path) -> str:
    with manifest.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return row["output_root"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime", required=True, choices=("deterministic", "stochastic")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stage_results = []
    for name in ALPHA_DIRS:
        alpha_dir = PROTOCOL_DIR / name
        manifest = filtered_manifest(alpha_dir, args.regime)
        root = output_root(manifest)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_manifest.py"),
            "--manifest",
            str(manifest),
            "--config-dir",
            str(alpha_dir / "final_generated_configs"),
            "--output-root",
            root,
            "--gpu-ids",
            "0",
            "--max-parallel",
            "1",
            "--resume-skip-completed",
            "--keep-going",
            "--results-json",
            str(alpha_dir / f"final_{args.regime}_launcher_results.json"),
        ]
        if args.dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        stage_results.append({"alpha_dir": name, "returncode": completed.returncode})
        if completed.returncode != 0 and not args.dry_run:
            break
    with (PROTOCOL_DIR / f"final_{args.regime}_queue_summary.json").open("w") as handle:
        json.dump(stage_results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if all(item["returncode"] == 0 for item in stage_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
