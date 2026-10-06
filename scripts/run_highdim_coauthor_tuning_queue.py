#!/usr/bin/env python3
"""Run one deterministic or stochastic high-dimensional tuning queue.

This wrapper is intended to be submitted as a one-GPU ``gpurun`` job.  It
processes alpha 0.5 first (reusing completed artifacts), then alpha 0.1 and
alpha 1.0.  Keeping deterministic and stochastic queues separate prevents two
CPU-heavy all-client runs from competing with one another.
"""

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


def read_output_root(manifest: Path) -> str:
    with manifest.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"empty manifest: {manifest}")
    return row["output_root"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime",
        required=True,
        choices=("deterministic", "stochastic"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage_results = []
    for alpha_dir_name in ALPHA_DIRS:
        alpha_dir = PROTOCOL_DIR / alpha_dir_name
        manifest = alpha_dir / f"tuning_manifest_{args.regime}.csv"
        output_root = read_output_root(manifest)
        results_json = alpha_dir / f"tuning_{args.regime}_launcher_results.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_manifest.py"),
            "--manifest",
            str(manifest),
            "--config-dir",
            str(alpha_dir / "generated_configs"),
            "--output-root",
            output_root,
            "--gpu-ids",
            "0",
            "--max-parallel",
            "1",
            "--resume-skip-completed",
            "--keep-going",
            "--results-json",
            str(results_json),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"QUEUE {alpha_dir_name} {args.regime}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        stage_results.append(
            {
                "alpha_dir": alpha_dir_name,
                "regime": args.regime,
                "returncode": completed.returncode,
                "manifest": str(manifest.relative_to(REPO_ROOT)),
                "output_root": output_root,
            }
        )
        if completed.returncode != 0 and not args.dry_run:
            break

    summary_path = PROTOCOL_DIR / f"tuning_{args.regime}_queue_summary.json"
    with summary_path.open("w") as handle:
        json.dump(stage_results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if all(item["returncode"] == 0 for item in stage_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
