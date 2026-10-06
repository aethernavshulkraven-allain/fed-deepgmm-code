#!/usr/bin/env python3
"""Quota-aware, multi-week orchestrator for the deterministic finals campaign.

The full finals manifest (108 runs, 81.0 GPU-h) exceeds even a single fresh
weekly quota (48 GPU-h), so it cannot be launched as one job -- gpurun does
not reject an over-budget job at submission, it starts it and the broker can
preempt mid-run once the budget is exhausted, which wastes whatever was
in-flight. This script instead launches in whole (dataset, alpha) cells
(6 runs each: both methods x 3 seeds), cheapest-first, submitting a wave via
gpurun only when it currently fits inside the remaining weekly quota with a
safety margin. When nothing fits, it sleeps and re-polls quota rather than
submitting anyway.

Meant to run as a long-lived background process (days to weeks). Safe to
kill and restart at any point: progress is tracked by real metrics.json
files on disk (via --resume-skip-completed), not by any internal state.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
FULL_MANIFEST = CAMPAIGN_DIR / "finals_manifest.csv"
CONFIG_DIR = CAMPAIGN_DIR / "generated_configs"
RESULT_ROOT = REPO_ROOT / "results/highdim_deterministic_finals_20260813"
ORCH_LOG = CAMPAIGN_DIR / "orchestrator_log.jsonl"
PYTHON_BIN = "python"

# (setup_seconds, seconds_per_round) -- same measured/interpolated table used
# throughout this campaign's cost estimates.
SCENARIO_COST = {
    "femnist_z":  (53.7, 2.478),
    "femnist_x":  (54.4, 2.453),
    "femnist_xz": (54.4, 5.723),
    "cifar10_z":  (164.9, 5.723),
    "cifar10_x":  (164.9, 5.723),
    "cifar10_xz": (164.9, 8.992),
}
COMM_ROUND = 500
SAFETY_MARGIN = 1.15  # require 15% headroom over the estimate before committing
POLL_INTERVAL_S = 6 * 3600  # re-check quota every 6h while blocked on budget


def cell_cost_hours(dataset: str, n_runs: int) -> float:
    setup, spr = SCENARIO_COST[dataset]
    return (setup + COMM_ROUND * spr) * n_runs / 3600


def log(event: dict) -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    ORCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ORCH_LOG.open("a") as handle:
        handle.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def load_rows() -> list[dict[str, str]]:
    with FULL_MANIFEST.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cell_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["dataset"], row["alpha"])


def is_row_complete(row: dict[str, str]) -> bool:
    run_dir = RESULT_ROOT / row["dataset"] / row["method"] / f"seed_{row['seed']}" / row["run_id"]
    return (run_dir / "metrics.json").exists()


def remaining_quota_hours() -> float:
    out = subprocess.run(["gpurun", "--status"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"remaining:\s*([\d.]+)\s*GPU-h", out)
    if not match:
        raise RuntimeError(f"could not parse gpurun --status output: {out!r}")
    return float(match.group(1))


def write_wave_manifest(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_wave(rows: list[dict[str, str]], wave_name: str, cells: list[tuple[str, str]]) -> int:
    wave_manifest = CAMPAIGN_DIR / f"wave_{wave_name}_manifest.csv"
    write_wave_manifest(rows, wave_manifest)
    results_json = CAMPAIGN_DIR / f"wave_{wave_name}_results.json"
    cmd = [
        "gpurun", "-g", "2",
        PYTHON_BIN, str(REPO_ROOT / "scripts/run_manifest.py"),
        "--manifest", str(wave_manifest),
        "--config-dir", str(CONFIG_DIR),
        "--output-root", str(RESULT_ROOT),
        "--gpu-ids", "0,1",
        "--max-parallel", "2",
        "--resume-skip-completed",
        "--keep-going",
        "--results-json", str(results_json),
    ]
    log({"event": "wave_start", "wave": wave_name, "runs": len(rows), "cells": cells})
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    log({"event": "wave_end", "wave": wave_name, "returncode": proc.returncode})
    return proc.returncode


def main() -> int:
    # RETIRED 2026-08-22: pre-BatchNorm-fix legacy entry point. The server
    # update this campaign ran under corrupted BatchNorm buffers
    # (running_var/num_batches_tracked) every round -- see
    # experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3/CORRECTION_ADDENDUM_20260822.md
    # and experiments/highdim_coauthor_protocol_v1/legacy_batchnorm_trajectories_20260822.json.
    # Not resumable as scientific evidence.
    print(
        "REFUSING TO RUN: this pre-BatchNorm-fix legacy entry point is retired.\n"
        "See experiments/highdim_coauthor_protocol_v1/legacy_batchnorm_trajectories_20260822.json\n"
        "Use: scripts/launch_highdim_deterministic_screen_post_bn_20260822.sh",
        file=sys.stderr,
    )
    return 1

    rows = load_rows()
    cells: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        cells.setdefault(cell_key(row), []).append(row)

    ordering = sorted(cells.keys(), key=lambda key: cell_cost_hours(key[0], len(cells[key])))

    log({
        "event": "orchestrator_start",
        "total_cells": len(ordering),
        "total_runs": len(rows),
        "order": [f"{ds}/alpha={a}" for ds, a in ordering],
    })

    pending = [key for key in ordering if not all(is_row_complete(r) for r in cells[key])]
    wave_idx = 0

    while pending:
        remaining = remaining_quota_hours()
        batch: list[tuple[str, str]] = []
        acc = 0.0
        for key in pending:
            cost = cell_cost_hours(key[0], len(cells[key]))
            if acc + cost * SAFETY_MARGIN <= remaining:
                batch.append(key)
                acc += cost

        if not batch:
            cheapest = min(cell_cost_hours(k[0], len(cells[k])) for k in pending)
            log({
                "event": "waiting_for_quota",
                "remaining_h": remaining,
                "cheapest_pending_cell_h": cheapest,
                "pending_cells": len(pending),
                "sleep_s": POLL_INTERVAL_S,
            })
            time.sleep(POLL_INTERVAL_S)
            continue

        wave_idx += 1
        wave_rows = [r for key in batch for r in cells[key]]
        run_wave(wave_rows, f"{wave_idx:02d}", [f"{ds}/alpha={a}" for ds, a in batch])
        pending = [key for key in ordering if not all(is_row_complete(r) for r in cells[key])]

    log({"event": "orchestrator_done", "total_cells": len(ordering), "total_runs": len(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
