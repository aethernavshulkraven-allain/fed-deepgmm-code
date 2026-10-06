"""Move aside run directories left partial by an interruption, so they are redone.

`run_stochastic_eg_campaign.run_stage` pauses on any pre-existing run directory
it cannot resolve. That is correct for a genuine failure -- the artifacts are
evidence and deserve diagnosis -- but wrong for a run the broker killed at quota
exhaustion, where the directory holds no evidence, only an unfinished attempt.
Left alone, the next launch resolves it as an infrastructure failure, pauses the
campaign, and the watcher stops. Unattended recovery never happens.

This runs from the live repository BEFORE the frozen runtime starts, so it also
repairs campaigns whose frozen copy predates it.

This REFUSES to run while the campaign holds its lock. A run in progress has a
directory full of partial artifacts and no terminal evidence yet, so it is
indistinguishable from an interrupted one by inspection alone; moving it would
destroy live work. The lock is the only reliable signal that nothing is running.

A directory is moved only when it carries NO terminal evidence. Anything holding
metrics.json, a validated failure artifact or a validation stop is left exactly
where it is, so real failures still pause the campaign. Nothing is ever deleted:
partial attempts are preserved under `_interrupted_attempts/`.
"""

import argparse
import csv
import fcntl
import json
from pathlib import Path
import time

# Any one of these means the directory is a resolvable outcome, not an interruption.
TERMINAL_EVIDENCE = ("metrics.json", "training_failure.json",
                     "pretraining_failure.json", "infrastructure_failure.json",
                     "validation_stop.json")


def interrupted_run_dirs(rows):
    for row in rows:
        run_dir = Path(row["final_result_dir"])
        if not run_dir.is_dir():
            continue
        entries = [p for p in run_dir.iterdir() if p.name != "_interrupted_attempts"]
        if not entries:
            continue
        if any((run_dir / name).exists() for name in TERMINAL_EVIDENCE):
            continue
        yield row, run_dir


def assert_no_run_in_progress(campaign):
    """Refuse while a run holds the campaign lock; its artifacts are live, not stale."""
    lock_path = campaign / "campaign.lock"
    if not lock_path.exists():
        return
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            raise SystemExit(
                f"REFUSING: a run is active in {campaign}. Its partial artifacts are "
                "live work, not an interrupted attempt.")
        fcntl.flock(handle, fcntl.LOCK_UN)


def quarantine(campaign, apply=True):
    campaign = Path(campaign).resolve()
    assert_no_run_in_progress(campaign)
    manifests = sorted(campaign.glob("*_manifest.csv"))
    if not manifests:
        raise SystemExit(f"No manifest found in {campaign}")
    rows = []
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    moved = []
    for row, run_dir in interrupted_run_dirs(rows):
        stamp = time.time_ns()
        destination = run_dir.parent / "_interrupted_attempts" / f"{run_dir.name}.{stamp}"
        record = {"run_id": row["run_id"], "run_dir": str(run_dir),
                  "preserved_at": str(destination), "time": time.time()}
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(destination)
        moved.append(record)
        print(f"interrupted: {row['run_id']} -> {destination}", flush=True)
    if moved and apply:
        ledger = campaign / "interrupted_attempts.jsonl"
        with ledger.open("a") as handle:
            for record in moved:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"quarantined {len(moved)} interrupted run(s)"
          f"{'' if apply else ' (dry run)'}", flush=True)
    return moved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    quarantine(args.campaign, apply=not args.dry_run)
