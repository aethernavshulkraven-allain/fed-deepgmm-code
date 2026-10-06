"""Gates on interrupted-run recovery.

A campaign killed at quota exhaustion leaves one run directory partially
written. `run_stage` cannot resolve it, records an infrastructure failure and
pauses; the watcher then stops, so unattended recovery never happens. Moving
that directory aside lets the run be redone -- but only when it really is stale,
never when it is live work or real evidence.
"""

import csv
import fcntl
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import quarantine_interrupted_runs as q          # noqa: E402

FIELDS = ["run_id", "final_result_dir"]


def build(tmp_path, runs):
    """runs: {run_id: [file names to create]}; [] means an empty directory."""
    campaign = tmp_path / "campaign"
    results = campaign / "results"
    rows = []
    for run_id, names in runs.items():
        run_dir = results / run_id
        run_dir.mkdir(parents=True)
        for name in names:
            (run_dir / name).write_text("{}\n")
        rows.append({"run_id": run_id, "final_result_dir": str(run_dir)})
    campaign.mkdir(exist_ok=True)
    with (campaign / "validation_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return campaign, results


def test_interrupted_run_is_moved_aside_and_preserved(tmp_path):
    campaign, results = build(tmp_path, {
        "interrupted": ["mse_by_round.csv", "effective_config.json"]})
    original = (results / "interrupted" / "mse_by_round.csv").read_text()
    moved = q.quarantine(campaign)
    assert len(moved) == 1
    assert not (results / "interrupted" / "mse_by_round.csv").exists()
    preserved = Path(moved[0]["preserved_at"])
    assert preserved.is_dir()
    assert (preserved / "mse_by_round.csv").read_text() == original
    assert (campaign / "interrupted_attempts.jsonl").exists()


@pytest.mark.parametrize("evidence", list(q.TERMINAL_EVIDENCE))
def test_terminal_evidence_is_never_moved(tmp_path, evidence):
    """Completed runs and real failures must survive untouched, so run_stage
    can resolve or pause on them."""
    campaign, results = build(tmp_path, {"resolved": ["mse_by_round.csv", evidence]})
    assert q.quarantine(campaign) == []
    assert (results / "resolved" / evidence).exists()


def test_empty_directory_is_left_alone(tmp_path):
    campaign, results = build(tmp_path, {"empty": []})
    assert q.quarantine(campaign) == []
    assert (results / "empty").is_dir()


def test_refuses_while_a_run_holds_the_campaign_lock(tmp_path):
    campaign, results = build(tmp_path, {"live": ["mse_by_round.csv"]})
    with (campaign / "campaign.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="REFUSING"):
            q.quarantine(campaign)
    assert (results / "live" / "mse_by_round.csv").exists()


def test_dry_run_changes_nothing(tmp_path):
    campaign, results = build(tmp_path, {"interrupted": ["mse_by_round.csv"]})
    moved = q.quarantine(campaign, apply=False)
    assert len(moved) == 1
    assert (results / "interrupted" / "mse_by_round.csv").exists()
    assert not (campaign / "interrupted_attempts.jsonl").exists()


def test_previously_quarantined_attempt_is_not_re_quarantined(tmp_path):
    """The _interrupted_attempts folder must not itself look like live artifacts."""
    campaign, results = build(tmp_path, {"interrupted": ["mse_by_round.csv"]})
    q.quarantine(campaign)
    (results / "interrupted").mkdir(exist_ok=True)
    assert q.quarantine(campaign) == []


def test_cli_runs_and_reports(tmp_path):
    campaign, _ = build(tmp_path, {"interrupted": ["mse_by_round.csv"]})
    out = subprocess.run([sys.executable, str(ROOT / "scripts/quarantine_interrupted_runs.py"),
                          str(campaign)], capture_output=True, text=True, check=True)
    assert "quarantined 1 interrupted run" in out.stdout
