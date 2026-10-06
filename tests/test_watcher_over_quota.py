"""The campaign watcher must survive running past the weekly GPU quota.

Past quota the broker still admits work as preemptible "scavenger" jobs: a new
submission runs only on an idle GPU and is REFUSED when none is free, and a
running job can be evicted when an in-budget user needs the GPU. Both are normal
over-quota behaviour. The watcher gives up only on a paused campaign or on
launches that genuinely ran and resolved nothing -- never on a refusal or a
preemption, or it would quit exactly when the quota runs out.

The broker and launcher are faked; no GPU is touched.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WATCHER = ROOT / "scripts/watch_fedeg_s_validation_20260922.sh"


def setup(tmp_path, launcher_body, gpurun_mode="run", status=None):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    if status:
        (campaign / "status.json").write_text(json.dumps(status))
    launcher = tmp_path / "fake_launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -e\n" + launcher_body + "\n")
    gpurun = tmp_path / "gpurun"
    refuse = "exit 1" if gpurun_mode == "refuse" else 'shift 2; exec "$@"'
    gpurun.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--status" ]; then echo "User: x quota: 48.0 GPU-h/weekly   '
        'remaining: 0.0 GPU-h"; exit 0; fi\n'
        + refuse + "\n")
    for path in (launcher, gpurun):
        path.chmod(0o755)
    env = {**os.environ, "CAMPAIGN_DIR": str(campaign), "WATCH_LAUNCHER": str(launcher),
           "WATCH_GPURUN": str(gpurun), "WATCH_INTERVAL": "1", "WATCH_SETTLE": "1",
           "WATCH_LOG": str(tmp_path / "watch.log"),
           "WATCH_RUN_LOG": str(tmp_path / "run.log")}
    return campaign, env


def watch(env, seconds):
    try:
        done = subprocess.run(["bash", str(WATCHER)], env=env, timeout=seconds,
                              capture_output=True, text=True)
        return done.returncode
    except subprocess.TimeoutExpired:
        return None   # still watching when the test stopped it


def write_status(state):
    return (f"python3 -c \"import json,time; json.dump({{'status':'{state}',"
            f"'time':time.time()}}, open('$CAMPAIGN_DIR/status.json','w'))\"")


def test_broker_refusals_are_retried_not_counted_as_stalls(tmp_path):
    _, env = setup(tmp_path, "true", gpurun_mode="refuse")
    assert watch(env, 12) is None
    log = (tmp_path / "watch.log").read_text()
    assert log.count("resubmitting") >= 3
    assert "exiting" not in log


def test_repeated_preemptions_are_not_counted_as_stalls(tmp_path):
    _, env = setup(tmp_path, write_status("interrupted"))
    assert watch(env, 12) is None
    log = (tmp_path / "watch.log").read_text()
    assert log.count("resubmitting") >= 3
    assert "exiting" not in log


def test_launches_that_run_but_resolve_nothing_stop_the_watcher(tmp_path):
    _, env = setup(tmp_path, write_status("running"))
    assert watch(env, 30) == 3


def test_progressing_campaign_keeps_going(tmp_path):
    body = ("python3 -c \"import json,os,time; p='$CAMPAIGN_DIR/validation_results.json';"
            " r=json.load(open(p)) if os.path.exists(p) else []; r.append({'run_id':len(r)});"
            " json.dump(r,open(p,'w')); json.dump({'status':'running','time':time.time()},"
            " open('$CAMPAIGN_DIR/status.json','w'))\"")
    _, env = setup(tmp_path, body)
    assert watch(env, 12) is None
    assert "exiting" not in (tmp_path / "watch.log").read_text()


def test_paused_campaign_stops_without_resubmitting(tmp_path):
    _, env = setup(tmp_path, "touch $CAMPAIGN_DIR/LAUNCHED",
                   status={"status": "paused", "reason": "x", "time": 1})
    assert watch(env, 10) == 2
    assert not (tmp_path / "campaign" / "LAUNCHED").exists()


def test_complete_campaign_exits_cleanly(tmp_path):
    _, env = setup(tmp_path, "true",
                   status={"status": "validation_complete_review_required", "time": 1})
    assert watch(env, 10) == 0
