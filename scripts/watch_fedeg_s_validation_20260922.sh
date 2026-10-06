#!/usr/bin/env bash
# Keep the FedEG-S validation stage advancing across quota periods.
#
# The campaign is resumable: completed runs are skipped, so a stop at quota
# exhaustion costs only the runs in flight. This watcher resubmits the launcher
# THROUGH gpurun when no run is active. It never bypasses the broker, never
# raises its own priority and never touches a GPU directly -- if the broker
# refuses for lack of budget, it waits and asks again.
#
# Past the weekly quota the broker still admits jobs, as preemptible
# "scavenger" work: a running job whose budget runs out keeps its GPU until an
# in-budget user needs it, and a new over-budget submission runs only on an idle
# GPU and is REFUSED outright when none is free. A refusal is therefore normal
# after quota exhaustion, not a failure, and must never count toward giving up.
#
# It resubmits ONLY for an interrupted run, never for a failed one. A campaign
# that writes status "paused" has hit an infrastructure or evidence failure that
# will recur identically on the next attempt, so retrying it just burns quota;
# the watcher stops and leaves it for a human. It also stops if launches that
# actually RAN keep producing no additional resolved runs, which catches a
# failure that somehow does not set "paused". A launch counts as having run only
# if it advanced the campaign's own status timestamp; a broker refusal writes
# nothing and is simply retried.
#
# "Active" is decided by the campaign's own advisory lock, not by matching
# process names: the launcher cd's into the frozen runtime directory and execs
# the bare script name, so the full path appears in no command line. A pending
# gpurun client counts as active too, so a merely-queued job is never
# double-submitted.
set -uo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
campaign="${CAMPAIGN_DIR:-$repo_root/experiments/highdim_coauthor_protocol_v1/fedeg_s_validation_v3_20260922}"
launcher="${WATCH_LAUNCHER:-$repo_root/scripts/launch_fedeg_s_validation_20260922.sh}"
gpurun_bin="${WATCH_GPURUN:-gpurun}"
settle="${WATCH_SETTLE:-90}"
probe="$repo_root/scripts/campaign_run_active.py"
log="${WATCH_RUN_LOG:-$repo_root/logs/fedeg_s_validation_20260922.log}"
watch_log="${WATCH_LOG:-$repo_root/logs/fedeg_s_validation_20260922_watch.log}"
interval="${WATCH_INTERVAL:-600}"
max_stalled="${WATCH_MAX_STALLED:-2}"
deadline=$(( $(date +%s) + 14*24*3600 ))   # hard stop after two weeks

note () { echo "[$(date -Is)] $*" >> "$watch_log"; }

field () {   # field <key> <default>
  python3 -c "
import json
try:
    print(json.load(open('$campaign/status.json')).get('$1', '$2'))
except Exception:
    print('$2')
" 2>/dev/null
}

stalled=0
last_resolved=-1
stamp_at_submit=""
note "watcher started; interval ${interval}s; campaign $campaign"
while [ "$(date +%s)" -lt "$deadline" ]; do
  status=$(field status "")
  # Count progress from the results ledger, not status.json: a preempted launch
  # leaves status "interrupted", which carries no resolved count at all.
  resolved=$(python3 -c "
import json
try:
    print(len(json.load(open('$campaign/validation_results.json'))))
except Exception:
    print(0)
" 2>/dev/null)
  stamp=$(field time "")

  case "$status" in
    validation_complete_review_required)
      note "campaign complete; watcher exiting"; exit 0 ;;
    paused)
      note "campaign PAUSED: $(field reason 'unknown')"
      note "a paused campaign fails identically on retry; not resubmitting. Watcher exiting for human review."
      exit 2 ;;
  esac

  if python3 "$probe" "$campaign" >/dev/null 2>&1; then
    note "run active (resolved=$resolved); nothing to do"
    stalled=0
  elif pgrep -f 'launch_fedeg_s_validation_20260922\.sh' >/dev/null 2>&1; then
    note "submission pending in the broker queue; waiting"
  else
    if [ -n "$stamp_at_submit" ] && [ "$stamp" != "$stamp_at_submit" ] \
        && [ "$status" != "interrupted" ]; then
      # The previous launch really ran and was not preempted. Did it resolve
      # anything? A preemption is expected over-quota behaviour, never a stall.
      if [ "$resolved" = "$last_resolved" ]; then
        stalled=$(( stalled + 1 ))
      else
        stalled=0
      fi
    fi
    last_resolved="$resolved"
    if [ "$stalled" -ge "$max_stalled" ]; then
      note "$stalled launches ran without resolving anything (resolved stuck at $resolved); watcher exiting for human review."
      exit 3
    fi
    stamp_at_submit="$stamp"
    remaining=$("$gpurun_bin" --status 2>/dev/null | sed -n 's/.*remaining: \([0-9.]*\) GPU-h.*/\1/p' | head -1)
    note "no run active (status='${status:-none}', resolved=$resolved, remaining=${remaining:-unknown} GPU-h); resubmitting"
    CAMPAIGN_DIR="$campaign" nohup "$gpurun_bin" -g 1 bash "$launcher" >> "$log" 2>&1 &
    sleep "$settle"
    if python3 "$probe" "$campaign" >/dev/null 2>&1; then
      note "resubmission running"
    elif pgrep -f 'launch_fedeg_s_validation_20260922\.sh' >/dev/null 2>&1; then
      note "resubmission queued, awaiting a GPU"
    else
      note "resubmission did not start (likely no budget); will retry"
    fi
  fi
  sleep "$interval"
done
note "watcher reached its two-week deadline; exiting"
