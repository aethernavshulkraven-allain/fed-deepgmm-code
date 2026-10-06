#!/usr/bin/env bash
# Polls until 2 GPUs are idle, then launches the corrected adjudication
# stage through the broker via the documented invocation. PREEMPTIBLE
# requests (over quota) refuse immediately rather than queuing if not
# enough GPUs are idle right now, so this loop is what provides the
# "run once possible" behavior instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "REFUSING TO RUN: the v2 adjudication campaign is legacy and scientifically ineligible." >&2
echo "Use the post-fix screening workflow; do not resume or mix v2 trajectories." >&2
exit 1

while true; do
  idle=$(gpurun --status | awk '$2=="idle"{c++} END{print c+0}')
  echo "$(date -Is) idle_gpus=${idle}"
  if [ "$idle" -ge 2 ]; then
    echo "2+ GPUs idle -- launching."
    exec gpurun -g 2 bash scripts/launch_highdim_psi_adjudication_20260819_v2.sh
  fi
  sleep 60
done
