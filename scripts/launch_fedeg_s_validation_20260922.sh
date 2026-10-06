#!/usr/bin/env bash
# FedEG-S validation stage only. It never launches the finals matrix automatically.
# Resumable: re-running this through gpurun skips completed runs, so a stop at
# quota exhaustion costs only the runs that were in flight.
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="${CAMPAIGN_DIR:-$repo_root/experiments/highdim_coauthor_protocol_v1/fedeg_s_validation_v3_20260922}"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: invoke this launcher through gpurun." >&2
  exit 1
fi
visible_count=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{ count = 0; for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) count++; print count }')
if [ "$visible_count" -lt 1 ] || [ "$visible_count" -gt 4 ]; then
  echo "REFUSING TO RUN: expected 1 to 4 broker-assigned GPUs; got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'." >&2
  exit 1
fi

export WANDB_MODE=disabled

# Redo runs an interruption left partial, so a stop at quota exhaustion recovers
# unattended. This lives in the launcher rather than the frozen runtime so it
# also repairs campaigns frozen before it existed. It refuses if a run is active
# and never touches a directory holding terminal evidence.
"$python_bin" "$repo_root/scripts/quarantine_interrupted_runs.py" "$campaign"

cd "$campaign/runtime/scripts"
exec "$python_bin" run_fedeg_s_validation_20260922.py run --campaign "$campaign"
