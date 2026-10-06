#!/usr/bin/env bash
# Launcher for the corrected Psi-adjudication/confirmation stage
# (psi_adjudication_20260819_v2/). MUST be invoked through the GPU broker,
# never bare:
#
#   gpurun -g 2 bash scripts/launch_highdim_psi_adjudication_20260819_v2.sh
#   gpurun -g 1 bash scripts/launch_highdim_psi_adjudication_20260819_v2.sh
#
# Accepts 1 or 2 broker-assigned GPUs -- use 1 when the other GPU belongs
# to someone else's job (check `gpurun --status` first: if GPU 0 isn't
# running under your own user, request -g 1, not -g 2). Running this
# script directly (without gpurun) would use whatever raw device indices
# happen to be requested without going through the broker's accounting or
# exclusivity guarantees -- e.g. it could collide with another user's job
# on a GPU the broker never actually granted us. The guard below refuses
# to proceed unless it detects it is running inside a broker job with 1 or
# 2 visible devices.
#
# Device indices are always 0 (single-GPU) or 0,1 (two-GPU) from inside
# the job: gpurun/main.py preset CUDA_VISIBLE_DEVICES to whichever
# physical GPU(s) the broker granted, so the visible devices are
# renumbered starting at 0 regardless of which physical GPUs were
# actually assigned.
#
# Progress safety: run_manifest.py writes --results-json after every job
# resolves (not just once at the end), so a hard kill (e.g. broker
# preemption, which gives a 60s warning under the PREEMPTIBLE class) mid-
# manifest still leaves an accurate results file for everything that
# finished before the kill. Per-run training artifacts (metrics.json,
# checkpoints) are always written directly by the training process on
# completion, independent of this script or its parent process staying
# alive. --resume-skip-completed --keep-going --overwrite-incomplete means
# re-running this exact command after any interruption (including this
# script's own bare-invocation guard refusing a bad launch) picks up only
# the incomplete/not-yet-started rows, discarding a partially-written
# run's directory and redoing it from scratch.
#
# Order (per the approved review packet): signal confirmation before _x
# adjudication.
#
# RETIRED 2026-08-22: the server-update math this stage ran under corrupted
# BatchNorm buffers (running_var/num_batches_tracked) on every round -- see
# experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3/CORRECTION_ADDENDUM_20260822.md
# and experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2/LEGACY_20260822.md.
# v2 is not resumable as scientific evidence -- its completed/partial
# artifacts are preserved in place for audit, untouched, but this launcher
# must not run again. Use scripts/launch_highdim_psi_adjudication_20260822_v3.sh.
echo "REFUSING TO RUN: this v2 stage is retired (post-fix v3 supersedes it)." >&2
echo "See experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2/LEGACY_20260822.md" >&2
echo "Use: gpurun -g <1 or 2> bash scripts/launch_highdim_psi_adjudication_20260822_v3.sh" >&2
exit 1

set -uo pipefail
repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
result_root="results/highdim_psi_adjudication_20260819_v2"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: GPU_BROKER_JOB is not set -- this script must be" >&2
  echo "invoked through the broker, not run bare:" >&2
  echo "  gpurun -g 2 bash scripts/launch_highdim_psi_adjudication_20260819_v2.sh" >&2
  exit 1
fi
visible_count=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c '[0-9]') ))
if [ "$visible_count" -lt 1 ] || [ "$visible_count" -gt 2 ]; then
  echo "REFUSING TO RUN: expected 1 or 2 broker-assigned GPUs, but" >&2
  echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' exposes ${visible_count}." >&2
  echo "Re-invoke as: gpurun -g <1 or 2> bash scripts/launch_highdim_psi_adjudication_20260819_v2.sh" >&2
  exit 1
fi
if [ "$visible_count" -eq 2 ]; then
  gpu_ids="0,1"
else
  gpu_ids="0"
fi

cd "$repo_root"
export WANDB_MODE=disabled

echo "Running under broker job ${GPU_BROKER_JOB}, device index list ${gpu_ids} (${visible_count} GPU(s))."

echo "=== Stage 1/2: signal confirmation (8 cells, 42 rows) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_signal_manifest.csv" \
  --config-dir "$campaign/generated_configs_signal" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/adjudication_signal_launcher_results.json"

echo "=== Stage 2/2: _x adjudication (4 cells, 21 rows) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_x_manifest.csv" \
  --config-dir "$campaign/generated_configs_x" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/adjudication_x_launcher_results.json"

echo "=== DONE ==="
