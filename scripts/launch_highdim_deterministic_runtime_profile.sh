#!/usr/bin/env bash
set -euo pipefail

# Deterministic runtime-profiling diagnostic.
#
# Pass 1 (unprofiled) gives true wall time per round.
# Pass 2 (profiled) gives the per-phase breakdown. The profiler issues a
# torch.cuda.synchronize() on entry and exit of every span, so pass-2 totals are
# inflated and must NOT be used for the GPU-hour projection -- use pass 1 for
# timing and pass 2 only for the relative shape of where time goes.

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_runtime_profile_20260805"
result_root="results/highdim_deterministic_runtime_profile_20260805"

cd "$repo_root"
export WANDB_MODE=disabled

echo "=== PASS 1: unprofiled, both aux arms (true timing + aux cost) ==="
env -u FEDGMM_PROFILE_RUNTIME \
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/profile_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root/unprofiled" \
  --gpu-ids 0,1 \
  --max-parallel 2 \
  --resume-skip-completed \
  --keep-going \
  --results-json "$campaign/unprofiled_launcher_results.json"

# Only the aux-off arm is profiled: that is the protocol-v2 configuration, and
# profiling both arms would double the (sync-inflated, slow) profiled pass for
# no extra information -- the aux cost is already measured by pass 1 wall time.
echo "=== PASS 2: profiled, aux-off only (phase breakdown) ==="
export FEDGMM_PROFILE_RUNTIME=1
export FEDGMM_PROFILE_GPU_TELEMETRY=1
export FEDGMM_PROFILE_GPU_INTERVAL_SECONDS=5
export FEDGMM_PROFILE_ROOT="$repo_root/results/_profiling/highdim_deterministic_runtime_profile_20260805"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/profile_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root/profiled" \
  --only auxiliary_regression=False \
  --gpu-ids 0,1 \
  --max-parallel 2 \
  --resume-skip-completed \
  --keep-going \
  --results-json "$campaign/profiled_launcher_results.json"

echo "=== DONE ==="
