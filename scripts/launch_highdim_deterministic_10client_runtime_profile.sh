#!/usr/bin/env bash
set -euo pipefail

# Runtime-profiling diagnostic for the adopted 10-client full-participation
# deterministic design (client_num_in_total = client_num_per_round = 10).
# Mirrors scripts/launch_highdim_deterministic_runtime_profile.sh (the
# 1000-client probe), aux-off only since that arm is already decided.
#
# Pass 1 (unprofiled) gives true wall time per round.
# Pass 2 (profiled) gives setup vs. per-round decomposition. The profiler
# issues a torch.cuda.synchronize() on entry/exit of every span, so pass-2
# totals are inflated and must NOT be used for the GPU-hour projection --
# pass 1 is for timing, pass 2 only for the setup/per-round split.

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_10client_runtime_profile_20260807"
result_root="results/highdim_deterministic_10client_runtime_profile_20260807"

cd "$repo_root"
export WANDB_MODE=disabled

echo "=== PASS 1: unprofiled (true timing) ==="
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

echo "=== PASS 2: profiled (setup vs. per-round breakdown) ==="
export FEDGMM_PROFILE_RUNTIME=1
export FEDGMM_PROFILE_GPU_TELEMETRY=1
export FEDGMM_PROFILE_GPU_INTERVAL_SECONDS=5
export FEDGMM_PROFILE_ROOT="$repo_root/results/_profiling/highdim_deterministic_10client_runtime_profile_20260807"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/profile_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root/profiled" \
  --gpu-ids 0,1 \
  --max-parallel 2 \
  --resume-skip-completed \
  --keep-going \
  --results-json "$campaign/profiled_launcher_results.json"

echo "=== DONE ==="
