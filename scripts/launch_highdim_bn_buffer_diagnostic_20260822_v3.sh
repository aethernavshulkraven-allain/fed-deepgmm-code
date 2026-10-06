#!/usr/bin/env bash
# Diagnostic-only entry point for the post-buffer-fix campaign
# (psi_adjudication_20260822_v3/). Runs ONLY the 120-round reproduction
# configuration (FEMNIST-Z, FedOGDA-D, seed 1, lr=0.001, cm=10 -- the exact
# config that produced the negative-running-variance failure under the old
# server update) and then stops. Does NOT start signal or X -- those stay
# gated behind scripts/launch_highdim_psi_adjudication_20260822_v3.sh, which
# re-runs this same preflight step before it will proceed to signal, then
# gates X behind a scored signal stage.
#
# Invoke only through gpurun, single GPU (this stage never needs more than
# one device):
#   gpurun -g 1 bash scripts/launch_highdim_bn_buffer_diagnostic_20260822_v3.sh
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
diagnostic_root="results/highdim_bn_buffer_diagnostic_20260822_v3"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: invoke this launcher through gpurun." >&2
  exit 1
fi
visible_count=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{ count = 0; for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) count++; print count }')
if [ "$visible_count" -ne 1 ]; then
  echo "REFUSING TO RUN: this diagnostic only ever needs 1 GPU; got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (${visible_count} device(s))." >&2
  exit 1
fi

cd "$repo_root"
export WANDB_MODE=disabled
echo "Broker job ${GPU_BROKER_JOB}."

"$python_bin" scripts/verify_protocol_hashes.py \
  --hashes "$campaign/diagnostic_launch_hashes.json"

echo "=== Diagnostic only: 120-round reproduction configuration ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/bn_buffer_diagnostic_manifest.csv" \
  --config-dir "$campaign/generated_configs_diagnostic" \
  --output-root "$diagnostic_root" \
  --gpu-ids "0" --max-parallel 1 \
  --resume-skip-completed --overwrite-incomplete \
  --results-json "$campaign/bn_buffer_diagnostic_launcher_results.json"
"$python_bin" scripts/check_manifest_stage_complete.py \
  --manifest "$campaign/bn_buffer_diagnostic_manifest.csv" \
  --results "$campaign/bn_buffer_diagnostic_launcher_results.json" \
  --require-clean --validate-artifacts
"$python_bin" scripts/certify_highdim_bn_diagnostic_20260822.py \
  --manifest "$campaign/bn_buffer_diagnostic_manifest.csv" \
  --launcher-results "$campaign/bn_buffer_diagnostic_launcher_results.json" \
  --launch-hashes "$campaign/diagnostic_launch_hashes.json" \
  --out "$campaign/bn_buffer_diagnostic_certification.json"

echo "=== DIAGNOSTIC CLEAN. Signal and X remain gated -- not started by this script. ==="
