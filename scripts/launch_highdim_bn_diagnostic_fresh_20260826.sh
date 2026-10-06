#!/usr/bin/env bash
# Fresh 120-round BatchNorm diagnostic (closeout plan SS6.1). One run only.
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/bn_diagnostic_fresh_20260826"
result_root="results/highdim_bn_diagnostic_fresh_20260826"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: invoke this launcher through gpurun." >&2
  exit 1
fi

cd "$repo_root"
export WANDB_MODE=disabled
"$python_bin" scripts/verify_protocol_hashes.py \
  --hashes "$campaign/diagnostic_launch_hashes.json"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/bn_buffer_diagnostic_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root" \
  --gpu-ids "0" --max-parallel 1 \
  --resume-skip-completed --overwrite-incomplete --keep-going \
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

echo "FRESH BN DIAGNOSTIC COMPLETE AND CERTIFIED."
