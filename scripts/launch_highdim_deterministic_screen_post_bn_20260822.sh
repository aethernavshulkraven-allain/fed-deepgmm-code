#!/usr/bin/env bash
# Full post-fix screen only. It never launches adjudication automatically.
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
result_root="results/highdim_deterministic_screen_post_bn_20260822"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: invoke this launcher through gpurun." >&2
  exit 1
fi
visible_count=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{ count = 0; for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) count++; print count }')
if [ "$visible_count" -lt 1 ] || [ "$visible_count" -gt 2 ]; then
  echo "REFUSING TO RUN: expected 1 or 2 broker-assigned GPUs; got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'." >&2
  exit 1
fi
gpu_ids="0"
if [ "$visible_count" -eq 2 ]; then
  gpu_ids="0,1"
fi

cd "$repo_root"
export WANDB_MODE=disabled
"$python_bin" scripts/verify_protocol_hashes.py \
  --hashes "$campaign/generated_artifact_hashes.json"
diagnostic_campaign="experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
"$python_bin" scripts/verify_highdim_bn_diagnostic_certification_20260822.py \
  --certification "$diagnostic_campaign/bn_buffer_diagnostic_certification.json" \
  --manifest "$diagnostic_campaign/bn_buffer_diagnostic_manifest.csv" \
  --launcher-results "$diagnostic_campaign/bn_buffer_diagnostic_launcher_results.json" \
  --launch-hashes "$diagnostic_campaign/diagnostic_launch_hashes.json"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/screen_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/screen_launcher_results.json"
"$python_bin" scripts/check_manifest_stage_complete.py \
  --manifest "$campaign/screen_manifest.csv" \
  --results "$campaign/screen_launcher_results.json" \
  --validate-artifacts
"$python_bin" scripts/score_highdim_screen_post_bn_20260822.py \
  --manifest "$campaign/screen_manifest.csv" \
  --out "$campaign/screen_results.json"

echo "SCREEN COMPLETE. Review boundary flags; adjudication was not started."
