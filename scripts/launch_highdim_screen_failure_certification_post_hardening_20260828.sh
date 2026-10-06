#!/usr/bin/env bash
# Fresh reproduction of the 4 screen rows that failed before round 0
# (closeout plan SS6.2), superseding the 2026-08-26 certification now that
# hash_bundle_sha256 is mandatory. --keep-going: a genuinely different
# failure on any one row must not block certifying the others.
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/screen_failure_certification_post_hardening_20260828"
result_root="results/highdim_screen_failure_certification_post_hardening_20260828"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: invoke this launcher through gpurun." >&2
  exit 1
fi
visible_count=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{ count = 0; for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) count++; print count }')
if [ "$visible_count" -lt 1 ]; then
  echo "REFUSING TO RUN: no broker-assigned GPUs; got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'." >&2
  exit 1
fi
gpu_ids="0"
if [ "$visible_count" -ge 2 ]; then
  gpu_ids="0,1"
fi

cd "$repo_root"
export WANDB_MODE=disabled
"$python_bin" scripts/verify_protocol_hashes.py \
  --hashes "$campaign/generated_artifact_hashes.json"

# Read by main.py's exception handler and recorded verbatim (both as a path
# and, now, as a content hash) in any pretraining_failure.json this launch
# produces, so a reader can trace exactly which frozen hash bundle
# certified the failure.
export FEDGMM_HASH_BUNDLE_ID="$campaign/generated_artifact_hashes.json"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/screen_failure_certification_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/screen_failure_certification_launcher_results.json"

echo "POST-HARDENING SCREEN FAILURE CERTIFICATION RUNS COMPLETE. Review outcomes against SS6.2's rule:"
echo "  same failure -> terminal-pretraining-ineligible; unexpected success -> investigate;"
echo "  different failure -> unresolved, investigate. Do not retry."
