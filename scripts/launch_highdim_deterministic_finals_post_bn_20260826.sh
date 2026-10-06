#!/usr/bin/env bash
# 3-alpha x 5-seed final matrix. Only the 132 genuinely new trajectories are
# ever launched (finals_launch_manifest.csv) -- the 48 reused V4/stability
# trajectories are never passed to run_manifest.py, since it has no reuse
# concept; they are accounted for solely in finals_evidence_ledger.json,
# which the aggregator reads directly against their real, original run dirs.
#
# This launcher only runs run_manifest.py against an already-prepared
# manifest; it does not invoke the preparer. If any cell's stability outcome
# was not "pass", the full Screen -> Rank -> Confirm -> Promote fallback
# (closeout plan SS9.1/SS9.3, starting from
# scripts/prepare_highdim_stability_retune_alpha0p1_20260827.py) must run
# first, then scripts/prepare_highdim_deterministic_finals_post_bn_20260826.py
# must be re-run with --retune-results pointing at
# scripts/score_highdim_stability_retune_promote_alpha0p1_20260827.py's
# output -- before this launcher is invoked.
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_finals_post_bn_20260826"
result_root="results/highdim_deterministic_finals_post_bn_20260826"

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
diagnostic_campaign="experiments/highdim_coauthor_protocol_v1/bn_diagnostic_fresh_20260826"
"$python_bin" scripts/verify_highdim_bn_diagnostic_certification_20260822.py \
  --certification "$diagnostic_campaign/bn_buffer_diagnostic_certification.json" \
  --manifest "$diagnostic_campaign/bn_buffer_diagnostic_manifest.csv" \
  --launcher-results "$diagnostic_campaign/bn_buffer_diagnostic_launcher_results.json" \
  --launch-hashes "$diagnostic_campaign/diagnostic_launch_hashes.json"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/finals_launch_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/finals_launcher_results.json"
"$python_bin" scripts/check_manifest_stage_complete.py \
  --manifest "$campaign/finals_launch_manifest.csv" \
  --results "$campaign/finals_launcher_results.json" \
  --validate-artifacts
"$python_bin" scripts/aggregate_highdim_deterministic_finals_post_bn_20260826.py \
  --ledger "$campaign/finals_evidence_ledger.json" \
  --out "$campaign/finals_report.json"

echo "FINAL DETERMINISTIC MATRIX COMPLETE. Test metrics unlocked in finals_report.json."
