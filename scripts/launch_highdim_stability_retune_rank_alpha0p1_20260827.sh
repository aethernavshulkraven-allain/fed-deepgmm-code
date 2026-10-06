#!/usr/bin/env bash
# Alpha=0.1 retune fallback Rank stage (closeout plan SS9.1 escape hatch;
# doe_review_and_revised_grid.md Part VI/VII): 500-round seed-0 runs for
# each retuned cell's Screen-stage top-2 candidates. Only launched after
# scripts/launch_highdim_stability_retune_alpha0p1_20260827.sh (the Screen
# stage) has produced retune_screen_results.json.
#
# This launcher only runs run_manifest.py against an already-prepared
# manifest; it does not invoke the preparer. Run
# scripts/prepare_highdim_stability_retune_rank_alpha0p1_20260827.py
# --screen-results retune_screen_results.json first.
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_stability_retune_rank_alpha0p1_20260827"
result_root="results/highdim_deterministic_stability_retune_rank_alpha0p1_20260827"

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
  --manifest "$campaign/rank_manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/rank_launcher_results.json"
"$python_bin" scripts/check_manifest_stage_complete.py \
  --manifest "$campaign/rank_manifest.csv" \
  --results "$campaign/rank_launcher_results.json" \
  --validate-artifacts

echo "ALPHA=0.1 RETUNE RANK STAGE COMPLETE. Next: launch_highdim_stability_retune_confirm_alpha0p1_20260827.sh."
