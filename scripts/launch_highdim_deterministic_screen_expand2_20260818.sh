#!/usr/bin/env bash
set -euo pipefail

# RETIRED 2026-08-22: pre-BatchNorm-fix legacy entry point. Not resumable as
# scientific evidence -- see
# experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3/CORRECTION_ADDENDUM_20260822.md
# and experiments/highdim_coauthor_protocol_v1/legacy_batchnorm_trajectories_20260822.json.
echo "REFUSING TO RUN: this pre-BatchNorm-fix legacy entry point is retired." >&2
echo "See experiments/highdim_coauthor_protocol_v1/legacy_batchnorm_trajectories_20260822.json" >&2
echo "Use: scripts/launch_highdim_deterministic_screen_post_bn_20260822.sh" >&2
exit 1

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
result_root="results/highdim_deterministic_screen_20260813"
cd "$repo_root"
export WANDB_MODE=disabled
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/screen_expand2_manifest.csv" \
  --config-dir "$campaign/generated_configs_expand2" \
  --output-root "$result_root" \
  --gpu-ids 0,1 --max-parallel 2 \
  --resume-skip-completed --keep-going \
  --results-json "$campaign/screen_expand2_launcher_results.json"
echo "=== DONE ==="
