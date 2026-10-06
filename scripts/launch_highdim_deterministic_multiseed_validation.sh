#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_multiseed_validation_20260803"
gpu_ids="${GPU_IDS:-0,1}"
max_parallel="${MAX_PARALLEL:-2}"

cd "$repo_root"
export WANDB_MODE=disabled

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/manifest.csv" \
  --config-dir "$campaign/generated_configs" \
  --output-root results/highdim_deterministic_multiseed_validation_20260803 \
  --gpu-ids "$gpu_ids" \
  --max-parallel "$max_parallel" \
  --resume-skip-completed \
  --keep-going \
  --results-json "$campaign/launcher_results.json"
