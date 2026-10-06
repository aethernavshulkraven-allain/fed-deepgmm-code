#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802"
result_root="results/highdim_deterministic_learning_gate_20260802"

cd "$repo_root"
export WANDB_MODE=disabled

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/equivalence_manifest.csv" \
  --config-dir "$campaign/generated_configs_equivalence" \
  --output-root "$result_root/equivalence" \
  --gpu-ids 0,1 \
  --max-parallel 2 \
  --resume-skip-completed \
  --results-json "$campaign/equivalence_launcher_results.json"

"$python_bin" scripts/check_highdim_deterministic_aux_equivalence.py \
  --root "$result_root/equivalence" \
  --output "$campaign/equivalence_report.json"

"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/gate_manifest.csv" \
  --config-dir "$campaign/generated_configs_gate" \
  --output-root "$result_root/gate" \
  --gpu-ids 0,1 \
  --max-parallel 2 \
  --resume-skip-completed \
  --keep-going \
  --results-json "$campaign/gate_launcher_results.json"
