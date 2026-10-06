#!/usr/bin/env bash
# Post-buffer-fix high-dimensional Psi campaign. Invoke only through gpurun:
#   gpurun -g 2 bash scripts/launch_highdim_psi_adjudication_20260822_v3.sh
#   gpurun -g 1 bash scripts/launch_highdim_psi_adjudication_20260822_v3.sh
set -euo pipefail

echo "REFUSING TO RUN: the v3 signal/X shortlist was selected from pre-fix BatchNorm trajectories." >&2
echo "Run the full post-fix image screen and generate a fresh frozen adjudication packet first." >&2
exit 1

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
campaign="experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
result_root="results/highdim_psi_adjudication_20260822_v3"
diagnostic_root="results/highdim_bn_buffer_diagnostic_20260822_v3"

if [ -z "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: invoke this launcher through gpurun." >&2
  exit 1
fi
visible_count=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{ count = 0; for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) count++; print count }')
if [ "$visible_count" -lt 1 ] || [ "$visible_count" -gt 2 ]; then
  echo "REFUSING TO RUN: expected 1 or 2 broker-assigned GPUs; got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'." >&2
  exit 1
fi
if [ "$visible_count" -eq 2 ]; then
  gpu_ids="0,1"
else
  gpu_ids="0"
fi

cd "$repo_root"
export WANDB_MODE=disabled
echo "Broker job ${GPU_BROKER_JOB}; logical GPU list ${gpu_ids}."

echo "=== Preflight: 120-round reproduction configuration ==="
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
  --require-clean

echo "=== Stage 1/2: signal confirmation (8 cells, 66 fresh runs) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_signal_manifest.csv" \
  --config-dir "$campaign/generated_configs_signal" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/adjudication_signal_launcher_results.json"
"$python_bin" scripts/check_manifest_stage_complete.py \
  --manifest "$campaign/adjudication_signal_manifest.csv" \
  --results "$campaign/adjudication_signal_launcher_results.json"
"$python_bin" scripts/score_highdim_adjudication_20260819.py \
  --cells signal \
  --campaign-dir "$campaign" \
  --results-root "$result_root" \
  --run-id-prefix det_adjudicate_v3

echo "=== Stage 2/2: X adjudication (4 cells, 33 fresh runs) ==="
"$python_bin" scripts/run_manifest.py \
  --manifest "$campaign/adjudication_x_manifest.csv" \
  --config-dir "$campaign/generated_configs_x" \
  --output-root "$result_root" \
  --gpu-ids "$gpu_ids" --max-parallel "$visible_count" \
  --resume-skip-completed --overwrite-incomplete --keep-going \
  --results-json "$campaign/adjudication_x_launcher_results.json"
"$python_bin" scripts/check_manifest_stage_complete.py \
  --manifest "$campaign/adjudication_x_manifest.csv" \
  --results "$campaign/adjudication_x_launcher_results.json"
"$python_bin" scripts/score_highdim_adjudication_20260819.py \
  --cells x \
  --campaign-dir "$campaign" \
  --results-root "$result_root" \
  --run-id-prefix det_adjudicate_v3

echo "=== V3 COMPLETE ==="
