#!/usr/bin/env bash
# Launch a budget-scoped high-dim queue under the gpu-broker.
#
# Run as:  gpurun scripts/launch_highdim_slice_gpurun.sh <manifest.csv> <output_root>
#
# gpurun presets CUDA_VISIBLE_DEVICES to the GPUs actually allocated to this job.
# This script refuses to start unless exactly one GPU is visible, because
# run_manifest.py is invoked with --gpu-ids 0 (logical index inside the allocation)
# and a wider visibility would let a run land on another user's GPU.

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Accepts one or more "manifest.csv=output_root" stages, run back to back inside a
# single broker allocation. Chaining avoids re-queueing between stages, which on a
# busy server can cost hours of waiting per stage.
if [ "$#" -lt 1 ]; then
  echo "usage: $0 <manifest.csv=output_root> [<manifest.csv=output_root> ...]" >&2
  echo "       $0 <manifest.csv> <output_root>   # legacy two-arg form" >&2
  exit 2
fi

STAGES=()
if [ "$#" -eq 2 ] && [ -f "$1" ] && [[ "$1" != *=* ]]; then
  STAGES+=("$1=$2")          # legacy two-argument form
else
  STAGES=("$@")
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== GPU allocation guard ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$PYTHON" - <<'PYCHECK'
import os, sys
import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not visible:
    sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is unset -- not running under gpurun")

n = torch.cuda.device_count()
print(f"visible spec : {visible!r}")
print(f"torch devices: {n}")
if not torch.cuda.is_available():
    sys.exit("REFUSING: CUDA is not available inside the allocation")
if n != 1:
    sys.exit(f"REFUSING: expected exactly 1 allocated GPU, torch sees {n}")
print(f"logical cuda:0 -> {torch.cuda.get_device_name(0)}")
print("guard passed")
PYCHECK

STAGE_NUM=0
for stage in "${STAGES[@]}"; do
  STAGE_NUM=$((STAGE_NUM + 1))
  MANIFEST="${stage%%=*}"
  OUTPUT_ROOT="${stage#*=}"
  SLICE_DIR="$(dirname "$MANIFEST")"
  BASE="$(basename "${MANIFEST%.csv}")"
  RESULTS_JSON="${SLICE_DIR}/${BASE}_run_results_${STAMP}.json"
  CONFIG_DIR="${SLICE_DIR}/${BASE}_generated_configs"

  echo
  echo "=== stage ${STAGE_NUM}/${#STAGES[@]} ==="
  echo "manifest    : $MANIFEST"
  echo "output root : $OUTPUT_ROOT"
  echo "results json: $RESULTS_JSON"
  echo "started     : $(date -Is)"
  echo

  # A stage that fails must not abort the remaining stages: each stage is an
  # independent set of runs, and run_manifest already records per-job outcomes.
  "$PYTHON" scripts/run_manifest.py \
    --manifest "$MANIFEST" \
    --config-dir "$CONFIG_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --gpu-ids 0 \
    --max-parallel 1 \
    --resume-skip-completed \
    --keep-going \
    --results-json "$RESULTS_JSON" || echo "stage ${STAGE_NUM} exited non-zero; continuing"

  echo "stage ${STAGE_NUM} finished: $(date -Is)"
done

echo
echo "=== all stages complete: $(date -Is) ==="
