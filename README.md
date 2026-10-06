# FedDeepGMM

Run federated experiments from `fedgmm/sp_decentralized_mnist_lr_example` with
the nested FedML `sp` backend. Client-local work can remain serial, use one
persistent process per GPU, or use several persistent processes on one GPU.
The coordinator always retains sampling, aggregation, optimizer history,
FedEG barriers, evaluation, checkpointing, and result generation.

## Client execution modes

Use logical CUDA device IDs in YAML. Multi-GPU execution uses one worker per
listed device:

```yaml
train_args:
  client_execution_mode: multi_gpu_processes
  enable_multiprocessing: false
  multiprocessing_num_workers: 4
  multiprocessing_gpu_ids: [0, 1, 2, 3]
```

True same-GPU multiprocessing deliberately assigns multiple spawned processes
to one logical device:

```yaml
train_args:
  client_execution_mode: multiprocessingsinglegpu
  enable_multiprocessing: false
  multiprocessingsinglegpu_num_workers: 2
  multiprocessingsinglegpu_gpu_id: 0

device_args:
  gpu_id: 0
```

Run an isolated physical GPU as follows:

```bash
cd fedgmm/sp_decentralized_mnist_lr_example
CUDA_VISIBLE_DEVICES=2 python main.py --cf <reviewed-config.yaml>
```

Here shell ID `2` selects the physical GPU; PyTorch remaps it to logical
`cuda:0`, so both YAML GPU IDs must be `0`. If `CUDA_VISIBLE_DEVICES` is
omitted, `main.py` exposes the repository default `0,1,2,3`. A shell-supplied
value is preserved. Both process modes fall back to `sp` when fewer than two
workers are selected. The legacy `enable_multiprocessing: true` selects
`multi_gpu_processes` only when `client_execution_mode` is absent.

Before GPU validation, use `nvidia-smi` to confirm the selected devices are
isolated. Deterministic methods must match SP by `torch.equal` for every `g`
and `f` checkpoint tensor, not merely by rounded metrics.

## FedEG methods

Both FedEG variants keep client sampling and sample-size or uniform-client
aggregation consistent with FedGDA and FedOGDA. The selected mode is controlled
by `aggregation_weighting`; the manifest launcher defaults to `sample_size`.

### FedEG

Use `client_optimizer: fed_eg` with manifest method `fed_eg_d` or
`fed_eg_s`. Clients perform ordinary local GDA/SGD. The server performs two
phases. From the round base parameters \(\theta_t=(g_t,f_t)\), it aggregates
predictor client models and forms

\[
\tilde\theta_t = \theta_t + \beta_p(\bar\theta^p-\theta_t).
\]

The same clients then train from \(\tilde\theta_t\). The server measures the
correction relative to that look-ahead point but applies it to the original
round parameters:

\[
\theta_{t+1}=\theta_t+\beta_c(\bar\theta^c-\tilde\theta_t).
\]

### FedEG_double

Use `client_optimizer: fed_eg_double` with manifest method
`fed_eg_double_d` or `fed_eg_double_s`. The server update is identical to
FedEG, but every local batch also uses ExtraGradient:

\[
\tilde\theta_{i,k}=\theta_{i,k}-\alpha F_i(\theta_{i,k}),
\qquad
\theta_{i,k+1}=\theta_{i,k}-\alpha F_i(\tilde\theta_{i,k}).
\]

Thus FedEG is local GDA plus server EG, whereas FedEG_double is local EG plus
server EG. `learning_rate` controls the local step; `eg_predictor_server_lr`
and `eg_corrector_server_lr` control the two server multipliers and fall back
to `server_learning_rate`. The critic local rate is
`critic_multiplier * learning_rate`. FedEG_double requires approximately
twice the client objective/gradient evaluations of FedEG and supports `sp`,
`multiprocessingsinglegpu`, and `multi_gpu_processes` execution.

## Auxiliary regression

Auxiliary regression is disabled by default because its learned state is not
used by federated GMM updates, model selection, metrics, predictions, centralized
runs, or eICU evaluation. Disabled checkpoints retain `reg_state_dict: null` and
`state.regression: null` for schema compatibility. Legacy experiments can opt in:

```yaml
train_args:
  auxiliary_regression: true
  auxiliary_regression_epochs: 2
```


## Known full-suite errors

The latest `python -m unittest discover -s tests -p 'test_*.py'` run executed 411 tests: 405 passed and six errored for existing repository test-environment or fixture reasons, not multiprocessing behavior.

- `test_experiment_utils`: collection fails because `scripts.run_abs_smoke` is shadowed or unresolved after test-path mutation.
- `test_preflight_full_gradient`: collection fails because `scripts.preflight_full_gradient` is shadowed or unresolved.
- `test_real_image_abs_pipeline`: collection fails because `scripts.analyze_real_image_abs_tuning` is shadowed or unresolved.
- `V1RealTreeTests.test_funnel_reaches_nine_rows_three_hospitals`
- `V1RealTreeTests.test_independent_scan_matches_completion_record_claim`
- `V1RealTreeTests.test_phase_is_final_complete`

The three `V1RealTreeTests` errors originate in `scripts/report_eicu_study_a_status.py`: the expected final-split record is absent, so `final_split` is `None` before `split_sizes` is read. Keep these limitations visible until test import isolation and the eICU v1 fixture tree are repaired.

# How to Run
Use the same idle physical GPU and run them sequentially.

  ## 1. Preview:

  SP:

    CUDA_VISIBLE_DEVICES=0 python scripts/run_manifest.py \
      --manifest experiments/my_campaign/manifest.csv \
      --config-dir results/fedgda_timing/spnew/configs \
      --output-root results/fedgda_timing/spnew/runs \
      --gpu-ids 0 \
      --max-parallel 1 \
      --only run_id=femnist_x_fedgda_d_seed0_alpha0p5_sp \
      --dry-run

  MP4:

  CUDA_VISIBLE_DEVICES=0 python scripts/run_manifest.py \
      --manifest experiments/my_campaign/manifest.csv \
      --config-dir results/fedgda_timing/mp4new/configs \
      --output-root results/fedgda_timing/mp4new/runs \
      --gpu-ids 0 \
      --max-parallel 1 \
      --only run_id=femnist_x_fedgda_d_seed0_alpha0p5 \
      --dry-run

  Each preview should show exactly one experiment.

  ## 2. GPU monitoring

  Open a second terminal and run:

    ### Before SP
  nvidia-smi -i 0 \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 \
    > results/fedgda_timing/spnew/logs/gpu.csv

    ### Before MP4
  nvidia-smi -i 0 \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 \
    > results/fedgda_timing/mp4new/logs/gpu.csv

  ## 3. Run SP

    CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v \
    -o results/fedgda_timing/spnew/logs/time.txt \
    python scripts/run_manifest.py \
      --manifest experiments/my_campaign/manifest.csv \
      --config-dir results/fedgda_timing/spnew/configs \
      --output-root results/fedgda_timing/spnew/runs \
      --gpu-ids 0 \
      --max-parallel 1 \
      --only run_id=femnist_x_fedgda_d_seed0_alpha0p5_sp \
      --results-json results/fedgda_timing/spnew/launcher.json \
      > results/fedgda_timing/spnew/logs/run.log 2>&1

  Run MP4 afterward:

  CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v \
    -o results/fedgda_timing/mp4new/logs/time.txt \
    python scripts/run_manifest.py \
      --manifest experiments/my_campaign/manifest.csv \
      --config-dir results/fedgda_timing/mp4new/configs \
      --output-root results/fedgda_timing/mp4new/runs \
      --gpu-ids 0 \
      --max-parallel 1 \
      --only run_id=femnist_x_fedgda_d_seed0_alpha0p5 \
      --results-json results/fedgda_timing/mp4new/launcher.json \
      > results/fedgda_timing/mp4new/logs/run.log 2>&1

  
  ## 5. Compare total wall time

  rg 'Elapsed \(wall clock\) time|Maximum resident set size|Exit status' \
    /tmp/fedogda_sp_time.txt \
    /tmp/fedogda_mp4_time.txt

  Elapsed (wall clock) time is the total duration, including data loading, training, evaluation, checkpointing, and shutdown.

  ## 6. Read training runtime from metrics.json

  SP metrics:

  results/fedogda_timing/sp/femnist_x/fedogda_d/seed_0/
    femnist_x_fedogda_d_sp_seed0_alpha0p5/metrics.json

  MP metrics:

  results/fedogda_timing/mp4/femnist_x/fedogda_d/seed_0/
    femnist_x_fedogda_d_mp4_seed0_alpha0p5/metrics.json

  Print the recorded runtime:

  python - <<'PY'
  import json

  paths = {
      "SP": (
          "results/fedogda_timing/sp/femnist_x/fedogda_d/seed_0/"
          "femnist_x_fedogda_d_sp_seed0_alpha0p5/metrics.json"
      ),
      "MP4": (
          "results/fedogda_timing/mp4/femnist_x/fedogda_d/seed_0/"
          "femnist_x_fedogda_d_mp4_seed0_alpha0p5/metrics.json"
      ),
  }

  for mode, path in paths.items():
      metrics = json.load(open(path))
      print(
          mode,
          "runtime_seconds =", metrics["runtime_seconds"],
          "final_test_mse =", metrics["final_test_mse"],
      )
  PY

  Calculate speedup:

  python - <<'PY'
  import json

  sp_path = (
      "results/fedogda_timing/sp/femnist_x/fedogda_d/seed_0/"
      "femnist_x_fedogda_d_sp_seed0_alpha0p5/metrics.json"
  )
  mp_path = (
      "results/fedogda_timing/mp4/femnist_x/fedogda_d/seed_0/"
      "femnist_x_fedogda_d_mp4_seed0_alpha0p5/metrics.json"
  )

  sp = json.load(open(sp_path))["runtime_seconds"]
  mp = json.load(open(mp_path))["runtime_seconds"]

  print(f"SP: {sp:.2f} seconds")
  print(f"MP4: {mp:.2f} seconds")
  print(f"Speedup: {sp / mp:.3f}x")
  print(f"MP time change: {(mp / sp - 1) * 100:+.2f}%")
  PY

  Interpretation:

  - Speedup above 1.0×: MP is faster.
  - Speedup below 1.0×: SP is faster.

  Use /usr/bin/time for the user-observed total duration and metrics.json for the repository-recorded run duration.
