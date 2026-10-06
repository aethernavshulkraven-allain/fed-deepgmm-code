import json
import os
import sys
import traceback

# Every model trained through this entry point is a tiny 1-2 hidden-layer MLP
# over at most a few thousand rows; PyTorch's CPU default is one BLAS thread
# per core, which on a shared multi-user machine lets one process grab dozens
# of cores for matmuls too small to benefit (measured: ~29 cores/process,
# almost pure thread-contention overhead, no speedup). Must be set before
# `import fedml`, which transitively imports numpy/torch and initializes
# their thread pools at that point.
_DEFAULT_CPU_THREADS = "4"
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_env_var, _DEFAULT_CPU_THREADS)

import fedml
from fedml import FedMLRunner
import torch
from experiment_utils import (
    ModelSelectionFailure,
    RuntimeProfiler,
    get_effective_config,
    run_dir_from_config,
    write_pretraining_failure_artifact,
)
from numerical_failure import write_training_failure

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
# Only claim all local GPUs when nothing upstream has scoped us to a subset.
# Under the gpu-broker (`gpurun`) CUDA_VISIBLE_DEVICES is preset to the GPUs
# actually allocated to this job; overwriting it would let the job run on a GPU
# belonging to another user, which the server policy forbids.
if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['FEDML_USING_MLOPS'] = 'false'
current_file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_dir)
if __name__ == "__main__":
    # init FedML framework
    print("DEBUG: Calling fedml.init()...")
    args = fedml.init()
    print("DEBUG: fedml.init() successful.")
    profile_config = get_effective_config(args)
    actual_run_dir = run_dir_from_config(profile_config)
    runtime_profiler = RuntimeProfiler.from_config(profile_config)
    runtime_profiler.configure(profile_config, actual_run_dir)
    setattr(args, "_fedgmm_runtime_profiler", runtime_profiler)
    runtime_profiler.record_environment(profile_config)
    runtime_profiler.start_telemetry()

    try:
        # init device
        with runtime_profiler.span("device_init"):
            device = fedml.device.get_device(args)
        runtime_profiler.record_environment(profile_config, device=device)

        # load data
        with runtime_profiler.span("data_load"):
            dataset, output_dim = fedml.data.load(args)

        with runtime_profiler.span("model_create"):
            # Always pass the full [g, f, reg] structure through. When
            # auxiliary_regression is disabled, model_hub still returns a
            # third slot holding None (it constructs and discards the model so
            # the initialization RNG stream stays aligned with aux-on runs);
            # FedAvgAPI reads model[2][0] and expects that slot to exist.
            model = fedml.model.create(args, output_dim)

        # # start training
        if os.environ.get("FEDGMM_CLIENT_EXECUTION_AUDIT") == "1":
            from client_execution_audit import install_execution_audit

            install_execution_audit()
        with runtime_profiler.span("runner_init"):
            fedml_runner = FedMLRunner(args, device, dataset, model)

        with runtime_profiler.span("runner_run"):
            fedml_runner.run()
        runtime_profiler.stop(extra={"exit_status": "completed"})
    except Exception as exc:
        runtime_profiler.stop(extra={"exit_status": "exception", "exception": repr(exc)})
        if bool(getattr(args, "stop_on_numerical_failure", False)) and isinstance(exc, FloatingPointError):
            traceback_text = traceback.format_exc()
            sys.stdout.flush()
            sys.stderr.write(traceback_text)
            sys.stderr.flush()
            write_training_failure(actual_run_dir, args, exc, traceback_text)
            sys.exit(2)
        # ModelSelectionFailure (closeout plan Phase 1 SS4.2) is the only
        # exception type that may classify this run as
        # terminal_pretraining_ineligible -- checked by type, not merely by
        # duck-typing a .diagnostics attribute, so no other exception can
        # accidentally qualify. Every other exception stays an unexplained
        # process failure with no such artifact written.
        if isinstance(exc, ModelSelectionFailure) and isinstance(exc.diagnostics, dict):
            traceback_text = traceback.format_exc()
            # FedAvgAPI.__init__ independently recomputes and writes its own
            # effective_config.json (get_effective_config(self.args) at a
            # later point than this module's own profile_config snapshot,
            # after args may have been mutated by intervening fedml calls)
            # -- read the real on-disk file so the recorded checksum always
            # matches a fresh recomputation, rather than risk a stale
            # early snapshot that looks like tampering later.
            effective_config_path = os.path.join(str(actual_run_dir), "effective_config.json")
            try:
                with open(effective_config_path) as handle:
                    written_effective_config = json.load(handle)
            except (OSError, json.JSONDecodeError):
                written_effective_config = profile_config
            sys.stdout.flush()
            # Write the traceback ourselves and exit via sys.exit() (raises
            # SystemExit, which the interpreter does not print a traceback
            # for) instead of re-raising the original exception. re-raising
            # would let Python's default top-level handler print this same
            # traceback to stderr *after* we have already hashed the file,
            # leaving stderr_sha256 permanently stale relative to the final
            # stderr.log. This way the recorded hash is of the complete,
            # final file, computed after nothing more can be appended.
            sys.stderr.write(traceback_text)
            sys.stderr.flush()
            write_pretraining_failure_artifact(
                actual_run_dir,
                run_id=str(getattr(args, "run_id", "")),
                effective_config=written_effective_config,
                per_epoch_diagnostics=exc.diagnostics.get("per_epoch"),
                best_score=exc.diagnostics.get("best_score"),
                terminal_reason=str(exc),
                traceback_text=traceback_text,
                stdout_path=os.environ.get("FEDGMM_JOB_STDOUT_LOG"),
                stderr_path=os.environ.get("FEDGMM_JOB_STDERR_LOG"),
                hash_bundle_id=os.environ.get("FEDGMM_HASH_BUNDLE_ID"),
            )
            sys.exit(1)
        raise
