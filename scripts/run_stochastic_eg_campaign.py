#!/usr/bin/env python3
"""Freeze and run the validation-only stochastic EG selection campaign.

Preparation is CPU-only. Run `run` under gpurun; each child executes the
supported main.py from a frozen source copy, on one allocated GPU.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_REL = Path("fedgmm/sp_decentralized_mnist_lr_example")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / EXAMPLE_REL))
import run_manifest as launcher  # noqa: E402
from experiment_utils import config_checksum  # noqa: E402
from numerical_failure import file_sha256  # noqa: E402

DATASETS = ("femnist_z", "femnist_x", "femnist_xz",
            "cifar10_z", "cifar10_x", "cifar10_xz")
METHODS = ("fed_eg_s", "fed_zo_eg_s")
ALPHAS = (0.1, 0.5, 1.0)
DEFAULT_CAMPAIGN = ROOT / "experiments/highdim_coauthor_protocol_v1/stochastic_eg_selection_20260906"


def save_json(path, payload, immutable=False):
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Refusing to change frozen artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_manifest(path, rows):
    import io

    handle = io.StringIO(newline="")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    text = handle.getvalue()
    path = Path(path)
    if path.exists() and path.read_bytes() != text.encode():
        raise RuntimeError(f"Refusing to replace manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(text.encode())


def candidates(method):
    # Equal six-trial budget and identical shared LR/server-rate pairs.
    # ZO uses a declared sparse joint design, not an exhaustive grid.
    recipes = ((1e-4, 1), (1e-3, 4), (1e-2, 1),
               (1e-3, 1), (1e-4, 4), (1e-2, 4))
    output = []
    for index, (lr, server_lr) in enumerate(
        (lr, server_lr) for lr in (0.001, 0.003, 0.01)
        for server_lr in (0.5, 1.5)
    ):
        mu, directions = recipes[index] if method == "fed_zo_eg_s" else (1e-3, 1)
        output.append({"candidate": index, "learning_rate": lr,
                       "server_learning_rate": server_lr, "zo_mu": mu,
                       "zo_num_directions": directions})
    return output


def make_row(campaign, source, stage, dataset, alpha, method, seed, candidate, rounds):
    row = dict(source)
    token = f"{alpha:g}".replace(".", "p")
    run_id = f"seg26_{stage}_{dataset}_{method}_a{token}_s{seed}_c{candidate['candidate']}"
    output = campaign / "results" / stage
    row.update(candidate)
    row.update({
        "run_id": run_id, "protocol_version": "stochastic_eg_selection_20260906_v1",
        "run_group": stage, "training_scope": "federated", "stage": stage,
        "dataset": dataset, "alpha": alpha, "partition_alpha": alpha,
        "method": method, "method_label": launcher.METHOD_LABEL[method],
        "client_optimizer": launcher.METHOD_TO_OPTIMIZER[method], "seed": seed,
        "client_num_in_total": 1000, "client_num_per_round": 10,
        "batch_size": 256, "epochs": 3, "comm_round": rounds,
        # Honour a candidate-supplied value: this update runs AFTER
        # row.update(candidate), so a hardcoded 10 here silently discarded
        # per-dataset multipliers and made whole campaigns run at cm=10.
        "critic_multiplier": candidate.get("critic_multiplier", 10),
        "weight_decay": 0.05, "gradient_clip_norm": 1,
        "objective_mode": "legacy", "objective_lambda_1": 0.1,
        "aggregation_weighting": "sample_size",
        "server_buffer_policy": "direct_client_aggregate",
        "client_execution_mode": "sp", "enable_multiprocessing": False,
        "stop_on_numerical_failure": True, "max_local_steps_per_round": 0,
        "require_multibatch_stochastic": False, "scenario_name": "main",
        "simple_model_selection_epochs": 100, "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200, "skip_model_selection": False,
        "skip_gmm_eval": False, "gmm_eval_proxy": "approx_psi",
        "auxiliary_regression": False, "auxiliary_regression_epochs": 0,
        "compact_predictions_only": True, "append_round_csv": True,
        "periodic_checkpoint_interval": 200, "log_test_mse_by_round": False,
        "test_mse_used_for_selection": False, "selection_metric_source": "validation",
        "primary_selection_metric": "pooled_validation_mse",
        "selection_source": "validation_only", "run_status": "not_started",
        "implementation_status": "ready", "preflight_required": False,
        "preflight_status": "not_required", "using_gpu": True, "gpu_id": 0,
        "data_cache_dir": str(ROOT / EXAMPLE_REL / "data"),
        "output_root": str(output),
        "final_result_dir": str(output / dataset / method / f"seed_{seed}" / run_id),
        "notes": "Frozen stochastic EG selection; tuning uses validation only; no automatic retuning.",
    })
    return row


def make_jobs(campaign, rows):
    kwargs = {name: None for name in inspect.signature(launcher.build_jobs).parameters
              if name.startswith("override_") or name.startswith("default_")}
    jobs, skipped = launcher.build_jobs(
        rows, python_executable=sys.executable,
        main_path=campaign / "runtime" / EXAMPLE_REL / "main.py",
        config_dir=campaign / "configs" / str(rows[0]["stage"]),
        output_root=Path(rows[0]["output_root"]), gpu_ids=[0], **kwargs,
    )
    if skipped or len(jobs) != len(rows):
        raise RuntimeError(f"Unlaunchable rows: {skipped}")
    return jobs


def snapshot(campaign):
    target = campaign / "runtime" / EXAMPLE_REL
    source = ROOT / EXAMPLE_REL
    if target.exists():
        raise RuntimeError(f"Runtime already exists: {target}; use the existing prepared campaign")

    def ignore(directory, names):
        skipped = {name for name in names if name in {"__pycache__", ".git"}
                   or Path(name).suffix in {".pyc", ".pt", ".npz", ".npy", ".log",
                                           ".png", ".jpg", ".pdf", ".mp4", ".ipynb"}}
        if Path(directory) == source:
            skipped.update(set(names) & {"data", "results", "csv", "checkpoints", "plots", "logs"})
        return skipped

    shutil.copytree(source, target, ignore=ignore)
    (target / "data").symlink_to(source / "data", target_is_directory=True)
    script_dir = campaign / "runtime/scripts"
    script_dir.mkdir(parents=True)
    for name in ("run_stochastic_eg_campaign.py", "run_manifest.py", "verify_protocol_hashes.py",
                 "highdim_protocol_hash_closure_20260822.py"):
        shutil.copy2(ROOT / "scripts" / name, script_dir / name)


def prepare(campaign):
    if (campaign / "freeze.json").exists():
        verify_freeze(campaign)
        print(f"Already prepared: {campaign}")
        return
    source_rows = launcher._load_rows(
        ROOT / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_stochastic.csv"
    )
    sources = {dataset: next(row for row in source_rows if row["dataset"] == dataset)
               for dataset in DATASETS}
    campaign.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1, "datasets": DATASETS, "alphas": ALPHAS, "methods": METHODS,
        "candidates": {method: candidates(method) for method in METHODS},
        "pilot_seed": 20, "screen_seed": 10, "confirmation_seeds": [11, 12],
        "final_seeds": [0, 1, 2, 3, 4], "rounds": {"pilot": 25, "screen": 150, "confirm": 500, "final": 1500},
        "counts": {"pilot": 18, "screen": 216, "confirm": 72, "final": 90},
        "selection": "validation_only; no test values enter scoring",
        "promotion": {"relative_improvement": 0.05, "minimum_cell_wins": 12,
                      "maximum_cell_regression_ratio": 1.25,
                      "maximum_median_final_best_validation_ratio": 2.0,
                      "maximum_p90_final_best_validation_ratio": 5.0,
                      "require_both_confirmation_seeds_improve": True,
                      "require_all_confirmation_runs_numerically_valid_for_winner": True},
        "uncertainty": "Two confirmation seeds support a budgeted selection gate, not a formal 95% population-superiority claim. Inconclusive or materially heterogeneous results stop finals.",
        "watchdog": {"startup_seconds": 1800, "stall_seconds": 600, "total_seconds": 10800},
        "baseline_comparison": "Old GDA/OGDA finals are historical only; refreshed baseline runs are outside this launch.",
        "batching": "Preserve batch_size=256 and actual existing loader semantics; record partition sizes. Many clients may be single-batch.",
        "zo": "Existing joint Rademacher SPSA correction; same batch within plus/minus pair; 2K objective probes per local step. Existing train-mode buffer updates are retained and audited, not silently changed.",
        "weight_decay": "0.05 preserved as compatibility metadata; not wired into G/F optimizers.",
        "sources": sources,
    }
    save_json(campaign / "protocol.json", protocol, immutable=True)
    pilot, screen = [], []
    for dataset in DATASETS:
        for method in METHODS:
            for index in ([3] if method == "fed_eg_s" else [3, 1]):
                pilot.append(make_row(campaign, sources[dataset], "pilot", dataset, 0.5,
                                      method, 20, candidates(method)[index], 25))
    for index in range(6):
        for alpha in ALPHAS:
            for dataset in DATASETS:
                for method in METHODS:
                    screen.append(make_row(campaign, sources[dataset], "screen", dataset, alpha,
                                           method, 10, candidates(method)[index], 150))
    write_manifest(campaign / "pilot_manifest.csv", pilot)
    write_manifest(campaign / "screen_manifest.csv", screen)
    snapshot(campaign)
    runtime = campaign / "runtime"
    source_hashes = [{"path": str(path.relative_to(runtime)), "sha256": file_sha256(path)}
                     for path in runtime.rglob("*.py") if "__pycache__" not in path.parts]
    save_json(runtime / "training_hashes.json", source_hashes, immutable=True)
    config_files = []
    for rows in (pilot, screen):
        for job in make_jobs(campaign, rows):
            launcher.write_config(job.config_path, job.config)
            config_files.append(job.config_path)
    data_files = [ROOT / EXAMPLE_REL / "data" / dataset / "main.npz" for dataset in DATASETS]
    runtime_files = [path for path in (campaign / "runtime").rglob("*")
                     if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts]
    files = runtime_files + config_files + [campaign / "protocol.json", campaign / "pilot_manifest.csv", campaign / "screen_manifest.csv"]
    records = {str(path.relative_to(campaign)): file_sha256(path) for path in files}
    data_records = {str(path): {"sha256": file_sha256(path), "size": path.stat().st_size,
                              "mtime_ns": path.stat().st_mtime_ns} for path in data_files}
    save_json(campaign / "freeze.json", {"files": records, "data": data_records,
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dirty_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)).hexdigest()}, immutable=True)
    print(json.dumps({"prepared": str(campaign), "counts": protocol["counts"],
                      "runtime_files": len(runtime_files)}, indent=2))


def verify_freeze(campaign, full_data=False):
    frozen = read_json(campaign / "freeze.json")
    for relative, expected in frozen["files"].items():
        if file_sha256(campaign / relative) != expected:
            raise RuntimeError(f"Frozen source/config changed: {relative}")
    for filename, expected in frozen["data"].items():
        path = Path(filename)
        if path.stat().st_size != expected["size"] or path.stat().st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError(f"Frozen dataset changed: {filename}")
        if full_data and file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen dataset hash changed: {filename}")


def validate_failure(job):
    path = job.run_dir / "training_failure.json"
    payload = read_json(path)
    config = read_json(job.run_dir / "effective_config.json")
    launcher._validate_effective_config(config, job.row, job.config)
    if not config.get("stop_on_numerical_failure"):
        raise RuntimeError("Unrequested terminal failure policy")
    if payload.get("schema_version") != 1 or payload.get("status") != "terminal_numerical_failure":
        raise RuntimeError("Invalid numerical failure schema")
    if payload.get("run_id") != job.row["run_id"] or not payload.get("reason") or not payload.get("traceback"):
        raise RuntimeError("Incomplete numerical failure evidence")
    if payload.get("effective_config_checksum") != config_checksum(config):
        raise RuntimeError("Failure config checksum mismatch")
    if not -1 <= int(payload["round"]) < int(job.config["comm_round"]):
        raise RuntimeError("Failure round outside configured horizon")
    evidence = payload.get("evidence_sha256", {})
    if "effective_config.json" not in evidence or "stderr.log" not in evidence:
        raise RuntimeError("Failure evidence lacks config/log hashes")
    for relative, expected in evidence.items():
        target = (job.run_dir / relative).resolve()
        if not target.is_relative_to(job.run_dir.resolve()) or file_sha256(target) != expected:
            raise RuntimeError(f"Failure evidence mismatch: {relative}")
    return {"status": "terminal_numerical_failure", "reason": payload["reason"],
            "round": payload["round"], "phase": payload["phase"]}


def resolve_job(job, returncode=0):
    if (job.run_dir / "infrastructure_failure.json").exists():
        raise RuntimeError("Prior infrastructure failure requires diagnosis; artifacts preserved")
    if (job.run_dir / "training_failure.json").exists():
        return validate_failure(job)
    if (job.run_dir / "pretraining_failure.json").exists():
        launcher.validate_pretraining_failure_artifact(job.run_dir, job.row)
        return {"status": "terminal_numerical_failure", "phase": "model_selection",
                "reason": "Validated terminal model-selection failure"}
    if returncode != 0:
        return {"status": "infrastructure_failure", "returncode": returncode,
                "reason": "Process failed without validated numerical-stop evidence"}
    for name in launcher.EXPECTED_ARTIFACTS:
        if not (job.run_dir / name).is_file():
            raise RuntimeError(f"Missing artifact: {name}")
    config = read_json(job.run_dir / "effective_config.json")
    launcher._validate_effective_config(config, job.row, job.config)
    for key in ("stop_on_numerical_failure", "zo_mu", "zo_num_directions", "objective_mode",
                "objective_lambda_1", "aggregation_weighting", "gradient_clip_norm",
                "auxiliary_regression", "skip_model_selection", "skip_gmm_eval",
                "simple_model_selection_epochs", "f_history_model_selection_epochs",
                "eg_predictor_server_lr", "eg_corrector_server_lr", "compact_predictions_only"):
        if config.get(key) != job.config.get(key):
            raise RuntimeError(f"Resolved configuration mismatch: {key}")
    _, terminal = launcher._validate_round_curve(job.run_dir / "mse_by_round.csv",
                                                int(job.config["comm_round"]), job.row["dataset"])
    metrics = read_json(job.run_dir / "metrics.json")
    if metrics.get("run_status") != "completed" or metrics.get("rounds_completed") != int(job.config["comm_round"]):
        raise RuntimeError("Incomplete training horizon")
    if config.get("test_mse_used_for_selection") or config.get("primary_selection_metric") != "pooled_validation_mse":
        raise RuntimeError("Selection policy mismatch")
    if job.row["stage"] == "final":
        launcher.validate_artifacts(job.run_dir, job.row, expected_config=job.config)
    # Tuning eligibility never reads test metric values, including finiteness.
    return {"status": "terminal_numerical_failure" if terminal or metrics.get("diverged") else "passed"}


def validation_record(job):
    with (job.run_dir / "mse_by_round.csv").open(newline="") as handle:
        curve = list(csv.DictReader(handle))
    values = [float(row["primary_val_mse"]) for row in curve]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError("Invalid validation curve")
    best = min(values)
    metrics = read_json(job.run_dir / "metrics.json")
    if not math.isclose(float(metrics["best_validation_mse"]), best, rel_tol=1e-10):
        raise RuntimeError("Selected validation score does not match round curve")
    return {"run_id": job.row["run_id"], "dataset": job.row["dataset"],
            "alpha": float(job.row["alpha"]), "method": job.row["method"],
            "seed": int(job.row["seed"]), "candidate": int(job.row["candidate"]),
            "best": best, "final": values[-1],
            "final_best_ratio": values[-1] / max(best, 1e-12),
            "last50_std": statistics.pstdev(values[-50:]),
            "rounds": len(values)}


def screen_winners(records):
    winners = []
    for alpha in ALPHAS:
        for dataset in DATASETS:
            for method in METHODS:
                group = [row for row in records if row["status"] == "passed"
                         and row["dataset"] == dataset and float(row["alpha"]) == alpha
                         and row["method"] == method]
                if not group:
                    raise RuntimeError(f"No eligible screening candidate: {dataset}/{alpha}/{method}")
                winners.append(min(group, key=lambda row: (
                    row["best"], row["last50_std"], row["final"] - row["best"], row["candidate"])))
    return winners


def choose_algorithm(records, protocol):
    rules = protocol["promotion"]
    expected = {(dataset, alpha, method, seed) for dataset in DATASETS for alpha in ALPHAS
                for method in METHODS for seed in protocol["confirmation_seeds"]}
    keys = [(row["dataset"], float(row["alpha"]), row["method"], int(row["seed"])) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Confirmation evidence is incomplete or duplicated")
    if any(row["status"] != "passed" for row in records):
        return {"status": "selection_inconclusive", "reason": "Confirmation contains a terminal failure; no unconditional winner promoted"}
    indexed = dict(zip(keys, records))
    assessments = {}
    for method in METHODS:
        other = next(name for name in METHODS if name != method)
        seed_ratios, cell_ratios = [], []
        for seed in protocol["confirmation_seeds"]:
            logs = [math.log(max(indexed[dataset, alpha, method, seed]["best"], 1e-12)
                             / max(indexed[dataset, alpha, other, seed]["best"], 1e-12))
                    for dataset in DATASETS for alpha in ALPHAS]
            seed_ratios.append(math.exp(statistics.mean(logs)))
        for dataset in DATASETS:
            for alpha in ALPHAS:
                means = {name: statistics.mean(indexed[dataset, alpha, name, seed]["best"]
                         for seed in protocol["confirmation_seeds"]) for name in METHODS}
                cell_ratios.append(means[method] / max(means[other], 1e-12))
        deterioration = sorted(row["final_best_ratio"] for row in records if row["method"] == method)
        median = statistics.median(deterioration)
        p90 = deterioration[math.ceil(0.9 * len(deterioration)) - 1]
        passed = (max(seed_ratios) <= 1 - rules["relative_improvement"]
                  and sum(ratio < 1 for ratio in cell_ratios) >= rules["minimum_cell_wins"]
                  and max(cell_ratios) <= rules["maximum_cell_regression_ratio"]
                  and median <= rules["maximum_median_final_best_validation_ratio"]
                  and p90 <= rules["maximum_p90_final_best_validation_ratio"])
        assessments[method] = {"passes": passed, "seed_geometric_mse_ratios": seed_ratios,
                               "cell_wins": sum(ratio < 1 for ratio in cell_ratios),
                               "worst_cell_ratio": max(cell_ratios),
                               "median_final_best_ratio": median, "p90_final_best_ratio": p90}
    winners = [method for method, assessment in assessments.items() if assessment["passes"]]
    return {"status": "selected" if len(winners) == 1 else "selection_inconclusive",
            "winner": winners[0] if len(winners) == 1 else None, "assessments": assessments,
            "selection_metric_source": "validation", "test_mse_used_for_selection": False}


def run_stage(campaign, stage, rows, allocated, watchdog, *, resolver=None, summarizer=None):
    resolver = resolver or resolve_job
    summarizer = summarizer or validation_record
    verify_freeze(campaign)
    write_manifest(campaign / f"{stage}_manifest.csv", rows)
    jobs = make_jobs(campaign, rows)
    pending = list(jobs)
    active, results = [], []
    paused = False
    ledger = campaign / f"{stage}_events.jsonl"
    results_path = campaign / f"{stage}_results.json"
    previous_results = {row["run_id"]: row for row in read_json(results_path)} if results_path.exists() else {}

    def event(payload):
        payload = {"time": time.time(), **payload}
        with ledger.open("a") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    def record(job, outcome, elapsed=None):
        result = {"run_id": job.row["run_id"], "dataset": job.row["dataset"],
                  "alpha": float(job.row["alpha"]), "method": job.row["method"],
                  "seed": int(job.row["seed"]), "candidate": int(job.row["candidate"]),
                  "run_dir": str(job.run_dir), **outcome}
        if outcome["status"] == "passed":
            result.update(summarizer(job))
        if elapsed is not None:
            result["wall_seconds"] = elapsed
        results.append(result)
        save_json(results_path, results)
        event({"event": "resolved", **result})
        print(f"{stage}: {outcome['status']} {job.row['run_id']}", flush=True)

    def stop_children(signum, _frame):
        for entry in active:
            if entry["process"].poll() is None:
                os.killpg(entry["process"].pid, signal.SIGTERM)
        save_json(campaign / "status.json", {"status": "interrupted", "stage": stage,
                  "signal": signum, "time": time.time()})
        raise SystemExit(128 + signum)

    old_handlers = {sig: signal.signal(sig, stop_children) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            while pending and len(active) < len(allocated) and not paused:
                job = pending.pop(0)
                if job.run_dir.exists() and any(job.run_dir.iterdir()):
                    try:
                        outcome = resolver(job)
                        elapsed = previous_results.get(job.row["run_id"], {}).get("wall_seconds")
                        if elapsed is None and outcome["status"] == "passed":
                            elapsed = read_json(job.run_dir / "metrics.json").get("runtime_seconds")
                        record(job, outcome, elapsed)
                        paused = outcome["status"] == "infrastructure_failure"
                    except Exception as exc:
                        record(job, {"status": "infrastructure_failure", "reason": f"Existing artifacts preserved: {exc}"})
                        paused = True
                    continue
                verify_freeze(campaign)
                occupied = {entry["gpu"] for entry in active}
                gpu = next(gpu for gpu in allocated if gpu not in occupied)
                job.run_dir.mkdir(parents=True)
                launcher.write_config(job.config_path, job.config)
                env = dict(job.env)
                env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                            "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                            "OPENBLAS_NUM_THREADS": "4", "FEDGMM_PROFILE_RUNTIME": "1",
                            "FEDGMM_PROFILE_ROOT": str(campaign / "profiles"),
                            "FEDGMM_PROFILE_GPU_TELEMETRY": "0",
                            "FEDGMM_HASH_BUNDLE_ID": "training_hashes.json"})
                stdout = (job.run_dir / "stdout.log").open("w")
                stderr = (job.run_dir / "stderr.log").open("w")
                process = subprocess.Popen(job.command, cwd=campaign / "runtime" / EXAMPLE_REL,
                                           env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                active.append({"job": job, "process": process, "stdout": stdout, "stderr": stderr,
                               "gpu": gpu, "started": time.monotonic(), "timed_out": None})
                event({"event": "started", "run_id": job.row["run_id"], "pid": process.pid,
                       "gpu": gpu, "logical_gpu": 0, "config": str(job.config_path),
                       "config_sha256": file_sha256(job.config_path)})
                print(f"{stage}: START {job.row['run_id']} GPU={gpu} PID={process.pid}", flush=True)
            for entry in list(active):
                job, process = entry["job"], entry["process"]
                elapsed = time.monotonic() - entry["started"]
                curve = job.run_dir / "mse_by_round.csv"
                if process.poll() is None:
                    reason = None
                    if elapsed > watchdog["total_seconds"]:
                        reason = "run wall-time limit exceeded"
                    elif curve.exists() and time.time() - curve.stat().st_mtime > watchdog["stall_seconds"]:
                        reason = "round CSV stopped progressing"
                    elif not curve.exists() and elapsed > watchdog["startup_seconds"]:
                        reason = "no first round within startup limit"
                    if reason and entry["timed_out"] is None:
                        entry["timed_out"] = time.monotonic()
                        entry["timeout_reason"] = reason
                        os.killpg(process.pid, signal.SIGTERM)
                    elif entry["timed_out"] and time.monotonic() - entry["timed_out"] > 15:
                        os.killpg(process.pid, signal.SIGKILL)
                    continue
                entry["stdout"].close()
                entry["stderr"].close()
                try:
                    if entry["timed_out"]:
                        outcome = {"status": "infrastructure_failure", "reason": entry["timeout_reason"]}
                    else:
                        outcome = resolver(job, process.returncode)
                except Exception as exc:
                    outcome = {"status": "infrastructure_failure", "reason": str(exc)}
                if outcome["status"] == "infrastructure_failure":
                    save_json(job.run_dir / "infrastructure_failure.json", outcome)
                    paused = True
                record(job, outcome, elapsed)
                active.remove(entry)
            save_json(campaign / "status.json", {"status": "pausing_after_failure" if paused else "running",
                      "stage": stage, "resolved": len(results), "total": len(jobs),
                      "active": [{"run_id": item["job"].row["run_id"], "pid": item["process"].pid,
                                  "gpu": item["gpu"]} for item in active], "pending": len(pending),
                      "time": time.time()})
            if paused and not active:
                raise RuntimeError(f"{stage}: infrastructure/evidence failure; pending work preserved")
            if active:
                time.sleep(2)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for entry in active:
            if entry["process"].poll() is None:
                os.killpg(entry["process"].pid, signal.SIGTERM)
            entry["stdout"].close()
            entry["stderr"].close()
    return results


def final_report(campaign, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for result in results:
        if result["status"] != "passed":
            rows.append(dict(result))
            continue
        run_dir = Path(result["run_dir"])
        metrics = read_json(run_dir / "metrics.json")
        rows.append({**result, "test_mse_at_best_validation": metrics["test_mse_at_best_validation"],
                     "final_test_mse": metrics["final_test_mse"]})
        with (run_dir / "mse_by_round.csv").open(newline="") as handle:
            curve = list(csv.DictReader(handle))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([int(row["round"]) for row in curve], [float(row["primary_val_mse"]) for row in curve])
        axes[0].set(xlabel="Communication round", ylabel="Validation structural MSE", yscale="log")
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as predictions:
            truth = predictions["true_g"].reshape(-1)
            estimate = predictions["best_validation_prediction"].reshape(-1)
            coordinate = predictions["sample_coordinate"].reshape(-1)
            np.save(run_dir / "plotting_arrays.npy", np.column_stack([coordinate, estimate, truth]), allow_pickle=False)
            axes[1].scatter(truth, estimate, s=3, alpha=0.25)
            low, high = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
            axes[1].plot([low, high], [low, high], color="black", linestyle="--")
            axes[1].set(xlabel="True structural value", ylabel="Validation-selected prediction")
        fig.suptitle(result["run_id"])
        fig.tight_layout()
        fig.savefig(run_dir / f"federated_{result['dataset']}_{result['method']}_stochastic_seed{result['seed']}.png", dpi=160)
        plt.close(fig)
    write_manifest(campaign / "final_metrics.csv", rows)
    aggregates = []
    for alpha in ALPHAS:
        for dataset in DATASETS:
            cell = [row for row in rows if row["dataset"] == dataset and row["alpha"] == alpha]
            values = [row["test_mse_at_best_validation"] for row in cell if row["status"] == "passed"]
            aggregates.append({"dataset": dataset, "alpha": alpha, "planned": len(cell),
                               "passed": len(values), "failures": len(cell) - len(values),
                               "test_mse_mean": statistics.mean(values) if len(values) == len(cell) else None,
                               "test_mse_std": statistics.stdev(values) if len(values) == len(cell) and len(values) > 1 else None})
    write_manifest(campaign / "final_aggregate.csv", aggregates)


def pilot_projection(campaign, pilots):
    measurements = {}
    for row in pilots:
        profile = read_json(campaign / "profiles" / row["dataset"] / row["method"]
                            / f"seed_{row['seed']}" / row["run_id"] / "profile_summary.json")
        seconds = profile["phase_totals_seconds"]["round_total"]
        per_round = seconds / row["rounds"]
        overhead = max(0.0, row["wall_seconds"] - seconds)
        directions = candidates(row["method"])[row["candidate"]]["zo_num_directions"]
        measurements[row["dataset"], row["method"], directions] = (overhead, per_round)

    def estimate(dataset, method, directions, rounds):
        overhead, per_round = measurements[dataset, method, directions]
        return overhead + rounds * per_round

    screen_seconds = sum(estimate(dataset, method, candidate["zo_num_directions"], 150)
                         for dataset in DATASETS for method in METHODS
                         for candidate in candidates(method)) * len(ALPHAS)
    confirmation_seconds = sum(max(estimate(dataset, method, candidate["zo_num_directions"], 500)
                                   for candidate in candidates(method))
                               for dataset in DATASETS for method in METHODS) * len(ALPHAS) * 2
    final_hours = {method: sum(max(estimate(dataset, method, candidate["zo_num_directions"], 1500)
                                    for candidate in candidates(method)) for dataset in DATASETS)
                  * len(ALPHAS) * 5 / 3600 for method in METHODS}
    return {"basis": "25-round alpha=0.5 pilots; startup separated from round cost; candidate maxima for unselected stages",
            "screen_gpu_hours": screen_seconds / 3600,
            "confirmation_gpu_hours_upper_candidate_estimate": confirmation_seconds / 3600,
            "final_gpu_hours_upper_candidate_estimate": final_hours,
            "ideal_remaining_two_gpu_hours": {method: (screen_seconds / 3600 + confirmation_seconds / 3600 + hours) / 2
                                              for method, hours in final_hours.items()},
            "allowance": "Allow at least 30% timing variation; excludes broker queue and preemption."}


def run(campaign, through):
    if ROOT != campaign / "runtime":
        raise RuntimeError("Launch the frozen runtime/scripts/run_stochastic_eg_campaign.py, not the live source")
    protocol = read_json(campaign / "protocol.json")
    allocated = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(allocated) != 2:
        raise RuntimeError("Run this campaign through gpurun -g 2; exactly two assigned GPUs are required")
    verify_freeze(campaign, full_data=True)
    pilots = run_stage(campaign, "pilot", launcher._load_rows(campaign / "pilot_manifest.csv"), allocated, protocol["watchdog"])
    if any(row["status"] != "passed" for row in pilots):
        raise RuntimeError("Pilot numerical failure; screening held for review")
    timing = pilot_projection(campaign, pilots)
    save_json(campaign / "pilot_timing.json", timing)
    print(json.dumps({"measured_pilot_projection": timing}, indent=2), flush=True)
    if through == "pilot":
        save_json(campaign / "status.json", {"status": "pilot_complete", "time": time.time()})
        return
    records = run_stage(campaign, "screen", launcher._load_rows(campaign / "screen_manifest.csv"), allocated, protocol["watchdog"])
    winners = screen_winners(records)
    save_json(campaign / "screen_winners.json", winners, immutable=True)
    confirm = []
    for winner in winners:
        for seed in protocol["confirmation_seeds"]:
            confirm.append(make_row(campaign, protocol["sources"][winner["dataset"]], "confirm",
                                   winner["dataset"], winner["alpha"], winner["method"], seed,
                                   candidates(winner["method"])[winner["candidate"]], 500))
    records = run_stage(campaign, "confirm", confirm, allocated, protocol["watchdog"])
    decision = choose_algorithm(records, protocol)
    save_json(campaign / "selection_report.json", decision, immutable=True)
    if decision["status"] != "selected":
        save_json(campaign / "status.json", {**decision, "stage": "selection", "time": time.time()})
        print("SELECTION INCONCLUSIVE: finals not launched; see selection_report.json", flush=True)
        return
    final = []
    for winner in winners:
        if winner["method"] != decision["winner"]:
            continue
        for seed in protocol["final_seeds"]:
            final.append(make_row(campaign, protocol["sources"][winner["dataset"]], "final",
                                 winner["dataset"], winner["alpha"], winner["method"], seed,
                                 candidates(winner["method"])[winner["candidate"]], 1500))
    results = run_stage(campaign, "final", final, allocated, protocol["watchdog"])
    final_report(campaign, results)
    save_json(campaign / "status.json", {"status": "finals_complete", "winner": decision["winner"],
              "passed": sum(row["status"] == "passed" for row in results), "total": len(results), "time": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--through", choices=("pilot", "final"), default="final")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "prepare":
        prepare(campaign)
    elif args.action == "verify":
        verify_freeze(campaign, full_data=True)
        print("Frozen sources, configs and datasets verified")
    else:
        with (campaign / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(campaign, args.through)
            except Exception as exc:
                save_json(campaign / "status.json", {"status": "paused", "reason": str(exc), "time": time.time()})
                raise


if __name__ == "__main__":
    main()
