#!/usr/bin/env python3
"""Materialize and run opt-in stochastic GPU utilization profiling jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1"
DEFAULT_SOURCE = PROTOCOL_DIR / "alpha1" / "final_manifest_stochastic.csv"
PROFILE_DIR = PROTOCOL_DIR / "gpu_util_profiling"
PROFILE_ROOT = REPO_ROOT / "results" / "_profiling" / "highdim_stochastic_gpu_util"
DEFAULT_DATASETS = ("cifar10_x", "cifar10_xz", "cifar10_z")
DEFAULT_METHODS = ("fedgda_s", "fedogda_s")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def materialize_manifest(
    *,
    source: Path,
    phase: str,
    rounds: int,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    seed: int,
    disable_aux_reg: bool,
    aux_reg_epochs: int | None,
    skip_model_selection: bool,
    disable_periodic_checkpoints: bool,
    periodic_checkpoint_interval: int | None,
    dataloader_num_workers: int | None,
    dataloader_pin_memory: bool,
    append_round_csv: bool,
    legacy_reg_state_cpu: bool,
) -> Path:
    source_rows = read_rows(source)
    selected = [
        row
        for row in source_rows
        if row["dataset"] in datasets
        and row["method"] in methods
        and int(row["seed"]) == int(seed)
    ]
    expected = len(datasets) * len(methods)
    if len(selected) != expected:
        found = sorted((row["dataset"], row["method"], row["seed"]) for row in selected)
        raise SystemExit(
            f"Expected {expected} profile rows from {source}, found {len(selected)}: {found}"
        )

    output_root = PROFILE_ROOT / phase
    manifest_path = PROFILE_DIR / phase / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0].keys())
    runtime_fields = [
        "skip_model_selection",
        "skip_gmm_eval",
        "gmm_eval_proxy",
        "auxiliary_regression",
        "auxiliary_regression_epochs",
        "auxiliary_regression_state_device",
        "append_round_csv",
        "periodic_checkpoint_interval",
        "dataloader_num_workers",
        "dataloader_pin_memory",
    ]
    for field in runtime_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    rows = []
    for row in selected:
        new_row = dict(row)
        source_run_id = row["run_id"]
        run_id = f"{phase}_{source_run_id}_rounds{int(rounds)}"
        new_row["run_id"] = run_id
        new_row["protocol_version"] = f"{row['protocol_version']}_gpu_util_{phase}"
        new_row["run_group"] = f"gpu_util_{phase}"
        new_row["comm_round"] = str(int(rounds))
        new_row["output_root"] = str(output_root.relative_to(REPO_ROOT))
        new_row["final_result_dir"] = str(
            output_root
            / row["dataset"]
            / row["method"]
            / f"seed_{int(seed)}"
            / run_id
        )
        new_row["run_status"] = "profiling"
        new_row["skip_model_selection"] = "true" if skip_model_selection else row.get("skip_model_selection", "")
        if skip_model_selection:
            new_row["skip_gmm_eval"] = "true"
            new_row["gmm_eval_proxy"] = "negative_val_mse"
        elif row.get("skip_gmm_eval"):
            new_row["skip_gmm_eval"] = row.get("skip_gmm_eval", "")
            new_row["gmm_eval_proxy"] = row.get("gmm_eval_proxy", "")
        new_row["auxiliary_regression"] = "false" if disable_aux_reg else row.get("auxiliary_regression", "false")
        if disable_aux_reg:
            new_row["auxiliary_regression_epochs"] = "0"
        elif aux_reg_epochs is not None:
            new_row["auxiliary_regression_epochs"] = str(int(aux_reg_epochs))
        else:
            new_row["auxiliary_regression_epochs"] = row.get("auxiliary_regression_epochs", "")
        new_row["auxiliary_regression_state_device"] = "cpu" if legacy_reg_state_cpu else row.get("auxiliary_regression_state_device", "device")
        new_row["append_round_csv"] = "true" if append_round_csv else "false"
        if disable_periodic_checkpoints:
            new_row["periodic_checkpoint_interval"] = "0"
        elif periodic_checkpoint_interval is not None:
            new_row["periodic_checkpoint_interval"] = str(int(periodic_checkpoint_interval))
        else:
            new_row["periodic_checkpoint_interval"] = row.get("periodic_checkpoint_interval", "")
        if dataloader_num_workers is not None:
            new_row["dataloader_num_workers"] = str(int(dataloader_num_workers))
        else:
            new_row["dataloader_num_workers"] = row.get("dataloader_num_workers", "")
        new_row["dataloader_pin_memory"] = "true" if dataloader_pin_memory else row.get("dataloader_pin_memory", "")
        runtime_notes = []
        if skip_model_selection:
            runtime_notes.append("skip_model_selection=true")
        if disable_aux_reg:
            runtime_notes.append("auxiliary_regression=false")
        elif aux_reg_epochs is not None:
            runtime_notes.append(f"auxiliary_regression_epochs={int(aux_reg_epochs)}")
        if disable_periodic_checkpoints:
            runtime_notes.append("periodic_checkpoint_interval=0")
        elif periodic_checkpoint_interval is not None:
            runtime_notes.append(f"periodic_checkpoint_interval={int(periodic_checkpoint_interval)}")
        if dataloader_num_workers is not None:
            runtime_notes.append(f"dataloader_num_workers={int(dataloader_num_workers)}")
        if dataloader_pin_memory:
            runtime_notes.append("dataloader_pin_memory=true")
        if not append_round_csv:
            runtime_notes.append("append_round_csv=false")
        if legacy_reg_state_cpu:
            runtime_notes.append("auxiliary_regression_state_device=cpu")
        new_row["notes"] = (
            f"GPU-utilization profiling copy of {source_run_id}; "
            f"{rounds} rounds; output isolated under results/_profiling. "
            f"Runtime overrides: {', '.join(runtime_notes) if runtime_notes else 'none'}. "
            + row.get("notes", "")
        )
        rows.append(new_row)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": phase,
        "rounds": int(rounds),
        "source_manifest": str(source.relative_to(REPO_ROOT)),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "datasets": list(datasets),
        "methods": list(methods),
        "seed": int(seed),
        "rows": len(rows),
        "runtime_overrides": {
            "skip_model_selection": bool(skip_model_selection),
            "disable_aux_reg": bool(disable_aux_reg),
            "aux_reg_epochs": aux_reg_epochs,
            "disable_periodic_checkpoints": bool(disable_periodic_checkpoints),
            "periodic_checkpoint_interval": periodic_checkpoint_interval,
            "dataloader_num_workers": dataloader_num_workers,
            "dataloader_pin_memory": bool(dataloader_pin_memory),
            "append_round_csv": bool(append_round_csv),
            "legacy_reg_state_cpu": bool(legacy_reg_state_cpu),
        },
    }
    with (manifest_path.parent / "manifest_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def run_manifest(
    manifest: Path,
    *,
    phase: str,
    dry_run: bool,
    max_parallel: int,
    gpu_ids: str,
    overwrite_incomplete: bool,
    profile_batches: bool,
    skip_aux_reg: bool,
) -> int:
    output_root = PROFILE_ROOT / phase
    config_dir = PROFILE_DIR / phase / "generated_configs"
    results_json = PROFILE_DIR / phase / "launcher_results.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_manifest.py"),
        "--manifest",
        str(manifest),
        "--config-dir",
        str(config_dir),
        "--output-root",
        str(output_root),
        "--gpu-ids",
        gpu_ids,
        "--max-parallel",
        str(int(max_parallel)),
        "--resume-skip-completed",
        "--keep-going",
        "--results-json",
        str(results_json),
    ]
    if dry_run:
        command.append("--dry-run")
    if overwrite_incomplete:
        command.append("--overwrite-incomplete")

    env = dict(os.environ)
    env["FEDGMM_PROFILE_RUNTIME"] = "1"
    env["FEDGMM_PROFILE_GPU_TELEMETRY"] = "1"
    env["FEDGMM_PROFILE_ROOT"] = str(output_root)
    env["FEDGMM_PROFILE_BATCHES"] = "1" if profile_batches else "0"
    env["FEDGMM_PROFILE_SKIP_AUX_REG"] = "1" if skip_aux_reg else "0"
    env.setdefault("FEDGMM_PROFILE_GPU_INTERVAL_SECONDS", "1")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--phase", default="baseline50")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument(
        "--batch-profile",
        action="store_true",
        help="Enable per-batch transfer/compute timers. This is higher overhead and should be used only for focused diagnostics.",
    )
    parser.add_argument(
        "--skip-aux-reg",
        action="store_true",
        help="Compatibility alias for --disable-aux-reg.",
    )
    parser.add_argument(
        "--disable-aux-reg",
        action="store_true",
        help="Diagnostic only: disable auxiliary regression train/aggregate/set work in the generated profile manifest.",
    )
    parser.add_argument(
        "--aux-reg-epochs",
        type=int,
        default=None,
        help="Override auxiliary regression local passes for profile rows.",
    )
    parser.add_argument(
        "--skip-model-selection",
        action="store_true",
        help="Diagnostic final-run fast path: bypass model-selection setup and use validation-MSE proxy for GMM eval.",
    )
    parser.add_argument(
        "--disable-periodic-checkpoints",
        action="store_true",
        help="Set periodic_checkpoint_interval=0 for profile rows; best/final checkpoints are still written.",
    )
    parser.add_argument("--periodic-checkpoint-interval", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--dataloader-pin-memory", action="store_true")
    parser.add_argument(
        "--rewrite-round-csv",
        action="store_true",
        help="Use the legacy full CSV rewrite each round instead of append-only writes.",
    )
    parser.add_argument(
        "--legacy-reg-state-cpu",
        action="store_true",
        help="Profile the old behavior that moves auxiliary regression state dicts through CPU.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = REPO_ROOT / source
    manifest = materialize_manifest(
        source=source,
        phase=args.phase,
        rounds=int(args.rounds),
        datasets=parse_csv_tuple(args.datasets),
        methods=parse_csv_tuple(args.methods),
        seed=int(args.seed),
        disable_aux_reg=bool(args.disable_aux_reg or args.skip_aux_reg),
        aux_reg_epochs=args.aux_reg_epochs,
        skip_model_selection=bool(args.skip_model_selection),
        disable_periodic_checkpoints=bool(args.disable_periodic_checkpoints),
        periodic_checkpoint_interval=args.periodic_checkpoint_interval,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=bool(args.dataloader_pin_memory),
        append_round_csv=not bool(args.rewrite_round_csv),
        legacy_reg_state_cpu=bool(args.legacy_reg_state_cpu),
    )
    return run_manifest(
        manifest,
        phase=args.phase,
        dry_run=bool(args.dry_run),
        max_parallel=int(args.max_parallel),
        gpu_ids=str(args.gpu_ids),
        overwrite_incomplete=bool(args.overwrite_incomplete),
        profile_batches=bool(args.batch_profile),
        skip_aux_reg=bool(args.disable_aux_reg or args.skip_aux_reg),
    )


if __name__ == "__main__":
    raise SystemExit(main())
