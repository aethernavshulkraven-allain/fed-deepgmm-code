#!/usr/bin/env python3
"""Run the FedOGDA-S focused v3 tuning pipeline stage by stage.

This supervisor is intentionally conservative. It will not start a duplicate
stage if a matching ``run_manifest.py`` process is already active. Failed
model-selection rows are handled by the analyzer as invalid candidates via
``--allow-partial``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_focused_v3"
EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
CONFIG_DIR = EXP_DIR / "generated_configs"
OUTPUT_ROOT = ROOT / "results" / "curve_fitting_tuning" / SCREEN_NAME
LOG_DIR = ROOT / "logs" / SCREEN_NAME
PYTHON = "python"
RUN_MANIFEST = ROOT / "scripts" / "run_manifest.py"
ANALYZER = ROOT / "scripts" / "analyze_fedogda_s_focused_v3.py"

ENV_PREFIX = [
    "OMP_NUM_THREADS=4",
    "MKL_NUM_THREADS=4",
    "OPENBLAS_NUM_THREADS=4",
    "NUMEXPR_NUM_THREADS=4",
    "VECLIB_MAXIMUM_THREADS=4",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metrics_count(manifest: Path) -> int:
    count = 0
    for row in read_rows(manifest):
        result_dir = ROOT / row["final_result_dir"]
        if (result_dir / "metrics.json").exists():
            count += 1
    return count


def has_active_stage(results_json: Path) -> bool:
    proc = subprocess.run(["pgrep", "-af", "run_manifest.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    marker = str(results_json)
    rel_marker = rel(results_json)
    for line in proc.stdout.splitlines():
        if SCREEN_NAME in line and (marker in line or rel_marker in line):
            return True
    return False


def run_checked(command: list[str], *, log_path: Path | None = None) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        print("+ " + " ".join(command), flush=True)
        return subprocess.run(command, cwd=ROOT).returncode
    with log_path.open("a") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def launch_stage(name: str, manifest: Path, results_json: Path) -> None:
    rows = read_rows(manifest)
    if not rows:
        raise SystemExit(f"{rel(manifest)} is missing or empty")
    if has_active_stage(results_json):
        print(f"{name}: already active; waiting for existing launcher", flush=True)
        return
    command = [
        "/usr/local/bin/gpurun",
        "-g",
        "1",
        "env",
        *ENV_PREFIX,
        PYTHON,
        str(RUN_MANIFEST),
        "--manifest",
        rel(manifest),
        "--config-dir",
        rel(CONFIG_DIR),
        "--output-root",
        rel(OUTPUT_ROOT),
        "--gpu-ids",
        "0",
        "--max-parallel",
        "1",
        "--resume-skip-completed",
        "--results-json",
        rel(results_json),
        "--keep-going",
    ]
    code = run_checked(command, log_path=LOG_DIR / f"{name}.log")
    print(f"{name}: launcher exited with code {code}", flush=True)


def wait_for_stage(name: str, manifest: Path, results_json: Path, poll_seconds: int) -> None:
    expected = len(read_rows(manifest))
    while True:
        completed = metrics_count(manifest)
        active = has_active_stage(results_json)
        print(f"{name}: completed_metrics={completed}/{expected}, active={active}", flush=True)
        if not active and results_json.exists():
            return
        if not active and completed >= expected:
            return
        time.sleep(poll_seconds)


def analyze(stage: str, *, materialize_next: bool) -> None:
    command = [PYTHON, str(ANALYZER), "--stage", stage, "--allow-partial"]
    if materialize_next:
        command.append("--materialize-next")
    code = run_checked(command, log_path=LOG_DIR / f"analyze_{stage}.log")
    if code != 0:
        raise SystemExit(f"analysis failed for {stage}; see {rel(LOG_DIR / f'analyze_{stage}.log')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--start-at", choices=["screen", "confirm_seed0", "confirm_seeds", "final"], default="screen")
    args = parser.parse_args()

    stages = {
        "screen": (EXP_DIR / "manifest.csv", EXP_DIR / "run_results_screen.json"),
        "confirm_seed0": (EXP_DIR / "confirm_seed0_manifest.csv", EXP_DIR / "run_results_confirm_seed0.json"),
        "confirm_seeds": (EXP_DIR / "confirm_seeds_manifest.csv", EXP_DIR / "run_results_confirm_seeds.json"),
    }

    if args.start_at in {"screen"}:
        launch_stage("screen", *stages["screen"])
        wait_for_stage("screen", *stages["screen"], poll_seconds=args.poll_seconds)
        analyze("screen", materialize_next=True)

    if args.start_at in {"screen", "confirm_seed0"}:
        launch_stage("confirm_seed0", *stages["confirm_seed0"])
        wait_for_stage("confirm_seed0", *stages["confirm_seed0"], poll_seconds=args.poll_seconds)
        analyze("confirm_seed0", materialize_next=True)

    if args.start_at in {"screen", "confirm_seed0", "confirm_seeds"}:
        launch_stage("confirm_seeds", *stages["confirm_seeds"])
        wait_for_stage("confirm_seeds", *stages["confirm_seeds"], poll_seconds=args.poll_seconds)

    analyze("final", materialize_next=False)
    summary = {
        "screen_metrics": metrics_count(stages["screen"][0]),
        "confirm_seed0_metrics": metrics_count(stages["confirm_seed0"][0]),
        "confirm_seeds_metrics": metrics_count(stages["confirm_seeds"][0]),
        "final_selected": rel(ROOT / "experiments" / "curve_fitting_plots" / "csv" / f"{SCREEN_NAME}_final_selected.csv"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
