#!/usr/bin/env python3
"""Run the fast Step FedOGDA-S v5 pipeline stage by stage."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_step_fast_v5"
EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
CONFIG_DIR = EXP_DIR / "generated_configs"
OUTPUT_ROOT = ROOT / "results" / "curve_fitting_tuning" / SCREEN_NAME
LOG_DIR = ROOT / "logs" / SCREEN_NAME
PYTHON = "python"
RUN_MANIFEST = ROOT / "scripts" / "run_manifest.py"
ANALYZER = ROOT / "scripts" / "analyze_fedogda_s_step_fast_v5.py"

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
        if (ROOT / row["final_result_dir"] / "metrics.json").exists():
            count += 1
    return count


def has_active_stage(results_json: Path) -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", "run_manifest.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
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


def launch_stage(name: str, manifest: Path, results_json: Path) -> bool:
    rows = read_rows(manifest)
    if not rows:
        print(f"{name}: no rows to launch", flush=True)
        return False
    if has_active_stage(results_json):
        print(f"{name}: already active; waiting for existing launcher", flush=True)
        return True
    command = [
        "/usr/local/bin/gpurun",
        "-g",
        "2",
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
        "0,1",
        "--max-parallel",
        "2",
        "--resume-skip-completed",
        "--results-json",
        rel(results_json),
        "--keep-going",
    ]
    code = run_checked(command, log_path=LOG_DIR / f"{name}.log")
    print(f"{name}: launcher exited with code {code}", flush=True)
    if code != 0:
        raise SystemExit(f"{name} launcher failed; see {rel(LOG_DIR / f'{name}.log')}")
    return True


def wait_for_stage(name: str, manifest: Path, results_json: Path, poll_seconds: int) -> None:
    expected = len(read_rows(manifest))
    if expected == 0:
        print(f"{name}: skipped empty manifest", flush=True)
        return
    while True:
        completed = metrics_count(manifest)
        active = has_active_stage(results_json)
        print(f"{name}: completed_metrics={completed}/{expected}, active={active}", flush=True)
        if not active and results_json.exists():
            return
        if not active and completed >= expected:
            return
        time.sleep(poll_seconds)


def analyze(materialize_next: bool, elapsed_seconds: int | None = None) -> None:
    command = [PYTHON, str(ANALYZER)]
    if materialize_next:
        command.append("--materialize-next")
    if elapsed_seconds is not None:
        command.extend(["--elapsed-seconds", str(elapsed_seconds)])
    code = run_checked(command, log_path=LOG_DIR / "analyze.log")
    if code != 0:
        raise SystemExit(f"analysis failed; see {rel(LOG_DIR / 'analyze.log')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    started_at = time.time()

    stage_a = ("stage_a", EXP_DIR / "stage_a_manifest.csv", EXP_DIR / "run_results_stage_a.json")
    stage_b = ("stage_b", EXP_DIR / "stage_b_manifest.csv", EXP_DIR / "run_results_stage_b.json")

    launch_stage(*stage_a)
    wait_for_stage(*stage_a, poll_seconds=args.poll_seconds)
    analyze(materialize_next=True)

    launched_b = launch_stage(*stage_b)
    if launched_b:
        wait_for_stage(*stage_b, poll_seconds=args.poll_seconds)
    elapsed_seconds = int(time.time() - started_at)
    analyze(materialize_next=False, elapsed_seconds=elapsed_seconds)

    summary = {
        "elapsed_seconds": elapsed_seconds,
        "stage_a_metrics": metrics_count(stage_a[1]),
        "stage_b_metrics": metrics_count(stage_b[1]),
        "stage_b_rows": len(read_rows(stage_b[1])),
        "final_selected": rel(
            ROOT
            / "experiments"
            / "curve_fitting_plots"
            / "csv"
            / f"{SCREEN_NAME}_final_selected.csv"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
