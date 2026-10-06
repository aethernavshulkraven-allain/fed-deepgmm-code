#!/usr/bin/env python3
"""Run rerun_protocol_v1 federated experiments in guarded sequential waves.

This is a small orchestration wrapper over ``scripts/run_manifest.py``.  It is
meant for long unattended launches: run the missing/guard pilot first, then run
method waves in a fixed order.  Each phase blocks until completion and the next
phase starts only if the previous phase exits cleanly.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "launch_candidate" / "manifest.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "queue_results"
RUN_MANIFEST = REPO_ROOT / "scripts" / "run_manifest.py"


@dataclass(frozen=True)
class Phase:
    name: str
    filters: tuple[str, ...]
    description: str


PHASES: tuple[Phase, ...] = (
    Phase(
        name="pilot_abs_seed0_alpha0p1",
        filters=("dataset=abs", "seed=0", "alpha=0.1"),
        description="Guard pilot; completed pilot rows are resume-skipped, missing rows run.",
    ),
    Phase(
        name="wave_fedgda_s",
        filters=("method=fedgda_s",),
        description="All stochastic FedGDA rows.",
    ),
    Phase(
        name="wave_fedogda_s",
        filters=("method=fedogda_s",),
        description="All stochastic FedOGDA rows.",
    ),
    Phase(
        name="wave_fedgda_d",
        filters=("method=fedgda_d",),
        description="All deterministic FedGDA rows.",
    ),
    Phase(
        name="wave_fedogda_d",
        filters=("method=fedogda_d",),
        description="All deterministic FedOGDA rows.",
    ),
)


def _phase_names() -> list[str]:
    return [phase.name for phase in PHASES]


def _select_phases(args: argparse.Namespace) -> list[Phase]:
    phases = list(PHASES)
    if args.skip_pilot:
        phases = [phase for phase in phases if not phase.name.startswith("pilot_")]
    if args.pilot_only:
        phases = [phase for phase in phases if phase.name.startswith("pilot_")]
    if args.stochastic_only:
        phases = [phase for phase in phases if phase.name in {"wave_fedgda_s", "wave_fedogda_s"}]
    if args.deterministic_only:
        phases = [phase for phase in phases if phase.name in {"wave_fedgda_d", "wave_fedogda_d"}]

    if args.start_at:
        names = _phase_names()
        start_index = names.index(args.start_at)
        allowed = set(names[start_index:])
        phases = [phase for phase in phases if phase.name in allowed]

    if args.stop_after:
        names = _phase_names()
        stop_index = names.index(args.stop_after)
        allowed = set(names[: stop_index + 1])
        phases = [phase for phase in phases if phase.name in allowed]

    return phases


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _phase_command(args: argparse.Namespace, phase: Phase, timestamp: str) -> list[str]:
    results_json = Path(args.results_dir) / f"{timestamp}_{phase.name}.json"
    if not results_json.is_absolute():
        results_json = REPO_ROOT / results_json

    command = [
        args.python,
        str(RUN_MANIFEST),
        "--manifest",
        str(args.manifest),
        "--gpu-ids",
        args.gpu_ids,
        "--max-parallel",
        str(args.max_parallel),
        "--resume-skip-completed",
        "--results-json",
        str(results_json),
    ]
    for item in phase.filters:
        command.extend(["--only", item])
    if args.overwrite_incomplete:
        command.append("--overwrite-incomplete")
    if args.keep_going_within_phase:
        command.append("--keep-going")
    if args.dry_run:
        command.append("--dry-run")
        if args.dry_run_limit is not None:
            command.extend(["--limit", str(args.dry_run_limit)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rerun_protocol_v1 in pilot/stochastic/deterministic waves."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Default is 1 for true one-after-another queueing; use 2 for one job per H100.",
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=3,
        help="Limit rows shown per phase during dry-run. Use -1 to show all.",
    )
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--stochastic-only", action="store_true")
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--start-at", choices=_phase_names())
    parser.add_argument("--stop-after", choices=_phase_names())
    parser.add_argument(
        "--overwrite-incomplete",
        action="store_true",
        help="Allow rerunning directories with partial artifacts. Completed rows are still skipped.",
    )
    parser.add_argument(
        "--keep-going-within-phase",
        action="store_true",
        help="Do not stop a run_manifest phase at the first row failure. The wrapper still stops before the next phase if the phase returns nonzero.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    args.manifest = manifest

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    args.results_dir = results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run_limit is not None and args.dry_run_limit < 0:
        args.dry_run_limit = None

    phases = _select_phases(args)
    if not phases:
        print("No phases selected.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "DRY RUN" if args.dry_run else "REAL LAUNCH"
    print(f"{mode}: {len(phases)} phase(s)", flush=True)
    print(f"manifest={manifest}", flush=True)
    print(f"gpu_ids={args.gpu_ids} max_parallel={args.max_parallel}", flush=True)
    print(f"results_dir={results_dir}", flush=True)

    for index, phase in enumerate(phases, start=1):
        print("", flush=True)
        print(f"=== Phase {index}/{len(phases)}: {phase.name} ===", flush=True)
        print(phase.description, flush=True)
        command = _phase_command(args, phase, timestamp)
        print(_quote_command(command), flush=True)
        result = subprocess.run(command, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"STOP: phase {phase.name} failed with return code {result.returncode}", file=sys.stderr, flush=True)
            return result.returncode

    print("", flush=True)
    print("All selected phases completed cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
