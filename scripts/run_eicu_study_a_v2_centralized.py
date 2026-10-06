#!/usr/bin/env python3
"""Run the centralized rows from a frozen Study A v2 final manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_centralized_lowdim.py"
METHOD = {"gda_d": "gda", "sgda_s": "sgda", "oadam_s": "oadam"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("training_scope") == "centralized"
            and row.get("role") == "centralized_baseline"
        ]


def build_command(
    row: dict[str, str],
    *,
    python: str,
    output_root: Path,
    overwrite: bool,
    iterations: int | None,
) -> tuple[list[str], Path]:
    result_dir = output_root / row["result_path"]
    batch_size = int(row["batch_size"])
    g_lr = float(row["learning_rate"])
    f_lr = g_lr * float(row["critic_multiplier"])
    command = [
        python,
        str(RUNNER),
        "--dataset",
        row["dataset"],
        "--scenario-name",
        row["scenario_name"],
        "--method",
        METHOD[row["method"]],
        "--seed",
        row["optimizer_seed"],
        "--scenario-seed",
        row["scenario_seed"],
        "--seed-pair-id",
        row["seed_pair_id"],
        "--protocol-version",
        row["protocol_version"],
        "--role",
        row["role"],
        "--g0",
        row["g0"],
        "--alignment-label",
        row["alignment_label"],
        "--primary-selection-metric",
        row["primary_selection_metric"],
        "--selection-source",
        row["selection_source"],
        "--scenario-scope",
        row["scenario_scope"],
        "--study-claim",
        row["study_claim"],
        "--scenario-checksum",
        row["scenario_checksum"],
        "--objective-mode",
        row["objective_mode"],
        "--output-dir",
        str(result_dir),
        "--iterations",
        str(iterations if iterations is not None else int(row["comm_round"])),
        "--batch-size",
        str(batch_size),
        "--g-lr",
        str(g_lr),
        "--f-lr",
        str(f_lr),
        "--weight-decay",
        row["weight_decay"],
        "--gradient-clip-norm",
        row["gradient_clip_norm"],
        "--hidden-widths",
        row["hidden_widths"],
        "--model-activation",
        row["model_activation"],
        "--data-dir",
        row["data_cache_dir"],
        "--run-id",
        row["run_id"],
        "--no-cuda",
    ]
    if overwrite:
        command.append("--overwrite")
    return command, result_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    rows = load_rows(manifest)
    if len(rows) != 45:
        raise ValueError(
            f"expected 45 centralized final rows, found {len(rows)} in {manifest}"
        )
    ledger = {
        "passed": [],
        "failed": [],
        "skipped_completed": [],
        "dry_run": [],
    }
    for row in rows:
        command, result_dir = build_command(
            row,
            python=args.python,
            output_root=output_root,
            overwrite=args.overwrite,
            iterations=args.iterations,
        )
        metrics = result_dir / "metrics.json"
        if metrics.is_file() and not args.overwrite:
            print(f"SKIP {row['run_id']} (metrics.json exists)")
            ledger["skipped_completed"].append(row["run_id"])
            continue
        print(f"START {row['run_id']}")
        if args.dry_run:
            print("  " + " ".join(command))
            ledger["dry_run"].append(row["run_id"])
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT)
        key = "passed" if completed.returncode == 0 else "failed"
        ledger[key].append(row["run_id"])
        print(f"{'PASS' if key == 'passed' else 'FAIL'} {row['run_id']}")
        if key == "failed" and args.stop_on_failure:
            break

    summary = {
        "total_rows": len(rows),
        **{key: len(value) for key, value in ledger.items()},
        "failed_run_ids": ledger["failed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.results_json:
        path = Path(args.results_json).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump({"summary": summary, "ledger": ledger}, handle, indent=2)
            handle.write("\n")
    return 1 if ledger["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
