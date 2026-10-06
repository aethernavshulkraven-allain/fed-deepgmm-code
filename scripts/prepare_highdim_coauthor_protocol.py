#!/usr/bin/env python3
"""Prepare the co-author-approved fixed-abs high-dimensional protocol.

The generated tuning manifests contain the two effective learning-rate
candidates per dataset/method.  The historical weight-decay dimension is
collapsed because the federated G/F optimizer factories do not consume that
argument.  Canonical alpha-0.5 run IDs and output paths are retained so valid
completed artifacts can be reused by ``run_manifest.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
SOURCE_BASE = SOURCE_DIR / "manifest.csv"
SOURCE_TUNING = SOURCE_DIR / "tuning" / "manifest.csv"
PROTOCOL_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1"

ALPHAS = (0.1, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4)
TUNING_ROUNDS = 150
DETERMINISTIC_FINAL_ROUNDS = 500
STOCHASTIC_FINAL_ROUNDS = 1500

LEARNING_RATES = {
    "deterministic": (0.001, 0.003),
    "stochastic": (0.003, 0.01),
}

# These values are retained only for configuration compatibility and artifact
# provenance.  They are fixed, not tuned, and do not reach the federated G/F
# optimizers in the current implementation.
CANONICAL_WEIGHT_DECAY = {
    "deterministic": 0.001,
    "stochastic": 0.05,
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def alpha_token(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def number_token(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


def regime(method: str) -> str:
    return "deterministic" if method.endswith("_d") else "stochastic"


def output_root(alpha: float, tuning: bool) -> str:
    if alpha == 0.5:
        base = "results/rerun_protocol_v1_real_images_abs_alpha0p5"
    else:
        base = f"results/rerun_protocol_v1_real_images_abs_alpha{alpha_token(alpha)}"
    return f"{base}_tuning" if tuning else base


def run_dir(root: str, row: dict[str, Any]) -> str:
    return (
        f"{root}/{row['dataset']}/{row['method']}/"
        f"seed_{int(row['seed'])}/{row['run_id']}"
    )


def source_indexes() -> tuple[list[str], dict[tuple[str, str, int], dict[str, str]], dict[tuple[str, str, str, str], dict[str, str]]]:
    fieldnames, base_rows = read_csv(SOURCE_BASE)
    _, tuning_rows = read_csv(SOURCE_TUNING)
    base = {
        (row["dataset"], row["method"], int(row["seed"])): row
        for row in base_rows
    }
    tuning = {
        (row["dataset"], row["method"], row["learning_rate"], row["weight_decay"]): row
        for row in tuning_rows
    }
    return fieldnames, base, tuning


def make_tuning_rows(
    alpha: float,
    base: dict[tuple[str, str, int], dict[str, str]],
    historical: dict[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, method, seed in sorted(base):
        if seed != 0:
            continue
        mode = regime(method)
        wd = CANONICAL_WEIGHT_DECAY[mode]
        for lr in LEARNING_RATES[mode]:
            historical_key = (dataset, method, str(lr), str(wd))
            if alpha == 0.5:
                row = dict(historical[historical_key])
                row["notes"] = (
                    row["notes"]
                    + " Canonical fixed weight-decay row retained for artifact reuse; "
                    "federated G/F selection varies learning rate only."
                )
            else:
                row = dict(base[(dataset, method, 0)])
                a = alpha_token(alpha)
                run_id = (
                    f"tune_{dataset}_{method}_seed0_alpha{a}_"
                    f"lr{number_token(lr)}_wd{number_token(wd)}"
                )
                root = output_root(alpha, tuning=True)
                row.update(
                    {
                        "run_id": run_id,
                        "protocol_version": f"highdim_coauthor_protocol_v1_alpha{a}_tuning",
                        "run_group": f"real_images_abs_alpha{a}_validation_tuning",
                        "alpha": str(alpha),
                        "partition_alpha": str(alpha),
                        "output_root": root,
                        "final_result_dir": "",
                        "implementation_status": "ready",
                        "run_status": "not_started",
                        "preflight_status": "passed" if row["preflight_required"].lower() == "true" else "not_required",
                        "comm_round": str(TUNING_ROUNDS),
                        "learning_rate": str(lr),
                        "learning_rate_status": "validation_tuning_candidate",
                        "weight_decay": str(wd),
                        "notes": (
                            "Fixed-abs high-dimensional validation tuning; learning-rate-only "
                            "selection on seed 0; Test MSE is excluded from selection. Weight "
                            "decay is a fixed compatibility field and is not connected to the "
                            "federated G/F optimizers."
                        ),
                    }
                )
                row["final_result_dir"] = run_dir(root, row)
            rows.append(row)
    return rows


def make_final_base_rows(
    alpha: float,
    base: dict[tuple[str, str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    a = alpha_token(alpha)
    root = output_root(alpha, tuning=False)
    for dataset, method, seed in sorted(base):
        if seed not in SEEDS:
            continue
        row = dict(base[(dataset, method, seed)])
        mode = regime(method)
        final_rounds = (
            DETERMINISTIC_FINAL_ROUNDS
            if mode == "deterministic"
            else STOCHASTIC_FINAL_ROUNDS
        )
        run_id = f"highdim_abs_{dataset}_{method}_seed{seed}_alpha{a}"
        row.update(
            {
                "run_id": run_id,
                "protocol_version": f"highdim_coauthor_protocol_v1_alpha{a}",
                "run_group": f"real_images_abs_alpha{a}_final",
                "alpha": str(alpha),
                "partition_alpha": str(alpha),
                "output_root": root,
                "final_result_dir": "",
                "implementation_status": "pending_validation_selection",
                "run_status": "not_started",
                "preflight_status": "passed" if row["preflight_required"].lower() == "true" else "not_required",
                "comm_round": str(final_rounds),
                "learning_rate": "",
                "learning_rate_status": "to_be_selected_from_seed0_validation_tuning",
                "weight_decay": str(CANONICAL_WEIGHT_DECAY[mode]),
                "notes": (
                    "Co-author-approved fixed-abs high-dimensional final run; five seeds; "
                    f"{final_rounds} rounds for the {mode} regime; inspect validation "
                    "convergence at round 200 without using Test MSE for stopping or selection."
                ),
            }
        )
        row["final_result_dir"] = run_dir(root, row)
        rows.append(row)
    return rows


def main() -> int:
    fieldnames, base, historical = source_indexes()
    totals: dict[str, Any] = {}
    for alpha in ALPHAS:
        a = alpha_token(alpha)
        alpha_dir = PROTOCOL_DIR / f"alpha{a}"
        tuning_rows = make_tuning_rows(alpha, base, historical)
        final_rows = make_final_base_rows(alpha, base)
        write_csv(alpha_dir / "tuning_manifest.csv", fieldnames, tuning_rows)
        write_csv(
            alpha_dir / "tuning_manifest_deterministic.csv",
            fieldnames,
            [row for row in tuning_rows if row["method"].endswith("_d")],
        )
        write_csv(
            alpha_dir / "tuning_manifest_stochastic.csv",
            fieldnames,
            [row for row in tuning_rows if row["method"].endswith("_s")],
        )
        write_csv(alpha_dir / "final_base_manifest.csv", fieldnames, final_rows)
        write_json(alpha_dir / "tuning_manifest.json", tuning_rows)
        write_json(alpha_dir / "final_base_manifest.json", final_rows)
        totals[f"alpha{a}"] = {
            "tuning": len(tuning_rows),
            "tuning_deterministic": sum(row["method"].endswith("_d") for row in tuning_rows),
            "tuning_stochastic": sum(row["method"].endswith("_s") for row in tuning_rows),
            "final": len(final_rows),
            "tuning_output_root": output_root(alpha, tuning=True),
            "final_output_root": output_root(alpha, tuning=False),
        }
    summary = {
        "protocol": "highdim_coauthor_protocol_v1",
        "g_function": "abs",
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "tuning_seed": 0,
        "tuning_rounds": TUNING_ROUNDS,
        "deterministic_final_rounds": DETERMINISTIC_FINAL_ROUNDS,
        "stochastic_final_rounds": STOCHASTIC_FINAL_ROUNDS,
        "selection": "validation_only",
        "test_mse_used_for_selection": False,
        "effective_tuning_dimensions": ["learning_rate"],
        "weight_decay_policy": "fixed compatibility field; not connected to federated G/F optimizers",
        "federated_tuning_runs": sum(item["tuning"] for item in totals.values()),
        "federated_final_runs": sum(item["final"] for item in totals.values()),
        "centralized_final_runs_planned": 6 * 3 * 5,
        "by_alpha": totals,
    }
    write_json(PROTOCOL_DIR / "protocol_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
