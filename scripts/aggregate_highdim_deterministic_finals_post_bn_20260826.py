#!/usr/bin/env python3
"""Final aggregation/reporting for the 3-alpha x 5-seed deterministic matrix
(closeout plan SS10). Test metrics are unlocked only after every one of the
180 planned trajectories has an auditable resolution (Phase 7 rule 2) --
this script refuses to report anything, including validation-only fields,
while any row is unresolved, so a partial run can never be read as if it
were the frozen final table.

Reads finals_evidence_ledger.json (produced by
prepare_highdim_deterministic_finals_post_bn_20260826.py), not a flat
manifest CSV: reused entries carry their real, original run_id, which is
what lets validate_artifacts() succeed for them (it requires the on-disk
effective_config.json's run_id to equal the row's run_id exactly).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import ManifestLaunchError, validate_artifacts  # noqa: E402
# run_manifest's own import above already added fedgmm/sp_decentralized_mnist_lr_example
# to sys.path as a side effect -- experiment_utils is only importable after that.
from experiment_utils import config_checksum  # noqa: E402

STABILITY_WINDOW = 50
EXPECTED_DATASETS = frozenset({
    "femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz",
})
EXPECTED_METHODS = frozenset({"fedgda_d", "fedogda_d"})
EXPECTED_ALPHAS = frozenset({0.1, 0.5, 1.0})
EXPECTED_SEEDS = frozenset({0, 1, 2, 3, 4})


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _last50_mean(run_dir: Path, comm_round: int, field: str) -> float:
    with (run_dir / "mse_by_round.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != comm_round:
        raise ValueError(f"mse_by_round.csv has {len(rows)} rows, expected {comm_round}")
    window = rows[comm_round - STABILITY_WINDOW:comm_round]
    values = [float(row[field]) for row in window]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{field} last-{STABILITY_WINDOW} window contains a nonfinite value")
    return sum(values) / len(values)


def _bn_min_running_var(metrics: dict) -> float | None:
    values = [
        metrics.get(field) for field in ("g_bn_min_running_var", "f_bn_min_running_var")
        if metrics.get(field) is not None
    ]
    if not values:
        return None
    return min(float(v) for v in values)


def _row_report(run_dir: Path, entry: dict) -> dict:
    # The ledger's own seed/alpha/learning_rate/critic_multiplier/
    # partition_alpha/comm_round labels are exactly the fields a preparer
    # bug (closeout review finding: alpha relabeled without moving
    # partition_alpha with it) could get wrong -- passing them through to
    # validate_artifacts() means _validate_effective_config() cross-checks
    # every one of them against the run's real, on-disk effective_config.json
    # rather than the ledger's labels being trusted un-verified. seed in
    # particular was missing here until a review caught it: without it, a
    # trajectory recorded as seed 4 in the ledger could silently be a run
    # whose effective_config.json actually says random_seed=0.
    row = {
        "run_id": entry["run_id"],
        "dataset": entry["dataset"],
        "method": entry["method"],
        "client_optimizer": entry["client_optimizer"],
        "seed": entry["seed"],
        "learning_rate": entry["learning_rate"],
        "critic_multiplier": entry["critic_multiplier"],
        "partition_alpha": entry["partition_alpha"],
        "comm_round": entry["comm_round"],
    }
    validation = validate_artifacts(run_dir, row)
    effective_config = _load_json(run_dir / "effective_config.json")
    report = {
        "run_id": entry["run_id"],
        "dataset": entry["dataset"],
        "method": entry["method"],
        "alpha": float(entry["alpha"]),
        "seed": int(entry["seed"]),
        "reused": bool(entry["reused"]),
        "source_stage": entry.get("source_stage", ""),
        # Freshly recomputed from the real on-disk config right now -- never
        # trusted from the ledger, which was written before the run existed
        # for new (non-reused) trajectories -- so downstream consumers have
        # something to pin/compare against.
        "effective_config_checksum": config_checksum(effective_config),
        "terminal_ineligible": validation["terminal_ineligible"],
        "terminal_reason": validation["terminal_reason"],
    }
    if validation["terminal_ineligible"]:
        # A terminal-divergent trajectory is a reportable result, not
        # something to silently rerun (closeout plan SS9.4) -- report what
        # is knowable and stop there.
        report.update({
            "best_validation_test_mse": None,
            "final_test_mse": None,
            "last50_psi": None,
            "last50_val_mse": None,
            "min_bn_running_var": None,
            "full_curve_stable": False,
        })
        return report
    metrics = _load_json(run_dir / "metrics.json")
    comm_round = int(metrics["rounds_completed"])
    report.update({
        "best_validation_test_mse": validation["test_mse_at_best_validation"],
        "final_test_mse": validation["final_test_mse"],
        "last50_psi": _last50_mean(run_dir, comm_round, "gmm_eval"),
        "last50_val_mse": _last50_mean(run_dir, comm_round, "primary_val_mse"),
        "min_bn_running_var": _bn_min_running_var(metrics),
        "full_curve_stable": True,
    })
    return report


def aggregate(ledger_path: Path) -> dict:
    ledger = _load_json(ledger_path)
    if ledger.get("status") != "complete":
        raise ValueError("finals evidence ledger is absent or incomplete")
    trajectories = ledger.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 180:
        raise ValueError(
            f"finals evidence ledger must list exactly 180 trajectories, "
            f"got {len(trajectories) if isinstance(trajectories, list) else 'invalid'}"
        )
    run_ids = [entry["run_id"] for entry in trajectories]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("finals evidence ledger contains duplicate run_ids")

    unresolved = []
    entries = []
    for trajectory in trajectories:
        run_dir = Path(trajectory["final_result_dir"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        try:
            entries.append(_row_report(run_dir, trajectory))
        except (ManifestLaunchError, OSError, KeyError, ValueError) as exc:
            unresolved.append({"run_id": trajectory["run_id"], "reason": str(exc)})

    if unresolved:
        raise ValueError(
            f"{len(unresolved)}/180 trajectories are not yet auditably resolved; "
            f"first={unresolved[0]} -- test metrics stay locked until every "
            "planned trajectory resolves (closeout plan Phase 7 rule 2)"
        )

    # A trajectory carrying an unexpected dataset, method, or alpha would
    # still pass the per-(cell, alpha) "5 seeds" check below as long as its
    # own group happened to also total 5 -- these set-equality checks catch
    # that class of error directly, rather than relying on counts alone.
    actual_datasets = {entry["dataset"] for entry in entries}
    if actual_datasets != EXPECTED_DATASETS:
        raise ValueError(
            f"finals evidence ledger covers unexpected datasets: "
            f"missing={sorted(EXPECTED_DATASETS - actual_datasets)}, "
            f"unexpected={sorted(actual_datasets - EXPECTED_DATASETS)}"
        )
    actual_methods = {entry["method"] for entry in entries}
    if actual_methods != EXPECTED_METHODS:
        raise ValueError(
            f"finals evidence ledger covers unexpected methods: "
            f"missing={sorted(EXPECTED_METHODS - actual_methods)}, "
            f"unexpected={sorted(actual_methods - EXPECTED_METHODS)}"
        )
    actual_alphas = {entry["alpha"] for entry in entries}
    if actual_alphas != EXPECTED_ALPHAS:
        raise ValueError(
            f"finals evidence ledger covers unexpected alphas: "
            f"missing={sorted(EXPECTED_ALPHAS - actual_alphas)}, "
            f"unexpected={sorted(actual_alphas - EXPECTED_ALPHAS)}"
        )

    by_cell_alpha: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for entry in entries:
        by_cell_alpha[(entry["dataset"], entry["method"], entry["alpha"])].append(entry)
    expected_groups = {
        (dataset, method, alpha)
        for dataset in EXPECTED_DATASETS for method in EXPECTED_METHODS for alpha in EXPECTED_ALPHAS
    }
    if set(by_cell_alpha) != expected_groups:
        raise ValueError(
            f"finals evidence ledger is missing (dataset, method, alpha) groups: "
            f"{sorted(expected_groups - set(by_cell_alpha))}"
        )

    cross_seed_summary = {}
    for (dataset, method, alpha), seed_entries in sorted(by_cell_alpha.items()):
        if len(seed_entries) != 5:
            raise ValueError(
                f"{dataset}/{method}/alpha={alpha}: expected 5 seeds, got {len(seed_entries)}"
            )
        actual_seeds = {e["seed"] for e in seed_entries}
        if actual_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{dataset}/{method}/alpha={alpha}: expected seeds "
                f"{sorted(EXPECTED_SEEDS)}, got {sorted(actual_seeds)}"
            )
        finite_final_test_mse = [
            e["final_test_mse"] for e in seed_entries if e["final_test_mse"] is not None
        ]
        cross_seed_summary[f"{dataset}|{method}|{alpha:g}"] = {
            "dataset": dataset, "method": method, "alpha": alpha,
            "seeds_terminal": sum(e["terminal_ineligible"] for e in seed_entries),
            "seeds_stable": sum(e["full_curve_stable"] for e in seed_entries),
            "median_final_test_mse": (
                median(finite_final_test_mse) if finite_final_test_mse else None
            ),
        }

    return {
        "status": "complete",
        "ledger": str(ledger_path.relative_to(REPO_ROOT)) if ledger_path.is_absolute() else str(ledger_path),
        "trajectories": entries,
        "cross_seed_summary": cross_seed_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = aggregate(args.ledger.resolve())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FINAL AGGREGATION BLOCKED: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "trajectories": len(result["trajectories"]),
        "cells_x_alphas": len(result["cross_seed_summary"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
