#!/usr/bin/env python3
"""Classify each alpha=0.1 stability run as pass or retune-required.

Implements the frozen escape hatch from doe_review_and_revised_grid.md:
"any cell whose winner diverges or fails the constant-predictor test at the
alpha=0.1 stability check must be re-tuned at that alpha specifically."
Written before any stability result exists (closeout plan SS4.7 / hard-stop
SS11) -- the two failure conditions below are fixed now, not chosen after
seeing which cells fail:

1. Divergence/nonfinite evidence, via the same validate_artifacts() every
   other stage uses.
2. The constant-predictor test: this run's last-50-round mean validation MSE
   (rounds 450..499 of 500, the same summary statistic frozen for Psi in
   PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md, applied here to MSE) is not
   strictly better than a constant predictor that always outputs the mean of
   the validation target -- i.e. mean((g_dev - mean(g_dev))**2), captured at
   training time as metrics.json's val_target_variance so this validator
   never needs raw per-sample validation targets or touches test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import ManifestLaunchError, validate_artifacts  # noqa: E402

STABILITY_COMM_ROUND = 500
LAST50_WINDOW = 50


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _last50_val_mse(run_dir: Path, run_id: str) -> float:
    curve_path = run_dir / "mse_by_round.csv"
    with curve_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != STABILITY_COMM_ROUND:
        raise ValueError(
            f"{run_id}: mse_by_round.csv has {len(rows)} rows, expected "
            f"exactly {STABILITY_COMM_ROUND}"
        )
    for index, row in enumerate(rows):
        try:
            round_index = int(row.get("round", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{run_id}: mse_by_round.csv[{index}].round is not an integer") from exc
        if round_index != index:
            raise ValueError(
                f"{run_id}: mse_by_round.csv[{index}].round is {round_index}, expected {index}"
            )
    window = rows[STABILITY_COMM_ROUND - LAST50_WINDOW:STABILITY_COMM_ROUND]
    values = []
    for offset, row in enumerate(window):
        round_index = STABILITY_COMM_ROUND - LAST50_WINDOW + offset
        raw = row.get("primary_val_mse")
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].primary_val_mse is blank")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].primary_val_mse is nonfinite")
        values.append(number)
    return sum(values) / len(values)


def classify_stability_run(run_dir: Path, row: dict[str, str]) -> dict:
    """Returns {"outcome": "pass"|"retune_required", "reason": str, ...}.
    Never raises for a legitimate divergent/failed run -- that IS a valid,
    classified outcome. Raises ManifestLaunchError/ValueError only for
    malformed or incomplete artifacts, which must block the stage rather
    than be silently classified."""
    validation = validate_artifacts(run_dir, row)
    if validation["terminal_ineligible"]:
        return {
            "outcome": "retune_required",
            "reason": f"diverged/nonfinite: {validation['terminal_reason']}",
            "last50_val_mse": None,
            "constant_predictor_mse": None,
        }
    metrics = _load_json(run_dir / "metrics.json")
    try:
        constant_predictor_mse = float(metrics["val_target_variance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{row['run_id']}: metrics.val_target_variance is missing or invalid") from exc
    if not math.isfinite(constant_predictor_mse) or constant_predictor_mse < 0.0:
        raise ValueError(f"{row['run_id']}: metrics.val_target_variance is not a valid variance")
    last50_val_mse = _last50_val_mse(run_dir, row["run_id"])
    if last50_val_mse >= constant_predictor_mse:
        return {
            "outcome": "retune_required",
            "reason": (
                f"fails constant-predictor test: last50_val_mse={last50_val_mse} >= "
                f"constant_predictor_mse={constant_predictor_mse}"
            ),
            "last50_val_mse": last50_val_mse,
            "constant_predictor_mse": constant_predictor_mse,
        }
    return {
        "outcome": "pass",
        "reason": "finite, nondivergent, and beats the constant predictor",
        "last50_val_mse": last50_val_mse,
        "constant_predictor_mse": constant_predictor_mse,
    }


def validate_stability(manifest_path: Path) -> dict:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        raise ValueError(f"stability manifest must contain exactly 12 rows (one per cell), got {len(rows)}")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("stability manifest contains duplicate run_ids")

    cells = {}
    retune_cells = []
    for row in rows:
        if row.get("server_buffer_policy") != "direct_client_aggregate":
            raise ValueError(f"{row['run_id']} does not freeze the corrected buffer policy")
        if str(row.get("alpha", row.get("partition_alpha", ""))) not in {"0.1"}:
            raise ValueError(f"{row['run_id']} is not an alpha=0.1 stability row")
        run_dir = Path(row["final_result_dir"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        try:
            classification = classify_stability_run(run_dir, row)
        except ManifestLaunchError as exc:
            raise ValueError(f"{row['run_id']}: artifact validation failed: {exc}") from exc
        cell_name = f"{row['dataset']}|{row['method']}"
        if cell_name in cells:
            raise ValueError(f"duplicate cell in stability manifest: {cell_name}")
        cells[cell_name] = {
            "dataset": row["dataset"],
            "method": row["method"],
            "run_id": row["run_id"],
            **classification,
        }
        if classification["outcome"] == "retune_required":
            retune_cells.append(cell_name)

    if len(cells) != 12:
        raise ValueError(f"expected 12 distinct dataset/method cells, got {len(cells)}")

    return {
        "status": "complete",
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "cells": cells,
        "retune_required_cells": sorted(retune_cells),
        "all_cells_pass": not retune_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_stability(args.manifest.resolve())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"STABILITY VALIDATION BLOCKED: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "cells": len(result["cells"]),
        "retune_required_cells": result["retune_required_cells"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
