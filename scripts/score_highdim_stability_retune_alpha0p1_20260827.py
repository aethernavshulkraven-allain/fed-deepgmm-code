#!/usr/bin/env python3
"""Score the alpha=0.1 per-cell retune fallback's 150-round Screen stage
(closeout plan SS9.1) and select each retuned cell's top 2 candidates.

This is the Screen stage of the frozen Screen->Rank->Confirm->Promote
extended-plan fallback (doe_review_and_revised_grid.md's escape hatch,
Part VI/VII): it never itself promotes a winner off a single noisy
seed-0/150-round score. Its "top2" output is consumed by
prepare_highdim_stability_retune_rank_alpha0p1_20260827.py and
prepare_highdim_stability_retune_confirm_alpha0p1_20260827.py, and the
actual winner is only frozen by
score_highdim_stability_retune_promote_alpha0p1_20260827.py after both
500-round stages complete (median-of-3-seeds rule, same as the frozen V4
adjudication rule).

Same last-50-round mean Psi ranking rule as the corrected screen
(PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md) -- reuses score_highdim_
screen_by_psi.rank_cell/boundary_flags directly rather than a parallel
implementation, over whichever cells prepare_highdim_stability_retune_
alpha0p1_20260827.py actually generated (any number of cells, unlike the
corrected screen's fixed 12).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import ManifestLaunchError, validate_artifacts  # noqa: E402
from score_highdim_screen_by_psi import boundary_flags, rank_cell  # noqa: E402

RETUNE_COMM_ROUND = 150
LAST50_WINDOW = 50


def _last50_mean(run_dir: Path, run_id: str) -> tuple[float, float]:
    curve_path = run_dir / "mse_by_round.csv"
    with curve_path.open(newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    if len(curve_rows) != RETUNE_COMM_ROUND:
        raise ValueError(
            f"{run_id}: mse_by_round.csv has {len(curve_rows)} rows, "
            f"expected exactly {RETUNE_COMM_ROUND}"
        )
    for index, curve_row in enumerate(curve_rows):
        try:
            round_index = int(curve_row.get("round", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{run_id}: mse_by_round.csv[{index}].round is not an integer") from exc
        if round_index != index:
            raise ValueError(
                f"{run_id}: mse_by_round.csv[{index}].round is {round_index}, expected {index}"
            )
    window = curve_rows[RETUNE_COMM_ROUND - LAST50_WINDOW:RETUNE_COMM_ROUND]
    psi_values, mse_values = [], []
    for offset, curve_row in enumerate(window):
        round_index = RETUNE_COMM_ROUND - LAST50_WINDOW + offset
        for field, sink in (("gmm_eval", psi_values), ("primary_val_mse", mse_values)):
            raw_value = curve_row.get(field)
            if raw_value is None or str(raw_value).strip() == "":
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is blank")
            number = float(raw_value)
            if not math.isfinite(number):
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is nonfinite")
            sink.append(number)
    return sum(psi_values) / len(psi_values), sum(mse_values) / len(mse_values)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def score_retune(manifest_path: Path) -> dict:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("retune manifest contains no rows")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("retune manifest contains duplicate run_ids")

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    invalid = []
    terminal = []
    baseline_failed = []
    for row in rows:
        if row.get("server_buffer_policy") != "direct_client_aggregate":
            raise ValueError(f"{row['run_id']} does not freeze the corrected buffer policy")
        if str(row.get("alpha")) != "0.1":
            raise ValueError(f"{row['run_id']} is not an alpha=0.1 retune row")
        run_dir = Path(row["final_result_dir"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        try:
            validation = validate_artifacts(run_dir, row)
        except ManifestLaunchError as exc:
            invalid.append({"run_id": row["run_id"], "reason": str(exc)})
            continue
        if validation["terminal_ineligible"]:
            terminal.append({"run_id": row["run_id"], "reason": validation["terminal_reason"]})
            continue
        try:
            psi_last50, mse_last50 = _last50_mean(run_dir, row["run_id"])
        except (OSError, ValueError) as exc:
            invalid.append({"run_id": row["run_id"], "reason": str(exc)})
            continue
        # Same constant-predictor eligibility rule as the stability stage's
        # own validator (validate_highdim_stability_alpha0p1_20260826.py) --
        # doe_review_and_revised_grid.md's escape hatch is only meaningful if
        # a retune candidate actually has to clear the baseline whose failure
        # triggered retuning in the first place, not merely be finite and
        # nondivergent. A candidate that fails this must never be ranked or
        # advanced to Rank/Confirm, even if its Psi score looks good.
        try:
            metrics = _load_json(run_dir / "metrics.json")
            constant_predictor_mse = float(metrics["val_target_variance"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            invalid.append({
                "run_id": row["run_id"],
                "reason": f"metrics.val_target_variance is missing or invalid: {exc}",
            })
            continue
        if not math.isfinite(constant_predictor_mse) or constant_predictor_mse < 0.0:
            invalid.append({
                "run_id": row["run_id"],
                "reason": "metrics.val_target_variance is not a valid variance",
            })
            continue
        if mse_last50 >= constant_predictor_mse:
            baseline_failed.append({
                "run_id": row["run_id"],
                "reason": (
                    f"fails constant-predictor test: last50_val_mse={mse_last50} >= "
                    f"constant_predictor_mse={constant_predictor_mse}"
                ),
            })
            continue
        cells[(row["dataset"], row["method"])].append({
            "run_id": row["run_id"],
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
            "gmm_eval": psi_last50,
            "val_mse": mse_last50,
        })

    if invalid:
        raise ValueError(f"retune is incomplete or malformed ({len(invalid)} runs); first={invalid[0]}")

    planned_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        planned_by_cell[(row["dataset"], row["method"])].append({
            "lr": float(row["learning_rate"]), "cm": float(row["critic_multiplier"]),
        })

    output_cells = {}
    terminal_ids = {entry["run_id"] for entry in terminal}
    baseline_failed_ids = {entry["run_id"] for entry in baseline_failed}
    for cell_key in sorted(planned_by_cell):
        eligible = cells.get(cell_key, [])
        if not eligible:
            raise ValueError(
                f"{cell_key} has zero eligible retune candidates -- every candidate "
                "either diverged or failed the constant-predictor test; this extended-"
                "plan fallback has no further automatic escalation, requires a dated "
                "protocol decision, not a rerun"
            )
        ranked = rank_cell(eligible)
        if len(ranked) < 2:
            raise ValueError(
                f"{cell_key} has only {len(ranked)} eligible screen candidate(s); "
                "the frozen extended-plan fallback (doe_review_and_revised_grid.md "
                "Part VI/VII escape hatch) requires a Rank stage over the top 2"
            )
        # This 150-round screen only selects WHICH two candidates advance to
        # the 500-round Rank/Confirm stages -- it must never itself pick a
        # cell's final winner (closeout plan SS9.1: "the fallback branch must
        # not be designed after observing which cells fail" applies equally
        # to not promoting off a single noisy seed-0/150-round score).
        boundary_key = (round(ranked[1]["gmm_eval"], 9), ranked[1]["val_mse"])
        boundary_tie = [
            c for c in ranked
            if (round(c["gmm_eval"], 9), c["val_mse"]) == boundary_key
        ]
        if len(boundary_tie) > 1:
            raise ValueError(
                f"{cell_key} has an unresolved exact Psi tie at the top-2/"
                f"excluded boundary among {[c['run_id'] for c in boundary_tie]}"
            )
        cell_name = f"{cell_key[0]}|{cell_key[1]}"
        output_cells[cell_name] = {
            "dataset": cell_key[0],
            "method": cell_key[1],
            "top2": ranked[:2],
            "eligible_candidates": len(eligible),
            "terminal_candidates": sum(
                row["run_id"] in terminal_ids
                for row in rows
                if (row["dataset"], row["method"]) == cell_key
            ),
            "baseline_failed_candidates": sum(
                row["run_id"] in baseline_failed_ids
                for row in rows
                if (row["dataset"], row["method"]) == cell_key
            ),
            "boundary_detail": boundary_flags(ranked[0], planned_by_cell[cell_key]),
        }
    return {
        "status": "complete",
        "stage": "screen",
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "server_buffer_policy": "direct_client_aggregate",
        "baseline_failed_runs": baseline_failed,
        "cells": output_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_retune(args.manifest.resolve())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"RETUNE SCORING BLOCKED: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "cells": len(result["cells"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
