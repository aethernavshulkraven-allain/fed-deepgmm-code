#!/usr/bin/env python3
"""Score the alpha=0.1 per-cell retune fallback's Promote stage (closeout
plan SS9.1 escape hatch; doe_review_and_revised_grid.md Part VI/VII) and
freeze a single winner per retuned cell.

This is the only stage of the fallback that is allowed to promote a
winner. It combines the Rank stage's seed-0 500-round run with the Confirm
stage's seed-{1,2} 500-round runs into a 3-seed Candidate per Rank
top-2 arm, then applies the exact same frozen median-of-3-seeds rule
(median last-50-round mean validation Psi, frozen pairwise practical-tie
rule, median-MSE tiebreak) that score_highdim_adjudication_20260819.py
uses for the V4 adjudication -- reusing that module's Candidate/SeedResult/
score_cell/load_seed_result directly rather than a parallel
implementation, since the frozen rule is dataset/campaign-agnostic by
design.

Output shape ("cells": {cell_name: {..., "winner": {"lr":..., "cm":...}}})
matches what prepare_highdim_deterministic_finals_post_bn_20260826.py's
--retune-results already expects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import METHOD_TO_OPTIMIZER  # noqa: E402
from score_highdim_adjudication_20260819 import (  # noqa: E402
    Candidate,
    candidate_to_dict,
    load_seed_result,
    score_cell,
)

CONFIRM_SEEDS = ("1", "2")
RANK_CAMPAIGN = "highdim_deterministic_stability_retune_rank_alpha0p1_20260827"
CONFIRM_CAMPAIGN = "highdim_deterministic_stability_retune_confirm_alpha0p1_20260827"


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _no_duplicates(pairs: list[tuple], what: str) -> dict:
    """dict(pairs) silently keeps only the last of a duplicate key -- for
    manifest run_ids and plan cell names that must mean a malformed or
    tampered input, not something to resolve by whichever entry happened to
    come last."""
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate {what}: {key!r}")
        result[key] = value
    return result


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return _no_duplicates([(row["run_id"], row) for row in rows], f"run_id in {path.name}")


def _require_summary_metadata(
    summary: dict, *, label: str, campaign: str, alpha: float, comm_round: int,
    seed_field: str, expected_seeds,
) -> None:
    """Rank/Confirm summaries are the only thing binding a Promote input to
    the correct campaign/alpha/horizon/seed set -- without this, --rank-*
    and --confirm-* could silently point at some other stage's or some
    other cell's output that merely happens to share the same JSON shape."""
    if summary.get("campaign") != campaign:
        raise ValueError(f"{label} summary campaign={summary.get('campaign')!r}, expected {campaign!r}")
    try:
        if float(summary.get("alpha", float("nan"))) != alpha:
            raise ValueError(f"{label} summary alpha={summary.get('alpha')!r}, expected {alpha!r}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} summary alpha={summary.get('alpha')!r}, expected {alpha!r}") from exc
    if int(summary.get("comm_round", -1)) != comm_round:
        raise ValueError(
            f"{label} summary comm_round={summary.get('comm_round')!r}, expected {comm_round}"
        )
    actual_seeds = summary.get(seed_field)
    if isinstance(expected_seeds, (list, tuple, set, frozenset)):
        if set(map(str, actual_seeds if isinstance(actual_seeds, list) else [])) != set(map(str, expected_seeds)):
            raise ValueError(
                f"{label} summary {seed_field}={actual_seeds!r}, expected {list(expected_seeds)}"
            )
    elif str(actual_seeds) != str(expected_seeds):
        raise ValueError(f"{label} summary {seed_field}={actual_seeds!r}, expected {expected_seeds!r}")


def _resolve_run_dir(row: dict[str, str]) -> Path:
    run_dir = Path(row["final_result_dir"])
    return run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir


def _expected_config(dataset: str, method: str, lr: float, cm: float) -> dict:
    return {
        "dataset": dataset,
        "variant": method,
        "client_optimizer": METHOD_TO_OPTIMIZER[method],
        "learning_rate": lr,
        "critic_multiplier": cm,
        "server_buffer_policy": "direct_client_aggregate",
    }


def promote(
    rank_manifest_path: Path,
    rank_summary_path: Path,
    confirm_manifest_path: Path,
    confirm_summary_path: Path,
) -> dict:
    rank_summary = _load_json(rank_summary_path)
    confirm_summary = _load_json(confirm_summary_path)
    _require_summary_metadata(
        rank_summary, label="rank", campaign=RANK_CAMPAIGN, alpha=0.1, comm_round=500,
        seed_field="seed", expected_seeds=0,
    )
    _require_summary_metadata(
        confirm_summary, label="confirm", campaign=CONFIRM_CAMPAIGN, alpha=0.1, comm_round=500,
        seed_field="seeds", expected_seeds=CONFIRM_SEEDS,
    )
    rank_rows = _load_rows(rank_manifest_path)
    confirm_rows = _load_rows(confirm_manifest_path)

    rank_plan = _no_duplicates(
        [(entry["cell"], entry) for entry in rank_summary.get("plan", [])], "rank plan cell",
    )
    confirm_plan = _no_duplicates(
        [(entry["cell"], entry) for entry in confirm_summary.get("plan", [])], "confirm plan cell",
    )
    if not rank_plan:
        raise ValueError("rank summary has no planned cells")
    if set(rank_plan) != set(confirm_plan):
        raise ValueError(
            f"rank/confirm cell mismatch: rank={sorted(rank_plan)} confirm={sorted(confirm_plan)}"
        )

    output_cells = {}
    for cell_name in sorted(rank_plan):
        rank_cell_plan = rank_plan[cell_name]
        confirm_cell_plan = confirm_plan[cell_name]
        dataset, method = rank_cell_plan["dataset"], rank_cell_plan["method"]
        if (confirm_cell_plan.get("dataset"), confirm_cell_plan.get("method")) != (dataset, method):
            raise ValueError(
                f"{cell_name}: confirm plan dataset/method "
                f"({confirm_cell_plan.get('dataset')!r}, {confirm_cell_plan.get('method')!r}) "
                f"disagrees with rank's ({dataset!r}, {method!r})"
            )
        rank_candidates = rank_cell_plan["candidates"]
        if len(rank_candidates) != 2:
            raise ValueError(f"{cell_name}: rank stage must carry exactly 2 candidates")
        # Two *entries* isn't two *candidates* if rank repeated the same
        # (lr, cm) arm -- that would let one confirm arm go completely
        # unused while still nominally satisfying every count check below.
        rank_lr_cm_keys = _no_duplicates(
            [((round(float(c["lr"]), 9), round(float(c["cm"]), 9)), c) for c in rank_candidates],
            f"{cell_name}: rank (lr, cm) candidate",
        )
        confirm_candidates = confirm_cell_plan["candidates"]
        if len(confirm_candidates) != 2:
            raise ValueError(
                f"{cell_name}: confirm stage must carry exactly 2 candidates, matching "
                f"rank's top-2 exactly -- got {len(confirm_candidates)}; extra or missing "
                "confirm candidates both indicate the preparers disagree on the top-2"
            )
        confirm_by_lr_cm = _no_duplicates(
            [
                ((round(float(c["lr"]), 9), round(float(c["cm"]), 9)), c)
                for c in confirm_candidates
            ],
            f"{cell_name}: confirm (lr, cm) candidate",
        )
        if set(confirm_by_lr_cm) != set(rank_lr_cm_keys):
            raise ValueError(
                f"{cell_name}: confirm candidates {sorted(confirm_by_lr_cm)} do not exactly "
                f"match rank's top-2 {sorted(rank_lr_cm_keys)}"
            )

        lr_cm_by_candidate_id: dict[str, tuple[float, float]] = {}
        candidates: list[Candidate] = []
        for rank_candidate in rank_candidates:
            lr, cm = float(rank_candidate["lr"]), float(rank_candidate["cm"])
            confirm_candidate = confirm_by_lr_cm[(round(lr, 9), round(cm, 9))]
            candidate_id = f"{dataset}/{method}/lr{lr:g}_cm{cm:g}"
            lr_cm_by_candidate_id[candidate_id] = (lr, cm)
            expected_config = _expected_config(dataset, method, lr, cm)

            rank_row = rank_rows.get(rank_candidate["run_id"])
            if rank_row is None:
                raise ValueError(
                    f"{cell_name}: rank manifest is missing run_id {rank_candidate['run_id']!r}"
                )
            seed_results = [
                load_seed_result(
                    _resolve_run_dir(rank_row), 0,
                    expected_config=expected_config, manifest_row=rank_row,
                )
            ]
            confirm_run_ids = confirm_candidate.get("run_ids") or {}
            if set(confirm_run_ids) != set(CONFIRM_SEEDS):
                raise ValueError(
                    f"{cell_name}: confirm candidate at lr={lr:g}, cm={cm:g} must carry "
                    f"run_ids for exactly seeds {list(CONFIRM_SEEDS)}, got {sorted(confirm_run_ids)}"
                )
            for seed_str in CONFIRM_SEEDS:
                run_id = confirm_run_ids[seed_str]
                confirm_row = confirm_rows.get(run_id) if run_id else None
                if confirm_row is None:
                    raise ValueError(
                        f"{cell_name}: confirm manifest is missing run_id for "
                        f"lr={lr:g}, cm={cm:g}, seed={seed_str}"
                    )
                seed_results.append(load_seed_result(
                    _resolve_run_dir(confirm_row), int(seed_str),
                    expected_config=expected_config, manifest_row=confirm_row,
                ))
            candidates.append(Candidate(
                candidate_id=candidate_id, label=f"lr={lr:g},cm={cm:g}",
                seeds=tuple(seed_results),
            ))

        outcome = score_cell(candidates)
        if outcome.winner is None:
            raise ValueError(
                f"{cell_name}: retune Confirm stage did not resolve to a promotable winner "
                f"(outcome={outcome.outcome}): {outcome.detail} -- this extended-plan "
                "fallback has no further automatic escalation; requires a dated protocol "
                "decision, not a rerun"
            )
        lr, cm = lr_cm_by_candidate_id[outcome.winner.candidate_id]
        output_cells[cell_name] = {
            "dataset": dataset,
            "method": method,
            "outcome": outcome.outcome,
            "winner": {"lr": lr, "cm": cm},
            "detail": outcome.detail,
            "all_candidates": [candidate_to_dict(c) for c in candidates],
        }

    return {"status": "complete", "stage": "promote", "cells": output_cells}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank-manifest", type=Path, required=True)
    parser.add_argument("--rank-summary", type=Path, required=True)
    parser.add_argument("--confirm-manifest", type=Path, required=True)
    parser.add_argument("--confirm-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = promote(
            args.rank_manifest.resolve(), args.rank_summary.resolve(),
            args.confirm_manifest.resolve(), args.confirm_summary.resolve(),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"PROMOTE SCORING BLOCKED: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {"status": result["status"], "cells": len(result["cells"])}, indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
