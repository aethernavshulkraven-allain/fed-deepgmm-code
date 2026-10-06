#!/usr/bin/env python3
"""Score only a complete, corrected high-dimensional image screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import (  # noqa: E402
    CORE_DATASET_FILES,
    CORE_PROTOCOL_DOCS,
    CORE_SOURCES,
    git_provenance,
)
from run_manifest import (  # noqa: E402
    ManifestLaunchError,
    load_certification_ledger,
    resolve_certified_run,
    validate_artifacts,
)
from score_highdim_screen_by_psi import boundary_flags, rank_cell  # noqa: E402

EXPECTED_CELLS = 12
SCREEN_COMM_ROUND = 150
LAST50_WINDOW = 50


def _last50_mean(run_dir: Path, run_id: str) -> tuple[float, float]:
    """Mean gmm_eval (Psi) and mean primary_val_mse over the last 50 of the
    screen's 150 communication rounds, per the frozen last-50 scorer rule
    (PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md). best_gmm_eval (the
    best-round value) is diagnostic only and must never be used here."""
    curve_path = run_dir / "mse_by_round.csv"
    with curve_path.open(newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    if len(curve_rows) != SCREEN_COMM_ROUND:
        raise ValueError(
            f"{run_id}: mse_by_round.csv has {len(curve_rows)} rows, "
            f"expected exactly {SCREEN_COMM_ROUND}"
        )
    for index, curve_row in enumerate(curve_rows):
        try:
            round_index = int(curve_row.get("round", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{run_id}: mse_by_round.csv[{index}].round is not an integer") from exc
        if round_index != index:
            raise ValueError(
                f"{run_id}: mse_by_round.csv[{index}].round is {round_index}, "
                f"expected {index} (missing, duplicated, or unordered rows)"
            )
    window = curve_rows[SCREEN_COMM_ROUND - LAST50_WINDOW:SCREEN_COMM_ROUND]
    psi_values = []
    mse_values = []
    for offset, curve_row in enumerate(window):
        round_index = SCREEN_COMM_ROUND - LAST50_WINDOW + offset
        for field, sink in (("gmm_eval", psi_values), ("primary_val_mse", mse_values)):
            raw_value = curve_row.get(field)
            if raw_value is None or str(raw_value).strip() == "":
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is blank")
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is not numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{run_id}: mse_by_round.csv[{round_index}].{field} is nonfinite")
            sink.append(number)
    return sum(psi_values) / len(psi_values), sum(mse_values) / len(mse_values)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The last-50-mean ranking that decides every winner in this file is
# computed from mse_by_round.csv, not metrics.json -- a result_sha256 that
# hashed only metrics.json (closeout review finding) could stay unchanged
# while the actual round curve the ranking was based on silently changed.
# Binds all three of a run's authoritative artifacts together, in a fixed,
# filename-prefixed order so no combination of edits produces a collision.
_ELIGIBLE_RESULT_FILES = ("effective_config.json", "metrics.json", "mse_by_round.csv")


def _eligible_result_sha256(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in _ELIGIBLE_RESULT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((run_dir / name).read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def score_screen(manifest_path: Path, certification_ledger_path: Path | None = None) -> dict:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 108:
        raise ValueError(f"corrected screen must contain 108 rows, got {len(rows)}")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("corrected screen manifest contains duplicate run_ids")
    ledger = load_certification_ledger(certification_ledger_path)

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    invalid = []
    terminal = []
    # Phase 4 SS7.1 requires screen_results.json to account for all 108
    # planned rows individually (not just the 12 per-cell summaries) plus a
    # result hash per row -- built alongside the existing per-row loop so it
    # can never drift from what that loop actually decided about each row.
    dispositions = []
    for row in rows:
        if row.get("server_buffer_policy") != "direct_client_aggregate":
            raise ValueError(f"{row['run_id']} does not freeze the corrected buffer policy")
        run_dir = Path(row["final_result_dir"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        # A certified pre-round-0 failure's real evidence lives in an
        # independent reproduction's own directory, never copied into this
        # row's own final_result_dir (closeout plan SS6.2) -- redirect only
        # for validation purposes; the run reported below is still this row.
        effective_run_dir, effective_row = resolve_certified_run(
            row["run_id"], row, run_dir, ledger
        )
        disposition_base = {
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "method": row["method"],
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
        }
        try:
            validation = validate_artifacts(effective_run_dir, effective_row)
        except ManifestLaunchError as exc:
            invalid.append({"run_id": row["run_id"], "reason": str(exc)})
            continue
        if validation["terminal_ineligible"]:
            terminal.append({
                "run_id": row["run_id"],
                "reason": validation["terminal_reason"],
            })
            dispositions.append({
                **disposition_base,
                "disposition": "terminal_pretraining_ineligible",
                "terminal_reason": validation["terminal_reason"],
                "result_sha256": _sha256(effective_run_dir / "pretraining_failure.json"),
            })
            continue
        metrics = _load_json(run_dir / "metrics.json")
        try:
            psi_last50, mse_last50 = _last50_mean(run_dir, row["run_id"])
        except (OSError, ValueError) as exc:
            invalid.append({"run_id": row["run_id"], "reason": str(exc)})
            continue
        try:
            best_gmm_eval_diagnostic = float(metrics["best_gmm_eval"])
            if not math.isfinite(best_gmm_eval_diagnostic):
                raise ValueError("nonfinite best_gmm_eval")
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({"run_id": row["run_id"], "reason": f"invalid diagnostic best_gmm_eval: {exc}"})
            continue
        cells[(row["dataset"], row["method"])].append({
            "run_id": row["run_id"],
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
            "gmm_eval": psi_last50,
            "val_mse": mse_last50,
            "best_gmm_eval_diagnostic": best_gmm_eval_diagnostic,
        })
        dispositions.append({
            **disposition_base,
            "disposition": "eligible",
            "gmm_eval": psi_last50,
            "val_mse": mse_last50,
            "best_gmm_eval_diagnostic": best_gmm_eval_diagnostic,
            "result_sha256": _eligible_result_sha256(run_dir),
        })

    if invalid:
        raise ValueError(
            f"screen is incomplete or malformed ({len(invalid)} runs); first={invalid[0]}"
        )
    if len(dispositions) != len(rows):
        raise ValueError(
            f"internal error: {len(dispositions)} row dispositions for {len(rows)} planned rows"
        )
    if len(cells) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} image cells, got {len(cells)}")

    planned_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        planned_by_cell[(row["dataset"], row["method"])].append({
            "lr": float(row["learning_rate"]),
            "cm": float(row["critic_multiplier"]),
        })

    output_cells = {}
    boundary_review_cells = []
    terminal_ids = {entry["run_id"] for entry in terminal}
    for cell_key in sorted(planned_by_cell):
        eligible = cells.get(cell_key, [])
        if len(eligible) < 2:
            raise ValueError(f"{cell_key} has fewer than two eligible candidates")
        psi_ranked = rank_cell(eligible)
        # rank_cell's own sort key is (-round(gmm_eval, 9), val_mse) -- val_mse
        # already acts as the documented secondary tiebreak for a gmm_eval tie,
        # so that alone is not an unresolved tie. It's unresolved only when
        # BOTH keys tie exactly, at which point Python's stable sort would
        # otherwise silently fall back to manifest order.
        psi_top_key = (round(psi_ranked[0]["gmm_eval"], 9), psi_ranked[0]["val_mse"])
        psi_tie_set = [
            candidate for candidate in psi_ranked
            if (round(candidate["gmm_eval"], 9), candidate["val_mse"]) == psi_top_key
        ]
        if len(psi_tie_set) > 1:
            raise ValueError(
                f"{cell_key} has an unresolved exact Psi tie among "
                f"{[candidate['run_id'] for candidate in psi_tie_set]}"
            )
        mse_minimum = min(candidate["val_mse"] for candidate in eligible)
        mse_winners = [
            candidate for candidate in eligible if candidate["val_mse"] == mse_minimum
        ]
        if len(mse_winners) != 1:
            raise ValueError(f"{cell_key} has an unresolved exact validation-MSE tie")
        mse_winner = mse_winners[0]
        # BOUNDARY_RULE_AMENDMENT_20260818.md SS"Replacement rule" step 1 is
        # scoped to "the Psi rank-1 candidate" only -- score_highdim_screen_
        # by_psi.py's reference implementation likewise only ever checks
        # rank_cell's top candidate. mse_flags is retained as diagnostic
        # metadata (it's informative that the MSE winner also sits at a
        # boundary) but must not itself trigger the frozen review/expansion
        # rule -- that would review/expand a candidate the rule never named.
        psi_flags = boundary_flags(psi_ranked[0], planned_by_cell[cell_key])
        mse_flags = boundary_flags(mse_winner, planned_by_cell[cell_key])
        review_required = bool(psi_flags)
        cell_name = f"{cell_key[0]}|{cell_key[1]}"
        if review_required:
            boundary_review_cells.append(cell_name)
        output_cells[cell_name] = {
            "dataset": cell_key[0],
            "method": cell_key[1],
            "psi_rank_1": psi_ranked[0],
            "psi_rank_2": psi_ranked[1],
            "mse_winner": mse_winner,
            "eligible_candidates": len(eligible),
            "terminal_candidates": sum(
                row["run_id"] in terminal_ids
                for row in rows
                if (row["dataset"], row["method"]) == cell_key
            ),
            "boundary_review_required": review_required,
            "boundary_detail": {"psi_rank_1": psi_flags, "mse_winner": mse_flags},
        }
    # Phase 4 SS7.1: screen_results.json must itself carry source, manifest,
    # and result hashes, not just a manifest path string -- source_hashes
    # mirrors every other stage's generated_artifact_hashes.json (CORE_
    # SOURCES + CORE_DATASET_FILES + CORE_PROTOCOL_DOCS), embedded here
    # instead of a sibling file since the plan asks for this file to
    # *contain* them. Also includes this scorer's own code, its ranking
    # dependency, and the protocol/DOE documents that define scoring and
    # boundary outcomes -- a silent edit to any of these would change this
    # very output without CORE_SOURCES (scoped to main.py's training
    # execution path, not post-hoc scoring) ever catching it.
    required_scorer_paths = (
        "scripts/score_highdim_screen_post_bn_20260822.py",
        "scripts/score_highdim_screen_by_psi.py",
        "experiments/highdim_coauthor_protocol_v1/doe_review_and_revised_grid.md",
        "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822/"
        "HIGH_DIM_DETERMINISTIC_CLOSEOUT_PLAN_20260826.md",
    )
    # BOUNDARY_REVIEW_20260827.md is a reactive decision artifact written
    # only after a first scoring pass identifies which cells are flagged
    # (closeout plan SS7.1 then SS7.2) -- hashed when present, so it's bound
    # into the *final*, post-review freeze, without making an earlier
    # discovery pass fail closed on a file that cannot exist yet.
    optional_scorer_paths = (
        "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822/"
        "BOUNDARY_REVIEW_20260827.md",
    )
    source_hashes = [
        {"path": path, "sha256": _sha256(REPO_ROOT / path)}
        for path in sorted({*CORE_SOURCES, *CORE_DATASET_FILES, *CORE_PROTOCOL_DOCS, *required_scorer_paths})
    ] + [
        {"path": path, "sha256": _sha256(REPO_ROOT / path)}
        for path in sorted(optional_scorer_paths)
        if (REPO_ROOT / path).is_file()
    ]
    return {
        "status": "complete",
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "server_buffer_policy": "direct_client_aggregate",
        "planned_runs": len(rows),
        "terminal_ineligible_runs": terminal,
        "boundary_review_required": bool(boundary_review_cells),
        "boundary_review_cells": boundary_review_cells,
        "cells": output_cells,
        "row_dispositions": dispositions,
        "screen_manifest_sha256": _sha256(manifest_path),
        "source_hashes": source_hashes,
        "git_provenance": git_provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--certification-ledger", type=Path, default=None,
        help="Links pre-round-0 terminal run_ids to their independent reproduction "
        "evidence (closeout plan SS6.2); see resolve_certified_run().",
    )
    args = parser.parse_args()
    try:
        result = score_screen(
            args.manifest.resolve(),
            args.certification_ledger.resolve() if args.certification_ledger else None,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"SCREEN SCORING BLOCKED: {exc}")
        return 2
    _atomic_json(args.out.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "cells": len(result["cells"]),
        "boundary_review_required": result["boundary_review_required"],
        "terminal_ineligible_runs": len(result["terminal_ineligible_runs"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
