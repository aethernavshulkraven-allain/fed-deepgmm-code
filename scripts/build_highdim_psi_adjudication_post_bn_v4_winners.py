#!/usr/bin/env python3
"""Combine the signal and X adjudication scoring outputs into a single
v4_winners.json (closeout plan Phase 6 input contract).

scripts/score_highdim_adjudication_20260819.py is the frozen, generic
Psi-vs-MSE adjudication scoring engine -- both the retired v2 line and this
campaign's post-BN V4 launchers already invoke it (--campaign-dir,
--run-id-prefix) to write adjudication_signal_results.json (8 cells: Z and
XZ datasets) and adjudication_x_results.json (4 cells: X datasets). Neither
file is directly consumable by prepare_highdim_deterministic_stability_
alpha0p1_20260826.py / prepare_highdim_deterministic_finals_post_bn_20260826.py,
which expect one merged, 12-cell v4_winners.json keyed "dataset|method" with
each winner's per-seed run_ids -- that adapter did not exist. This script is
it; it does not re-score anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
DEFAULT_CAMPAIGN_DIR = PROTOCOL_ROOT / "psi_adjudication_post_bn_v4"

DATASETS = ("femnist_x", "femnist_z", "femnist_xz", "cifar10_x", "cifar10_z", "cifar10_xz")
METHODS = ("fedgda_d", "fedogda_d")
# Matches prepare_highdim_psi_adjudication_post_bn_v4.py's own stage split
# (signal = not dataset.endswith("_x"), x = dataset.endswith("_x")) -- the
# exact 12-cell grid this adapter must produce, no more, no fewer, no
# substitutes. Slash-separated to match score_highdim_adjudication_20260819.py's
# own "dataset/method" JSON key convention (its input); the pipe-separated
# "dataset|method" convention is v4_winners.json's own output contract, kept
# separate below as ALL_CELL_NAMES.
SIGNAL_CELLS = frozenset(f"{d}/{m}" for d in DATASETS if not d.endswith("_x") for m in METHODS)
X_CELLS = frozenset(f"{d}/{m}" for d in DATASETS if d.endswith("_x") for m in METHODS)
ALL_CELL_NAMES = frozenset(f"{d}|{m}" for d in DATASETS for m in METHODS)

# score_highdim_adjudication_20260819.py's score_cell(): a cell only carries
# a real winner at these two outcomes; every other outcome (incomplete,
# retune_required, mse_tie_unresolved) leaves winner=None.
WINNING_OUTCOMES = {"promoted", "tie_resolved_by_mse"}

_CANDIDATE_ID_RE = re.compile(
    r"^(?P<dataset>[^/]+)/(?P<method>[^/]+)/lr(?P<lr>-?[0-9.eE+-]+)_cm(?P<cm>-?[0-9.eE+-]+)$"
)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_candidate_id(candidate_id: str) -> tuple[str, str, float, float]:
    match = _CANDIDATE_ID_RE.match(candidate_id)
    if not match:
        raise ValueError(f"cannot parse dataset/method/lr/cm out of candidate_id {candidate_id!r}")
    lr, cm = float(match.group("lr")), float(match.group("cm"))
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"candidate_id {candidate_id!r} has a non-positive or nonfinite lr: {lr}")
    if not math.isfinite(cm) or cm <= 0.0:
        raise ValueError(f"candidate_id {candidate_id!r} has a non-positive or nonfinite cm: {cm}")
    return match.group("dataset"), match.group("method"), lr, cm


def _combine_stage(
    source_label: str, source: dict, expected_cells: frozenset[str], cells: dict[str, dict],
) -> None:
    actual_keys = set(source)
    if actual_keys != expected_cells:
        raise ValueError(
            f"{source_label} results cover the wrong cells: "
            f"missing={sorted(expected_cells - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_cells)}"
        )
    for cell_key, cell in sorted(source.items()):
        if not isinstance(cell, dict):
            raise ValueError(f"{source_label}/{cell_key}: cell entry must be an object")
        dataset, method = cell.get("dataset"), cell.get("method")
        if cell_key != f"{dataset}/{method}":
            raise ValueError(
                f"{source_label}: JSON key {cell_key!r} does not match its own "
                f"dataset/method fields ({dataset!r}, {method!r})"
            )
        cell_name = f"{dataset}|{method}"
        outcome = cell.get("outcome")
        if outcome not in WINNING_OUTCOMES:
            raise ValueError(
                f"{source_label}/{cell_key}: outcome={outcome!r} has no frozen winner -- "
                "v4_winners.json cannot be built until every cell resolves to "
                "'promoted' or 'tie_resolved_by_mse' (closeout plan Phase 5)"
            )
        winner = cell.get("winner")
        if not isinstance(winner, dict):
            raise ValueError(f"{source_label}/{cell_key}: outcome={outcome!r} but winner is missing")
        candidate_dataset, candidate_method, lr, cm = _parse_candidate_id(winner.get("candidate_id", ""))
        if (candidate_dataset, candidate_method) != (dataset, method):
            raise ValueError(
                f"{source_label}/{cell_key}: winner candidate_id names "
                f"{candidate_dataset!r}/{candidate_method!r}, not this cell's own "
                f"{dataset!r}/{method!r}"
            )
        seeds = winner.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != 3:
            raise ValueError(
                f"{source_label}/{cell_key}: winner must carry exactly 3 seed records, "
                f"got {len(seeds) if isinstance(seeds, list) else 'invalid'}"
            )
        run_ids: dict[str, str] = {}
        for seed_result in seeds:
            if not isinstance(seed_result, dict):
                raise ValueError(f"{source_label}/{cell_key}: seed record must be an object")
            seed = seed_result.get("seed")
            # score_highdim_adjudication_20260819.py's score_cell() only
            # ever assigns `winner` from a Candidate whose .eligible property
            # already required every seed to be complete/finite/nondivergent
            # -- re-checked here anyway so a hand-edited or malformed results
            # file can't slip an unfit seed through this adapter.
            if not (
                seed_result.get("status") == "complete"
                and seed_result.get("artifacts_complete") is True
                and seed_result.get("finite") is True
                and seed_result.get("diverged") is False
            ):
                raise ValueError(
                    f"{source_label}/{cell_key}: winner seed {seed} is not complete/"
                    "finite/nondivergent"
                )
            run_dir = seed_result.get("run_dir") or ""
            if not run_dir:
                raise ValueError(f"{source_label}/{cell_key}: winner seed {seed} has no run_dir recorded")
            if str(seed) in run_ids:
                raise ValueError(f"{source_label}/{cell_key}: duplicate seed {seed} among winner seeds")
            run_ids[str(seed)] = Path(run_dir).name
        if set(run_ids) != {"0", "1", "2"}:
            raise ValueError(
                f"{source_label}/{cell_key}: winner must carry run_ids for seeds 0, 1, 2, "
                f"got {sorted(run_ids)}"
            )
        if cell_name in cells:
            raise ValueError(f"duplicate cell {cell_name!r} across signal/X results")
        cells[cell_name] = {
            "dataset": dataset,
            "method": method,
            "winner": {"lr": lr, "cm": cm, "run_ids": run_ids},
        }


def combine(signal_results_path: Path, x_results_path: Path, output_path: Path) -> dict:
    signal = _load_json(signal_results_path)
    x = _load_json(x_results_path)

    cells: dict[str, dict] = {}
    _combine_stage("signal", signal, SIGNAL_CELLS, cells)
    _combine_stage("x", x, X_CELLS, cells)

    if set(cells) != ALL_CELL_NAMES:
        raise ValueError(
            f"combined output covers the wrong cells: "
            f"missing={sorted(ALL_CELL_NAMES - set(cells))}, "
            f"unexpected={sorted(set(cells) - ALL_CELL_NAMES)}"
        )

    result = {
        "status": "complete",
        "alpha": 0.5,
        "seeds": [0, 1, 2],
        "cells": cells,
        "source_hashes": [
            {"path": str(p.relative_to(REPO_ROOT)), "sha256": _sha256(p)}
            for p in sorted((signal_results_path, x_results_path))
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal-results", type=Path, default=DEFAULT_CAMPAIGN_DIR / "adjudication_signal_results.json")
    parser.add_argument("--x-results", type=Path, default=DEFAULT_CAMPAIGN_DIR / "adjudication_x_results.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_CAMPAIGN_DIR / "v4_winners.json")
    args = parser.parse_args()
    try:
        result = combine(args.signal_results.resolve(), args.x_results.resolve(), args.out.resolve())
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"V4 WINNERS ADAPTER BLOCKED: {exc}")
        return 2
    print(json.dumps({"status": result["status"], "cells": len(result["cells"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
