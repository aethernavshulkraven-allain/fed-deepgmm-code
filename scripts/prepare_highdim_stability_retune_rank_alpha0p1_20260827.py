#!/usr/bin/env python3
"""Prepare the alpha=0.1 retune fallback's Rank stage (closeout plan SS9.1
escape hatch; doe_review_and_revised_grid.md Part VI/VII): for each retuned
cell, launch its Screen-stage top-2 candidates for 500 rounds at seed 0.

Consumes scripts/score_highdim_stability_retune_alpha0p1_20260827.py's
"top2"-shaped output. Each row is built from a real screen-manifest
template row for the cell, exactly like every other preparer in this
campaign, so every column run_manifest.py's build_config() requires without
a default is present.

Rank's seed-0 runs are reused directly by the Confirm stage (the frozen
Rank/Confirm de-duplication in doe_review_and_revised_grid.md Part VII:
"Confirm now runs seeds {1,2} only and takes seed 0 from Rank") -- Confirm
never re-launches seed 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import (  # noqa: E402
    CORE_DATASET_FILES,
    CORE_PROTOCOL_DOCS,
    CORE_SOURCES,
)

PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
DEFAULT_SCREEN_MANIFEST = (
    PROTOCOL_ROOT / "deterministic_screen_post_bn_20260822" / "screen_manifest.csv"
)
DEFAULT_OUTPUT_DIR = PROTOCOL_ROOT / "deterministic_stability_retune_rank_alpha0p1_20260827"
RESULT_ROOT = "results/highdim_deterministic_stability_retune_rank_alpha0p1_20260827"
RANK_COMM_ROUND = 500
EXTRA_FIELDS = ("compact_predictions_only", "source_screen_results")


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _run_id(dataset: str, method: str, lr: float, cm: float) -> str:
    return (
        f"det_stability_retune_rank_alpha0p1_{dataset}_{method}_seed0_"
        f"lr{_token(lr)}_cm{_token(cm)}"
    )


def prepare(screen_results_path: Path, screen_manifest_path: Path, output_dir: Path) -> dict:
    screen = _load_json(screen_results_path)
    if screen.get("status") != "complete":
        raise ValueError("screen results are absent or incomplete")
    if screen.get("stage") != "screen":
        raise ValueError(f"expected the Screen stage's own output, got stage={screen.get('stage')!r}")
    cells = screen.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("screen results must contain at least one cell")

    with screen_manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        templates: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            templates.setdefault((row["dataset"], row["method"]), row)
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    plan = []
    for cell_name in sorted(cells):
        cell = cells[cell_name]
        dataset, method = cell["dataset"], cell["method"]
        if f"{dataset}|{method}" != cell_name:
            raise ValueError(f"cell key {cell_name!r} does not match its dataset/method fields")
        top2 = cell.get("top2")
        if not isinstance(top2, list) or len(top2) != 2:
            raise ValueError(f"{cell_name}: screen results must carry exactly 2 top2 candidates")
        template = templates.get((dataset, method))
        if template is None:
            raise ValueError(f"{cell_name}: no screen-manifest template row for this cell")
        candidates = []
        for candidate in top2:
            lr, cm = float(candidate["lr"]), float(candidate["cm"])
            run_id = _run_id(dataset, method, lr, cm)
            row = dict(template)
            row.update({
                "run_id": run_id,
                "protocol_version": "highdim_deterministic_stability_retune_rank_alpha0p1_v1",
                "run_group": "highdim_deterministic_stability_retune_rank_alpha0p1_20260827",
                "seed": "0",
                "alpha": "0.1",
                "partition_alpha": "0.1",
                "learning_rate": f"{lr:g}",
                "critic_multiplier": f"{cm:g}",
                "comm_round": str(RANK_COMM_ROUND),
                "output_root": RESULT_ROOT,
                "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_0/{run_id}",
                "run_status": "not_started",
                "preflight_required": "True",
                "preflight_status": "bn_buffer_diagnostic_certified",
                "server_buffer_policy": "direct_client_aggregate",
                "compact_predictions_only": "True",
                "source_manifest": "",
                "source_run_id": "",
                "source_screen_results": _relative_or_str(screen_results_path),
                "notes": (
                    f"alpha=0.1 retune Rank stage for {cell_name} (closeout plan SS9.1 "
                    f"fallback): top-2 Screen candidate at lr={lr:g}, cm={cm:g}, "
                    f"seed 0, {RANK_COMM_ROUND} rounds."
                ),
            })
            rows.append(row)
            candidates.append({"lr": lr, "cm": cm, "run_id": run_id})
        plan.append({"cell": cell_name, "dataset": dataset, "method": method, "candidates": candidates})

    manifest_path = output_dir / "rank_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "rank_summary.json"
    summary_path.write_text(json.dumps({
        "campaign": "highdim_deterministic_stability_retune_rank_alpha0p1_20260827",
        "run_count": len(rows),
        "alpha": 0.1,
        "seed": 0,
        "comm_round": RANK_COMM_ROUND,
        "server_buffer_policy": "direct_client_aggregate",
        "source_screen_results": _relative_or_str(screen_results_path),
        "source_screen_manifest": _relative_or_str(screen_manifest_path),
        "plan": plan,
    }, indent=2, sort_keys=True) + "\n")

    hashed_paths = [
        manifest_path, summary_path, screen_results_path, screen_manifest_path,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        *(REPO_ROOT / doc for doc in CORE_PROTOCOL_DOCS),
        REPO_ROOT / "scripts/prepare_highdim_stability_retune_confirm_alpha0p1_20260827.py",
        REPO_ROOT / "scripts/score_highdim_stability_retune_promote_alpha0p1_20260827.py",
        REPO_ROOT / "scripts/launch_highdim_stability_retune_rank_alpha0p1_20260827.sh",
        Path(__file__),
    ]
    (output_dir / "generated_artifact_hashes.json").write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in sorted(hashed_paths)
    ], indent=2, sort_keys=True) + "\n")

    return {"manifest": str(manifest_path), "cells": len(plan), "runs": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-results", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, default=DEFAULT_SCREEN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = prepare(
            args.screen_results.resolve(), args.screen_manifest.resolve(), args.output_dir.resolve(),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"RANK PREPARATION BLOCKED: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
