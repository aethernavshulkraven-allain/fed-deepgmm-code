#!/usr/bin/env python3
"""Prepare the per-cell alpha=0.1 retune fallback (closeout plan SS9.1's
"the fallback branch must be implemented before launching [the stability]
stage. It must not be designed after observing which cells fail.").

Written and tested before any real stability result exists -- there is no
retune_required cell yet, since V4 and stability have not been launched
(Phase 5/6 are out of scope for this closeout pass).

Grid choice: reuses the exact (learning_rate, critic_multiplier) grid
already tested for that (dataset, method) cell in the frozen corrected
screen -- not a newly-invented grid -- at alpha=0.1 instead of alpha=0.5,
seed 0, the screen's 150-round length (this is a re-screen for one cell, not
a 500-round confirmation). This mirrors doe_review_and_revised_grid.md's
extended-plan fallback ("per-(alpha, scenario, method) tuning") using
machinery already frozen for the corrected screen, rather than inventing a
new procedure post-hoc.

Only prepares rows for cells listed in --stability-results'
retune_required_cells; a cell that passed is never touched.
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
DEFAULT_OUTPUT_DIR = PROTOCOL_ROOT / "deterministic_stability_retune_alpha0p1_20260827"
RESULT_ROOT = "results/highdim_deterministic_stability_retune_alpha0p1_20260827"
# compact_predictions_only isn't a screen_manifest.csv column, and
# source_screen_run_id is new to this stage -- both must be added to the
# generated manifest's own fieldnames or DictWriter's extrasaction="ignore"
# silently drops them from every written row.
EXTRA_FIELDS = ("compact_predictions_only", "source_screen_run_id")


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


def prepare(stability_results_path: Path, screen_manifest_path: Path, output_dir: Path) -> dict:
    stability = _load_json(stability_results_path)
    if stability.get("status") != "complete":
        raise ValueError("stability results are absent or incomplete")
    stability_cells = stability.get("cells")
    if not isinstance(stability_cells, dict) or len(stability_cells) != 12:
        raise ValueError("stability results must contain exactly 12 cells")
    retune_cells = sorted(
        name for name, cell in stability_cells.items() if cell.get("outcome") != "pass"
    )
    if not retune_cells:
        raise ValueError(
            "no cell requires retuning -- nothing to prepare; this script is "
            "the SS9.1 fallback branch, not a general-purpose re-screen"
        )

    with screen_manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        screen_rows_by_cell: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            cell_name = f"{row['dataset']}|{row['method']}"
            screen_rows_by_cell.setdefault(cell_name, []).append(row)
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    plan = []
    for cell_name in retune_cells:
        dataset, method = cell_name.split("|", 1)
        source_rows = screen_rows_by_cell.get(cell_name)
        if not source_rows:
            raise ValueError(f"{cell_name}: no screen-manifest rows to build a retune grid from")
        cell_rows = []
        for source in source_rows:
            lr = float(source["learning_rate"])
            cm = float(source["critic_multiplier"])
            run_id = (
                f"det_stability_retune_alpha0p1_{dataset}_{method}_seed0_"
                f"lr{_token(lr)}_cm{_token(cm)}"
            )
            row = dict(source)
            row.update({
                "run_id": run_id,
                "protocol_version": "highdim_deterministic_stability_retune_alpha0p1_v1",
                "run_group": "highdim_deterministic_stability_retune_alpha0p1_20260827",
                "seed": "0",
                "alpha": "0.1",
                "partition_alpha": "0.1",
                "learning_rate": f"{lr:g}",
                "critic_multiplier": f"{cm:g}",
                "comm_round": "150",
                "output_root": RESULT_ROOT,
                "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_0/{run_id}",
                "run_status": "not_started",
                "preflight_required": "True",
                "preflight_status": "bn_buffer_diagnostic_certified",
                "server_buffer_policy": "direct_client_aggregate",
                "compact_predictions_only": "True",
                # This row's real, direct lineage is the exact screen row
                # whose (lr, cm) it reuses -- recorded precisely here.
                # source_manifest/source_run_id on `source` describe a
                # different, two-hops-removed thing (where the screen row's
                # own hyperparameter combo was discovered), which is no
                # longer accurate for this new retune row and is cleared
                # rather than carried forward.
                "source_manifest": "",
                "source_run_id": "",
                "source_screen_run_id": source["run_id"],
                "notes": (
                    f"alpha=0.1 per-cell retune for {cell_name} (closeout plan SS9.1 "
                    "fallback) -- same (lr, cm) grid already tested for this cell in "
                    "the corrected screen, re-run at alpha=0.1."
                ),
            })
            rows.append(row)
            cell_rows.append({"lr": lr, "cm": cm, "run_id": run_id})
        plan.append({"cell": cell_name, "dataset": dataset, "method": method, "candidates": cell_rows})

    manifest_path = output_dir / "retune_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "retune_summary.json"
    summary_path.write_text(json.dumps({
        "campaign": "highdim_deterministic_stability_retune_alpha0p1_20260827",
        "retune_cells": retune_cells,
        "run_count": len(rows),
        "alpha": 0.1,
        "seed": 0,
        "comm_round": 150,
        "server_buffer_policy": "direct_client_aggregate",
        "source_stability_results": _relative_or_str(stability_results_path),
        "source_screen_manifest": _relative_or_str(screen_manifest_path),
        "plan": plan,
    }, indent=2, sort_keys=True) + "\n")

    hashed_paths = [
        manifest_path, summary_path, stability_results_path, screen_manifest_path,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        *(REPO_ROOT / doc for doc in CORE_PROTOCOL_DOCS),
        REPO_ROOT / "scripts/score_highdim_stability_retune_alpha0p1_20260827.py",
        REPO_ROOT / "scripts/launch_highdim_stability_retune_alpha0p1_20260827.sh",
        Path(__file__),
    ]
    (output_dir / "generated_artifact_hashes.json").write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in sorted(hashed_paths)
    ], indent=2, sort_keys=True) + "\n")

    return {"manifest": str(manifest_path), "retune_cells": retune_cells, "runs": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stability-results", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, default=DEFAULT_SCREEN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = prepare(
            args.stability_results.resolve(), args.screen_manifest.resolve(), args.output_dir.resolve(),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"RETUNE PREPARATION BLOCKED: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
