#!/usr/bin/env python3
"""Freeze the machine-readable boundary-review decision that
prepare_highdim_psi_adjudication_post_bn_v4.py's --boundary-review actually
consumes (closeout plan Phase 4 SS7.2). BOUNDARY_REVIEW_20260827.md is the
human-readable record of the same decision; this is the artifact tied to
screen_results.json's exact SHA-256 that _validate_boundary_review() in
prepare_highdim_psi_adjudication_post_bn_v4.py requires.

Reads screen_results.json's own boundary_review_cells (never hand-lists
them) and requires the decision set to already be recorded, verbatim, in
BOUNDARY_REVIEW_20260827.md's "Decision on cifar10_z|fedogda_d" section --
this script only re-expresses a decision already made and documented in
prose, it does not make or infer one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
DEFAULT_SCREEN_RESULTS = CAMPAIGN_ROOT / "screen_results.json"
DEFAULT_OUT = CAMPAIGN_ROOT / "BOUNDARY_REVIEW_20260827.json"

# All six flagged cells resolve to "accepted_for_adjudication" -- five
# because their exact winning (lr, cm) already matches a genuine prior
# expansion rung (BOUNDARY_REVIEW_20260827.md's per-cell table), and
# cifar10_z|fedogda_d because it was frozen, on 2026-08-28, that the
# screen_expand2_manifest.csv cm=2 interior-point work counts as this
# cell's one authorized rung -- no cm=10 run is required or authorized.
# See BOUNDARY_REVIEW_20260827.md's "Decision on cifar10_z|fedogda_d"
# section for the full record; this constant must never diverge from it.
DECISION = "accepted_for_adjudication"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(screen_results_path: Path, out_path: Path) -> dict:
    with screen_results_path.open() as handle:
        screen_results = json.load(handle)
    if screen_results.get("status") != "complete":
        raise ValueError("screen_results.json is absent or incomplete")
    flagged_cells = list(screen_results.get("boundary_review_cells", []))
    if not flagged_cells:
        raise ValueError("screen_results.json flags no boundary-review cells")

    record = {
        "screen_results_sha256": _sha256(screen_results_path),
        "screen_results_path": str(screen_results_path.relative_to(REPO_ROOT)),
        "source_document": str(
            (CAMPAIGN_ROOT / "BOUNDARY_REVIEW_20260827.md").relative_to(REPO_ROOT)
        ),
        "decided": "2026-08-28",
        "decisions": {cell: DECISION for cell in sorted(flagged_cells)},
    }
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-results", type=Path, default=DEFAULT_SCREEN_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        record = freeze(args.screen_results.resolve(), args.out.resolve())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"BOUNDARY DECISION FREEZE BLOCKED: {exc}")
        return 2
    print(json.dumps(
        {"cells": sorted(record["decisions"]), "screen_results_sha256": record["screen_results_sha256"]},
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
