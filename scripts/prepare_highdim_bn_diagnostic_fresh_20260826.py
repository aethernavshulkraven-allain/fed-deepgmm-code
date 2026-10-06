#!/usr/bin/env python3
"""Prepare the fresh 120-round BatchNorm diagnostic (closeout plan SS6.1),
in a namespace separate from the retrospectively certified v3 diagnostic,
under the expanded post-2026-08-26 hash closure (data loaders, Psi/model-
selection/optimizer modules, dataset NPZs, git-revision + dirty-diff
provenance) and the accompanying source edits.

Exact same scientific configuration as v3's diagnostic row (dataset,
method, seed, comm_round, learning_rate, critic_multiplier, and every other
train_args field) -- only identifiers and output paths change. This is a
repeat of the same diagnostic under a fresh freeze, not a new diagnostic.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import (  # noqa: E402
    CORE_DATASET_FILES,
    CORE_PROTOCOL_DOCS,
    CORE_SOURCES,
    git_provenance,
)
from verify_protocol_hashes import sha256_file  # noqa: E402

V3_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
    / "bn_buffer_diagnostic_manifest.csv"
)
OUTPUT_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/bn_diagnostic_fresh_20260826"
RESULT_ROOT = "results/highdim_bn_diagnostic_fresh_20260826"
RUN_ID = "bn_diagnostic_fresh_20260826_femnist_z_fedogda_d_seed1_lr0p001_cm10"

# Fields that change between the v3 row and this fresh row. Every other
# field (the full scientific configuration) is copied verbatim.
OVERRIDES = {
    "run_id": RUN_ID,
    "run_group": "highdim_bn_diagnostic_fresh_20260826",
    "output_root": RESULT_ROOT,
    "final_result_dir": f"{RESULT_ROOT}/femnist_z/fedogda_d/seed_1/{RUN_ID}",
    "notes": (
        "Fresh repeat of the v3 120-round BatchNorm diagnostic under the "
        "post-2026-08-26 expanded hash closure (closeout plan SS6.1); same "
        "scientific configuration as psi_adjudication_20260822_v3's row, "
        "new namespace and provenance only."
    ),
}


def main() -> int:
    with V3_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one v3 diagnostic row, got {len(rows)}")
    row = dict(rows[0])
    row.update(OVERRIDES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "bn_buffer_diagnostic_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    hashed_paths = sorted({
        manifest_path,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        *(REPO_ROOT / doc for doc in CORE_PROTOCOL_DOCS),
        Path(__file__),
    })
    launch_hashes_path = OUTPUT_DIR / "diagnostic_launch_hashes.json"
    launch_hashes_path.write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for path in hashed_paths
    ], indent=2, sort_keys=True) + "\n")

    provenance_path = OUTPUT_DIR / "git_provenance.json"
    provenance_path.write_text(json.dumps(git_provenance(), indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "manifest": str(manifest_path),
        "launch_hashes": str(launch_hashes_path),
        "git_provenance": str(provenance_path),
        "run_id": row["run_id"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
