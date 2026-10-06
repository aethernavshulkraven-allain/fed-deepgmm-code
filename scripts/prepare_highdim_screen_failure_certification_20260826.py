#!/usr/bin/env python3
"""Prepare the dedicated certification namespace for the four screen rows
that failed before federated round 0 (closeout plan SS6.2). Exact same
hyperparameters as their original screen rows -- this is a deterministic
reproduction check, not a retry with different hyperparameters -- in a new
result namespace so the original screen directories are never touched.
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

SCREEN_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
    / "screen_manifest.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/screen_failure_certification_20260826"
)
RESULT_ROOT = "results/highdim_screen_failure_certification_20260826"

# The four rows POST_BN_SCREEN_REVIEW_20260826.md identified as never
# starting federated training (best_score=-inf before round 0).
TARGETS = (
    ("femnist_z", "fedgda_d", 0.1, 5.0),
    ("cifar10_x", "fedgda_d", 0.333333, 10.0),
    ("femnist_x", "fedgda_d", 0.333333, 10.0),
    ("femnist_x", "fedgda_d", 0.333333, 20.0),
)


def main() -> int:
    with SCREEN_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        screen_rows = list(reader)

    selected = []
    for dataset, method, lr, cm in TARGETS:
        matches = [
            row for row in screen_rows
            if row["dataset"] == dataset and row["method"] == method
            and abs(float(row["learning_rate"]) - lr) < 1e-9
            and abs(float(row["critic_multiplier"]) - cm) < 1e-9
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one screen row for {(dataset, method, lr, cm)}, got {len(matches)}"
            )
        selected.append(matches[0])

    rows = []
    for source in selected:
        run_id = f"screen_failure_cert_20260826_{source['run_id']}"
        row = dict(source)
        row.update({
            "run_id": run_id,
            "run_group": "highdim_screen_failure_certification_20260826",
            "output_root": RESULT_ROOT,
            "final_result_dir": (
                f"{RESULT_ROOT}/{source['dataset']}/{source['method']}/"
                f"seed_{source['seed']}/{run_id}"
            ),
            "notes": (
                f"Deterministic reproduction of screen row {source['run_id']} "
                "(closeout plan SS6.2) -- exact same hyperparameters, new "
                "namespace, original screen directory never touched."
            ),
        })
        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "screen_failure_certification_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    hashed_paths = sorted({
        manifest_path,
        SCREEN_MANIFEST,
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        *(REPO_ROOT / doc for doc in CORE_PROTOCOL_DOCS),
        Path(__file__),
    })
    hashes_path = OUTPUT_DIR / "generated_artifact_hashes.json"
    hashes_path.write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for path in hashed_paths
    ], indent=2, sort_keys=True) + "\n")

    provenance_path = OUTPUT_DIR / "git_provenance.json"
    provenance_path.write_text(json.dumps(git_provenance(), indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "manifest": str(manifest_path),
        "hashes": str(hashes_path),
        "run_ids": [row["run_id"] for row in rows],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
