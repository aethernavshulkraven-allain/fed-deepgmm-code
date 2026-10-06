#!/usr/bin/env python3
"""Build the certification ledger linking each of the 4 screen rows that
failed before federated round 0 to its POST-HARDENING reproduction
(closeout plan SS6.2), superseding apply_screen_failure_certification_
20260826.py's ledger now that hash_bundle_sha256 is mandatory.

Never writes into the original screen row directories, nor into the
2026-08-26 certification directory: PROTOCOL_DECISION_ADDENDUM_20260826.md
requires terminal-pretraining evidence to be "written by the training
process itself at the moment of failure." This script only records the
link between the original screen run_id and the fresh reproduction's real
run_id, for score_highdim_screen_post_bn_20260822.py and
check_manifest_stage_complete.py to resolve via
run_manifest.resolve_certified_run().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import validate_pretraining_failure_artifact  # noqa: E402

SCREEN_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822"
    / "screen_manifest.csv"
)
CERT_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/screen_failure_certification_post_hardening_20260828"
    / "screen_failure_certification_manifest.csv"
)
LEDGER_PATH = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/screen_failure_certification_post_hardening_20260828"
    / "certification_ledger.json"
)
RUN_ID_PREFIX = "screen_failure_cert_post_hardening_20260828_"


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def build_ledger() -> dict:
    import csv

    with SCREEN_MANIFEST.open(newline="") as handle:
        screen_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    with CERT_MANIFEST.open(newline="") as handle:
        cert_rows = list(csv.DictReader(handle))

    ledger = {}
    for cert_row in cert_rows:
        cert_run_id = cert_row["run_id"]
        cert_run_dir = Path(cert_row["final_result_dir"])
        if not cert_run_dir.is_absolute():
            cert_run_dir = REPO_ROOT / cert_run_dir

        # The certification's own row (real run_id, real directory) -- must
        # validate as real, process-authored evidence before it goes in the
        # ledger at all.
        validate_pretraining_failure_artifact(cert_run_dir, cert_row)

        original_run_id = cert_run_id.removeprefix(RUN_ID_PREFIX)
        if original_run_id not in screen_rows:
            raise ValueError(f"cannot find original screen row for {cert_run_id}")
        original_run_dir = Path(screen_rows[original_run_id]["final_result_dir"])
        if not original_run_dir.is_absolute():
            original_run_dir = REPO_ROOT / original_run_dir
        if (original_run_dir / "pretraining_failure.json").exists():
            raise ValueError(
                f"{original_run_id}: the original screen directory already contains a "
                "pretraining_failure.json -- refusing to build a ledger over an "
                "unexpectedly non-pristine original directory"
            )

        ledger[original_run_id] = {
            "certified_run_id": cert_run_id,
            "certified_run_dir": str(cert_run_dir.relative_to(REPO_ROOT)) if cert_run_dir.is_relative_to(REPO_ROOT) else str(cert_run_dir),
        }

    return ledger


def main() -> int:
    ledger = build_ledger()
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ledger": str(LEDGER_PATH), "entries": sorted(ledger)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
