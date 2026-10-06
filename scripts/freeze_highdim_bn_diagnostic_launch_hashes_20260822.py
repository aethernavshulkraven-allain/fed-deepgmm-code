#!/usr/bin/env python3
"""Freeze diagnostic_launch_hashes.json for the 120-round BatchNorm
diagnostic: the manifest plus the full decision/execution-critical source
closure (scripts/highdim_protocol_hash_closure_20260822.py), so
certify_highdim_bn_diagnostic_20260822.py and
verify_highdim_bn_diagnostic_certification_20260822.py can prove the
diagnostic's result actually came from this exact code, not just the two
files (experiment_utils.py, fedavg_api.py) narrowly checked before.

Re-run this whenever any file in the closure changes for a real reason
(such as fedavg_api.py's requires_grad filtering fix) -- then re-run
certify_highdim_bn_diagnostic_20260822.py so the certification matches
again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import CORE_SOURCES  # noqa: E402
from verify_protocol_hashes import sha256_file  # noqa: E402

CAMPAIGN_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
OUTPUT = CAMPAIGN_DIR / "diagnostic_launch_hashes.json"
MANIFEST = CAMPAIGN_DIR / "bn_buffer_diagnostic_manifest.csv"


def main() -> int:
    paths = sorted({str(MANIFEST.relative_to(REPO_ROOT)), *CORE_SOURCES})
    records = [
        {"path": path, "sha256": sha256_file(REPO_ROOT / path)}
        for path in paths
    ]
    OUTPUT.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(REPO_ROOT)), "files": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
