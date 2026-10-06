#!/usr/bin/env python3
"""Downstream verifier for bn_buffer_diagnostic_certification.json.

A JSON field saying certification_status=passed is not sufficient evidence
that the diagnostic actually still passes -- the file could be stale (code
changed underneath it) or hand-edited. This recomputes the certification
from scratch (re-hashing every artifact file, re-deriving every criterion
from the raw manifest/metrics/curve data via the same certify() the
producer uses) and requires the fresh result to match the stored file
exactly, field for field. It never trusts the stored file's own claims and
never overwrites it -- a read-only gate, meant to run before every
downstream stage that depends on the diagnostic having passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from certify_highdim_bn_diagnostic_20260822 import certify  # noqa: E402


def verify(
    certification_path: Path,
    manifest_path: Path,
    launcher_results_path: Path,
    launch_hashes_path: Path,
) -> dict:
    if not certification_path.is_file():
        raise ValueError(f"no certification on disk at {certification_path}")
    stored = json.loads(certification_path.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        scratch_path = Path(tmp) / "recomputed_certification.json"
        fresh = certify(manifest_path, launcher_results_path, launch_hashes_path, scratch_path)

    if fresh != stored:
        raise ValueError(
            "stored certification does not match a fresh recomputation from "
            f"the same inputs -- stored={json.dumps(stored, sort_keys=True)} "
            f"fresh={json.dumps(fresh, sort_keys=True)}"
        )
    if stored.get("certification_status") != "passed":
        raise ValueError(f"certification_status is {stored.get('certification_status')!r}, not 'passed'")
    return {"verified": True, "run_id": stored.get("run_id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launcher-results", type=Path, required=True)
    parser.add_argument("--launch-hashes", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(
            args.certification.resolve(),
            args.manifest.resolve(),
            args.launcher_results.resolve(),
            args.launch_hashes.resolve(),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"DIAGNOSTIC CERTIFICATION VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
