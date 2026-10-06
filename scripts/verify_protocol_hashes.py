#!/usr/bin/env python3
"""Verify a protocol packet's recorded SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import git_provenance  # noqa: E402

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(record_path: Path, root: Path = REPO_ROOT) -> dict[str, int]:
    with record_path.open() as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("hash record must be a nonempty JSON list")

    root = root.resolve()
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"hash record {index} must be an object")
        relative = str(record.get("path", ""))
        expected = str(record.get("sha256", ""))
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"hash record {index} has unsafe path {relative!r}")
        if relative in seen:
            raise ValueError(f"duplicate hash path: {relative}")
        seen.add(relative)
        if not SHA256_RE.fullmatch(expected):
            raise ValueError(f"invalid SHA-256 for {relative}")
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"hashed file is missing or outside the repository: {relative}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(
                f"hash mismatch for {relative}: expected {expected}, got {actual}"
            )
    return {"verified_hashes": len(records)}


def verify_git_provenance(record_path: Path, root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Requires the frozen git_revision/dirty_diff_sha256 (see
    highdim_protocol_hash_closure_20260822.git_provenance) to exactly match a
    fresh recomputation against the live tree -- the source-hash analogue of
    verify_hashes, but for repository state rather than individual files."""
    with record_path.open() as handle:
        stored = json.load(handle)
    if not isinstance(stored, dict) or "git_revision" not in stored:
        raise ValueError(f"{record_path} does not contain a git_provenance record")
    fresh = git_provenance(root)
    if stored.get("git_revision") != fresh["git_revision"]:
        raise ValueError(
            f"git_revision mismatch: recorded {stored.get('git_revision')!r}, "
            f"current {fresh['git_revision']!r}"
        )
    if stored.get("dirty_diff_sha256") != fresh["dirty_diff_sha256"]:
        raise ValueError(
            "dirty_diff_sha256 mismatch: the working tree diff has changed since "
            f"freeze -- recorded {stored.get('dirty_diff_sha256')!r}, current "
            f"{fresh['dirty_diff_sha256']!r}"
        )
    return fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--git-provenance",
        type=Path,
        default=None,
        help="Optional path to a JSON object with git_revision/dirty_diff_sha256, "
        "also verified against the live tree if given.",
    )
    args = parser.parse_args()
    try:
        summary = verify_hashes(args.hashes, args.root)
        if args.git_provenance is not None:
            summary["git_provenance"] = verify_git_provenance(args.git_provenance, args.root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"HASH VERIFICATION FAILED: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
