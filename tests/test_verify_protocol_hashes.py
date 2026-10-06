"""Tests for immutable protocol hash verification."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from verify_protocol_hashes import sha256_file, verify_hashes  # noqa: E402


class ProtocolHashTests(unittest.TestCase):
    def test_matching_hash_passes_and_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "packet.txt"
            target.write_text("frozen\n")
            records = root / "hashes.json"
            records.write_text(json.dumps([
                {"path": "packet.txt", "sha256": sha256_file(target)}
            ]))
            self.assertEqual(verify_hashes(records, root)["verified_hashes"], 1)
            target.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_hashes(records, root)

    def test_unsafe_and_duplicate_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "hashes.json"
            records.write_text(json.dumps([
                {"path": "../outside", "sha256": "0" * 64}
            ]))
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                verify_hashes(records, root)


if __name__ == "__main__":
    unittest.main()
