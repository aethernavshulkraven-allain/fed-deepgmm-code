"""Tests for the manifest-stage barrier used by sequential campaign launchers."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_manifest_stage_complete import check_stage  # noqa: E402


class StageCompleteTests(unittest.TestCase):
    def _files(self, root: Path, results: list[dict]) -> tuple[Path, Path]:
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id"])
            writer.writeheader()
            writer.writerows([{"run_id": "a"}, {"run_id": "b"}])
        result_path = root / "results.json"
        result_path.write_text(json.dumps(results))
        return manifest, result_path

    def test_all_resolved_statuses_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, results = self._files(Path(tmp), [
                {"run_id": "a", "status": "passed"},
                {"run_id": "b", "status": "skipped_terminal_ineligible"},
            ])
            summary = check_stage(manifest, results)
            self.assertEqual(summary["resolved_rows"], 2)
            self.assertEqual(summary["terminal_ineligible"], 1)
            with self.assertRaisesRegex(ValueError, "skipped_terminal_ineligible"):
                check_stage(manifest, results, require_clean=True)

    def test_missing_or_failed_row_blocks_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, results = self._files(Path(tmp), [
                {"run_id": "a", "status": "passed"},
            ])
            with self.assertRaisesRegex(ValueError, r"missing=\['b'\]"):
                check_stage(manifest, results)

            results.write_text(json.dumps([
                {"run_id": "a", "status": "passed"},
                {"run_id": "b", "status": "failed_validation"},
            ]))
            with self.assertRaisesRegex(ValueError, "failed_validation"):
                check_stage(manifest, results)

    def test_duplicate_result_blocks_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, results = self._files(Path(tmp), [
                {"run_id": "a", "status": "passed"},
                {"run_id": "a", "status": "passed"},
                {"run_id": "b", "status": "passed"},
            ])
            with self.assertRaisesRegex(ValueError, "duplicate launcher-result"):
                check_stage(manifest, results)

    def test_empty_manifest_blocks_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text("run_id\n")
            results = root / "results.json"
            results.write_text("[]")
            with self.assertRaisesRegex(ValueError, "no runs"):
                check_stage(manifest, results)

    def test_artifact_status_must_agree_with_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, results = self._files(root, [
                {"run_id": "a", "status": "passed", "run_dir": str(root)},
                {"run_id": "b", "status": "skipped_completed", "run_dir": str(root)},
            ])
            rows = list(csv.DictReader(manifest.open()))
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run_id", "final_result_dir"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({"run_id": row["run_id"], "final_result_dir": root})
            with mock.patch(
                "check_manifest_stage_complete.validate_artifacts",
                return_value={"terminal_ineligible": False},
            ) as validator:
                check_stage(manifest, results, validate_stage_artifacts=True)
            self.assertEqual(validator.call_count, 2)

            with mock.patch(
                "check_manifest_stage_complete.validate_artifacts",
                return_value={"terminal_ineligible": True},
            ):
                with self.assertRaisesRegex(ValueError, "classification mismatch"):
                    check_stage(manifest, results, validate_stage_artifacts=True)


if __name__ == "__main__":
    unittest.main()
