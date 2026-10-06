import csv
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "preflight_eicu_release.py"
SPEC_PATH = (
    REPO_ROOT
    / "experiments"
    / "eicu_full_data_preflight"
    / "required_tables.json"
)
MODULE_SPEC = importlib.util.spec_from_file_location("preflight_eicu_release", SCRIPT)
preflight = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(preflight)


class PreflightEicuReleaseTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        with SPEC_PATH.open(encoding="utf-8") as handle:
            self.specification = json.load(handle)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _write_gzip_csv(path, columns, rows=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows or [])
        os.utime(path, (1_700_000_000, 1_700_000_000))

    def _make_release(
        self,
        root,
        *,
        include_optional=True,
        omit_table=None,
        omit_column=None,
        case_variant=None,
        patient_rows=None,
    ):
        categories = ["required_tables"]
        if include_optional:
            categories += ["optional_tables", "sensitivity_tables"]
        for category in categories:
            for table in self.specification[category]:
                if table["name"] == omit_table:
                    continue
                columns = list(table["columns"])
                if omit_column and table["name"] == omit_column[0]:
                    columns.remove(omit_column[1])
                filename = f"{table['name']}.csv.gz"
                if table["name"] == case_variant:
                    filename = f"{table['name'].upper()}.CSV.GZ"
                rows = []
                if table["name"] == "patient" and patient_rows:
                    rows = patient_rows
                self._write_gzip_csv(root / filename, columns, rows)
        return root

    def _run_cli(self, root, out, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--eicu-root",
                str(root),
                "--out",
                str(out),
                *extra,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_complete_mini_demo_like_release(self):
        root = self._make_release(self.base / "eicu-crd-demo" / "2.0.1")
        out = self.base / "out"

        completed = self._run_cli(root, out)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((out / preflight.OUTPUT_JSON).read_text())
        self.assertEqual(report["release_classification"], "demo")
        self.assertTrue(report["launchable_for_demo_smoke"])
        self.assertFalse(report["launchable_for_full_cohort_build"])
        self.assertTrue(report["require_full_satisfied"])
        self.assertFalse(report["blocking_reasons"])
        self.assertTrue((out / preflight.OUTPUT_MARKDOWN).is_file())

    def test_missing_required_table_is_blocking(self):
        root = self._make_release(
            self.base / "release", include_optional=False, omit_table="lab"
        )
        out = self.base / "out"

        completed = self._run_cli(root, out)

        self.assertEqual(completed.returncode, 1)
        report = json.loads((out / preflight.OUTPUT_JSON).read_text())
        self.assertFalse(report["launchable_for_demo_smoke"])
        self.assertIn("Missing required table: lab.", report["blocking_reasons"])

    def test_missing_required_column_is_blocking(self):
        root = self._make_release(
            self.base / "release",
            include_optional=False,
            omit_column=("diagnosis", "icd9code"),
        )
        out = self.base / "out"

        completed = self._run_cli(root, out)

        self.assertEqual(completed.returncode, 1)
        report = json.loads((out / preflight.OUTPUT_JSON).read_text())
        diagnosis = next(item for item in report["tables"] if item["name"] == "diagnosis")
        self.assertEqual(diagnosis["header_status"], "missing_columns")
        self.assertEqual(diagnosis["missing_required_columns"], ["icd9code"])

    def test_filename_resolution_is_case_insensitive(self):
        root = self._make_release(
            self.base / "release", include_optional=False, case_variant="infusiondrug"
        )
        report = preflight.run_preflight(root, self.base / "out")
        infusion = next(item for item in report["tables"] if item["name"] == "infusiondrug")

        self.assertTrue(report["launchable_for_demo_smoke"])
        self.assertEqual(infusion["resolved_filename"], "INFUSIONDRUG.CSV.GZ")
        self.assertEqual(infusion["header_status"], "ok")

    def test_require_full_rejects_demo_path(self):
        root = self._make_release(self.base / "eicu-crd-demo" / "2.0.1")
        out = self.base / "out"

        completed = self._run_cli(root, out, "--require-full")

        self.assertEqual(completed.returncode, 1)
        report = json.loads((out / preflight.OUTPUT_JSON).read_text())
        self.assertFalse(report["require_full_satisfied"])
        self.assertTrue(
            any("Full eICU was required" in reason for reason in report["blocking_reasons"])
        )

    def test_standard_non_demo_release_path_is_likely_full(self):
        root = self._make_release(self.base / "eicu-crd" / "2.0")
        out = self.base / "out"

        completed = self._run_cli(root, out, "--require-full")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((out / preflight.OUTPUT_JSON).read_text())
        self.assertEqual(report["release_classification"], "likely_full")
        self.assertTrue(report["launchable_for_full_cohort_build"])
        self.assertTrue(report["require_full_satisfied"])

    def test_dry_run_writes_nothing(self):
        root = self._make_release(self.base / "eicu-crd-demo" / "2.0.1")
        out = self.base / "not-created"

        completed = self._run_cli(root, out, "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["release_classification"], "demo")
        self.assertFalse(out.exists())

    def test_checksum_is_opt_in_and_sha256_is_over_file_bytes(self):
        root = self._make_release(self.base / "release", include_optional=False)
        without = preflight.run_preflight(root, self.base / "out")
        with_checksum = preflight.run_preflight(
            root, self.base / "out", checksum=True
        )

        required_without = [
            item for item in without["tables"] if item["category"] == "required"
        ]
        self.assertTrue(all("sha256" not in item for item in required_without))
        patient = next(item for item in with_checksum["tables"] if item["name"] == "patient")
        expected = hashlib.sha256(Path(patient["path"]).read_bytes()).hexdigest()
        self.assertEqual(patient["sha256"], expected)

    def test_outputs_do_not_contain_patient_values_or_identifiers(self):
        sentinel_id = "PATIENT_IDENTIFIER_987654321"
        sentinel_value = "PRIVATE_PATIENT_VALUE"
        patient_spec = next(
            item
            for item in self.specification["required_tables"]
            if item["name"] == "patient"
        )
        row = [sentinel_id] + [sentinel_value] * (len(patient_spec["columns"]) - 1)
        root = self._make_release(
            self.base / "eicu-crd-demo" / "2.0.1",
            patient_rows=[row],
        )
        out = self.base / "out"

        completed = self._run_cli(root, out, "--count-patient-rows")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        json_text = (out / preflight.OUTPUT_JSON).read_text()
        markdown_text = (out / preflight.OUTPUT_MARKDOWN).read_text()
        self.assertNotIn(sentinel_id, json_text)
        self.assertNotIn(sentinel_value, json_text)
        self.assertNotIn(sentinel_id, markdown_text)
        self.assertNotIn(sentinel_value, markdown_text)
        self.assertEqual(json.loads(json_text)["patient_table_rows"], 1)

    def test_json_keys_and_table_order_are_deterministic(self):
        root = self._make_release(self.base / "release")
        out = self.base / "out"
        completed = self._run_cli(root, out)
        self.assertEqual(completed.returncode, 0, completed.stderr)

        report = json.loads((out / preflight.OUTPUT_JSON).read_text())
        self.assertEqual(list(report), sorted(report))
        actual = [(item["category"], item["name"].lower()) for item in report["tables"]]
        self.assertEqual(actual, sorted(actual))


if __name__ == "__main__":
    unittest.main()
