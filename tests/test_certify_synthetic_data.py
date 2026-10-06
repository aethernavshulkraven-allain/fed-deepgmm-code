import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "certify_synthetic_data.py"
SPEC = importlib.util.spec_from_file_location("certify_synthetic_data", SCRIPT_PATH)
certify = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = certify
SPEC.loader.exec_module(certify)


class SyntheticDataCertificationTests(unittest.TestCase):
    def test_array_checksum_is_stable_and_includes_key_dtype_and_shape(self):
        values = np.array([[1.0, 2.0]], dtype=np.float64)
        self.assertEqual(
            certify.array_checksum("train_x", values),
            certify.array_checksum("train_x", values.copy()),
        )
        self.assertNotEqual(
            certify.array_checksum("train_x", values),
            certify.array_checksum("train_y", values),
        )
        self.assertNotEqual(
            certify.array_checksum("train_x", values),
            certify.array_checksum("train_x", values.astype(np.float32)),
        )
        self.assertNotEqual(
            certify.array_checksum("train_x", values),
            certify.array_checksum("train_x", values.reshape(2, 1)),
        )

    def test_exact_comparison_and_container_difference_status(self):
        left = {"x": np.array([1.0, 2.0])}
        comparison = certify.compare_arrays(left, {"x": left["x"].copy()})
        self.assertTrue(comparison["all_arrays_exact"])
        comparison.update({"left_file_sha256": "one", "right_file_sha256": "two"})
        self.assertEqual(
            certify.content_status(comparison, semantic_match=True),
            "content_match_file_container_differs",
        )
        comparison["right_file_sha256"] = "one"
        self.assertEqual(certify.content_status(comparison, semantic_match=True), "exact_content_match")

    def test_dtype_and_shape_mismatches_are_detected(self):
        dtype_result = certify.compare_arrays(
            {"x": np.array([1.0], dtype=np.float64)},
            {"x": np.array([1.0], dtype=np.float32)},
        )
        self.assertFalse(dtype_result["all_dtypes_match"])
        self.assertFalse(dtype_result["all_arrays_exact"])
        shape_result = certify.compare_arrays(
            {"x": np.array([1.0, 2.0])},
            {"x": np.array([[1.0, 2.0]])},
        )
        self.assertFalse(shape_result["all_shapes_match"])
        self.assertFalse(shape_result["all_arrays_exact"])

    def test_step_paper_function_uses_nonnegative_threshold(self):
        values = np.array([[-1.0], [0.0], [1.0]])
        actual = certify.paper_true_function("step", values)
        np.testing.assert_array_equal(actual, np.array([[0.0], [1.0], [1.0]]))

    def test_nonlegacy_output_guard_rejects_data_zoo(self):
        with self.assertRaises(ValueError):
            certify._require_nonlegacy_output(certify.LEGACY_DATA_ROOT / "blocked.npz")

    def test_status_assignment_for_nonexact_semantic_data(self):
        comparison = {
            "all_arrays_exact": False,
            "left_file_sha256": "left",
            "right_file_sha256": "right",
        }
        self.assertEqual(certify.content_status(comparison, semantic_match=True), "semantic_match_nonexact")
        self.assertEqual(certify.content_status(comparison, semantic_match=False), "mismatch")


if __name__ == "__main__":
    unittest.main()
