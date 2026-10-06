"""The image loader must not silently train on an archive the run does not name.

fedml/data/MNIST/data_loader.py searches a cwd-relative ``data/<dataset>/`` path
BEFORE ``args.data_cache_dir``. Runs launch from the example directory, whose
``data/`` holds the ORIGINAL archives, so a config pointing data_cache_dir at a
corrected dataset would have loaded the original one with no visible sign --
and ``scenario_checksum`` was an effective-config field that nothing verified
on this path.

These tests pin the two protections: refuse the ambiguity when both candidate
archives exist, and refuse a declared checksum that does not match the bytes.
"""

import hashlib
import os
import sys
import types
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE)

from fedml.data.MNIST.data_loader import load_partition_data_mnist  # noqa: E402


def write_archive(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 8
    np.savez(
        path,
        splits=np.array(["train", "dev", "test"]),
        **{f"{s}_{k}": np.full((n, 1), value, dtype=np.float64)
           for s in ("train", "dev", "test") for k in ("x", "z", "y", "g", "w")},
    )


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_args(cache_dir, scenario_name="main", checksum=""):
    return types.SimpleNamespace(
        dataset="femnist_x", scenario_name=scenario_name,
        data_cache_dir=cache_dir, scenario_checksum=checksum,
        client_num_in_total=2, partition_alpha=0.5, partition_method="hetero",
        batch_size=4,
    )


class ArchiveResolutionTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(os.chdir, self._cwd)

    def test_ambiguous_archives_are_refused(self):
        """Both candidates present -> the cwd copy would silently win. Refuse."""
        write_archive(os.path.join(self.tmp, "data", "femnist_x", "main.npz"), 1.0)
        cache = os.path.join(self.tmp, "corrected")
        write_archive(os.path.join(cache, "femnist_x", "main.npz"), 2.0)

        with self.assertRaises(ValueError) as ctx:
            load_partition_data_mnist(make_args(cache), batch_size=4)
        self.assertIn("Ambiguous scenario archive", str(ctx.exception))

    def test_declared_checksum_mismatch_is_refused(self):
        """A run may not train on bytes it does not describe."""
        cache = os.path.join(self.tmp, "corrected")
        write_archive(os.path.join(cache, "femnist_x", "main.npz"), 2.0)

        args = make_args(cache, checksum="0" * 64)
        with self.assertRaises(ValueError) as ctx:
            load_partition_data_mnist(args, batch_size=4)
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_matching_checksum_records_resolved_identity(self):
        """A correct checksum passes and the run self-describes what it loaded."""
        cache = os.path.join(self.tmp, "corrected")
        archive = os.path.join(cache, "femnist_x", "main.npz")
        write_archive(archive, 2.0)

        args = make_args(cache, checksum=sha256_of(archive))
        try:
            load_partition_data_mnist(args, batch_size=4)
        except Exception as exc:
            # Only the identity guards matter here. A minimal stub archive is
            # not expected to survive full partitioning, but it must get past
            # the checksum and ambiguity checks to reach that point.
            if "checksum" in str(exc) or "Ambiguous scenario archive" in str(exc):
                raise
        self.assertEqual(args._resolved_scenario_sha256, sha256_of(archive))
        self.assertEqual(args._resolved_scenario_path, os.path.abspath(archive))


if __name__ == "__main__":
    unittest.main()
