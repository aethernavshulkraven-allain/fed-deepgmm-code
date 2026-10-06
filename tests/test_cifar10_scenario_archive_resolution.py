"""The CIFAR-10 loader must reach absolute archives and verify what it loads.

It used to concatenate "data/" + dataset + "/" + scenario_name, which turned an
absolute scenario_name into "data/cifar10_z//abs/path.npz" and killed every
cifar10_z validation run six seconds after launch. Absolute paths are also the
only route to the corrected CIFAR-X/XZ archives, whose originals share the
relative name -- so a checksum mismatch must refuse rather than train.
"""

import hashlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "fedgmm/sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(EXAMPLE))

from fedml.data.cifar10.efficient_loader import resolve_cifar10_scenario_archive  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_archive(path, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, marker=np.array([marker]))
    return path


def args_for(dataset, scenario_name, checksum=""):
    return types.SimpleNamespace(dataset=dataset, scenario_name=scenario_name,
                                 scenario_checksum=checksum)


def test_absolute_scenario_name_resolves(tmp_path):
    archive = write_archive(tmp_path / "corrected" / "cifar10_x" / "main.npz", 1)
    args = args_for("cifar10_x", str(archive.with_suffix("")), sha(archive))
    assert resolve_cifar10_scenario_archive(args) == str(archive)
    assert args._resolved_scenario_sha256 == sha(archive)


def test_relative_scenario_name_keeps_legacy_location(tmp_path, monkeypatch):
    archive = write_archive(tmp_path / "data" / "cifar10_z" / "main.npz", 2)
    monkeypatch.chdir(tmp_path)
    args = args_for("cifar10_z", "main")
    assert resolve_cifar10_scenario_archive(args) == str(archive)


def test_checksum_mismatch_refuses(tmp_path):
    """The failure this guards: naming the corrected archive, loading the original."""
    archive = write_archive(tmp_path / "cifar10_x" / "main.npz", 3)
    args = args_for("cifar10_x", str(archive.with_suffix("")), "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_cifar10_scenario_archive(args)


def test_relative_original_cannot_pass_for_the_corrected_archive(tmp_path, monkeypatch):
    """A config declaring the corrected checksum but resolving to the original
    by relative name must refuse."""
    corrected = write_archive(tmp_path / "corrected" / "cifar10_x" / "main.npz", 4)
    write_archive(tmp_path / "data" / "cifar10_x" / "main.npz", 5)
    monkeypatch.chdir(tmp_path)
    args = args_for("cifar10_x", "main", sha(corrected))
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_cifar10_scenario_archive(args)


def test_missing_archive_names_the_absolute_path(tmp_path):
    args = args_for("cifar10_z", str(tmp_path / "nope" / "main"))
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        resolve_cifar10_scenario_archive(args)


def test_real_cifar10_z_archive_as_the_campaign_declares_it():
    """The exact shape that failed in production, against the real archive."""
    archive = EXAMPLE / "data" / "cifar10_z" / "main.npz"
    if not archive.is_file():
        pytest.skip("real cifar10_z archive not present")
    args = args_for("cifar10_z", str(archive.with_suffix("")), sha(archive))
    assert resolve_cifar10_scenario_archive(args) == str(archive)
