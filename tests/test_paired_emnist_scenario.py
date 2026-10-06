from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "fedgmm/sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(EXAMPLE))

from scenarios.abstract_scenario import AbstractScenario, Dataset
from scenarios.paired_emnist_scenario import (
    PairedClassConsistentEMNISTScenario, reconstruct_scalar_splits,
)
from generate_paired_emnist_data import generate, sha256


def make_source(path, image_z=False, corrupt=False):
    sizes = {"train": 20, "dev": 10, "test": 10}
    numeric = reconstruct_scalar_splits(sizes)
    mean, std = numeric["train"][2].mean(), numeric["train"][2].std()
    rng = np.random.RandomState(20)
    source = AbstractScenario()
    for split, (x, z, y, g, _) in numeric.items():
        digit = np.clip(1.5*x+5, 0, 9).round()
        class_g = np.abs((digit-5)/1.5)
        if corrupt:
            y = y + .1
        source.splits[split] = Dataset(
            rng.randn(len(x), 1, 28, 28),
            rng.randn(len(x), 1, 28, 28) if image_z else z,
            (y-mean)/std, (class_g-mean)/std, digit,
        )
    source.to_file(str(path))
    return numeric, std


@pytest.mark.parametrize("image_z", [False, True])
def test_only_outcome_changes_and_residual_is_original_noise(tmp_path, image_z):
    source = tmp_path / "source.npz"
    numeric, std = make_source(source, image_z)
    before = source.read_bytes()
    scenario = PairedClassConsistentEMNISTScenario(source)
    scenario.setup(20, 10, 10)
    output = tmp_path / "corrected.npz"
    scenario.to_file(str(output))
    assert source.read_bytes() == before
    with np.load(source) as a, np.load(output) as b:
        for key in a.files:
            if not key.endswith("_y"):
                assert np.array_equal(a[key], b[key]), key
        for split, (_, _, y, g, _) in numeric.items():
            np.testing.assert_allclose(b[f"{split}_y"]-b[f"{split}_g"],
                                       (y-g)/std, rtol=0, atol=1e-12)
            assert not np.array_equal(a[f"{split}_y"], b[f"{split}_y"])


def test_wrong_source_scalar_alignment_fails_closed(tmp_path):
    source = tmp_path / "bad.npz"
    make_source(source, corrupt=True)
    scenario = PairedClassConsistentEMNISTScenario(source)
    with pytest.raises(ValueError, match="reconstruction failed"):
        scenario.setup(20, 10, 10)


def test_replay_preserves_callers_rng():
    np.random.seed(45)
    expected = np.random.randn(3)
    np.random.seed(45)
    reconstruct_scalar_splits({"train": 20, "dev": 10, "test": 10})
    assert np.array_equal(np.random.randn(3), expected)


def test_split_size_mismatch_rejected(tmp_path):
    source = tmp_path / "source.npz"
    make_source(source)
    scenario = PairedClassConsistentEMNISTScenario(source)
    with pytest.raises(ValueError, match="Expected train size"):
        scenario.generate_data(21)


def test_generator_writes_new_archives_and_refuses_overwrite(tmp_path):
    source_root, output_root = tmp_path / "source", tmp_path / "corrected"
    for dataset in ("femnist_x", "femnist_xz"):
        make_source(source_root / dataset / "main.npz", image_z=dataset.endswith("_xz"))
    evidence = generate(source_root, output_root)
    for record in evidence["datasets"].values():
        assert sha256(record["source"]) == record["source_sha256"]
        assert sha256(record["generated"]) == record["generated_sha256"]
        assert all(record["unchanged_arrays"].values())
    assert (output_root / "generation_audit.json").is_file()
    with pytest.raises(FileExistsError):
        generate(source_root, output_root)
