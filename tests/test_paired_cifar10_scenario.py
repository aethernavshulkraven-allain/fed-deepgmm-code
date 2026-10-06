"""Gates on the class-consistent CIFAR-10 paired scenario.

The scenario's job is to reuse original image draws that cannot be replayed, so
its safety rests entirely on refusing to touch them unless the replayable scalar
stream reproduces the source archive exactly. These tests check that the refusal
actually fires, not merely that the happy path runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedgmm/sp_decentralized_mnist_lr_example"))

from scenarios.paired_cifar10_scenario import (            # noqa: E402
    IMAGE_SHAPE, SPLIT_ORDER, PairedClassConsistentCIFAR10Scenario,
    reconstruct_scalar_splits)

SIZES = {"train": 40, "dev": 12, "test": 8}


def build_archive(path, sizes=SIZES, z_image=False, corrupt=None):
    """Write a small archive the replay reproduces, so the gates can be tested."""
    toy, numeric = reconstruct_scalar_splits(sizes)
    mean = float(numeric["train"][2].mean())
    std = float(numeric["train"][2].std())
    arrays, rng = {}, np.random.RandomState(0)
    for split in SPLIT_ORDER:
        toy_x, toy_z, toy_y, _, _ = numeric[split]
        n = sizes[split]
        digits = np.clip(1.5*toy_x[:, 0] + 5., 0, 9).round().reshape(-1, 1)
        arrays[f"{split}_x"] = rng.rand(n, *IMAGE_SHAPE).astype(np.float32)
        arrays[f"{split}_z"] = (rng.rand(n, *IMAGE_SHAPE).astype(np.float32)
                                if z_image else toy_z)
        arrays[f"{split}_y"] = (toy_y-mean)/std
        arrays[f"{split}_g"] = (toy._true_g_function_np((digits-5.)/1.5)-mean)/std
        arrays[f"{split}_w"] = digits
    if corrupt:
        corrupt(arrays)
    np.savez(path, **arrays)
    return arrays, mean, std


def run(path):
    scenario = PairedClassConsistentCIFAR10Scenario(path)
    scenario.setup(num_train=SIZES["train"], num_dev=SIZES["dev"],
                   num_test=SIZES["test"])
    return scenario


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "main.npz"
    arrays, mean, std = build_archive(path)
    return path, arrays, mean, std


def test_corrected_outcome_is_structural_function_plus_original_noise(archive):
    """The whole point: y becomes g plus the noise the original archive drew."""
    path, arrays, mean, std = archive
    scenario = run(path)
    _, numeric = reconstruct_scalar_splits(SIZES)
    for split in SPLIT_ORDER:
        _, _, toy_y, toy_g, _ = numeric[split]
        noise = (toy_y-toy_g)/std
        produced = scenario.audit[split]
        assert produced["corrected_residual_vs_original_noise_max_abs"] < 1e-12
        # The correction must actually change something, or it is a no-op test.
        assert produced["outcome_change_mse"] > 1e-6
        assert np.allclose(arrays[f"{split}_y"], (toy_y-mean)/std)
        assert np.max(np.abs(noise)) > 0


def test_images_instrument_and_structural_function_pass_through_untouched(archive):
    path, arrays, _, _ = archive
    scenario = PairedClassConsistentCIFAR10Scenario(path)
    for split in SPLIT_ORDER:
        x, z, y, g, w = scenario.generate_data(SIZES[split])
        assert np.array_equal(x, arrays[f"{split}_x"])
        assert np.array_equal(z, arrays[f"{split}_z"])
        assert np.array_equal(g, arrays[f"{split}_g"])
        assert np.array_equal(w, arrays[f"{split}_w"])
        assert not np.array_equal(y, arrays[f"{split}_y"])


def test_image_instrument_archive_is_accepted(tmp_path):
    path = tmp_path / "main.npz"
    build_archive(path, z_image=True)
    scenario = run(path)
    assert set(scenario.audit) == set(SPLIT_ORDER)


@pytest.mark.parametrize("key", ["y", "g", "w", "z"])
def test_refuses_when_source_scalars_do_not_replay(tmp_path, key):
    """Mutation gate: perturbing any replayable array must block image reuse."""
    path = tmp_path / "main.npz"

    def corrupt(arrays):
        arrays[f"train_{key}"] = arrays[f"train_{key}"] + 1e-6

    build_archive(path, corrupt=corrupt)
    with pytest.raises(ValueError, match="Source reconstruction failed"):
        run(path)


def test_refuses_wrong_treatment_image_shape(tmp_path):
    path = tmp_path / "main.npz"

    def corrupt(arrays):
        for split in SPLIT_ORDER:
            arrays[f"{split}_x"] = arrays[f"{split}_x"][:, :1, :28, :28]

    build_archive(path, corrupt=corrupt)
    with pytest.raises(ValueError, match="Expected image-X CIFAR-10 archive"):
        run(path)


def test_refuses_unrecognized_instrument_shape(tmp_path):
    path = tmp_path / "main.npz"

    def corrupt(arrays):
        for split in SPLIT_ORDER:
            arrays[f"{split}_z"] = np.zeros((len(arrays[f"{split}_z"]), 7))

    build_archive(path, corrupt=corrupt)
    with pytest.raises(ValueError, match="scalar or .* instrument"):
        run(path)


def test_refuses_wrong_split_size_and_extra_draw(archive):
    path, _, _, _ = archive
    scenario = PairedClassConsistentCIFAR10Scenario(path)
    with pytest.raises(ValueError, match="Expected train size"):
        scenario.generate_data(SIZES["train"] + 1)
    for split in SPLIT_ORDER:
        scenario.generate_data(SIZES[split])
    with pytest.raises(ValueError, match="exactly train/dev/test"):
        scenario.generate_data(SIZES["train"])


def test_replay_does_not_disturb_caller_rng_state():
    np.random.seed(11)
    before = np.random.rand(3)
    np.random.seed(11)
    reconstruct_scalar_splits(SIZES)
    assert np.array_equal(np.random.rand(3), before)


def test_normalization_comes_from_training_outcome_only(archive):
    path, _, mean, std = archive
    scenario = PairedClassConsistentCIFAR10Scenario(path)
    _, numeric = reconstruct_scalar_splits(SIZES)
    assert scenario.mean == pytest.approx(float(numeric["train"][2].mean()))
    assert scenario.std == pytest.approx(float(numeric["train"][2].std()))
    assert scenario.mean == pytest.approx(mean)
    assert scenario.std == pytest.approx(std)

def test_construction_formula_is_pinned(archive):
    """Pin the algebra the scenario's internal residual check defends.

    That check (`residual_error > 1e-12`) cannot fire on its own: subtracting
    the noise from the corrected outcome cancels the `toy_y - toy_g` term and
    leaves exactly the normalized class-based `g` that the reconstruction gate
    has already verified. It is defence-in-depth against a future edit to the
    construction formula, and removing it alone is unobservable. So the property
    is pinned here from outside instead: corrected outcome minus original noise
    must equal the archive's own structural function.
    """
    path, arrays, mean, std = archive
    scenario = PairedClassConsistentCIFAR10Scenario(path)
    _, numeric = reconstruct_scalar_splits(SIZES)
    for split in SPLIT_ORDER:
        _, _, y, g, _ = scenario.generate_data(SIZES[split])
        _, _, toy_y, toy_g, _ = numeric[split]
        noise = (toy_y-toy_g)/std
        assert np.max(np.abs((y-noise) - g)) < 1e-12
        assert np.array_equal(g, arrays[f"{split}_g"])
