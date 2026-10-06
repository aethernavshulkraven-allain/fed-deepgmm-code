"""Class-consistent CIFAR-10 DGP with the original image draws held fixed.

CIFAR-10 image-X archives carry the same internal inconsistency the paired
FEMNIST work corrected. In `cifar10_scenario.AbstractCIFAR10Scenario`, when
`use_x_images` is set, the structural function is recomputed on the image class

    g = true_g((x_digits - 5) / 1.5)

while the outcome returned is the untouched continuous-treatment `toy_y`, whose
own signal is `toy_g = true_g(toy_x)`. Rounding to a class label makes those two
different quantities, so the archive's `y` is not `g` plus noise: the regression
target and the structural function it is scored against disagree. Correcting it
means rebuilding `y` on the same class the image actually shows,

    y_raw = g_class + (toy_y - toy_g)

which keeps the original noise draw exactly and changes only the signal.

This mirrors `scenarios.paired_emnist_scenario`, deliberately as a separate
module rather than a generalization of it: the FEMNIST archives are already
generated, audited and hash-recorded, and they must stay reproducible by the
unchanged code that produced them.

Source images are reused, never redrawn. The original generator seeds NumPy and
torch but not Python's `random`, and it is `random.shuffle` and `random.choice`
that pick the images -- so the image draws cannot be replayed. The scalar draws
can be, and this scenario refuses to reuse any image until the replayed scalars
reproduce the source archive's `y`, `g`, `w` (and scalar `z`) exactly.
"""

from pathlib import Path

import numpy as np

from scenarios.abstract_scenario import AbstractScenario
from scenarios.toy_scenarios import AGMMZoo


SPLIT_ORDER = ("train", "dev", "test")
IMAGE_SHAPE = (3, 32, 32)


def reconstruct_scalar_splits(sizes, g_function="abs", seed=527):
    """Replay AGMMZoo without disturbing the caller's NumPy RNG state.

    The generator seeds NumPy immediately before building AGMMZoo and then draws
    train, dev and test in that order, so replaying that sequence reproduces the
    scalar stream exactly.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        toy = AGMMZoo(g_function=g_function, two_gps=False,
                      n_instruments=1, iv_strength=0.5)
        return toy, {split: toy.generate_data(sizes[split]) for split in SPLIT_ORDER}
    finally:
        np.random.set_state(state)


class PairedClassConsistentCIFAR10Scenario(AbstractScenario):
    """Preserve original X/Z/g/w; replace the continuous-treatment outcome.

    Both arms keep the ORIGINAL training-outcome normalization so that `g` and
    the error term stay in identical units across the paired comparison and the
    only difference is the corrected outcome.
    """

    def __init__(self, source_path, g_function="abs", seed=527):
        super().__init__()
        self.source_path = Path(source_path).resolve()
        with np.load(self.source_path, allow_pickle=False) as source:
            self.sizes = {split: len(source[f"{split}_y"]) for split in SPLIT_ORDER}
        self.toy, self.numeric = reconstruct_scalar_splits(self.sizes, g_function, seed)
        train_y = self.numeric["train"][2]
        self.mean = float(train_y.mean())
        self.std = float(train_y.std())
        self.next_split = 0
        self.audit = {}

    def generate_data(self, num_data, **kwargs):
        if self.next_split >= len(SPLIT_ORDER):
            raise ValueError("Paired scenario supports exactly train/dev/test draws")
        split = SPLIT_ORDER[self.next_split]
        if num_data != self.sizes[split]:
            raise ValueError(f"Expected {split} size {self.sizes[split]}, got {num_data}")
        toy_x, toy_z, toy_y, toy_g, _ = self.numeric[split]
        # Reproduce the generator's own class mapping, including its column take.
        digits = np.clip(1.5*toy_x[:, 0] + 5., 0, 9).round().reshape(-1, 1)
        g_class = self.toy._true_g_function_np((digits-5.)/1.5)
        with np.load(self.source_path, allow_pickle=False) as source:
            x, z, saved_y, saved_g, w = (
                source[f"{split}_{key}"] for key in ("x", "z", "y", "g", "w")
            )
        if x.shape[1:] != IMAGE_SHAPE:
            raise ValueError(f"Expected image-X CIFAR-10 archive with {IMAGE_SHAPE} "
                             f"treatment images, got {x.shape[1:]}")
        expected = {"y": (toy_y-self.mean)/self.std,
                    "g": (g_class-self.mean)/self.std, "w": digits}
        actual = {"y": saved_y, "g": saved_g, "w": w}
        if z.shape[1:] == (1,):
            expected["z"] = toy_z
            actual["z"] = z
        elif z.shape[1:] != IMAGE_SHAPE:
            raise ValueError(f"Expected scalar or {IMAGE_SHAPE} CIFAR-10 instrument, "
                             f"got {z.shape[1:]}")
        errors = {}
        for key, value in expected.items():
            if actual[key].shape != value.shape:
                raise ValueError(f"Source {split}_{key} shape mismatch: "
                                 f"{actual[key].shape} vs {value.shape}")
            errors[key] = float(np.max(np.abs(actual[key]-value)))
            if not np.array_equal(actual[key], value):
                raise ValueError(f"Source reconstruction failed: {split}_{key}: {errors[key]}")
        corrected_raw_y = g_class + (toy_y-toy_g)
        corrected_y = (corrected_raw_y-self.mean)/self.std
        original_noise = (toy_y-toy_g)/self.std
        residual_error = float(np.max(np.abs(corrected_y-saved_g-original_noise)))
        if residual_error > 1e-12:
            raise ValueError("Corrected target residual does not equal original noise")
        self.audit[split] = {
            "source_reconstruction_max_abs": errors,
            "corrected_residual_vs_original_noise_max_abs": residual_error,
            "outcome_change_mse": float(np.mean((corrected_y-saved_y)**2)),
            "n": num_data,
        }
        self.next_split += 1
        return x, z, corrected_y, saved_g, w
