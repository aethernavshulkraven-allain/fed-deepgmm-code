"""Class-consistent EMNIST DGP with the original image draws held fixed.

This is a new scenario, not an in-place NPZ repair. Seeded scalar draws must
reconstruct the source archive before its images may be reused. Both arms use
the original training-outcome normalization to isolate the outcome change.
"""

from pathlib import Path

import numpy as np

from scenarios.abstract_scenario import AbstractScenario
from scenarios.toy_scenarios import AGMMZoo


SPLIT_ORDER = ("train", "dev", "test")


def reconstruct_scalar_splits(sizes, seed=527):
    """Replay AGMMZoo without changing the caller's NumPy RNG state."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        toy = AGMMZoo(g_function="abs", two_gps=False,
                      n_instruments=1, iv_strength=0.5)
        return {split: toy.generate_data(sizes[split]) for split in SPLIT_ORDER}
    finally:
        np.random.set_state(state)


class PairedClassConsistentEMNISTScenario(AbstractScenario):
    """Preserve original X/Z/g/w; replace continuous-treatment outcome signal.

    Source images are fixed exogenous image draws, not regenerated using a new
    Python RNG seed (the original generator did not seed random.shuffle/choice).
    The raw construction is y_class = g_class + (toy_y - toy_g), followed by
    the ORIGINAL train-Y affine normalization. This keeps g and error units
    identical across the paired experiments.
    """

    def __init__(self, source_path, seed=527):
        super().__init__()
        self.source_path = Path(source_path).resolve()
        with np.load(self.source_path, allow_pickle=False) as source:
            self.sizes = {split: len(source[f"{split}_y"]) for split in SPLIT_ORDER}
        self.numeric = reconstruct_scalar_splits(self.sizes, seed)
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
        digits = np.clip(1.5 * toy_x + 5., 0, 9).round()
        g_class = np.abs((digits - 5.) / 1.5)
        with np.load(self.source_path, allow_pickle=False) as source:
            x, z, saved_y, saved_g, w = (
                source[f"{split}_{key}"] for key in ("x", "z", "y", "g", "w")
            )
        if x.shape[1:] != (1, 28, 28):
            raise ValueError("Expected image-X EMNIST archive")
        expected = {"y": (toy_y-self.mean)/self.std,
                    "g": (g_class-self.mean)/self.std, "w": digits}
        actual = {"y": saved_y, "g": saved_g, "w": w}
        if z.shape[1:] == (1,):
            expected["z"] = toy_z
            actual["z"] = z
        elif z.shape[1:] != (1, 28, 28):
            raise ValueError("Expected scalar or image EMNIST instrument")
        errors = {}
        for key, value in expected.items():
            if actual[key].shape != value.shape:
                raise ValueError(f"Source {split}_{key} shape mismatch")
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
