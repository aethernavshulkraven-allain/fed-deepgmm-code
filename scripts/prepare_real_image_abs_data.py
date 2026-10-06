#!/usr/bin/env python3
"""Prepare abs real-image DeepGMM scenario files.

Generated files:

* ``data/femnist_x/main.npz``
* ``data/femnist_z/main.npz``
* ``data/femnist_xz/main.npz``
* ``data/cifar10_x/main.npz``
* ``data/cifar10_z/main.npz``
* ``data/cifar10_xz/main.npz``

FEMNIST is intentionally sourced from TensorFlow Federated with
``only_digits=True``. If TensorFlow Federated is not installed, the script
fails with an explicit dependency message instead of using the older
torchvision EMNIST proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
DEFAULT_OUTPUT_ROOT = EXAMPLE_ROOT / "data"
DEFAULT_CACHE_DIR = EXAMPLE_ROOT / "datasets"
DEFAULT_METADATA_PATH = (
    REPO_ROOT
    / "experiments"
    / "rerun_protocol_v1_real_images_abs_alpha0p5"
    / "data_generation_manifest.json"
)
GENERATOR_SEED = 527
G_FUNCTION = "abs"
SPLIT_SIZES = {"train": 20_000, "dev": 10_000, "test": 10_000}
MODES = {
    "x": (True, False),
    "z": (False, True),
    "xz": (True, True),
}

if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from scenarios.abstract_scenario import AbstractScenario  # noqa: E402
from scenarios.toy_scenarios import AGMMZoo, Standardizer  # noqa: E402


class MissingFEMNISTDependency(RuntimeError):
    pass


@dataclass
class ImagePool:
    dataset_key: str
    source_name: str
    image_shape: tuple[int, ...]
    images_by_split_and_digit: dict[str, dict[int, np.ndarray]]
    metadata: dict[str, object]

    def sample(
        self,
        digits: np.ndarray,
        rng: np.random.Generator,
        split: str,
    ) -> np.ndarray:
        digits = np.asarray(digits, dtype=np.int64).reshape(-1)
        output = np.empty((digits.shape[0], *self.image_shape), dtype=np.float32)
        for digit in sorted(set(int(item) for item in digits.tolist())):
            candidates = self.images_by_split_and_digit[split][digit]
            positions = np.flatnonzero(digits == digit)
            choices = rng.integers(0, candidates.shape[0], size=positions.shape[0])
            output[positions] = candidates[choices]
        return output

    @property
    def counts_by_split_and_digit(self) -> dict[str, dict[str, int]]:
        return {
            split: {
                str(digit): int(images.shape[0])
                for digit, images in sorted(images_by_digit.items())
            }
            for split, images_by_digit in self.images_by_split_and_digit.items()
        }


class ImageMappedAGMMScenario(AbstractScenario):
    def __init__(
        self,
        image_pool: ImagePool,
        *,
        use_x_images: bool,
        use_z_images: bool,
        g_function: str,
        seed: int,
    ):
        super().__init__()
        self.image_pool = image_pool
        self.use_x_images = use_x_images
        self.use_z_images = use_z_images
        self.rng = np.random.default_rng(seed)
        self.toy_scenario = AGMMZoo(
            g_function=g_function,
            two_gps=False,
            n_instruments=1,
            iv_strength=0.5,
        )
        self._generated_split_order = ("train", "dev", "test")
        self._generated_split_index = 0

    @staticmethod
    def _digits_from_latent(values: np.ndarray) -> np.ndarray:
        return np.clip(1.5 * values[:, 0] + 5.0, 0, 9).round().astype(np.int64)

    def generate_data(self, num_data: int, **kwargs):
        if self._generated_split_index >= len(self._generated_split_order):
            raise RuntimeError("ImageMappedAGMMScenario received more than three split generations")
        split = self._generated_split_order[self._generated_split_index]
        self._generated_split_index += 1
        toy_x, toy_z, toy_y, toy_g, _ = self.toy_scenario.generate_data(num_data)

        if self.use_x_images:
            x_digits = self._digits_from_latent(toy_x)
            x = self.image_pool.sample(x_digits, self.rng, split)
            centered_x = ((x_digits.reshape(-1, 1) - 5.0) / 1.5).astype(np.float64)
            g = self.toy_scenario._true_g_function_np(centered_x).reshape(-1, 1)
            w = x_digits.reshape(-1, 1).astype(np.float64)
        else:
            x = toy_x.reshape(-1, 1) * 1.5 + 5.0
            g = toy_g.reshape(-1, 1)
            w = x

        if self.use_z_images:
            z_digits = self._digits_from_latent(toy_z)
            z = self.image_pool.sample(z_digits, self.rng, split)
        else:
            z = toy_z.reshape(-1, 1)

        return x, z, toy_y.reshape(-1, 1), g, w

    def true_g_function(self, x):
        raise NotImplementedError()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_limited(
    parts: dict[int, list[np.ndarray]],
    counts: dict[int, int],
    images: np.ndarray,
    labels: np.ndarray,
    max_images_per_digit: int | None,
) -> None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    for digit in range(10):
        positions = np.flatnonzero(labels == digit)
        if positions.size == 0:
            continue
        if max_images_per_digit is not None:
            remaining = max_images_per_digit - counts[digit]
            if remaining <= 0:
                continue
            positions = positions[:remaining]
        parts[digit].append(np.asarray(images[positions], dtype=np.float32))
        counts[digit] += int(positions.shape[0])


def _pool_from_parts(
    *,
    dataset_key: str,
    source_name: str,
    train_parts: dict[int, list[np.ndarray]],
    test_parts: dict[int, list[np.ndarray]],
    split_seed: int,
    metadata: dict[str, object],
) -> ImagePool:
    source_train: dict[int, np.ndarray] = {}
    source_test: dict[int, np.ndarray] = {}
    for digit in range(10):
        if not train_parts[digit] or not test_parts[digit]:
            raise RuntimeError(f"{source_name} has an empty source split for digit/class {digit}")
        source_train[digit] = np.concatenate(train_parts[digit], axis=0)
        source_test[digit] = np.concatenate(test_parts[digit], axis=0)

    generated_pools, split_diagnostics = _make_content_disjoint_generated_pools(
        source_train,
        source_test,
        seed=split_seed,
    )
    first = generated_pools["train"][0]
    return ImagePool(
        dataset_key=dataset_key,
        source_name=source_name,
        image_shape=tuple(int(item) for item in first.shape[1:]),
        images_by_split_and_digit=generated_pools,
        metadata={
            **metadata,
            "generated_split_policy": (
                "content_disjoint: source test -> generated test; source train unique "
                "content -> deterministic 80/20 generated train/dev; content appearing "
                "in source test excluded from train/dev"
            ),
            "generated_split_seed": int(split_seed),
            "content_disjoint_split_diagnostics": split_diagnostics,
        },
    )


def _content_unique(images: np.ndarray) -> tuple[np.ndarray, list[bytes]]:
    unique_images: list[np.ndarray] = []
    unique_hashes: list[bytes] = []
    seen: set[bytes] = set()
    for image in images:
        contiguous = np.ascontiguousarray(image)
        digest = hashlib.sha256(contiguous.view(np.uint8)).digest()
        if digest in seen:
            continue
        seen.add(digest)
        unique_images.append(contiguous)
        unique_hashes.append(digest)
    if not unique_images:
        raise RuntimeError("Content deduplication removed every source image")
    return np.stack(unique_images), unique_hashes


def _make_content_disjoint_generated_pools(
    source_train: dict[int, np.ndarray],
    source_test: dict[int, np.ndarray],
    *,
    seed: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, object]]:
    pools = {split: {} for split in ("train", "dev", "test")}
    diagnostics: dict[str, object] = {}
    for digit in range(10):
        test_unique, test_hashes = _content_unique(source_test[digit])
        test_hash_set = set(test_hashes)
        train_unique, train_hashes = _content_unique(source_train[digit])
        keep = np.array([digest not in test_hash_set for digest in train_hashes], dtype=bool)
        train_unique = train_unique[keep]
        if train_unique.shape[0] < 2:
            raise RuntimeError(f"Insufficient disjoint source-train images for digit {digit}")

        rng = np.random.default_rng(seed + digit)
        order = rng.permutation(train_unique.shape[0])
        dev_size = max(1, int(round(0.2 * train_unique.shape[0])))
        dev_indices = order[:dev_size]
        train_indices = order[dev_size:]
        pools["train"][digit] = train_unique[train_indices]
        pools["dev"][digit] = train_unique[dev_indices]
        pools["test"][digit] = test_unique
        diagnostics[str(digit)] = {
            "source_train_records": int(source_train[digit].shape[0]),
            "source_test_records": int(source_test[digit].shape[0]),
            "generated_train_unique_content": int(train_indices.shape[0]),
            "generated_dev_unique_content": int(dev_indices.shape[0]),
            "generated_test_unique_content": int(test_unique.shape[0]),
            "source_train_content_excluded_due_to_test_overlap": int(np.count_nonzero(~keep)),
        }
    return pools, diagnostics


def _normalize_femnist_pixels(pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim == 3:
        array = array[:, None, :, :]
    elif array.ndim == 4 and array.shape[-1] == 1:
        array = np.transpose(array, (0, 3, 1, 2))
    if array.max(initial=0.0) > 1.5:
        array = array / 255.0
    return (array - 0.1307) / 0.3081


def load_tff_femnist_digits_pool(
    cache_dir: Path,
    *,
    max_images_per_digit: int | None,
    batch_size: int,
) -> ImagePool:
    try:
        import tensorflow_federated as tff  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingFEMNISTDependency(
            "TensorFlow Federated is required for FEMNIST only_digits=True. "
            "Install tensorflow and tensorflow_federated in the fedgmm environment, "
            "then rerun this script. The current generator intentionally does not "
            "fall back to torchvision EMNIST."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_client_data, test_client_data = tff.simulation.datasets.emnist.load_data(
        cache_dir=str(cache_dir),
        only_digits=True,
    )
    source_parts = {
        "train": {digit: [] for digit in range(10)},
        "test": {digit: [] for digit in range(10)},
    }
    split_client_counts: dict[str, int] = {}

    for split_name, client_data in (
        ("train", train_client_data),
        ("test", test_client_data),
    ):
        counts = {digit: 0 for digit in range(10)}
        split_client_counts[split_name] = len(client_data.client_ids)
        for client_id in client_data.client_ids:
            dataset = client_data.create_tf_dataset_for_client(client_id).batch(batch_size)
            for batch in dataset.as_numpy_iterator():
                images = _normalize_femnist_pixels(batch["pixels"])
                labels = np.asarray(batch["label"], dtype=np.int64)
                _append_limited(
                    source_parts[split_name], counts, images, labels, max_images_per_digit
                )
            if max_images_per_digit is not None and all(
                counts[digit] >= max_images_per_digit for digit in range(10)
            ):
                break

    return _pool_from_parts(
        dataset_key="femnist",
        source_name="tff_federated_emnist_only_digits",
        train_parts=source_parts["train"],
        test_parts=source_parts["test"],
        split_seed=GENERATOR_SEED,
        metadata={
            "source": "tensorflow_federated.simulation.datasets.emnist.load_data",
            "only_digits": True,
            "cache_dir": str(cache_dir),
            "split_client_counts": split_client_counts,
            "max_images_per_digit": max_images_per_digit,
        },
    )


def load_cifar10_pool(
    cache_dir: Path,
    *,
    max_images_per_digit: int | None,
    batch_size: int,
) -> ImagePool:
    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    source_parts = {
        "train": {digit: [] for digit in range(10)},
        "test": {digit: [] for digit in range(10)},
    }
    split_sizes: dict[str, int] = {}

    for split_name, train in (("train", True), ("test", False)):
        counts = {digit: 0 for digit in range(10)}
        dataset = datasets.CIFAR10(
            root=str(cache_dir),
            train=train,
            download=True,
            transform=transform,
        )
        split_sizes[split_name] = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for images, labels in loader:
            _append_limited(
                source_parts[split_name],
                counts,
                images.numpy(),
                labels.numpy(),
                max_images_per_digit,
            )
        if max_images_per_digit is not None and all(
            counts[digit] >= max_images_per_digit for digit in range(10)
        ):
            break

    return _pool_from_parts(
        dataset_key="cifar10",
        source_name="torchvision_cifar10",
        train_parts=source_parts["train"],
        test_parts=source_parts["test"],
        split_seed=GENERATOR_SEED,
        metadata={
            "source": "torchvision.datasets.CIFAR10",
            "official_source": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "cache_dir": str(cache_dir),
            "split_sizes": split_sizes,
            "max_images_per_digit": max_images_per_digit,
        },
    )


def scenario_dataset_name(dataset_key: str, mode: str) -> str:
    if dataset_key == "femnist":
        return f"femnist_{mode}"
    if dataset_key == "cifar10":
        return f"cifar10_{mode}"
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def generate_scenario_file(
    image_pool: ImagePool,
    *,
    mode: str,
    output_root: Path,
    seed: int,
    overwrite: bool,
    skip_existing: bool,
) -> dict[str, object]:
    use_x_images, use_z_images = MODES[mode]
    dataset_name = scenario_dataset_name(image_pool.dataset_key, mode)
    output_stem = output_root / dataset_name / "main"
    output_path = output_stem.with_suffix(".npz")
    if output_path.exists():
        if skip_existing:
            return {
                "dataset": dataset_name,
                "path": str(output_path.relative_to(REPO_ROOT)),
                "status": "skipped_existing",
                "sha256": file_sha256(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite or --skip-existing"
            )

    np.random.seed(seed)
    torch.manual_seed(seed)
    scenario = Standardizer(
        ImageMappedAGMMScenario(
            image_pool,
            use_x_images=use_x_images,
            use_z_images=use_z_images,
            g_function=G_FUNCTION,
            seed=seed,
        )
    )
    scenario.setup(
        num_train=SPLIT_SIZES["train"],
        num_dev=SPLIT_SIZES["dev"],
        num_test=SPLIT_SIZES["test"],
    )
    scenario.info()
    scenario.to_file(str(output_stem))
    return {
        "dataset": dataset_name,
        "path": str(output_path.relative_to(REPO_ROOT)),
        "status": "generated",
        "sha256": file_sha256(output_path),
        "size_bytes": output_path.stat().st_size,
        "use_x_images": use_x_images,
        "use_z_images": use_z_images,
        "standardizer_mean": float(scenario._mean),
        "standardizer_std": float(scenario._std),
    }


def _none_if_zero(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare abs FEMNIST/CIFAR-10 scenario files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("femnist", "cifar10"),
        default=["femnist", "cifar10"],
        help="Dataset sources to prepare.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODES),
        default=list(MODES),
        help="Image mapping modes to generate.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--seed", type=int, default=GENERATOR_SEED)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-images-per-digit",
        type=int,
        default=0,
        help="Optional source-pool cap per digit/class. Use 0 for all images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_pool(dataset_key: str, cache_dir: Path, args: argparse.Namespace) -> ImagePool:
    max_images_per_digit = _none_if_zero(args.max_images_per_digit)
    source_cache = cache_dir / dataset_key
    if dataset_key == "femnist":
        return load_tff_femnist_digits_pool(
            source_cache,
            max_images_per_digit=max_images_per_digit,
            batch_size=int(args.batch_size),
        )
    if dataset_key == "cifar10":
        return load_cifar10_pool(
            source_cache,
            max_images_per_digit=max_images_per_digit,
            batch_size=int(args.batch_size),
        )
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    cache_dir = Path(args.cache_dir)
    metadata_out = Path(args.metadata_out)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    if not metadata_out.is_absolute():
        metadata_out = REPO_ROOT / metadata_out
    if args.overwrite and args.skip_existing:
        raise SystemExit("--overwrite and --skip-existing are mutually exclusive")

    generated: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    for dataset_key in args.datasets:
        try:
            image_pool = load_pool(dataset_key, cache_dir, args)
        except MissingFEMNISTDependency as exc:
            print(str(exc), file=sys.stderr)
            return 2
        sources[dataset_key] = {
            **image_pool.metadata,
            "source_name": image_pool.source_name,
            "image_shape": list(image_pool.image_shape),
            "counts_by_split_and_digit": image_pool.counts_by_split_and_digit,
        }
        for mode in args.modes:
            generated.append(
                generate_scenario_file(
                    image_pool,
                    mode=mode,
                    output_root=output_root,
                    seed=int(args.seed),
                    overwrite=bool(args.overwrite),
                    skip_existing=bool(args.skip_existing),
                )
            )

    metadata = {
        "protocol_version": "rerun_protocol_v1_real_images_abs_alpha0p5",
        "g_function": G_FUNCTION,
        "seed": int(args.seed),
        "split_sizes": SPLIT_SIZES,
        "sources": sources,
        "generated_files": generated,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with metadata_out.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({
        "metadata": str(metadata_out.relative_to(REPO_ROOT)),
        "generated_or_checked": len(generated),
        "files": generated,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
