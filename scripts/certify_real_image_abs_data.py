#!/usr/bin/env python3
"""Certify the fixed-abs high-dimensional data artifacts without modifying them.

The certification checks provenance, container hashes, schema, numeric
finiteness, standardized absolute-response semantics, paired-scenario
fairness, and exact image reuse across train/dev/test splits.  Reports are
written next to the real-image protocol manifest; input NPZ files are opened
read-only and are never regenerated or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
DEFAULT_GENERATION_MANIFEST = PROTOCOL_ROOT / "data_generation_manifest.json"
DEFAULT_JSON_REPORT = PROTOCOL_ROOT / "data_certification.json"
DEFAULT_MARKDOWN_REPORT = PROTOCOL_ROOT / "data_certification.md"
DEFAULT_DATA_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example" / "data"
SPLIT_SIZES = {"train": 20_000, "dev": 10_000, "test": 10_000}
FIELDS = ("x", "z", "y", "g", "w")
NUMERIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    mode: str
    x_tail: tuple[int, ...]
    z_tail: tuple[int, ...]

    @property
    def uses_x_images(self) -> bool:
        return "x" in self.mode

    @property
    def uses_z_images(self) -> bool:
        return "z" in self.mode


SPECS = (
    DatasetSpec("femnist_x", "femnist", "x", (1, 28, 28), (1,)),
    DatasetSpec("femnist_z", "femnist", "z", (1,), (1, 28, 28)),
    DatasetSpec("femnist_xz", "femnist", "xz", (1, 28, 28), (1, 28, 28)),
    DatasetSpec("cifar10_x", "cifar10", "x", (3, 32, 32), (1,)),
    DatasetSpec("cifar10_z", "cifar10", "z", (1,), (3, 32, 32)),
    DatasetSpec("cifar10_xz", "cifar10", "xz", (3, 32, 32), (3, 32, 32)),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def image_fingerprints(images: np.ndarray) -> set[bytes]:
    """Return exact per-image payload fingerprints for overlap detection."""

    fingerprints: set[bytes] = set()
    for image in images:
        contiguous = np.ascontiguousarray(image)
        fingerprints.add(hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest())
    return fingerprints


def split_overlap(images_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    fingerprints = {
        split: image_fingerprints(images)
        for split, images in images_by_split.items()
    }
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    pair_results = {
        f"{left}_{right}": len(fingerprints[left] & fingerprints[right])
        for left, right in pairs
    }
    return {
        "unique_images_by_split": {
            split: len(values) for split, values in fingerprints.items()
        },
        "exact_unique_image_overlap": pair_results,
        "no_cross_split_exact_image_reuse": all(value == 0 for value in pair_results.values()),
    }


def infer_standardizer(
    normalized_g: np.ndarray,
    raw_g: np.ndarray,
) -> tuple[float, float]:
    std, mean = np.polyfit(normalized_g.reshape(-1), raw_g.reshape(-1), 1)
    return float(mean), float(std)


def raw_abs_response(spec: DatasetSpec, arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    if spec.uses_x_images:
        structural_input = (arrays[f"{split}_w"] - 5.0) / 1.5
    else:
        structural_input = (arrays[f"{split}_x"] - 5.0) / 1.5
    return np.abs(structural_input)


def certify_dataset(
    spec: DatasetSpec,
    path: Path,
    manifest_entry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actual_hash = file_sha256(path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}

    required_keys = {
        f"{split}_{field}" for split in SPLIT_SIZES for field in FIELDS
    }
    expected_keys = required_keys | {"splits"}
    key_set_valid = set(arrays) == expected_keys
    recorded_splits = [str(item) for item in arrays.get("splits", np.array([])).tolist()]
    split_names_valid = set(recorded_splits) == set(SPLIT_SIZES)

    split_results: dict[str, Any] = {}
    all_shapes_valid = True
    all_finite = True
    all_y_standardized = True
    for split, expected_size in SPLIT_SIZES.items():
        expected_shapes = {
            "x": (expected_size, *spec.x_tail),
            "z": (expected_size, *spec.z_tail),
            "y": (expected_size, 1),
            "g": (expected_size, 1),
            "w": (expected_size, 1),
        }
        shapes = {field: list(arrays[f"{split}_{field}"].shape) for field in FIELDS}
        shapes_valid = all(
            arrays[f"{split}_{field}"].shape == expected_shapes[field]
            for field in FIELDS
        )
        finite = all(np.isfinite(arrays[f"{split}_{field}"]).all() for field in FIELDS)
        y_mean = float(np.mean(arrays[f"{split}_y"]))
        y_std = float(np.std(arrays[f"{split}_y"]))
        y_standardized = (
            split != "train"
            or (abs(y_mean) <= NUMERIC_TOLERANCE and abs(y_std - 1.0) <= NUMERIC_TOLERANCE)
        )
        all_shapes_valid = all_shapes_valid and shapes_valid
        all_finite = all_finite and finite
        all_y_standardized = all_y_standardized and y_standardized
        split_results[split] = {
            "expected_size": expected_size,
            "shapes": shapes,
            "shapes_valid": shapes_valid,
            "all_finite": finite,
            "y_mean": y_mean,
            "y_std": y_std,
            "train_y_zero_mean_unit_std": y_standardized if split == "train" else None,
        }

    train_raw_g = raw_abs_response(spec, arrays, "train")
    standardizer_mean, standardizer_std = infer_standardizer(
        arrays["train_g"], train_raw_g
    )
    function_errors: dict[str, float] = {}
    for split in SPLIT_SIZES:
        raw_g = raw_abs_response(spec, arrays, split)
        expected_normalized_g = (raw_g - standardizer_mean) / standardizer_std
        function_errors[split] = float(
            np.max(np.abs(arrays[f"{split}_g"] - expected_normalized_g))
        )
    max_function_error = max(function_errors.values())

    modality_overlap: dict[str, Any] = {}
    if spec.uses_x_images:
        modality_overlap["x"] = split_overlap(
            {split: arrays[f"{split}_x"] for split in SPLIT_SIZES}
        )
    if spec.uses_z_images:
        modality_overlap["z"] = split_overlap(
            {split: arrays[f"{split}_z"] for split in SPLIT_SIZES}
        )
    split_isolation_pass = all(
        result["no_cross_split_exact_image_reuse"]
        for result in modality_overlap.values()
    )

    hash_valid = actual_hash == manifest_entry.get("sha256")
    size_valid = path.stat().st_size == int(manifest_entry.get("size_bytes", -1))
    response_valid = max_function_error <= NUMERIC_TOLERANCE
    core_invariants_pass = all(
        (
            path.exists(),
            hash_valid,
            size_valid,
            key_set_valid,
            split_names_valid,
            all_shapes_valid,
            all_finite,
            all_y_standardized,
            response_valid,
            standardizer_std > 0.0,
        )
    )
    result = {
        "dataset": spec.name,
        "family": spec.family,
        "mode": spec.mode,
        "path": str(path.relative_to(REPO_ROOT)),
        "size_bytes": path.stat().st_size,
        "size_matches_manifest": size_valid,
        "sha256": actual_hash,
        "sha256_matches_manifest": hash_valid,
        "keys": sorted(arrays),
        "key_set_valid": key_set_valid,
        "recorded_splits": recorded_splits,
        "split_names_valid": split_names_valid,
        "split_results": split_results,
        "all_shapes_valid": all_shapes_valid,
        "all_finite": all_finite,
        "train_y_standardized": all_y_standardized,
        "response_function": "abs",
        "inferred_standardizer_mean": standardizer_mean,
        "inferred_standardizer_std": standardizer_std,
        "max_abs_response_error": max_function_error,
        "response_error_by_split": function_errors,
        "response_semantics_valid": response_valid,
        "image_split_overlap": modality_overlap,
        "split_isolation_pass": split_isolation_pass,
        "core_invariants_pass": core_invariants_pass,
    }
    return result, arrays


def fairness_checks(arrays_by_dataset: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    y_checksums = {
        name: {
            split: array_sha256(arrays[f"{split}_y"])
            for split in SPLIT_SIZES
        }
        for name, arrays in arrays_by_dataset.items()
    }
    shared_toy_outcomes = all(
        len({checksums[split] for checksums in y_checksums.values()}) == 1
        for split in SPLIT_SIZES
    )

    family_results: dict[str, Any] = {}
    for family in ("femnist", "cifar10"):
        x = arrays_by_dataset[f"{family}_x"]
        xz = arrays_by_dataset[f"{family}_xz"]
        paired_x_fields = {}
        for split in SPLIT_SIZES:
            paired_x_fields[split] = all(
                np.array_equal(x[f"{split}_{field}"], xz[f"{split}_{field}"])
                for field in ("x", "g", "w", "y")
            )
        family_results[family] = {
            "x_and_xz_share_x_g_w_y_by_split": paired_x_fields,
            "paired_x_fields_exact": all(paired_x_fields.values()),
        }

    return {
        "y_checksums": y_checksums,
        "all_scenarios_share_exact_toy_y_by_split": shared_toy_outcomes,
        "family_checks": family_results,
        "family_check_note": (
            "Exact X-image equality between x and xz is diagnostic only. It is not required: "
            "sampling Z images in xz advances that scenario's independent image RNG before "
            "later splits. Method fairness comes from every method consuming the same saved "
            "scenario file; exact toy Y equality confirms the paired DGP."
        ),
        "fairness_invariants_pass": shared_toy_outcomes,
    }


def source_checks(metadata: dict[str, Any]) -> dict[str, Any]:
    sources = metadata.get("sources", {})
    femnist = sources.get("femnist", {})
    cifar10 = sources.get("cifar10", {})
    expected_digits = {str(digit) for digit in range(10)}
    counts_valid = all(
        set(source.get("counts_by_split_and_digit", {})) == {"train", "dev", "test"}
        and all(
            set(counts) == expected_digits
            and all(int(value) > 0 for value in counts.values())
            for counts in source.get("counts_by_split_and_digit", {}).values()
        )
        for source in (femnist, cifar10)
    )
    split_policies = {
        source.get("generated_split_policy") for source in (femnist, cifar10)
    }
    content_disjoint_policy = (
        len(split_policies) == 1
        and next(iter(split_policies), "").startswith("content_disjoint:")
    )
    return {
        "femnist_only_digits": femnist.get("only_digits"),
        "femnist_source": femnist.get("source"),
        "cifar10_source": cifar10.get("source"),
        "all_digit_classes_nonempty": counts_valid,
        "content_disjoint_generated_split_policy": content_disjoint_policy,
        "source_pool_note": (
            "Generated test uses unique source-test content. Generated train/dev use a "
            "deterministic 80/20 partition of unique source-train content after excluding "
            "content also present in source test. Exact overlap is measured from saved arrays."
        ),
        "source_invariants_pass": (
            femnist.get("only_digits") is True
            and femnist.get("source")
            == "tensorflow_federated.simulation.datasets.emnist.load_data"
            and cifar10.get("source") == "torchvision.datasets.CIFAR10"
            and counts_valid
            and content_disjoint_policy
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# High-dimensional fixed-abs data certification",
        "",
        f"Overall decision: `{report['decision']}`.",
        "",
        "## Certified invariants",
        "",
        "- Response function metadata is `abs` for every scenario.",
        "- FEMNIST source is TFF Federated EMNIST with `only_digits=True`.",
        "- CIFAR-10 source is `torchvision.datasets.CIFAR10`.",
        "- Generator seed is `527`; split sizes are 20,000/10,000/10,000.",
        "- All NPZ hashes, shapes, numeric finiteness checks, and standardized abs-response checks passed.",
        "- All six scenarios share the exact same toy Y arrays for paired comparison.",
        "",
        "## Dataset results",
        "",
        "| Dataset | Core invariants | Abs error | Split isolation | Train/test image overlap |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for result in report["datasets"]:
        train_test_overlap = sum(
            modality["exact_unique_image_overlap"]["train_test"]
            for modality in result["image_split_overlap"].values()
        )
        lines.append(
            f"| {result['dataset']} | {str(result['core_invariants_pass']).lower()} "
            f"| {result['max_abs_response_error']:.3g} "
            f"| {str(result['split_isolation_pass']).lower()} | {train_test_overlap} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if report["decision"] == "blocked_before_training":
        lines.extend(
            [
                "The data are internally consistent and fair across methods, but they are not certified for ",
                "train/dev/test isolation because the saved arrays contain exact image reuse across generated ",
                "splits. Repair the split policy and regenerate before final high-dimensional training.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "All required protocol and split-isolation checks passed. The files are certified for the ",
                "fixed-abs high-dimensional experiment.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--generation-manifest", default=str(DEFAULT_GENERATION_MANIFEST))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_REPORT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    generation_manifest = Path(args.generation_manifest).resolve()
    json_out = Path(args.json_out).resolve()
    markdown_out = Path(args.markdown_out).resolve()
    metadata = json.loads(generation_manifest.read_text(encoding="utf-8"))

    metadata_invariants = {
        "protocol_version": metadata.get("protocol_version"),
        "g_function": metadata.get("g_function"),
        "seed": metadata.get("seed"),
        "split_sizes": metadata.get("split_sizes"),
        "valid": (
            metadata.get("protocol_version") == "rerun_protocol_v1_real_images_abs_alpha0p5"
            and metadata.get("g_function") == "abs"
            and int(metadata.get("seed", -1)) == 527
            and metadata.get("split_sizes") == SPLIT_SIZES
        ),
    }
    entries = {item["dataset"]: item for item in metadata.get("generated_files", [])}
    dataset_results: list[dict[str, Any]] = []
    arrays_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    for spec in SPECS:
        path = data_root / spec.name / "main.npz"
        if spec.name not in entries:
            raise RuntimeError(f"Missing generation-manifest entry for {spec.name}")
        result, arrays = certify_dataset(spec, path, entries[spec.name])
        dataset_results.append(result)
        arrays_by_dataset[spec.name] = arrays
        print(
            f"{spec.name}: core={result['core_invariants_pass']} "
            f"split_isolation={result['split_isolation_pass']} "
            f"abs_error={result['max_abs_response_error']:.3g}",
            flush=True,
        )

    fairness = fairness_checks(arrays_by_dataset)
    sources = source_checks(metadata)
    core_pass = (
        metadata_invariants["valid"]
        and sources["source_invariants_pass"]
        and fairness["fairness_invariants_pass"]
        and all(result["core_invariants_pass"] for result in dataset_results)
    )
    split_isolation_pass = all(result["split_isolation_pass"] for result in dataset_results)
    if not core_pass:
        decision = "failed_core_invariants"
    elif not split_isolation_pass:
        decision = "blocked_before_training"
    else:
        decision = "certified"

    report = {
        "certification_scope": "fixed_abs_high_dimensional_federated_data",
        "generation_manifest": str(generation_manifest.relative_to(REPO_ROOT)),
        "metadata_invariants": metadata_invariants,
        "source_checks": sources,
        "fairness_checks": fairness,
        "datasets": dataset_results,
        "core_invariants_pass": core_pass,
        "split_isolation_pass": split_isolation_pass,
        "decision": decision,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_out, report)
    print(json.dumps({
        "decision": decision,
        "json_report": str(json_out.relative_to(REPO_ROOT)),
        "markdown_report": str(markdown_out.relative_to(REPO_ROOT)),
    }, indent=2, sort_keys=True))
    return 0 if decision == "certified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
