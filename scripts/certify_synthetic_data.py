#!/usr/bin/env python3
"""Certify legacy low-dimensional synthetic data without modifying it.

The script intentionally reuses ``generate_zoo_data.create_dataset`` for
candidate generation.  It writes only to ``results/_data_certification`` and
``experiments/data_certification`` unless the repository generator itself is
certified as paper-compatible and legacy contents need replacement.  In that
case only, it may create a separate ``data/paper_v1`` directory; it never
writes to ``data/zoo``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
LEGACY_DATA_ROOT = EXAMPLE_ROOT / "data" / "zoo"
CERTIFICATION_RUN_ROOT = REPO_ROOT / "results" / "_data_certification"
ARTIFACT_ROOT = REPO_ROOT / "experiments" / "data_certification"
PAPER_V1_ROOT = EXAMPLE_ROOT / "data" / "paper_v1" / "seed_527"
GENERATOR_SEED = 527
SPLIT_SIZES = {"train": 20_000, "dev": 20_000, "test": 20_000}
NUMERIC_TOLERANCE = 1e-12

if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

# Import the exact public helper used by the repository's low-dimensional
# generator.  The scenario imports below are only used to inspect its metadata
# and validate the saved structural function after that helper has generated
# candidates.
from generate_zoo_data import create_dataset  # noqa: E402
from scenarios.toy_scenarios import AGMMZoo, Standardizer  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    repository_dataset_name: str


SPECS = (
    DatasetSpec("absolute", "abs"),
    DatasetSpec("step", "step"),
    DatasetSpec("linear", "linear"),
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _require_repo_path(path: Path) -> Path:
    resolved = path.resolve()
    if not _is_relative_to(resolved, REPO_ROOT):
        raise ValueError(f"Refusing to write outside the repository: {resolved}")
    return resolved


def _require_nonlegacy_output(path: Path) -> Path:
    resolved = _require_repo_path(path)
    if _is_relative_to(resolved, LEGACY_DATA_ROOT):
        raise ValueError(f"Refusing to write into protected legacy data: {resolved}")
    return resolved


def array_checksum(key: str, array: np.ndarray) -> str:
    """Hash array identity and payload without depending on NPZ container bytes."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lower_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def value_at(array: np.ndarray, index: tuple[int, ...] | None) -> Any:
    if index is None:
        return None
    value = array[index]
    return value.item() if hasattr(value, "item") else value


def first_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[tuple[int, ...] | None, Any, Any]:
    if left.shape != right.shape:
        return None, None, None
    unequal = np.not_equal(left, right)
    locations = np.argwhere(unequal)
    if locations.size == 0:
        return None, None, None
    index = tuple(int(item) for item in locations[0])
    return index, value_at(left, index), value_at(right, index)


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or not (
        np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number)
    ):
        return None
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def compare_arrays(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    per_key: dict[str, Any] = {}
    first_key = None
    first_index = None
    first_left = None
    first_right = None
    maximum = 0.0
    has_numeric_difference = False

    for key in sorted(left_keys | right_keys):
        entry: dict[str, Any] = {
            "present_in_left": key in left,
            "present_in_right": key in right,
        }
        if key not in left or key not in right:
            entry.update(
                {
                    "shape_equal": False,
                    "dtype_equal": False,
                    "array_equal": False,
                    "max_abs_difference": None,
                    "first_differing_index": None,
                    "left_value": None,
                    "right_value": None,
                    "left_checksum": array_checksum(key, left[key]) if key in left else None,
                    "right_checksum": array_checksum(key, right[key]) if key in right else None,
                }
            )
        else:
            left_array = left[key]
            right_array = right[key]
            # Array payload equality alone is insufficient for certification:
            # dtype is part of the required array identity.
            equal = bool(
                left_array.shape == right_array.shape
                and left_array.dtype == right_array.dtype
                and np.array_equal(left_array, right_array)
            )
            index, left_value, right_value = first_difference(left_array, right_array)
            max_difference = max_abs_difference(left_array, right_array)
            entry.update(
                {
                    "shape_equal": left_array.shape == right_array.shape,
                    "dtype_equal": left_array.dtype == right_array.dtype,
                    "array_equal": equal,
                    "max_abs_difference": max_difference,
                    "first_differing_index": list(index) if index is not None else None,
                    "left_value": json_safe(left_value),
                    "right_value": json_safe(right_value),
                    "left_checksum": array_checksum(key, left_array),
                    "right_checksum": array_checksum(key, right_array),
                }
            )
            if max_difference is not None:
                maximum = max(maximum, max_difference)
                has_numeric_difference = has_numeric_difference or max_difference != 0.0
            if not equal and first_key is None:
                first_key = key
                first_index = list(index) if index is not None else None
                first_left = json_safe(left_value)
                first_right = json_safe(right_value)
        per_key[key] = entry

    all_shapes_match = all(item["shape_equal"] for item in per_key.values())
    all_dtypes_match = all(item["dtype_equal"] for item in per_key.values())
    all_arrays_exact = left_keys == right_keys and all(item["array_equal"] for item in per_key.values())
    return {
        "key_set_equal": left_keys == right_keys,
        "all_shapes_match": all_shapes_match,
        "all_dtypes_match": all_dtypes_match,
        "all_arrays_exact": all_arrays_exact,
        "max_abs_difference": maximum if has_numeric_difference else 0.0,
        "first_differing_key": first_key,
        "first_differing_index": first_index,
        "first_left_value": first_left,
        "first_right_value": first_right,
        "per_key": per_key,
    }


def paper_true_function(dataset: str, x: np.ndarray) -> np.ndarray:
    if dataset == "absolute":
        return np.abs(x)
    if dataset == "step":
        return (x >= 0).astype(np.float64)
    if dataset == "linear":
        return x
    raise ValueError(f"Unsupported certification dataset: {dataset}")


def generate_metadata(repository_dataset_name: str) -> dict[str, Any]:
    """Run the existing scenario in memory to expose its stored normalization."""

    np.random.seed(GENERATOR_SEED)
    scenario = Standardizer(
        AGMMZoo(repository_dataset_name, two_gps=False, n_instruments=2)
    )
    scenario.setup(
        num_train=SPLIT_SIZES["train"],
        num_dev=SPLIT_SIZES["dev"],
        num_test=SPLIT_SIZES["test"],
    )
    raw_scenario = scenario._scenario
    return {
        "standardizer_mean": float(scenario._mean),
        "standardizer_std": float(scenario._std),
        "two_gps": bool(raw_scenario._two_gps),
        "n_instruments": int(raw_scenario._n_instruments),
        "iv_strength": float(raw_scenario._iv_strength),
        "generator_true_function": raw_scenario._true_g_function_np,
    }


def generate_candidate_run(run_dir: Path) -> dict[str, Path]:
    run_dir = _require_nonlegacy_output(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for spec in SPECS:
        output_prefix = run_dir / spec.repository_dataset_name
        _require_nonlegacy_output(output_prefix)
        # This is the repository generator itself. It resets NumPy to seed 527
        # before each dataset, exactly as the author script does.
        create_dataset(spec.repository_dataset_name, dir=f"{run_dir}{os.sep}")
        output_path = output_prefix.with_suffix(".npz")
        if not output_path.exists():
            raise RuntimeError(f"Generator did not create expected candidate: {output_path}")
        outputs[spec.dataset] = output_path
    return outputs


def discover_split_names(arrays: dict[str, np.ndarray]) -> list[str]:
    if "splits" in arrays:
        values = arrays["splits"].tolist()
        return [str(value) for value in values]
    prefixes = {key.split("_", 1)[0] for key in arrays if "_" in key}
    return sorted(prefixes)


def split_key(split: str, field: str) -> str:
    return f"{split}_{field}"


def check_semantics(
    spec: DatasetSpec,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    split_names = discover_split_names(arrays)
    expected_split_names = {"train", "dev", "test"}
    split_results: dict[str, Any] = {}
    all_finite = True
    split_sizes_match = set(split_names) == expected_split_names
    all_dimensions_compatible = True
    z_in_expected_range = True
    max_function_error = 0.0
    max_generator_function_error = 0.0

    generator_function = metadata["generator_true_function"]
    for split in split_names:
        required = {field: split_key(split, field) for field in ("x", "y", "z", "g", "w")}
        missing = [key for key in required.values() if key not in arrays]
        entry: dict[str, Any] = {"missing_keys": missing}
        if missing:
            split_results[split] = entry
            split_sizes_match = False
            all_dimensions_compatible = False
            continue

        x = arrays[required["x"]]
        y = arrays[required["y"]]
        z = arrays[required["z"]]
        g = arrays[required["g"]]
        w = arrays[required["w"]]
        sizes = {name: int(value.shape[0]) for name, value in {"x": x, "y": y, "z": z, "g": g, "w": w}.items()}
        split_size = sizes["x"]
        compatible = len(set(sizes.values())) == 1 and z.ndim == 2 and z.shape[1] == 2
        finite = all(
            np.all(np.isfinite(value))
            for value in (x, y, z, g, w)
            if np.issubdtype(value.dtype, np.number)
        )
        paper_raw_g = paper_true_function(spec.dataset, x)
        generator_raw_g = generator_function(x)
        normalized_paper_g = (paper_raw_g - metadata["standardizer_mean"]) / metadata["standardizer_std"]
        normalized_generator_g = (generator_raw_g - metadata["standardizer_mean"]) / metadata["standardizer_std"]
        paper_error = float(np.max(np.abs(g - normalized_paper_g)))
        generator_error = float(np.max(np.abs(g - normalized_generator_g)))
        max_function_error = max(max_function_error, paper_error)
        max_generator_function_error = max(max_generator_function_error, generator_error)
        expected_size = SPLIT_SIZES.get(split)
        split_sizes_match = split_sizes_match and expected_size == split_size
        all_dimensions_compatible = all_dimensions_compatible and compatible
        all_finite = all_finite and finite
        z_in_expected_range = z_in_expected_range and bool(np.all(z >= -3.0) and np.all(z <= 3.0))
        entry.update(
            {
                "size": split_size,
                "expected_size": expected_size,
                "array_sizes": sizes,
                "z_shape": list(z.shape),
                "dimensions_compatible": compatible,
                "all_finite": finite,
                "z_in_uniform_support": bool(np.all(z >= -3.0) and np.all(z <= 3.0)),
                "max_abs_true_function_error": paper_error,
                "max_abs_generator_true_function_error": generator_error,
            }
        )
        split_results[split] = entry

    code_matches_paper_function = max_function_error <= NUMERIC_TOLERANCE
    return {
        "split_names": split_names,
        "split_results": split_results,
        "split_sizes_match": split_sizes_match,
        "all_dimensions_compatible": all_dimensions_compatible,
        "all_finite": all_finite,
        "z_in_expected_range": z_in_expected_range,
        "max_abs_true_function_error": max_function_error,
        "max_abs_generator_true_function_error": max_generator_function_error,
        "paper_true_function_match": code_matches_paper_function,
        "standardization_stage": "global_train_y_mean_std_applied_to_y_and_g_before_save",
    }


def code_dgp_report(metadata: dict[str, Any]) -> dict[str, Any]:
    # This reflects the executed AGMMZoo branch; the equations are not
    # reimplemented here.
    matches_paper = (
        metadata["two_gps"] is True
        and metadata["n_instruments"] == 2
        and metadata["iv_strength"] == 1.0
    )
    return {
        "implemented": (
            "AGMMZoo(two_gps=False, n_instruments=2, iv_strength=0.5): "
            "X=Z1+confounder+Normal(0, std=0.1); "
            "Y=g(X)+2*confounder+Normal(0, std=0.1)"
        ),
        "paper_expected": (
            "X=Z1+Z2+e+gamma_noise; Y=g0(X)+e+delta; "
            "gamma_noise/delta Normal(0, std=0.1)"
        ),
        "matches_paper_expected_dgp": matches_paper,
        "noise_parameter_interpretation": "standard_deviation",
        "noise_std": 0.1,
    }


def compare_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = load_npz(left_path)
    right = load_npz(right_path)
    comparison = compare_arrays(left, right)
    comparison.update(
        {
            "left_file": str(left_path),
            "right_file": str(right_path),
            "left_file_sha256": file_checksum(left_path),
            "right_file_sha256": file_checksum(right_path),
        }
    )
    return comparison


def content_status(comparison: dict[str, Any], semantic_match: bool) -> str:
    if comparison["all_arrays_exact"]:
        if comparison["left_file_sha256"] == comparison["right_file_sha256"]:
            return "exact_content_match"
        return "content_match_file_container_differs"
    if semantic_match:
        return "semantic_match_nonexact"
    return "mismatch"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _require_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


CSV_FIELDS = (
    "dataset",
    "repository_dataset_name",
    "legacy_file",
    "generator_source",
    "generator_seed",
    "generator_deterministic",
    "train_size",
    "validation_size",
    "test_size",
    "npz_keys",
    "all_shapes_match",
    "all_dtypes_match",
    "all_arrays_exact",
    "legacy_file_sha256",
    "generated_file_sha256",
    "max_abs_difference",
    "first_differing_key",
    "first_differing_index",
    "true_function",
    "max_abs_true_function_error",
    "noise_parameter_interpretation",
    "standardization_stage",
    "certification_status",
    "reuse_legacy_data",
    "selected_data_path",
    "decision_status",
    "notes",
)


def csv_row(result: dict[str, Any]) -> dict[str, Any]:
    comparison = result["legacy_comparison"]
    semantics = result["semantics"]
    split_results = semantics["split_results"]
    return {
        "dataset": result["dataset"],
        "repository_dataset_name": result["repository_dataset_name"],
        "legacy_file": result["legacy_file"],
        "generator_source": result["generator_source"],
        "generator_seed": GENERATOR_SEED,
        "generator_deterministic": lower_bool(result["generator_deterministic"]),
        "train_size": split_results.get("train", {}).get("size", ""),
        "validation_size": split_results.get("dev", {}).get("size", ""),
        "test_size": split_results.get("test", {}).get("size", ""),
        "npz_keys": "|".join(result["npz_keys"]),
        "all_shapes_match": lower_bool(comparison["all_shapes_match"]),
        "all_dtypes_match": lower_bool(comparison["all_dtypes_match"]),
        "all_arrays_exact": lower_bool(comparison["all_arrays_exact"]),
        "legacy_file_sha256": comparison["left_file_sha256"],
        "generated_file_sha256": comparison["right_file_sha256"],
        "max_abs_difference": comparison["max_abs_difference"],
        "first_differing_key": comparison["first_differing_key"] or "",
        "first_differing_index": json.dumps(comparison["first_differing_index"]),
        "true_function": result["paper_true_function"],
        "max_abs_true_function_error": semantics["max_abs_true_function_error"],
        "noise_parameter_interpretation": result["dgp"]["noise_parameter_interpretation"],
        "standardization_stage": semantics["standardization_stage"],
        "certification_status": result["certification_status"],
        "reuse_legacy_data": lower_bool(result["reuse_legacy_data"]),
        "selected_data_path": result["selected_data_path"] or "",
        "decision_status": result["decision_status"],
        "notes": result["notes"],
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _require_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Low-dimensional synthetic-data certification",
        "",
        "This certification compares array contents rather than relying only on `.npz` file hashes. ",
        "An NPZ container can differ because of ZIP metadata or layout while every stored array is identical.",
        "",
        "## Method",
        "",
        "Candidates were generated twice with the repository's own `generate_zoo_data.create_dataset` helper, ",
        f"resetting its documented NumPy seed `{GENERATOR_SEED}` for every dataset. Candidates are stored only under ",
        "`results/_data_certification/`; no file under `data/zoo/` is written or modified.",
        "",
        "Array checksums hash the key name, shape, dtype, and contiguous bytes. The code uses ",
        "`np.random.normal(0, 0.1, ...)`; NumPy's second positional parameter is the standard deviation, ",
        "so the code implements standard deviation 0.1 (variance 0.01).",
        "",
        "## Standardization",
        "",
        "The generator's `Standardizer` captures mean/std from the first generated (train) Y split and applies ",
        "that transform to both Y and stored true-g for train, dev, and test before writing the NPZ file.",
        "",
        "## Decision",
        "",
        f"Overall decision: `{report['decision_status']}`.",
        "",
    ]
    for result in report["datasets"]:
        lines.extend(
            [
                f"### {result['dataset'].title()}",
                "",
                f"- Legacy comparison: `{result['certification_status']}`",
                f"- Generator deterministic: `{lower_bool(result['generator_deterministic'])}`",
                f"- Legacy reusable for paper-aligned replication: `{lower_bool(result['reuse_legacy_data'])}`",
                f"- Paper-function max error: `{result['semantics']['max_abs_true_function_error']}`",
                f"- Selected data path: `{result['selected_data_path'] or 'none (blocked)'}`",
                f"- Notes: {result['notes']}",
                "",
            ]
        )
    if report["decision_status"] == "blocked":
        lines.extend(
            [
                "## Blocker",
                "",
                "The current author generator does not implement the requested paper DGP: it is configured with ",
                "`two_gps=False`, so X omits Z2, and it uses a `2 * confounder` term in Y. Its Step function is ",
                "also `1` below zero and `2.5` at/above zero rather than `1{x >= 0}`. Therefore the legacy files may ",
                "be exactly reproducible author-code artifacts, but they are not certified for paper-aligned reuse.",
                "No `data/paper_v1/` replacement was created.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_paper_v1(candidates: dict[str, Path], report: dict[str, Any]) -> dict[str, str]:
    """Create separate versioned data only after all paper checks have passed."""

    _require_nonlegacy_output(PAPER_V1_ROOT)
    PAPER_V1_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for spec in SPECS:
        source = candidates[spec.dataset]
        destination = PAPER_V1_ROOT / f"{spec.dataset}.npz"
        _require_nonlegacy_output(destination)
        shutil.copy2(source, destination)
        paths[spec.dataset] = str(destination.relative_to(REPO_ROOT))
    recipe = {
        "recipe_id": "paper_v1_synthetic_author_data",
        "datasets": [spec.dataset for spec in SPECS],
        "generator_seed": GENERATOR_SEED,
        "regenerate_per_final_run": False,
        "noise_parameter_interpretation": "standard_deviation",
        "noise_std": 0.1,
        "data_source_status": "author_code_derived_exactly_certified",
        "certification_status": "passed",
        "paths": paths,
        "source_code": report["generator_source"],
    }
    write_json(PAPER_V1_ROOT / "data_recipe.json", recipe)
    return paths


def certify() -> dict[str, Any]:
    CERTIFICATION_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_a = CERTIFICATION_RUN_ROOT / "run_a"
    run_b = CERTIFICATION_RUN_ROOT / "run_b"
    candidates_a = generate_candidate_run(run_a)
    candidates_b = generate_candidate_run(run_b)

    deterministic_results: dict[str, dict[str, Any]] = {}
    for spec in SPECS:
        comparison = compare_files(candidates_a[spec.dataset], candidates_b[spec.dataset])
        deterministic_results[spec.dataset] = comparison
        if not comparison["all_arrays_exact"]:
            raise RuntimeError(
                "generator_nondeterministic: "
                f"dataset={spec.dataset}, key={comparison['first_differing_key']}, "
                f"index={comparison['first_differing_index']}, "
                f"run_a={comparison['first_left_value']}, run_b={comparison['first_right_value']}"
            )

    dgp_by_dataset: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for spec in SPECS:
        metadata = generate_metadata(spec.repository_dataset_name)
        dgp = code_dgp_report(metadata)
        dgp_by_dataset[spec.dataset] = dgp
        generated_arrays = load_npz(candidates_a[spec.dataset])
        semantics = check_semantics(spec, generated_arrays, metadata)
        legacy_path = LEGACY_DATA_ROOT / f"{spec.repository_dataset_name}.npz"
        if legacy_path.exists():
            legacy_comparison = compare_files(legacy_path, candidates_a[spec.dataset])
            legacy_status = content_status(
                legacy_comparison,
                semantics["split_sizes_match"]
                and semantics["all_finite"]
                and semantics["all_dimensions_compatible"],
            )
        else:
            legacy_comparison = {
                "left_file_sha256": None,
                "right_file_sha256": file_checksum(candidates_a[spec.dataset]),
                "key_set_equal": False,
                "all_shapes_match": False,
                "all_dtypes_match": False,
                "all_arrays_exact": False,
                "max_abs_difference": None,
                "first_differing_key": None,
                "first_differing_index": None,
                "per_key": {},
            }
            legacy_status = "missing_legacy_file"

        paper_semantics_pass = bool(
            semantics["split_sizes_match"]
            and semantics["all_finite"]
            and semantics["all_dimensions_compatible"]
            and semantics["z_in_expected_range"]
            and semantics["paper_true_function_match"]
            and dgp["matches_paper_expected_dgp"]
        )
        notes = []
        if not dgp["matches_paper_expected_dgp"]:
            notes.append("generator_dgp_differs_from_paper_expected")
        if not semantics["paper_true_function_match"]:
            notes.append("stored_true_g_does_not_match_paper_true_function")
        if legacy_status in {"exact_content_match", "content_match_file_container_differs"}:
            notes.append("legacy_file_is_exactly_reproducible_from_current_generator")
        results.append(
            {
                "dataset": spec.dataset,
                "repository_dataset_name": spec.repository_dataset_name,
                "legacy_file": str(legacy_path.relative_to(REPO_ROOT)) if legacy_path.exists() else str(legacy_path),
                "generated_file": str(candidates_a[spec.dataset].relative_to(REPO_ROOT)),
                "generator_source": "fedgmm/sp_decentralized_mnist_lr_example/generate_zoo_data.py:create_dataset",
                "generator_deterministic": True,
                "npz_keys": sorted(generated_arrays),
                "paper_true_function": {
                    "absolute": "abs(x)",
                    "step": "1{x >= 0}",
                    "linear": "x",
                }[spec.dataset],
                "dgp": dgp,
                "semantics": semantics,
                "determinism_comparison": deterministic_results[spec.dataset],
                "legacy_comparison": legacy_comparison,
                "certification_status": legacy_status,
                "paper_semantics_pass": paper_semantics_pass,
                "reuse_legacy_data": False,
                "selected_data_path": None,
                "decision_status": "blocked" if not paper_semantics_pass else "pending_legacy_decision",
                "notes": ";".join(notes),
            }
        )

    all_semantics_pass = all(result["paper_semantics_pass"] for result in results)
    all_legacy_exact = all(
        result["certification_status"] in {"exact_content_match", "content_match_file_container_differs"}
        for result in results
    )
    decision_status = "blocked"
    selected_paths: dict[str, str] = {}
    if all_semantics_pass and all_legacy_exact:
        decision_status = "passed"
        for result in results:
            result["reuse_legacy_data"] = True
            result["selected_data_path"] = result["legacy_file"]
            result["decision_status"] = "reuse_legacy_data"
            selected_paths[result["dataset"]] = result["legacy_file"]
        recipe = {
            "recipe_id": "paper_v1_synthetic_author_data",
            "datasets": [spec.dataset for spec in SPECS],
            "generator_seed": GENERATOR_SEED,
            "regenerate_per_final_run": False,
            "noise_parameter_interpretation": "standard_deviation",
            "noise_std": 0.1,
            "data_source_status": "author_code_derived_exactly_certified",
            "certification_status": "passed",
            "paths": selected_paths,
        }
        write_json(ARTIFACT_ROOT / "paper_v1_data_recipe.json", recipe)
    elif all_semantics_pass:
        decision_status = "replication_dataset_created_from_author_code"
        report_stub = {
            "generator_source": "fedgmm/sp_decentralized_mnist_lr_example/generate_zoo_data.py:create_dataset"
        }
        selected_paths = generate_paper_v1(candidates_a, report_stub)
        for result in results:
            result["selected_data_path"] = selected_paths[result["dataset"]]
            result["decision_status"] = decision_status
    else:
        for result in results:
            result["decision_status"] = "blocked"

    report = {
        "certification_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "generator_source": "fedgmm/sp_decentralized_mnist_lr_example/generate_zoo_data.py:create_dataset",
        "generator_seed": GENERATOR_SEED,
        "candidate_runs": {
            "run_a": str(run_a.relative_to(REPO_ROOT)),
            "run_b": str(run_b.relative_to(REPO_ROOT)),
        },
        "decision_status": decision_status,
        "selected_data_paths": selected_paths,
        "dgp_by_dataset": dgp_by_dataset,
        "datasets": results,
    }
    return report


def main() -> int:
    report = certify()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(ARTIFACT_ROOT / "synthetic_data_certification.csv", [csv_row(item) for item in report["datasets"]])
    write_json(ARTIFACT_ROOT / "synthetic_data_certification.json", report)
    write_readme(ARTIFACT_ROOT / "README.md", report)
    print(f"Synthetic data certification decision: {report['decision_status']}")
    for result in report["datasets"]:
        print(
            f"{result['dataset']}: {result['certification_status']}; "
            f"paper_semantics_pass={lower_bool(result['paper_semantics_pass'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
