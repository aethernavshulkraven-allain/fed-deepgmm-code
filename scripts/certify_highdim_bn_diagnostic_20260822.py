#!/usr/bin/env python3
"""Certify the exact 120-round post-BatchNorm-fix diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import CORE_SOURCES  # noqa: E402
from run_manifest import ManifestLaunchError, validate_artifacts  # noqa: E402
from verify_protocol_hashes import sha256_file  # noqa: E402

CORE_NUMERICAL_SOURCES = CORE_SOURCES
EXPECTED = {
    "dataset": "femnist_z",
    "method": "fedogda_d",
    "client_optimizer": "ogda",
    "seed": "1",
    "comm_round": "120",
    "learning_rate": "0.001",
    "critic_multiplier": "10",
    "server_buffer_policy": "direct_client_aggregate",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def certify(
    manifest_path: Path,
    launcher_results_path: Path,
    launch_hashes_path: Path,
    output_path: Path,
) -> dict:
    rows = _load_rows(manifest_path)
    if len(rows) != 1:
        raise ValueError(f"diagnostic manifest must contain exactly one run, got {len(rows)}")
    row = rows[0]
    for field, expected in EXPECTED.items():
        if str(row.get(field)) != expected:
            raise ValueError(
                f"diagnostic {field}={row.get(field)!r}; expected {expected!r}"
            )

    launcher_results = _load_json(launcher_results_path)
    if not isinstance(launcher_results, list) or len(launcher_results) != 1:
        raise ValueError("diagnostic launcher results must contain exactly one result")
    launcher_result = launcher_results[0]
    if str(launcher_result.get("run_id")) != str(row["run_id"]):
        raise ValueError("diagnostic launcher result has the wrong run_id")
    if launcher_result.get("status") not in {"passed", "skipped_completed"}:
        raise ValueError(
            f"diagnostic launcher status is not clean: {launcher_result.get('status')!r}"
        )

    run_dir = Path(str(row["final_result_dir"]))
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    try:
        validation = validate_artifacts(run_dir, row)
    except ManifestLaunchError as exc:
        raise ValueError(f"diagnostic artifact validation failed: {exc}") from exc
    if validation["terminal_ineligible"]:
        raise ValueError(f"diagnostic is terminal-ineligible: {validation['terminal_reason']}")

    metrics = _load_json(run_dir / "metrics.json")
    curve = _load_rows(run_dir / "mse_by_round.csv")
    f_bn_values = [float(curve_row["f_bn_min_running_var"]) for curve_row in curve]
    if not all(math.isfinite(value) and value >= 0.0 for value in f_bn_values):
        raise ValueError("diagnostic has negative or nonfinite critic BatchNorm variance")
    if metrics.get("nonfinite_first_round") is not None:
        raise ValueError("diagnostic recorded a nonfinite first round")
    if metrics.get("nonfinite_diagnostics") != []:
        raise ValueError("diagnostic recorded nonfinite critic diagnostics")

    launch_hash_records = _load_json(launch_hashes_path)
    if not isinstance(launch_hash_records, list):
        raise ValueError("launch hash record must be a JSON list")
    launch_hashes = {
        str(record.get("path")): str(record.get("sha256"))
        for record in launch_hash_records
        if isinstance(record, dict)
    }
    for relative in (*CORE_NUMERICAL_SOURCES, str(manifest_path.relative_to(REPO_ROOT))):
        expected_hash = launch_hashes.get(relative)
        if expected_hash is None:
            raise ValueError(f"launch hashes do not contain {relative}")
        actual_hash = sha256_file(REPO_ROOT / relative)
        if actual_hash != expected_hash:
            raise ValueError(
                f"launch provenance mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    artifact_names = (
        "effective_config.json",
        "metrics.json",
        "mse_by_round.csv",
        "predictions.npz",
        "checkpoints/best_validation.pt",
        "checkpoints/final.pt",
    )
    certification = {
        "certification_status": "passed",
        "run_id": row["run_id"],
        "criteria": {
            "exact_rounds_0_through_119": True,
            "completed_and_nondivergent": True,
            "finite_critic_outputs": True,
            "nonnegative_batchnorm_running_variance": True,
            "server_buffer_policy": "direct_client_aggregate",
        },
        "minimum_f_bn_running_variance": min(f_bn_values),
        "launch_hash_record": str(launch_hashes_path.relative_to(REPO_ROOT)),
        # A path string alone doesn't protect against the launch-hashes file
        # itself being silently replaced with a different set of records
        # after certification (e.g. regenerated to match newer source) --
        # verify_highdim_bn_diagnostic_certification_20260822.py's fresh-
        # recomputation comparison only re-derives from *whatever* currently
        # lives at that path, so it can't detect that kind of swap on its
        # own. This checksum pins the launch-hashes file's own content at
        # certification time, closing that gap.
        "launch_hash_record_sha256": sha256_file(launch_hashes_path),
        "artifact_hashes": [
            {"path": name, "sha256": sha256_file(run_dir / name)}
            for name in artifact_names
        ],
    }
    _atomic_json(output_path, certification)
    return certification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launcher-results", type=Path, required=True)
    parser.add_argument("--launch-hashes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        certification = certify(
            args.manifest.resolve(),
            args.launcher_results.resolve(),
            args.launch_hashes.resolve(),
            args.out.resolve(),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"DIAGNOSTIC CERTIFICATION FAILED: {exc}")
        return 1
    print(json.dumps(certification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
