#!/usr/bin/env python3
"""Inventory pre-fix BatchNorm-bearing image trajectories as legacy evidence."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
OUTPUT = PROTOCOL_ROOT / "legacy_batchnorm_trajectories_20260822.json"
SOURCES = (
    ("screen", PROTOCOL_ROOT / "deterministic_screen_20260813/screen_manifest.csv", None),
    ("screen_expand", PROTOCOL_ROOT / "deterministic_screen_20260813/screen_expand_manifest.csv", None),
    ("screen_expand2", PROTOCOL_ROOT / "deterministic_screen_20260813/screen_expand2_manifest.csv", None),
    (
        "screen_expand2_corrected",
        PROTOCOL_ROOT / "deterministic_screen_20260813/screen_expand2_corrected_v1_manifest.csv",
        None,
    ),
    ("finals", PROTOCOL_ROOT / "deterministic_finals_20260813/finals_manifest.csv", None),
    (
        "v2_signal",
        PROTOCOL_ROOT / "psi_adjudication_20260819_v2/adjudication_signal_manifest.csv",
        None,
    ),
    (
        "v2_x",
        PROTOCOL_ROOT / "psi_adjudication_20260819_v2/adjudication_x_manifest.csv",
        None,
    ),
    (
        "10client_runtime_profile",
        PROTOCOL_ROOT / "deterministic_10client_runtime_profile_20260807/profile_manifest.csv",
        None,
    ),
    (
        "learning_gate_equivalence",
        PROTOCOL_ROOT / "deterministic_learning_gate_20260802/equivalence_manifest.csv",
        None,
    ),
    (
        "learning_gate",
        PROTOCOL_ROOT / "deterministic_learning_gate_20260802/gate_manifest.csv",
        None,
    ),
    (
        "multiseed_validation",
        PROTOCOL_ROOT / "deterministic_multiseed_validation_20260803/manifest.csv",
        None,
    ),
    (
        "runtime_profile",
        PROTOCOL_ROOT / "deterministic_runtime_profile_20260805/profile_manifest.csv",
        None,
    ),
)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _artifact_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "absent"
    required = (
        "effective_config.json",
        "metrics.json",
        "mse_by_round.csv",
        "predictions.npz",
        "checkpoints/best_validation.pt",
        "checkpoints/final.pt",
    )
    return "complete" if all((run_dir / name).exists() for name in required) else "partial"


def build_registry() -> dict:
    records = []
    resolved_dirs: list[str] = []
    for campaign, manifest_path, _ in SOURCES:
        if not manifest_path.exists():
            continue
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            dataset = str(row.get("dataset", ""))
            if not dataset.startswith(("mnist_", "femnist_", "cifar10_", "cifar_")):
                continue
            run_dir_text = str(row.get("final_result_dir", ""))
            run_dir = Path(run_dir_text)
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir
            resolved_dirs.append(str(run_dir))
            config = _load_json(run_dir / "effective_config.json")
            records.append({
                "campaign": campaign,
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "run_id": str(row.get("run_id", "")),
                "dataset": dataset,
                "method": str(row.get("method", "")),
                "seed": str(row.get("seed", "")),
                "result_dir": run_dir_text,
                "artifact_state": _artifact_state(run_dir),
                "recorded_server_buffer_policy": config.get("server_buffer_policy"),
                "scientific_status": "legacy_ineligible",
                "reason": "trajectory began before the BatchNorm server-buffer correction",
            })
    counts: dict[str, int] = {}
    for record in records:
        key = str(record["artifact_state"])
        counts[key] = counts.get(key, 0) + 1

    # A manifest entry is a row; a physical trajectory is the actual on-disk
    # run_dir it points at. Two manifests can legitimately name the same
    # run_dir (e.g. a candidate reused across campaigns), so entries and
    # trajectories are not automatically equal -- report both explicitly
    # rather than assuming, and name the specific duplicates if any exist.
    dir_counts: dict[str, int] = {}
    for resolved in resolved_dirs:
        dir_counts[resolved] = dir_counts.get(resolved, 0) + 1
    duplicated_trajectories = sorted(path for path, count in dir_counts.items() if count > 1)

    return {
        "schema_version": 1,
        "policy": "legacy trajectories are preserved but may not be resumed, mixed, or promoted",
        "records": records,
        "summary": {
            "total": len(records),
            "artifact_states": counts,
            "manifest_entries": len(records),
            "unique_physical_trajectories": len(set(resolved_dirs)),
            "duplicated_trajectories": duplicated_trajectories,
        },
    }


def main() -> int:
    registry = build_registry()
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(registry["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
