#!/usr/bin/env python3
"""Repair manifest provenance omitted from completed Study A federated JSONs.

This tool changes metadata only. Before replacing either JSON artifact it
copies the original file to a caller-supplied archival root. It refuses to
touch a run unless the existing immutable identity fields agree with the
manifest, and it refuses to replace any non-empty provenance value with a
different value.

The bug this script works around -- the federated result serializer omitting
manifest provenance fields from ``effective_config.json`` -- is fixed at the
source in ``experiment_utils.py`` and passed through by ``run_manifest.py``,
and new campaign runs are (or will be) hard-stopped at write time if
provenance is empty. That write-time guard is the correct fix going forward.
This script's repair path exists only to patch already-completed campaigns
(for example, the Study A v1 demo campaign) where re-running is not an
option, so actually modifying artifacts requires the explicit ``--allow-repair``
flag; ``--dry-run`` keeps working without it since a preview never modifies
anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


PROVENANCE_FIELDS = (
    "role",
    "scenario_name",
    "g0",
    "alignment_label",
    "primary_selection_metric",
    "selection_source",
    "scenario_scope",
    "study_claim",
)


class ReconciliationError(RuntimeError):
    """Raised when an artifact cannot be reconciled without ambiguity."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReconciliationError(f"{path} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temp_path, path)


def _config_checksum(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _assert_identity(
    row: dict[str, str],
    effective_config: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
) -> None:
    checks = (
        ("run_id", row["run_id"], effective_config.get("run_id")),
        ("dataset", row["dataset"], effective_config.get("dataset")),
        ("method", row["method"], effective_config.get("variant")),
        ("optimizer seed", int(row["seed"]), effective_config.get("random_seed")),
        (
            "scenario checksum",
            row["scenario_checksum"],
            effective_config.get("scenario_checksum"),
        ),
        ("metrics run_id", row["run_id"], metrics.get("run_id")),
        ("metrics method", row["method"], metrics.get("method")),
        (
            "metrics scenario seed",
            int(row["scenario_seed"]),
            metrics.get("scenario_seed"),
        ),
        (
            "metrics scenario checksum",
            row["scenario_checksum"],
            metrics.get("scenario_checksum"),
        ),
    )
    for label, expected, actual in checks:
        if actual != expected:
            raise ReconciliationError(
                f"{run_dir}: {label} mismatch: expected {expected!r}, got {actual!r}"
            )


def _set_if_missing(
    artifact: dict[str, Any],
    field: str,
    expected: Any,
    *,
    artifact_path: Path,
) -> bool:
    current = artifact.get(field)
    if current is None or current == "":
        artifact[field] = expected
        return True
    if current != expected:
        raise ReconciliationError(
            f"{artifact_path}: refusing to replace {field}={current!r} "
            f"with {expected!r}"
        )
    return False


def _archive_original(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ReconciliationError(
                f"archive destination exists with different bytes: {destination}"
            )
        return
    shutil.copy2(source, destination)


def reconcile_campaign(
    manifest_path: Path,
    results_root: Path,
    backup_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reconcile federated rows and return an auditable summary."""

    manifest_path = manifest_path.resolve()
    results_root = results_root.resolve()
    backup_root = backup_root.resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "results_root": str(results_root),
        "backup_root": str(backup_root),
        "dry_run": dry_run,
        "manifest_rows": len(rows),
        "federated_rows_checked": 0,
        "runs_changed": 0,
        "runs_already_consistent": 0,
        "artifacts_archived": 0,
        "fields_added_to_effective_config": 0,
        "fields_added_to_metrics": 0,
    }

    for row in rows:
        if row.get("training_scope") != "federated":
            continue
        summary["federated_rows_checked"] += 1
        run_dir = results_root / row["result_path"]
        effective_path = run_dir / "effective_config.json"
        metrics_path = run_dir / "metrics.json"
        if not effective_path.is_file() or not metrics_path.is_file():
            raise ReconciliationError(f"missing completed artifacts in {run_dir}")

        effective_config = _load_json(effective_path)
        metrics = _load_json(metrics_path)
        _assert_identity(row, effective_config, metrics, run_dir)

        effective_changed = 0
        for field in PROVENANCE_FIELDS:
            effective_changed += int(
                _set_if_missing(
                    effective_config,
                    field,
                    row[field],
                    artifact_path=effective_path,
                )
            )

        metric_values: dict[str, Any] = {
            "selection_metric": row["primary_selection_metric"],
            "selection_source": row["selection_source"],
            "alignment_label": row["alignment_label"],
            "is_primary": row["role"] == "confirmatory",
        }
        metrics_changed = 0
        for field, expected in metric_values.items():
            metrics_changed += int(
                _set_if_missing(
                    metrics,
                    field,
                    expected,
                    artifact_path=metrics_path,
                )
            )

        checksum = _config_checksum(effective_config)
        if metrics.get("config_checksum") != checksum:
            metrics["config_checksum"] = checksum
            metrics_changed += 1

        if not effective_changed and not metrics_changed:
            summary["runs_already_consistent"] += 1
            continue

        summary["runs_changed"] += 1
        summary["fields_added_to_effective_config"] += effective_changed
        summary["fields_added_to_metrics"] += metrics_changed
        if dry_run:
            continue

        relative_run = Path(row["result_path"])
        effective_backup = backup_root / relative_run / effective_path.name
        metrics_backup = backup_root / relative_run / metrics_path.name
        _archive_original(effective_path, effective_backup)
        _archive_original(metrics_path, metrics_backup)
        summary["artifacts_archived"] += 2
        _atomic_write_json(effective_path, effective_config)
        _atomic_write_json(metrics_path, metrics)

    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-repair",
        action="store_true",
        help=(
            "Required to actually modify artifacts. Post-hoc repair is a "
            "deliberate act reserved for already-completed campaigns (for "
            "example, Study A v1); the write-time guard in "
            "experiment_utils.py/run_manifest.py is the correct fix for new "
            "campaign runs. Without this flag, only --dry-run is permitted."
        ),
    )
    return parser.parse_args(argv)


def _repair_refusal_message() -> str:
    return (
        "Refusing to modify Study A artifacts: --allow-repair was not "
        "passed. Post-hoc metadata repair must be a deliberate, explicitly "
        "opted-into act, not a routine step -- it is reserved for "
        "already-completed campaigns (for example, the Study A v1 demo "
        "campaign) where re-running is not an option. The root cause this "
        "script works around (the federated serializer omitting provenance "
        "fields from effective_config.json) is now fixed at the source in "
        "experiment_utils.py and passed through by run_manifest.py, and is "
        "enforced going forward by a write-time guard that hard-fails a "
        "campaign run with empty provenance -- that write-time guard is the "
        "correct fix for new runs, not this script. Pass --allow-repair to "
        "opt in to repairing an already-completed campaign, or --dry-run to "
        "preview without modifying anything."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dry_run and not args.allow_repair:
        print(json.dumps({"error": _repair_refusal_message()}, indent=2))
        return 1
    try:
        summary = reconcile_campaign(
            args.manifest,
            args.results_root,
            args.backup_root,
            dry_run=args.dry_run,
        )
    except ReconciliationError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    if args.report is not None:
        _atomic_write_json(args.report.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
