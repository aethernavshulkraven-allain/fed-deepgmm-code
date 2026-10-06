#!/usr/bin/env python3
"""Read-only release preflight for the eICU cohort builder.

This tool intentionally does not import the evolving eICU cohort implementation.
Its table contract is frozen in experiments/eicu_full_data_preflight/
required_tables.json and was derived from scripts/prepare_eicu_cohort.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = (
    REPO_ROOT / "experiments" / "eicu_full_data_preflight" / "required_tables.json"
)
OUTPUT_JSON = "eicu_release_preflight.json"
OUTPUT_MARKDOWN = "eicu_release_preflight.md"
TABLE_SUFFIXES = (".csv.gz", ".csv")
LIKELY_FULL_PATIENT_ROWS = 100_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check an eICU table root without loading patient data or importing "
            "the cohort implementation."
        )
    )
    parser.add_argument("--eicu-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="fail unless the release can be classified as likely full",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="compute SHA-256 over each resolved table file (potentially expensive)",
    )
    parser.add_argument(
        "--count-patient-rows",
        action="store_true",
        help="stream the patient table to count CSV records without loading it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform all checks but do not create the output directory or files",
    )
    return parser.parse_args(argv)


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8-sig", errors="replace", newline="")
    return path.open(mode="rt", encoding="utf-8-sig", errors="replace", newline="")


def _read_header(path: Path) -> list[str]:
    with _open_text(path) as handle:
        return [column.strip() for column in next(csv.reader(handle))]


def _stream_csv_row_count(path: Path) -> int:
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _load_requirements(path: Path = DEFAULT_REQUIREMENTS) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        requirements = json.load(handle)
    for category in ("required_tables", "optional_tables", "sensitivity_tables"):
        requirements[category] = sorted(
            requirements.get(category, []), key=lambda item: item["name"].lower()
        )
    return requirements


def _safe_entries(root: Path) -> tuple[list[Path], str | None]:
    try:
        return sorted(root.iterdir(), key=lambda path: path.name.lower()), None
    except OSError as exc:
        return [], f"eICU root is not readable: {exc}"


def _table_index(entries: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for entry in entries:
        if not entry.is_file():
            continue
        lower = entry.name.lower()
        for suffix in TABLE_SUFFIXES:
            if lower.endswith(suffix):
                table_name = lower[: -len(suffix)]
                index.setdefault(table_name, []).append(entry)
                break
    return index


def _resolved_table(
    specification: dict[str, Any],
    category: str,
    index: dict[str, list[Path]],
    checksum: bool,
) -> tuple[dict[str, Any], list[str]]:
    name = specification["name"]
    matches = sorted(index.get(name.lower(), []), key=lambda path: path.name.lower())
    warnings: list[str] = []
    status: dict[str, Any] = {
        "category": category,
        "exists": bool(matches),
        "header_status": "not_present",
        "missing_required_columns": list(specification["columns"]),
        "name": name,
        "optional_columns": list(specification.get("optional_columns", [])),
        "path": None,
        "purpose": specification["purpose"],
        "required_columns": list(specification["columns"]),
        "resolved_filename": None,
        "size": {
            "compressed_bytes": None,
            "on_disk_bytes": None,
            "uncompressed_bytes": None,
        },
    }
    if not matches:
        return status, warnings

    path = matches[0]
    if len(matches) > 1:
        warnings.append(
            f"Multiple case-insensitive matches for {name}; using {path.name}."
        )
    status["resolved_filename"] = path.name
    status["path"] = str(path.resolve())
    try:
        stat = path.stat()
        is_gzip = path.name.lower().endswith(".gz")
        status["size"] = {
            "compressed_bytes": stat.st_size if is_gzip else None,
            "on_disk_bytes": stat.st_size,
            "uncompressed_bytes": None if is_gzip else stat.st_size,
        }
        status["mtime_utc"] = _mtime_utc(stat.st_mtime)
    except OSError as exc:
        status["header_status"] = "unreadable"
        status["read_error"] = str(exc)
        return status, warnings

    try:
        header = _read_header(path)
        header_set = set(header)
        missing = sorted(set(specification["columns"]) - header_set)
        status["column_count"] = len(header)
        status["missing_required_columns"] = missing
        status["present_optional_columns"] = sorted(
            set(specification.get("optional_columns", [])) & header_set
        )
        status["header_status"] = "missing_columns" if missing else "ok"
    except (OSError, EOFError, csv.Error, StopIteration) as exc:
        status["header_status"] = "unreadable"
        status["read_error"] = f"{type(exc).__name__}: {exc}"

    if checksum:
        try:
            status["sha256"] = _sha256(path)
        except OSError as exc:
            status["checksum_error"] = str(exc)
    return status, warnings


def _classify_release(root: Path, patient_rows: int | None) -> tuple[str, str]:
    components = [component.lower() for component in root.expanduser().parts]
    joined = "/".join(components)
    if "eicu-crd-demo" in joined or any(
        component == "demo" or component.startswith("demo-") for component in components
    ):
        return "demo", "path contains an explicit eICU demo marker"
    if patient_rows is not None and patient_rows >= LIKELY_FULL_PATIENT_ROWS:
        return "likely_full", f"patient table has at least {LIKELY_FULL_PATIENT_ROWS:,} rows"
    if "eicu-crd" in components:
        return "likely_full", "path contains the standard non-demo eicu-crd release marker"
    return "unknown", "no reliable demo or likely-full marker was detected"


def _detected_version(root: Path) -> str | None:
    version_pattern = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?)$", re.IGNORECASE)
    for component in reversed(root.expanduser().parts):
        match = version_pattern.match(component)
        if match:
            return match.group(1)
    return None


def run_preflight(
    eicu_root: Path,
    out: Path,
    *,
    require_full: bool = False,
    checksum: bool = False,
    count_patient_rows: bool = False,
) -> dict[str, Any]:
    root = eicu_root.expanduser().absolute()
    requirements = _load_requirements()
    root_exists = root.exists()
    root_is_directory = root.is_dir()
    root_readable = root_exists and root_is_directory and os.access(root, os.R_OK | os.X_OK)
    entries: list[Path] = []
    entry_error: str | None = None
    if root_readable:
        entries, entry_error = _safe_entries(root)
        root_readable = entry_error is None

    table_index = _table_index(entries)
    tables: list[dict[str, Any]] = []
    warnings: list[str] = []
    category_map = (
        ("required", requirements["required_tables"]),
        ("optional", requirements["optional_tables"]),
        ("sensitivity", requirements["sensitivity_tables"]),
    )
    for category, specifications in category_map:
        for specification in specifications:
            table_status, table_warnings = _resolved_table(
                specification, category, table_index, checksum
            )
            tables.append(table_status)
            warnings.extend(table_warnings)

    tables.sort(key=lambda item: (item["category"], item["name"].lower()))
    patient_status = next(table for table in tables if table["name"] == "patient")
    patient_rows: int | None = None
    if count_patient_rows and patient_status["exists"]:
        try:
            patient_rows = _stream_csv_row_count(Path(patient_status["path"]))
        except (OSError, EOFError, csv.Error) as exc:
            warnings.append(f"Could not stream patient row count: {type(exc).__name__}: {exc}")

    classification, classification_basis = _classify_release(root, patient_rows)
    required_status = [table for table in tables if table["category"] == "required"]
    required_ok = root_readable and all(
        table["exists"] and table["header_status"] == "ok" for table in required_status
    )
    launchable_demo = bool(required_ok)
    launchable_full = bool(required_ok and classification == "likely_full")

    blocking_reasons: list[str] = []
    if not root_exists:
        blocking_reasons.append("eICU root does not exist.")
    elif not root_is_directory:
        blocking_reasons.append("eICU root is not a directory.")
    elif not root_readable:
        blocking_reasons.append(entry_error or "eICU root is not readable.")
    for table in required_status:
        if not table["exists"]:
            blocking_reasons.append(f"Missing required table: {table['name']}.")
        elif table["header_status"] == "missing_columns":
            columns = ", ".join(table["missing_required_columns"])
            blocking_reasons.append(
                f"Required table {table['name']} is missing columns: {columns}."
            )
        elif table["header_status"] != "ok":
            blocking_reasons.append(
                f"Could not validate header for required table {table['name']}."
            )
        if checksum and table.get("checksum_error"):
            blocking_reasons.append(
                f"Could not checksum required table {table['name']}: "
                f"{table['checksum_error']}"
            )

    for table in tables:
        if table["category"] in {"optional", "sensitivity"}:
            if not table["exists"]:
                warnings.append(
                    f"{table['category'].capitalize()} table {table['name']} is absent."
                )
            elif table["header_status"] != "ok":
                warnings.append(
                    f"{table['category'].capitalize()} table {table['name']} "
                    "is present but its expected columns were not fully validated."
                )

    if classification == "demo":
        warnings.append(
            "This is a demo release: it is suitable only for pipeline smoke testing."
        )
    elif classification == "unknown":
        warnings.append(
            "Release size/class could not be verified; full-cohort launch remains disabled."
        )
    warnings.append(
        "This preflight checks data readiness for cohort construction only; it does "
        "not establish scientific validity or paper alignment."
    )
    if any(
        table["exists"] and table["size"]["compressed_bytes"] is not None
        for table in tables
    ):
        warnings.append(
            "Exact uncompressed byte sizes are not reported for gzip files because "
            "obtaining them safely can require a full decompression pass."
        )

    if require_full and not launchable_full:
        blocking_reasons.append(
            "Full eICU was required, but this root is not a validated likely-full "
            "release ready for cohort construction."
        )

    output_filesystem = _existing_ancestor(out)
    disk = shutil.disk_usage(output_filesystem)
    known_table_bytes = sum(
        table["size"]["on_disk_bytes"] or 0 for table in tables if table["exists"]
    )
    require_full_satisfied = (not require_full) or launchable_full
    report: dict[str, Any] = {
        "blocking_reasons": sorted(set(blocking_reasons)),
        "detected": {
            "classification_basis": classification_basis,
            "eicu_root": str(root),
            "version": _detected_version(root),
        },
        "launchable_for_demo_smoke": launchable_demo,
        "launchable_for_full_cohort_build": launchable_full,
        "patient_table_rows": patient_rows,
        "release_classification": classification,
        "require_full_requested": require_full,
        "require_full_satisfied": bool(require_full_satisfied),
        "root_status": {
            "exists": root_exists,
            "is_directory": root_is_directory,
            "readable": root_readable,
        },
        "schema": {
            "cohort_builder_reference": requirements["cohort_builder_reference"],
            "required_tables_spec": str(DEFAULT_REQUIREMENTS.relative_to(REPO_ROOT)),
            "version": requirements["schema_version"],
        },
        "storage": {
            "available_bytes_on_output_filesystem": disk.free,
            "known_table_on_disk_bytes": known_table_bytes,
            "output_filesystem_probe_path": str(output_filesystem.resolve()),
            "output_target": str(out.expanduser().absolute()),
            "total_bytes_on_output_filesystem": disk.total,
        },
        "tables": tables,
        "warnings": sorted(set(warnings)),
    }
    return report


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "not safely available"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def render_markdown(report: dict[str, Any]) -> str:
    detected = report["detected"]
    lines = [
        "# eICU release preflight",
        "",
        "This is a data-readiness check for cohort construction. It does **not** "
        "certify scientific validity, causal identification, or paper alignment.",
        "",
        "## Release",
        "",
        f"- Classification: `{report['release_classification']}`",
        f"- Classification basis: {detected['classification_basis']}",
        f"- Resolved root: `{detected['eicu_root']}`",
        f"- Detected version: `{detected['version'] or 'unknown'}`",
        f"- Launchable for demo smoke: `{str(report['launchable_for_demo_smoke']).lower()}`",
        (
            "- Launchable for full cohort build: "
            f"`{str(report['launchable_for_full_cohort_build']).lower()}`"
        ),
        f"- Require-full requested: `{str(report['require_full_requested']).lower()}`",
        f"- Require-full satisfied: `{str(report['require_full_satisfied']).lower()}`",
    ]
    if report["patient_table_rows"] is not None:
        lines.append(f"- Streamed patient-table rows: `{report['patient_table_rows']}`")

    lines.extend(["", "## Table status", ""])
    for category in ("required", "optional", "sensitivity"):
        lines.extend(
            [
                f"### {category.capitalize()} tables",
                "",
                "| Table | File | Header | On-disk size | Uncompressed size |",
                "|---|---|---|---:|---:|",
            ]
        )
        for table in (item for item in report["tables"] if item["category"] == category):
            lines.append(
                "| {name} | {filename} | {header} | {disk} | {uncompressed} |".format(
                    name=table["name"],
                    filename=table["resolved_filename"] or "missing",
                    header=table["header_status"],
                    disk=_human_bytes(table["size"]["on_disk_bytes"]),
                    uncompressed=_human_bytes(table["size"]["uncompressed_bytes"]),
                )
            )
            if table["missing_required_columns"] and table["exists"]:
                lines.append(
                    f"\nMissing expected columns in `{table['name']}`: "
                    + ", ".join(f"`{column}`" for column in table["missing_required_columns"])
                    + "."
                )
        lines.append("")

    storage = report["storage"]
    lines.extend(
        [
            "## Storage",
            "",
            f"- Known required/optional table bytes on disk: "
            f"{_human_bytes(storage['known_table_on_disk_bytes'])}",
            f"- Available on output filesystem: "
            f"{_human_bytes(storage['available_bytes_on_output_filesystem'])}",
            f"- Output filesystem probe: `{storage['output_filesystem_probe_path']}`",
            "",
            "## Blocking reasons",
            "",
        ]
    )
    if report["blocking_reasons"]:
        lines.extend(f"- {reason}" for reason in report["blocking_reasons"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _write_outputs(out: Path, report: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / OUTPUT_JSON
    markdown_path = out / OUTPUT_MARKDOWN
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_preflight(
        args.eicu_root,
        args.out,
        require_full=args.require_full,
        checksum=args.checksum,
        count_patient_rows=args.count_patient_rows,
    )
    if args.dry_run:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        try:
            _write_outputs(args.out, report)
        except OSError as exc:
            print(f"error: could not write preflight outputs: {exc}", file=sys.stderr)
            return 2

    if report["blocking_reasons"] or not report["require_full_satisfied"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
