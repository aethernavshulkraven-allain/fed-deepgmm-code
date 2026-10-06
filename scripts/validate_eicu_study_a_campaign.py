#!/usr/bin/env python3
"""Read-only, contract-driven validation for the eICU Study A campaign.

This module deliberately does not import campaign-generation or training code.
It reads a CSV manifest and, when requested, JSON/scenario/result artifacts.
Only the optional report directory is ever written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_JSON = "eicu_study_a_validation.json"
REPORT_MARKDOWN = "eicu_study_a_validation.md"
MISSING = object()


# ---------------------------------------------------------------------------
# Canonical split naming
# ---------------------------------------------------------------------------
# Study A v2's frozen artifacts are the source of truth for the middle split's
# name: cohort_metadata.json keys it "dev" (``split_sizes.dev``,
# ``clients_per_split.dev``), as does the v1 scenario metadata
# (``split_sizes: {dev, test, train}``), ``prepare_eicu_study_a_v2_cohort.py``,
# and the ``missing_train_dev_or_test`` exclusion reason in
# ``client_eligibility_audit.csv``. Some prose write-ups instead call the same
# split "Validation" (protocol_v2.md: "Train+Validation"; RUNBOOK.md: "Select
# from Validation only"). Both spellings name the identical split.
#
# "dev" is canonical because it is what the checksummed/frozen artifacts use;
# renaming those artifacts to "validation" is out of scope (it would break
# ``cohort_sha256`` / ``frozen_client_list_sha256`` provenance checks). This
# map instead lets any caller normalize either spelling to the canonical
# artifact key, additively, without touching frozen metadata.
CANONICAL_SPLIT_NAMES = ("train", "dev", "test")
SPLIT_NAME_ALIASES = {
    "train": "train",
    "dev": "dev",
    "validation": "dev",
    "val": "dev",
    "test": "test",
}


def normalize_split_name(name: str) -> str:
    """Map a split spelling (``dev``, ``Validation``, ``val``, ...) to its
    canonical artifact name (``train``, ``dev``, or ``test``).

    Raises ``ValueError`` on a name that is neither the canonical spelling
    nor a recognized alias, so a typo fails loudly instead of silently
    dropping a split.
    """
    key = str(name).strip().lower()
    try:
        return SPLIT_NAME_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unrecognized split name {name!r}; expected one of "
            f"{sorted(SPLIT_NAME_ALIASES)}"
        ) from exc


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    run_id: str | None = None
    role: str | None = None
    path: str | None = None


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_value(document: Any, paths: Sequence[str]) -> Any:
    for dotted_path in paths:
        current = document
        found = True
        for part in dotted_path.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            return current
    return MISSING


def _flatten(document: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(document, Mapping):
        for key, value in document.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value, child))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            child = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten(value, child))
    else:
        flattened[prefix] = document
    return flattened


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"not a boolean: {value!r}")


def _equivalent(left: Any, right: Any) -> bool:
    if left is MISSING or right is MISSING:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        try:
            return _parse_bool(left) == _parse_bool(right)
        except ValueError:
            return False
    if isinstance(left, int) and not isinstance(left, bool):
        try:
            return left == int(str(right))
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonfinite_paths(document: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_nonfinite_paths(value, child))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            child = f"{prefix}.{index}" if prefix else str(index)
            found.extend(_nonfinite_paths(value, child))
    elif _is_numeric(document) and not math.isfinite(float(document)):
        found.append(prefix)
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _path_from(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


class CampaignValidator:
    def __init__(
        self,
        *,
        manifest_path: Path,
        contract_path: Path,
        config_dir: Path | None = None,
        scenario_root: Path | None = None,
        results_root: Path | None = None,
        phase: str = "prelaunch",
        allow_demo: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.contract_path = contract_path.resolve()
        self.config_dir = config_dir.resolve() if config_dir else None
        self.scenario_root = scenario_root.resolve() if scenario_root else None
        self.results_root = results_root.resolve() if results_root else None
        self.phase = phase
        self.allow_demo = bool(allow_demo)
        self.contract: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.raw_rows: list[dict[str, str]] = []
        self.headers: list[str] = []
        self.errors: list[Issue] = []
        self.warnings: list[Issue] = []
        self.configs: dict[str, dict[str, Any]] = {}
        self.scenarios: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.resolved_result_paths: dict[str, Path] = {}
        self.result_directories_found = 0
        self.audit: dict[str, Any] = {
            "pairing": {"groups_checked": 0, "violations": 0},
            "provenance": {
                "configs_checked": 0,
                "scenarios_checked": 0,
                "artifact_checksums_recomputed": 0,
                "violations": 0,
            },
            "selection_policy": {"rows_checked": 0, "violations": 0},
        }

    def error(
        self,
        code: str,
        message: str,
        row: Mapping[str, Any] | None = None,
        path: Path | None = None,
    ) -> None:
        self.errors.append(
            Issue(
                code,
                message,
                str(row.get("run_id")) if row and row.get("run_id") else None,
                str(row.get("role")) if row and row.get("role") else None,
                str(path) if path else None,
            )
        )

    def warn(
        self,
        code: str,
        message: str,
        row: Mapping[str, Any] | None = None,
        path: Path | None = None,
    ) -> None:
        self.warnings.append(
            Issue(
                code,
                message,
                str(row.get("run_id")) if row and row.get("run_id") else None,
                str(row.get("role")) if row and row.get("role") else None,
                str(path) if path else None,
            )
        )

    def run(self) -> dict[str, Any]:
        self._load_inputs()
        if self.contract and self.headers:
            self._validate_manifest_rows()
            self._validate_coverage()
            self._validate_pairing()
            self._validate_ablation()
            self._validate_claims()
            if self.phase == "postrun":
                self._validate_results()
        return self._report()

    def _load_inputs(self) -> None:
        if not self.contract_path.is_file():
            self.error(
                "contract_missing",
                "Contract JSON does not exist.",
                path=self.contract_path,
            )
            return
        try:
            contract = _json_load(self.contract_path)
        except (OSError, json.JSONDecodeError) as exc:
            self.error(
                "contract_invalid",
                f"Could not read contract JSON: {exc}",
                path=self.contract_path,
            )
            return
        if not isinstance(contract, dict) or not isinstance(contract.get("roles"), dict):
            self.error(
                "contract_invalid",
                "Contract must be an object containing a roles object.",
                path=self.contract_path,
            )
            return
        if self.allow_demo:
            for rules in contract["roles"].values():
                scopes = rules.get("scenario_scope")
                if isinstance(scopes, list) and "demo" not in scopes:
                    scopes.append("demo")
                rules["reject_demo"] = False
        self.contract = contract

        if not self.manifest_path.is_file():
            self.error(
                "manifest_missing",
                "Manifest CSV does not exist.",
                path=self.manifest_path,
            )
            return
        try:
            with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.headers = list(reader.fieldnames or [])
                self.raw_rows = [dict(row) for row in reader]
        except (OSError, csv.Error) as exc:
            self.error(
                "manifest_invalid",
                f"Could not read manifest CSV: {exc}",
                path=self.manifest_path,
            )
            return
        self._canonicalize_rows()

    def _canonicalize_rows(self) -> None:
        manifest = self.contract.get("manifest", {})
        field_map = manifest.get("fields", {})
        required = manifest.get("required_fields", [])
        integer_fields = set(manifest.get("integer_fields", []))
        boolean_fields = set(manifest.get("boolean_fields", []))

        for canonical in required:
            column = field_map.get(canonical)
            if not column or column not in self.headers:
                self.error(
                    "manifest_column_missing",
                    f"Required canonical field {canonical!r} maps to missing column {column!r}.",
                )

        for line_number, raw in enumerate(self.raw_rows, start=2):
            row: dict[str, Any] = {"_line": line_number, "_raw": raw}
            for canonical, column in field_map.items():
                value: Any = raw.get(column, "")
                if isinstance(value, str):
                    value = value.strip()
                if value == "":
                    value = None
                elif canonical in integer_fields:
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        self.error(
                            "manifest_value_invalid",
                            f"Line {line_number}: {canonical} must be an integer.",
                            row,
                        )
                elif canonical in boolean_fields:
                    try:
                        value = _parse_bool(value)
                    except ValueError:
                        self.error(
                            "manifest_value_invalid",
                            f"Line {line_number}: {canonical} must be a boolean.",
                            row,
                        )
                row[canonical] = value
            for canonical in required:
                if row.get(canonical) is None:
                    self.error(
                        "manifest_value_missing",
                        f"Line {line_number}: required field {canonical} is empty.",
                        row,
                    )
            self.rows.append(row)

    def _validate_manifest_rows(self) -> None:
        run_ids: dict[str, dict[str, Any]] = {}
        result_paths: dict[str, dict[str, Any]] = {}
        role_contracts = self.contract["roles"]
        global_rules = self.contract.get("global_rules", {})
        patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in global_rules.get("unresolved_placeholder_patterns", [])
        ]

        for row in self.rows:
            run_id = row.get("run_id")
            if run_id in run_ids:
                self.error(
                    "duplicate_run_id",
                    f"run_id {run_id!r} is duplicated (first seen on line {run_ids[run_id]['_line']}).",
                    row,
                )
            elif run_id is not None:
                run_ids[str(run_id)] = row

            role = row.get("role")
            rules = role_contracts.get(role)
            if rules is None:
                self.error("unknown_role", f"Role {role!r} is not declared by the contract.", row)
                continue

            self._validate_role_values(row, rules)
            self._validate_selection_fields(row)
            self._validate_result_path(row, result_paths)
            config = self._load_config(row)
            scenario = self._load_scenario(row)
            if config is not None:
                self._compare_json_to_manifest(row, config, "config")
                if rules.get("require_frozen_config"):
                    for flattened_path, value in _flatten(config).items():
                        if isinstance(value, str) and any(
                            pattern.search(value) for pattern in patterns
                        ):
                            self.error(
                                "unresolved_tuning_placeholder",
                                f"Frozen config contains unresolved placeholder at {flattened_path}.",
                                row,
                            )
            if scenario is not None:
                self._validate_scenario_metadata(row, scenario, rules, config)

    def _validate_role_values(
        self, row: Mapping[str, Any], rules: Mapping[str, Any]
    ) -> None:
        for field in (
            "g0",
            "seed",
            "method",
            "aggregation_weighting",
            "objective_mode",
            "alignment_label",
            "scenario_scope",
        ):
            allowed = rules.get(field)
            value = row.get(field)
            if allowed is not None and value not in allowed:
                self.error(
                    "role_value_violation",
                    f"{field}={value!r} is not allowed for role {row.get('role')!r}; expected one of {allowed!r}.",
                    row,
                )

        sample_roles = self.contract.get("global_rules", {}).get(
            "sample_size_allowed_roles", []
        )
        if (
            row.get("aggregation_weighting") == "sample_size"
            and row.get("role") not in sample_roles
        ):
            self.error(
                "sample_size_role_violation",
                "sample_size aggregation is allowed only for contract-declared ablation roles.",
                row,
            )

        if row.get("role") == "aggregation_ablation" and not str(
            row.get("alignment_label") or ""
        ).startswith("non_paper_aligned"):
            self.error(
                "ablation_label_missing",
                "Aggregation ablation must carry an explicit non-paper-aligned label.",
                row,
            )

        if rules.get("federated") is False and row.get("aggregation_weighting") != "none":
            self.error(
                "centralized_aggregation_violation",
                "Centralized rows must not use federated aggregation.",
                row,
            )

    def _validate_selection_fields(self, row: Mapping[str, Any]) -> None:
        expected = self.contract.get("global_rules", {})
        before = len(self.errors)
        for field in (
            "primary_selection_metric",
            "test_mse_used_for_selection",
            "selection_source",
        ):
            if field in expected and not _equivalent(row.get(field), expected[field]):
                code = (
                    "test_mse_selection_violation"
                    if field == "test_mse_used_for_selection"
                    else "selection_policy_violation"
                )
                self.error(
                    code,
                    f"{field}={row.get(field)!r}; contract requires {expected[field]!r}.",
                    row,
                )
        self.audit["selection_policy"]["rows_checked"] += 1
        self.audit["selection_policy"]["violations"] += len(self.errors) - before

    def _declared_results_root(self, row: Mapping[str, Any]) -> Path | None:
        if self.results_root is not None:
            return self.results_root
        output_root = row.get("output_root")
        if output_root:
            return _path_from(self.manifest_path.parent, output_root).resolve()
        contract_root = self.contract.get("paths", {}).get("results_root")
        if contract_root:
            return _path_from(self.contract_path.parent, contract_root).resolve()
        return None

    def _validate_result_path(
        self,
        row: Mapping[str, Any],
        seen: dict[str, Mapping[str, Any]],
    ) -> None:
        result_value = row.get("result_path")
        if not result_value:
            return
        root = self._declared_results_root(row)
        if root is None:
            self.error(
                "output_root_missing",
                "No output root is declared by --results-root, the manifest, or the contract.",
                row,
            )
            return
        result_path = _path_from(root, result_value).resolve(strict=False)
        if not _safe_relative(result_path, root):
            self.error(
                "output_path_escape",
                f"Result path escapes declared output root {root}.",
                row,
                result_path,
            )
            return
        key = str(result_path)
        if key in seen:
            self.error(
                "duplicate_result_path",
                f"Result path duplicates run {seen[key].get('run_id')!r}.",
                row,
                result_path,
            )
        else:
            seen[key] = row
            if row.get("run_id") is not None:
                self.resolved_result_paths[str(row["run_id"])] = result_path

    def _resource_path(
        self, row: Mapping[str, Any], field: str, override_root: Path | None
    ) -> Path | None:
        value = row.get(field)
        if not value:
            return None
        base = override_root or self.manifest_path.parent
        return _path_from(base, value).resolve(strict=False)

    def _load_config(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._resource_path(row, "config_path", self.config_dir)
        if path is None:
            return None
        must_exist = self.config_dir is not None or bool(row.get("config_path"))
        if not path.is_file():
            if must_exist:
                self.error("config_missing", "Config JSON does not exist.", row, path)
            return None
        try:
            config = _json_load(path)
        except (OSError, json.JSONDecodeError) as exc:
            self.error("config_invalid", f"Could not read config JSON: {exc}", row, path)
            return None
        if not isinstance(config, dict):
            self.error("config_invalid", "Config JSON must contain an object.", row, path)
            return None
        if row.get("run_id") is not None:
            self.configs[str(row["run_id"])] = config
        self.audit["provenance"]["configs_checked"] += 1
        return config

    def _load_scenario(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._resource_path(
            row, "scenario_metadata_path", self.scenario_root
        )
        if path is None:
            return None
        must_exist = self.scenario_root is not None or bool(
            row.get("scenario_metadata_path")
        )
        if not path.is_file():
            if must_exist:
                self.error(
                    "scenario_metadata_missing",
                    "Scenario metadata JSON does not exist.",
                    row,
                    path,
                )
            return None
        try:
            scenario = _json_load(path)
        except (OSError, json.JSONDecodeError) as exc:
            self.error(
                "scenario_metadata_invalid",
                f"Could not read scenario metadata JSON: {exc}",
                row,
                path,
            )
            return None
        if not isinstance(scenario, dict):
            self.error(
                "scenario_metadata_invalid",
                "Scenario metadata JSON must contain an object.",
                row,
                path,
            )
            return None
        if row.get("run_id") is not None:
            self.scenarios[str(row["run_id"])] = scenario
        self.audit["provenance"]["scenarios_checked"] += 1
        return scenario

    def _compare_json_to_manifest(
        self, row: Mapping[str, Any], document: Mapping[str, Any], source: str
    ) -> None:
        paths = self.contract.get("json_paths", {}).get(source, {})
        before = len(self.errors)
        for field, alternatives in paths.items():
            manifest_value = row.get(field)
            if manifest_value is None:
                continue
            document_value = _json_value(document, alternatives)
            if document_value is MISSING:
                self.error(
                    f"{source}_field_missing",
                    f"{source.title()} does not provide {field} via declared JSON paths.",
                    row,
                )
            elif not _equivalent(manifest_value, document_value):
                self.error(
                    f"{source}_manifest_mismatch",
                    f"{source.title()} {field}={document_value!r} disagrees with manifest value {manifest_value!r}.",
                    row,
                )
        self.audit["provenance"]["violations"] += len(self.errors) - before

    def _validate_scenario_metadata(
        self,
        row: Mapping[str, Any],
        scenario: Mapping[str, Any],
        role_rules: Mapping[str, Any],
        config: Mapping[str, Any] | None,
    ) -> None:
        before = len(self.errors)
        paths = self.contract.get("json_paths", {}).get("scenario", {})
        required = self.contract.get("scenario_rules", {}).get(
            "required_metadata_fields", []
        )
        for field in required:
            value = _json_value(scenario, paths.get(field, [field]))
            empty_container = isinstance(value, (Mapping, list)) and not value
            if value is MISSING or value is None or value == "" or empty_container:
                self.error(
                    "scenario_metadata_field_missing",
                    f"Scenario metadata lacks required field {field}.",
                    row,
                )

        for field in (
            "g0",
            "seed",
            "input_dim",
            "instrument_dim",
            "scenario_checksum",
            "scenario_scope",
        ):
            scenario_value = _json_value(scenario, paths.get(field, [field]))
            if scenario_value is not MISSING and not _equivalent(
                row.get(field), scenario_value
            ):
                self.error(
                    "scenario_manifest_mismatch",
                    f"Scenario {field}={scenario_value!r} disagrees with manifest value {row.get(field)!r}.",
                    row,
                )
        scenario_checksum = _json_value(
            scenario, paths.get("scenario_checksum", ["scenario_checksum"])
        )
        algorithm = self.contract.get("scenario_rules", {}).get(
            "checksum_algorithm", "sha256"
        )
        if (
            algorithm == "sha256"
            and scenario_checksum is not MISSING
            and not re.fullmatch(r"[0-9a-fA-F]{64}", str(scenario_checksum))
        ):
            self.error(
                "scenario_checksum_invalid",
                f"Scenario checksum is not a 64-character SHA-256 digest: {scenario_checksum!r}.",
                row,
            )

        demo_value = _json_value(scenario, paths.get("is_demo", ["is_demo"]))
        if role_rules.get("reject_demo") and demo_value is not MISSING:
            try:
                is_demo = _parse_bool(demo_value)
            except ValueError:
                self.error(
                    "scenario_demo_flag_invalid",
                    f"Scenario is_demo value {demo_value!r} is not boolean.",
                    row,
                )
            else:
                if is_demo:
                    self.error(
                        "demo_confirmatory_violation",
                        "Demo scenario is forbidden for this fixed full-eICU role.",
                        row,
                    )

        if config is not None:
            config_paths = self.contract.get("json_paths", {}).get("config", {})
            for field in ("input_dim", "instrument_dim", "scenario_checksum"):
                scenario_value = _json_value(scenario, paths.get(field, [field]))
                config_value = _json_value(config, config_paths.get(field, [field]))
                if (
                    scenario_value is not MISSING
                    and config_value is not MISSING
                    and not _equivalent(scenario_value, config_value)
                ):
                    self.error(
                        "scenario_config_mismatch",
                        f"Scenario {field}={scenario_value!r} disagrees with config value {config_value!r}.",
                        row,
                    )

        artifact_value = _json_value(
            scenario, paths.get("artifact_path", ["artifact_path"])
        )
        if artifact_value is not MISSING and artifact_value:
            metadata_path = self._resource_path(
                row, "scenario_metadata_path", self.scenario_root
            )
            artifact_path = _path_from(metadata_path.parent, artifact_value).resolve()
            if not artifact_path.is_file():
                self.error(
                    "scenario_artifact_missing",
                    "Scenario artifact named by metadata does not exist.",
                    row,
                    artifact_path,
                )
            else:
                if algorithm != "sha256":
                    self.error(
                        "checksum_algorithm_unsupported",
                        f"Unsupported scenario checksum algorithm {algorithm!r}.",
                        row,
                    )
                else:
                    actual = _sha256(artifact_path)
                    declared = _json_value(
                        scenario,
                        paths.get("scenario_checksum", ["scenario_checksum"]),
                    )
                    if declared is not MISSING and actual != str(declared):
                        self.error(
                            "scenario_artifact_checksum_mismatch",
                            f"Scenario artifact sha256 {actual} disagrees with metadata {declared!r}.",
                            row,
                            artifact_path,
                        )
                    self.audit["provenance"][
                        "artifact_checksums_recomputed"
                    ] += 1
        else:
            self.warn(
                "scenario_artifact_checksum_not_recomputed",
                "Metadata has no scenario artifact path; checksum consistency was checked, but artifact bytes were not re-hashed.",
                row,
            )
        self.audit["provenance"]["violations"] += len(self.errors) - before

    def _validate_coverage(self) -> None:
        coverage: dict[str, Any] = {}
        for role, rules in self.contract.get("roles", {}).items():
            role_rows = [row for row in self.rows if row.get("role") == role]
            observed_tuples = [
                (row.get("g0"), row.get("seed"), row.get("method"))
                for row in role_rows
            ]
            entry: dict[str, Any] = {
                "observed_rows": len(role_rows),
                "expected_rows": rules.get("fixed_row_count"),
                "g0": sorted(
                    {str(row.get("g0")) for row in role_rows if row.get("g0") is not None}
                ),
                "seeds": sorted(
                    {row.get("seed") for row in role_rows if isinstance(row.get("seed"), int)}
                ),
                "methods": sorted(
                    {
                        str(row.get("method"))
                        for row in role_rows
                        if row.get("method") is not None
                    }
                ),
            }
            fixed_count = rules.get("fixed_row_count")
            if fixed_count is not None:
                expected_tuples = set(
                    itertools.product(
                        rules.get("g0", []),
                        rules.get("seeds", []),
                        rules.get("methods", []),
                    )
                )
                observed_set = set(observed_tuples)
                missing = sorted(expected_tuples - observed_set, key=str)
                extra = sorted(observed_set - expected_tuples, key=str)
                duplicates = sorted(
                    [item for item, count in Counter(observed_tuples).items() if count > 1],
                    key=str,
                )
                entry.update(
                    {
                        "complete": not missing
                        and not extra
                        and not duplicates
                        and len(role_rows) == fixed_count,
                        "missing_combinations": [list(item) for item in missing],
                        "extra_combinations": [list(item) for item in extra],
                        "duplicate_combinations": [list(item) for item in duplicates],
                    }
                )
                if len(role_rows) != fixed_count:
                    self.error(
                        "role_row_count_mismatch",
                        f"Role {role!r} has {len(role_rows)} rows; contract requires {fixed_count}.",
                    )
                if missing:
                    self.error(
                        "matrix_incomplete",
                        f"Role {role!r} is missing {len(missing)} g0/seed/method combinations: {missing[:8]!r}.",
                    )
                if extra:
                    self.error(
                        "matrix_extra",
                        f"Role {role!r} has {len(extra)} undeclared combinations: {extra[:8]!r}.",
                    )
                if duplicates:
                    self.error(
                        "matrix_duplicate",
                        f"Role {role!r} duplicates {len(duplicates)} combinations: {duplicates[:8]!r}.",
                    )
            else:
                entry["complete"] = True
            coverage[role] = entry
        self.coverage = coverage

    def _identity(
        self, row: Mapping[str, Any], fields: Iterable[str]
    ) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in fields)

    def _validate_pairing(self) -> None:
        pairing = self.contract.get("pairing", {})
        key_fields = pairing.get("key_fields", ["g0", "seed"])
        identity_fields = pairing.get(
            "scenario_identity_fields", ["scenario", "scenario_checksum"]
        )
        pair = pairing.get("federated_pair", {})
        pair_rows = [
            row for row in self.rows if row.get("role") == pair.get("role")
        ]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in pair_rows:
            groups[self._identity(row, key_fields)].append(row)
        expected_methods = set(pair.get("methods", []))
        for key, rows in groups.items():
            self.audit["pairing"]["groups_checked"] += 1
            before = len(self.errors)
            methods = {row.get("method") for row in rows}
            if methods != expected_methods:
                self.error(
                    "federated_pair_incomplete",
                    f"Pair {key!r} has methods {sorted(methods, key=str)!r}; expected {sorted(expected_methods)!r}.",
                )
            identities = {self._identity(row, identity_fields) for row in rows}
            if len(identities) > 1:
                self.error(
                    "federated_pair_provenance_mismatch",
                    f"Paired federated methods for {key!r} do not share scenario provenance.",
                )
            self.audit["pairing"]["violations"] += len(self.errors) - before

        same_roles = pairing.get("same_scenario_roles", [])
        all_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            if row.get("role") in same_roles:
                all_groups[self._identity(row, key_fields)].append(row)
        for key, rows in all_groups.items():
            self.audit["pairing"]["groups_checked"] += 1
            before = len(self.errors)
            identities = {self._identity(row, identity_fields) for row in rows}
            roles_present = {row.get("role") for row in rows}
            if len(identities) > 1:
                self.error(
                    "cross_role_scenario_mismatch",
                    f"Fixed campaign roles for {key!r} do not share identical scenario provenance.",
                )
            if not set(same_roles).issubset(roles_present):
                self.error(
                    "cross_role_pair_incomplete",
                    f"Scenario key {key!r} lacks one or more required roles {same_roles!r}.",
                )
            self.audit["pairing"]["violations"] += len(self.errors) - before

    def _validate_ablation(self) -> None:
        manifest_rules = self.contract.get("manifest", {})
        allowed_columns = {
            manifest_rules.get("fields", {}).get(field, field)
            for field in manifest_rules.get("ablation_allowed_differences", [])
        }
        allowed_config_paths = set(
            manifest_rules.get("config_ablation_allowed_differences", [])
        )
        roles = self.contract.get("roles", {})
        field_map = manifest_rules.get("fields", {})
        for role, role_rules in roles.items():
            compare_role = role_rules.get("compare_to_role")
            if not compare_role:
                continue
            reference: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
            for row in self.rows:
                if row.get("role") == compare_role:
                    reference[(row.get("g0"), row.get("seed"), row.get("method"))] = row
            for row in self.rows:
                if row.get("role") != role:
                    continue
                key = (row.get("g0"), row.get("seed"), row.get("method"))
                base = reference.get(key)
                if base is None:
                    self.error(
                        "ablation_reference_missing",
                        f"No {compare_role!r} reference row exists for {key!r}.",
                        row,
                    )
                    continue
                raw = row["_raw"]
                base_raw = base["_raw"]
                changed = sorted(
                    column
                    for column in self.headers
                    if column not in allowed_columns
                    and (raw.get(column) or "") != (base_raw.get(column) or "")
                )
                if changed:
                    self.error(
                        "ablation_changed_forbidden_manifest_field",
                        f"Ablation differs from {compare_role} in non-allowed manifest fields: {changed!r}.",
                        row,
                    )
                run_id = str(row.get("run_id"))
                base_id = str(base.get("run_id"))
                if run_id in self.configs and base_id in self.configs:
                    current_flat = _flatten(self.configs[run_id])
                    base_flat = _flatten(self.configs[base_id])
                    changed_paths = sorted(
                        path
                        for path in set(current_flat) | set(base_flat)
                        if path not in allowed_config_paths
                        and current_flat.get(path, MISSING)
                        != base_flat.get(path, MISSING)
                    )
                    if changed_paths:
                        self.error(
                            "ablation_changed_forbidden_config_field",
                            f"Ablation config differs beyond contract allowances: {changed_paths[:12]!r}.",
                            row,
                        )

    def _validate_claims(self) -> None:
        study = self.contract.get("study", {})
        expected_claim = study.get("claim")
        patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in study.get("forbidden_claim_patterns", [])
        ]
        for row in self.rows:
            claim = str(row.get("study_claim") or "")
            if expected_claim is not None and claim != expected_claim:
                self.error(
                    "study_claim_violation",
                    f"Study claim {claim!r} must be {expected_claim!r}; Study A has no published numeric target.",
                    row,
                )
            if any(pattern.search(claim) for pattern in patterns):
                self.error(
                    "paper_numeric_reproduction_claim",
                    "Manifest claims paper numeric reproduction, which is forbidden for Study A.",
                    row,
                )

    def _validate_results(self) -> None:
        postrun = self.contract.get("postrun", {})
        completed_by_role: Counter[str] = Counter()
        for row in self.rows:
            run_id = str(row.get("run_id"))
            result_path = self.resolved_result_paths.get(run_id)
            if result_path is None:
                continue
            if result_path.is_dir():
                self.result_directories_found += 1
            before = len(self.errors)
            required = postrun.get("required_files", [])
            for filename in required:
                artifact = result_path / filename
                if not artifact.is_file():
                    self.error(
                        "result_artifact_missing",
                        f"Required result artifact {filename!r} is missing.",
                        row,
                        artifact,
                    )
            for alternatives in postrun.get("required_file_alternatives", []):
                if not any((result_path / filename).is_file() for filename in alternatives):
                    self.error(
                        "checkpoint_missing",
                        f"Expected one checkpoint from {alternatives!r}.",
                        row,
                        result_path,
                    )
            role_rules = self.contract.get("roles", {}).get(row.get("role"), {})
            if role_rules.get("require_per_client_artifact"):
                alternatives = postrun.get("per_client_artifact_alternatives", [])
                if not any((result_path / filename).is_file() for filename in alternatives):
                    self.error(
                        "per_client_artifact_missing",
                        f"Expected per-client evaluation artifact from {alternatives!r}.",
                        row,
                        result_path,
                    )

            effective = self._read_result_json(
                result_path / "effective_config.json", row, "effective_config"
            )
            metrics = self._read_result_json(
                result_path / "metrics.json", row, "metrics"
            )
            if effective is not None:
                compare_fields = postrun.get(
                    "effective_config_compare_fields", []
                )
                config_paths = self.contract.get("json_paths", {}).get("config", {})
                for field in compare_fields:
                    effective_value = _json_value(
                        effective, config_paths.get(field, [field])
                    )
                    if effective_value is MISSING:
                        self.error(
                            "effective_config_field_missing",
                            f"Effective config lacks {field}.",
                            row,
                        )
                    elif not _equivalent(row.get(field), effective_value):
                        self.error(
                            "effective_config_manifest_mismatch",
                            f"Effective config {field}={effective_value!r} disagrees with manifest value {row.get(field)!r}.",
                            row,
                        )
            if metrics is not None:
                self._validate_metrics(row, metrics, result_path)
                self.results[run_id] = metrics
            if len(self.errors) == before:
                completed_by_role[str(row.get("role"))] += 1

        completion: dict[str, Any] = {}
        for role, rules in self.contract.get("roles", {}).items():
            expected = rules.get("fixed_row_count")
            completed = completed_by_role[role]
            completion[role] = {"completed": completed, "expected": expected}
            if expected is not None and completed != expected:
                self.error(
                    "completion_count_mismatch",
                    f"Role {role!r} has {completed} complete valid results; contract requires {expected}.",
                )
        self.completion = completion

    def _read_result_json(
        self, path: Path, row: Mapping[str, Any], artifact_name: str
    ) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            document = _json_load(path)
        except (OSError, json.JSONDecodeError) as exc:
            self.error(
                "result_json_invalid",
                f"Could not read {artifact_name}: {exc}",
                row,
                path,
            )
            return None
        if not isinstance(document, dict):
            self.error(
                "result_json_invalid",
                f"{artifact_name} must contain a JSON object.",
                row,
                path,
            )
            return None
        return document

    def _metric(
        self, metrics: Mapping[str, Any], field: str, default: Any = MISSING
    ) -> Any:
        paths = self.contract.get("json_paths", {}).get("metrics", {})
        value = _json_value(metrics, paths.get(field, [field]))
        return default if value is MISSING else value

    def _validate_metrics(
        self, row: Mapping[str, Any], metrics: Mapping[str, Any], result_path: Path
    ) -> None:
        for field in ("run_id", "method", "seed", "scenario_checksum"):
            value = self._metric(metrics, field)
            if value is MISSING:
                self.error("metrics_field_missing", f"Metrics lacks {field}.", row)
            elif not _equivalent(row.get(field), value):
                self.error(
                    "metrics_manifest_mismatch",
                    f"Metrics {field}={value!r} disagrees with manifest value {row.get(field)!r}.",
                    row,
                )

        selection_before = len(self.errors)
        expected_selection = {
            "selection_metric": row.get("primary_selection_metric"),
            "selection_source": row.get("selection_source"),
        }
        for field, expected in expected_selection.items():
            value = self._metric(metrics, field)
            if value is MISSING or not _equivalent(expected, value):
                self.error(
                    "metrics_selection_policy_mismatch",
                    f"Metrics {field}={None if value is MISSING else value!r} disagrees with {expected!r}.",
                    row,
                )

        best_round = self._metric(metrics, "best_validation_round")
        selected_round = self._metric(metrics, "selected_round")
        if best_round is MISSING or selected_round is MISSING:
            self.error(
                "metrics_selection_round_missing",
                "Metrics must record best_validation_round and selected_round.",
                row,
            )
        elif not _equivalent(best_round, selected_round):
            self.error(
                "metrics_selection_round_mismatch",
                f"selected_round={selected_round!r} is not best_validation_round={best_round!r}.",
                row,
            )

        reported_after = self._metric(metrics, "test_mse_reported_after_selection")
        try:
            reported_after_bool = _parse_bool(reported_after)
        except ValueError:
            reported_after_bool = False
        if not reported_after_bool:
            self.error(
                "test_mse_reporting_order_violation",
                "Metrics must attest test MSE was reported only after validation selection.",
                row,
            )
        self.audit["selection_policy"]["violations"] += (
            len(self.errors) - selection_before
        )

        diverged_value = self._metric(metrics, "diverged")
        try:
            diverged = _parse_bool(diverged_value)
        except ValueError:
            diverged = False
            self.error(
                "diverged_flag_invalid",
                f"Metrics diverged value {diverged_value!r} is not boolean.",
                row,
            )
        nonfinite_paths = _nonfinite_paths(metrics)
        evidence_parameters = self._metric(metrics, "nonfinite_parameters", False)
        evidence_metrics = self._metric(metrics, "nonfinite_metrics", False)
        try:
            nonfinite_parameters = _parse_bool(evidence_parameters)
            nonfinite_metrics_claimed = _parse_bool(evidence_metrics)
        except ValueError:
            nonfinite_parameters = False
            nonfinite_metrics_claimed = False
            self.error(
                "divergence_evidence_invalid",
                "Divergence evidence flags must be boolean.",
                row,
            )
        if nonfinite_metrics_claimed and not nonfinite_paths:
            self.error(
                "nonfinite_metric_evidence_mismatch",
                "nonfinite_metrics=true but every numeric value recorded in metrics.json is finite.",
                row,
            )
        has_nonfinite_evidence = nonfinite_parameters or bool(nonfinite_paths)
        if diverged and not has_nonfinite_evidence:
            self.error(
                "false_divergence_label",
                "diverged=true has no non-finite parameter or metric evidence; poor finite MSE is not divergence.",
                row,
            )
        if has_nonfinite_evidence and not diverged:
            self.error(
                "missing_divergence_label",
                f"Non-finite state/metrics ({nonfinite_paths[:5]!r}) require diverged=true.",
                row,
            )

        postrun = self.contract.get("postrun", {})
        if not diverged:
            for field in postrun.get("required_primary_metric_fields", []):
                value = self._metric(metrics, field)
                if not _is_numeric(value) or not math.isfinite(float(value)):
                    self.error(
                        "primary_metric_invalid",
                        f"Finite non-diverged run requires numeric finite primary metric {field}.",
                        row,
                    )
            for field in postrun.get("required_secondary_metric_fields", []):
                value = self._metric(metrics, field)
                if not _is_numeric(value) or not math.isfinite(float(value)):
                    self.error(
                        "secondary_metric_invalid",
                        f"Finite non-diverged run requires numeric finite secondary metric {field}.",
                        row,
                    )
            curve_before = len(self.errors)
            self._validate_round_curve(row, metrics, result_path)
            self.audit["selection_policy"]["violations"] += (
                len(self.errors) - curve_before
            )

        role_rules = self.contract.get("roles", {}).get(row.get("role"), {})
        if row.get("role") == "aggregation_ablation":
            is_primary = self._metric(metrics, "is_primary")
            label = self._metric(metrics, "alignment_label")
            try:
                primary_bool = _parse_bool(is_primary)
            except ValueError:
                primary_bool = True
            if primary_bool or not str(label or "").startswith("non_paper_aligned"):
                self.error(
                    "ablation_not_visibly_nonprimary",
                    "Aggregation ablation metrics must record is_primary=false and a non-paper-aligned label.",
                    row,
                )
        elif role_rules.get("primary"):
            is_primary = self._metric(metrics, "is_primary")
            try:
                if not _parse_bool(is_primary):
                    self.error(
                        "primary_result_not_marked",
                        "Primary confirmatory metrics must record is_primary=true.",
                        row,
                    )
            except ValueError:
                self.error(
                    "primary_result_not_marked",
                    "Primary confirmatory metrics must record is_primary=true.",
                    row,
                )

    def _validate_round_curve(
        self, row: Mapping[str, Any], metrics: Mapping[str, Any], result_path: Path
    ) -> None:
        curve_path = result_path / "mse_by_round.csv"
        if not curve_path.is_file():
            return
        columns = self.contract.get("postrun", {}).get("round_curve_columns", {})
        round_column = columns.get("round", "round")
        metric_column = columns.get(
            "primary_validation_metric", "equal_client_validation_mse"
        )
        try:
            with curve_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            finite_records = [
                (int(record[round_column]), float(record[metric_column]))
                for record in records
                if math.isfinite(float(record[metric_column]))
            ]
        except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
            self.error(
                "round_curve_invalid",
                f"Could not audit validation-selection curve: {exc}",
                row,
                curve_path,
            )
            return
        if not finite_records:
            self.error(
                "round_curve_invalid",
                "Round curve has no finite primary validation values.",
                row,
                curve_path,
            )
            return
        curve_best_round, curve_best_value = min(finite_records, key=lambda item: item[1])
        selected_round = self._metric(metrics, "selected_round")
        reported_best = self._metric(metrics, "primary_equal_client_validation_mse")
        if not _equivalent(curve_best_round, selected_round):
            self.error(
                "round_curve_selection_mismatch",
                f"Curve selects round {curve_best_round}, but metrics selected {selected_round!r}.",
                row,
                curve_path,
            )
        if not _is_numeric(reported_best) or not math.isclose(
            curve_best_value,
            float(reported_best),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            self.error(
                "round_curve_metric_mismatch",
                f"Curve best validation MSE {curve_best_value} disagrees with metrics {reported_best!r}.",
                row,
                curve_path,
            )

    def _report(self) -> dict[str, Any]:
        roles = self.contract.get("roles", {}) if self.contract else {}
        fixed_expected = sum(
            int(rules["fixed_row_count"])
            for rules in roles.values()
            if rules.get("fixed_row_count") is not None
        )
        fixed_observed = sum(
            1
            for row in self.rows
            if roles.get(row.get("role"), {}).get("fixed_row_count") is not None
        )
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "contract_version": self.contract.get("contract_version")
            if self.contract
            else None,
            "manifest": str(self.manifest_path),
            "contract": str(self.contract_path),
            "allow_demo": self.allow_demo,
            "counts": {
                "manifest_rows": len(self.rows),
                "fixed_rows_expected": fixed_expected,
                "fixed_rows_observed": fixed_observed,
                "result_directories_found": self.result_directories_found,
                "results_validated": len(self.results),
                "valid_completed_results": sum(
                    item["completed"]
                    for item in getattr(self, "completion", {}).values()
                ),
                "blocking_errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "blocking_errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
            "coverage": getattr(self, "coverage", {}),
            "pairing_audit": self.audit["pairing"],
            "provenance_audit": self.audit["provenance"],
            "selection_policy_audit": self.audit["selection_policy"],
        }
        if self.phase == "postrun":
            report["completion"] = getattr(self, "completion", {})
            report["reportable"] = not self.errors
        else:
            report["launchable"] = not self.errors
        return report


def validate_campaign(
    *,
    manifest: Path,
    contract: Path,
    config_dir: Path | None = None,
    scenario_root: Path | None = None,
    results_root: Path | None = None,
    phase: str = "prelaunch",
    allow_demo: bool = False,
) -> dict[str, Any]:
    """Validate without writing or mutating any campaign artifact."""
    return CampaignValidator(
        manifest_path=manifest,
        contract_path=contract,
        config_dir=config_dir,
        scenario_root=scenario_root,
        results_root=results_root,
        phase=phase,
        allow_demo=allow_demo,
    ).run()


def _markdown(report: Mapping[str, Any]) -> str:
    outcome_name = "reportable" if report["phase"] == "postrun" else "launchable"
    lines = [
        "# eICU Study A campaign validation",
        "",
        f"- Phase: `{report['phase']}`",
        f"- Contract version: `{report.get('contract_version')}`",
        f"- Manifest rows: {report['counts']['manifest_rows']}",
        f"- Fixed rows: {report['counts']['fixed_rows_observed']} / {report['counts']['fixed_rows_expected']}",
        f"- Result directories found: {report['counts']['result_directories_found']}",
        f"- Results validated: {report['counts']['results_validated']}",
        f"- Valid completed results: {report['counts']['valid_completed_results']}",
        f"- Blocking errors: {report['counts']['blocking_errors']}",
        f"- Warnings: {report['counts']['warnings']}",
        f"- **{outcome_name}: `{str(report[outcome_name]).lower()}`**",
        "",
        "## Role coverage",
        "",
        "| Role | Observed | Expected | Complete | g0 | Seeds | Methods |",
        "|---|---:|---:|:---:|---|---|---|",
    ]
    for role, coverage in report.get("coverage", {}).items():
        lines.append(
            "| {role} | {observed} | {expected} | {complete} | {g0} | {seeds} | {methods} |".format(
                role=role,
                observed=coverage["observed_rows"],
                expected=coverage["expected_rows"]
                if coverage["expected_rows"] is not None
                else "variable",
                complete="yes" if coverage.get("complete") else "no",
                g0=", ".join(map(str, coverage["g0"])) or "—",
                seeds=", ".join(map(str, coverage["seeds"])) or "—",
                methods=", ".join(map(str, coverage["methods"])) or "—",
            )
        )
    for title, key in (
        ("Pairing audit", "pairing_audit"),
        ("Provenance audit", "provenance_audit"),
        ("Selection-policy audit", "selection_policy_audit"),
    ):
        lines.extend(["", f"## {title}", ""])
        for name, value in report.get(key, {}).items():
            lines.append(f"- {name.replace('_', ' ')}: {value}")
    for title, key in (
        ("Blocking errors", "blocking_errors"),
        ("Warnings", "warnings"),
    ):
        lines.extend(["", f"## {title}", ""])
        issues = report.get(key, [])
        if not issues:
            lines.append("None.")
        else:
            for issue in issues:
                context = " / ".join(
                    str(issue[name])
                    for name in ("role", "run_id", "path")
                    if issue.get(name)
                )
                suffix = f" ({context})" if context else ""
                lines.append(f"- `{issue['code']}`: {issue['message']}{suffix}")
    return "\n".join(lines) + "\n"


def _write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON
    markdown_path = output_dir / REPORT_MARKDOWN
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(_markdown(report))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an eICU Study A manifest and completed artifacts."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--phase", choices=("prelaunch", "postrun"), default="prelaunch"
    )
    parser.add_argument(
        "--allow-demo",
        action="store_true",
        help=(
            "Allow demo-scoped artifacts for an explicitly non-reportable pipeline "
            "campaign; the default full-eICU rejection remains unchanged."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional report directory. If omitted, no files are written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_campaign(
        manifest=args.manifest,
        contract=args.contract,
        config_dir=args.config_dir,
        scenario_root=args.scenario_root,
        results_root=args.results_root,
        phase=args.phase,
        allow_demo=args.allow_demo,
    )
    if args.out is not None:
        _write_report(report, args.out)
    outcome = report["reportable"] if args.phase == "postrun" else report["launchable"]
    print(
        json.dumps(
            {
                "phase": args.phase,
                "blocking_errors": report["counts"]["blocking_errors"],
                "warnings": report["counts"]["warnings"],
                "reportable" if args.phase == "postrun" else "launchable": outcome,
            },
            sort_keys=True,
        )
    )
    return 0 if outcome else 1


if __name__ == "__main__":
    sys.exit(main())
