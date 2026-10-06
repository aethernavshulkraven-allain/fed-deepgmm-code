#!/usr/bin/env python3
"""Validate the definitive paper-reproduction and extension target registry."""

import argparse
import csv
import math
import sys


COLUMNS = (
    "target_id",
    "track",
    "source_version",
    "source_location",
    "dataset_family",
    "dataset",
    "input_mode",
    "scenario",
    "true_function",
    "algorithm_id",
    "algorithm_family",
    "training_scope",
    "stochasticity",
    "paper_method_label",
    "paper_mapping_status",
    "paper_target_available",
    "target_status",
    "target_mean_test_mse",
    "target_std_test_mse",
    "paper_reported_value",
    "target_metric",
    "target_num_runs",
    "seed_values",
    "curve_required",
    "aggregate_test_mse_required",
    "train_size",
    "validation_size",
    "test_size",
    "model_architecture",
    "partition_method",
    "partition_alpha",
    "paper_batch_size",
    "paper_batch_size_status",
    "y_standardization",
    "learning_rate_tuning",
    "checkpoint_selection_method",
    "communication_rounds",
    "total_clients",
    "clients_per_round",
    "local_epochs",
    "server_learning_rate",
    "weight_decay",
    "gradient_clipping",
    "paper_protocol_match_required",
    "comparison_reference",
    "tuning_rule",
    "fairness_rule",
    "test_tuning_allowed",
    "our_result_status",
    "our_completed_seeds",
    "our_result_dir_pattern",
    "our_mean_test_mse",
    "our_std_test_mse",
    "absolute_mean_gap",
    "relative_mean_gap_pct",
    "protocol_match_status",
    "notes",
)

SCENARIOS = (
    "absolute",
    "step",
    "linear",
    "sine",
    "femnist_x",
    "femnist_xz",
    "femnist_z",
    "cifar10_x",
    "cifar10_xz",
    "cifar10_z",
)
ALGORITHMS = (
    "gda_d",
    "sgda_s",
    "oadam_d",
    "oadam_s",
    "fedgda_d",
    "fedgda_s",
    "fedogda_d",
    "fedogda_s",
)
BOOLEAN_COLUMNS = (
    "paper_target_available",
    "curve_required",
    "aggregate_test_mse_required",
    "paper_protocol_match_required",
    "test_tuning_allowed",
)
NUMERIC_COLUMNS = (
    "target_mean_test_mse",
    "target_std_test_mse",
    "target_num_runs",
    "train_size",
    "validation_size",
    "test_size",
    "partition_alpha",
    "paper_batch_size",
    "communication_rounds",
    "total_clients",
    "clients_per_round",
    "local_epochs",
    "server_learning_rate",
    "weight_decay",
    "gradient_clipping",
    "our_mean_test_mse",
    "our_std_test_mse",
    "absolute_mean_gap",
    "relative_mean_gap_pct",
)


def _targets():
    values = {
        "absolute": {
            "oadam_s": (0.03, 0.01, "0.03 ± 0.01"),
            "gda_d": (0.013, 0.01, "0.013 ± 0.01"),
            "fedgda_d": (0.40, 0.01, "0.40 ± 0.01"),
            "sgda_s": (0.009, 0.01, "0.009 ± 0.01"),
            "fedgda_s": (0.20, 0.00, "0.20 ± 0.00"),
        },
        "step": {
            "oadam_s": (0.30, 0.00, "0.30 ± 0.00"),
            "gda_d": (0.03, 0.00, "0.03 ± 0.00"),
            "fedgda_d": (0.04, 0.01, "0.04 ± 0.01"),
            "sgda_s": (0.112, 0.00, "0.112 ± 0.00"),
            "fedgda_s": (0.23, 0.01, "0.23 ± 0.01"),
        },
        "linear": {
            "oadam_s": (0.01, 0.00, "0.01 ± 0.00"),
            "gda_d": (0.02, 0.00, "0.02 ± 0.00"),
            "fedgda_d": (0.01, 0.00, "0.01 ± 0.00"),
            "sgda_s": (0.03, 0.00, "0.03 ± 0.00"),
            "fedgda_s": (0.04, 0.00, "0.04 ± 0.00"),
        },
        "femnist_x": {
            "oadam_s": (0.50, 0.00, "0.50 ± 0.00"),
            "gda_d": (1.11, 0.01, "1.11 ± 0.01"),
            "fedgda_d": (0.21, 0.02, "0.21 ± 0.02"),
            "sgda_s": (0.40, 0.01, "0.40 ± 0.01"),
            "fedgda_s": (0.19, 0.01, "0.19 ± 0.01"),
        },
        "femnist_xz": {
            "oadam_s": (0.24, 0.00, "0.24 ± 0.00"),
            "gda_d": (0.46, 0.09, "0.46 ± 0.09"),
            "fedgda_d": (0.19, 0.03, "0.19 ± 0.03"),
            "sgda_s": (0.14, 0.02, "0.14 ± 0.02"),
            "fedgda_s": (0.20, 0.00, "0.20 ± 0.00"),
        },
        "femnist_z": {
            "oadam_s": (0.10, 0.00, "0.10 ± 0.00"),
            "gda_d": (0.42, 0.01, "0.42 ± 0.01"),
            "fedgda_d": (0.24, 0.01, "0.24 ± 0.01"),
            "sgda_s": (0.11, 0.02, "0.11 ± 0.02"),
            "fedgda_s": (0.23, 0.01, "0.23 ± 0.01"),
        },
        "cifar10_x": {
            "oadam_s": (0.55, 0.30, "0.55 ± 0.30"),
            "gda_d": (0.19, 0.01, "0.19 ± 0.01"),
            "fedgda_d": (0.25, 0.03, "0.25 ± 0.03"),
            "sgda_s": (0.20, 0.08, "0.20 ± 0.08"),
            "fedgda_s": (0.22, 0.08, "0.22 ± 0.08"),
        },
        "cifar10_xz": {
            "oadam_s": (0.40, 0.11, "0.40 ± 0.11"),
            "gda_d": (0.24, 0.00, "0.24 ± 0.00"),
            "fedgda_d": (0.24, 0.03, "0.24 ± 0.03"),
            "sgda_s": (0.19, 0.03, "0.19 ± 0.03"),
            "fedgda_s": (0.22, 0.02, "0.22 ± 0.02"),
        },
        "cifar10_z": {
            "oadam_s": (0.13, 0.03, "0.13 ± 0.03"),
            "gda_d": (0.13, 0.01, "0.13 ± 0.01"),
            "fedgda_d": (1.70, 2.60, "1.70 ± 2.60"),
            "sgda_s": (0.24, 0.01, "0.24 ± 0.01"),
            "fedgda_s": (0.52, 0.60, "0.52 ± 0.60"),
        },
    }
    return {(scenario, algorithm): target for scenario, methods in values.items() for algorithm, target in methods.items()}


PAPER_TARGETS = _targets()
FEDOGDA_FAIRNESS = (
    "same data splits, seeds, model architecture, client partition, training budget, "
    "and comparable tuning budget as paired FedGDA"
)


def _number(value, field, row_number, errors):
    try:
        number = float(value)
    except ValueError:
        errors.append(f"row {row_number}: {field} must be numeric or blank, got {value!r}")
        return None
    if not math.isfinite(number):
        errors.append(f"row {row_number}: {field} must be finite, got {value!r}")
        return None
    return number


def _is_true(row, field):
    return row[field] == "true"


def validate(path):
    errors = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            errors.append("CSV header does not match the required column order")
        rows = []
        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                errors.append(f"row {row_number}: contains more cells than the header")
            missing = [field for field in COLUMNS if raw_row.get(field) is None]
            if missing:
                errors.append(f"row {row_number}: missing explicit cells for {', '.join(missing)}")
            rows.append({field: raw_row.get(field) or "" for field in COLUMNS})

    if len(rows) != 80:
        errors.append(f"expected 80 rows, found {len(rows)}")

    ids = [row["target_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("target_id values must be unique")

    scenario_set = {row["scenario"] for row in rows}
    if scenario_set != set(SCENARIOS):
        errors.append(f"expected scenarios {sorted(SCENARIOS)}, found {sorted(scenario_set)}")
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        algorithms = [row["algorithm_id"] for row in scenario_rows]
        if len(scenario_rows) != 8 or set(algorithms) != set(ALGORITHMS):
            errors.append(f"scenario {scenario}: expected exactly one row for each of 8 algorithms")

    pairs = [(row["scenario"], row["algorithm_id"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate scenario/algorithm combination found")

    paper_rows = [row for row in rows if _is_true(row, "paper_target_available")]
    extension_rows = [row for row in rows if not _is_true(row, "paper_target_available")]
    if len(paper_rows) != 45:
        errors.append(f"expected 45 paper target rows, found {len(paper_rows)}")
    if len(extension_rows) != 35:
        errors.append(f"expected 35 extension rows, found {len(extension_rows)}")

    for row_number, row in enumerate(rows, start=2):
        for field in BOOLEAN_COLUMNS:
            if row[field] not in {"true", "false"}:
                errors.append(f"row {row_number}: {field} must be lowercase true or false")
        for field in NUMERIC_COLUMNS:
            value = row[field]
            if value:
                _number(value, field, row_number, errors)
        if row["test_tuning_allowed"] != "false":
            errors.append(f"row {row_number}: test_tuning_allowed must be false")

        key = (row["scenario"], row["algorithm_id"])
        has_paper_target = _is_true(row, "paper_target_available")
        if has_paper_target:
            if row["track"] != "paper_reproduction":
                errors.append(f"row {row_number}: paper target must use paper_reproduction track")
            if row["paper_protocol_match_required"] != "true":
                errors.append(f"row {row_number}: paper target must require paper protocol matching")
            if key not in PAPER_TARGETS:
                errors.append(f"row {row_number}: unexpected paper target for {key}")
            else:
                expected_mean, expected_std, expected_display = PAPER_TARGETS[key]
                actual_mean = _number(row["target_mean_test_mse"], "target_mean_test_mse", row_number, errors)
                actual_std = _number(row["target_std_test_mse"], "target_std_test_mse", row_number, errors)
                if actual_mean is not None and actual_mean != expected_mean:
                    errors.append(f"row {row_number}: incorrect paper mean for {key}")
                if actual_std is not None and actual_std != expected_std:
                    errors.append(f"row {row_number}: incorrect paper std for {key}")
                if row["paper_reported_value"] != expected_display:
                    errors.append(f"row {row_number}: incorrect paper display value for {key}")
            if row["partition_alpha"] != "0.3":
                errors.append(f"row {row_number}: paper target must use partition_alpha 0.3")
            if row["algorithm_id"] == "fedgda_s" and row["paper_batch_size"] != "256":
                errors.append(f"row {row_number}: paper fedgda_s target must use batch size 256")
        else:
            if row["track"] != "extension":
                errors.append(f"row {row_number}: no-paper-target row must use extension track")
            if row["target_status"] != "new_experiment_no_paper_target":
                errors.append(f"row {row_number}: extension row has incorrect target_status")
            if row["paper_protocol_match_required"] != "false":
                errors.append(f"row {row_number}: extension row must not require paper protocol matching")
            for field in ("target_mean_test_mse", "target_std_test_mse", "paper_reported_value"):
                if row[field] != "":
                    errors.append(f"row {row_number}: extension {field} must be blank")

        if row["scenario"] in {"absolute", "step", "linear", "sine"} and row["curve_required"] != "true":
            errors.append(f"row {row_number}: low-dimensional scenario must require a curve")
        if row["scenario"] == "sine":
            if has_paper_target:
                errors.append(f"row {row_number}: sine cannot have a paper target")
            if row["true_function"] != "g0(x) = sin(x)":
                errors.append(f"row {row_number}: sine true function must match the repository")
        if row["dataset_family"] == "high_dimensional":
            if row["aggregate_test_mse_required"] != "true" or row["target_num_runs"] != "5":
                errors.append(f"row {row_number}: high-dimensional rows require five-run MSE aggregation")
            if row["curve_required"] != "false":
                errors.append(f"row {row_number}: high-dimensional rows must not require curves")
        if row["algorithm_id"] == "fedogda_d":
            if row["comparison_reference"] != "fedgda_d":
                errors.append(f"row {row_number}: fedogda_d must reference fedgda_d")
            if row["tuning_rule"] != "validation_only" or row["fairness_rule"] != FEDOGDA_FAIRNESS:
                errors.append(f"row {row_number}: fedogda_d fairness/tuning fields are incorrect")
        if row["algorithm_id"] == "fedogda_s":
            if row["comparison_reference"] != "fedgda_s":
                errors.append(f"row {row_number}: fedogda_s must reference fedgda_s")
            if row["tuning_rule"] != "validation_only" or row["fairness_rule"] != FEDOGDA_FAIRNESS:
                errors.append(f"row {row_number}: fedogda_s fairness/tuning fields are incorrect")

    expected_keys = set(PAPER_TARGETS)
    actual_keys = {(row["scenario"], row["algorithm_id"]) for row in paper_rows}
    if actual_keys != expected_keys:
        errors.append("paper target scenario/algorithm mappings do not exactly match the supplied targets")

    if errors:
        raise ValueError("\n".join(errors))
    return len(rows), len(paper_rows), len(extension_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    try:
        rows, paper_rows, extension_rows = validate(args.csv_path)
    except (OSError, ValueError) as exc:
        print(f"validation: failed\n{exc}", file=sys.stderr)
        return 1
    print(f"rows: {rows}")
    print(f"paper_target_rows: {paper_rows}")
    print(f"extension_rows: {extension_rows}")
    print(f"scenarios: {len(SCENARIOS)}")
    print(f"algorithms_per_scenario: {len(ALGORITHMS)}")
    print("validation: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
