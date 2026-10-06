#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import os
import struct
import zipfile


EXPECTED_ARTIFACTS = (
    "effective_config.json",
    "mse_by_round.csv",
    "metrics.json",
    "predictions.npz",
    os.path.join("checkpoints", "best_validation.pt"),
    os.path.join("checkpoints", "final.pt"),
)
PREDICTION_KEYS = (
    "x",
    "true_g",
    "best_validation_prediction",
    "final_prediction",
)


class ValidationError(Exception):
    pass


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _as_float(value, field):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} is not numeric: {value!r}")


def _assert_finite_number(value, field):
    number = _as_float(value, field)
    if not math.isfinite(number):
        raise ValidationError(f"{field} is not finite: {value!r}")
    return number


def _assert_int(value, field):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} is not an integer: {value!r}")


def _assert_json_numbers_finite(value, prefix="json"):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_numbers_finite(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_numbers_finite(item, f"{prefix}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValidationError(f"{prefix} is not finite: {value!r}")


def _load_csv_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _read_npy_bytes(payload):
    if not payload.startswith(b"\x93NUMPY"):
        raise ValidationError("Invalid npy payload")
    major = payload[6]
    offset = 8
    if major == 1:
        header_len = struct.unpack("<H", payload[offset:offset + 2])[0]
        offset += 2
    else:
        header_len = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
    header = ast.literal_eval(payload[offset:offset + header_len].decode("latin1").strip())
    offset += header_len
    shape = header["shape"]
    if isinstance(shape, int):
        shape = (shape,)
    descr = header["descr"]
    if header.get("fortran_order"):
        raise ValidationError("Fortran-order npy arrays are not supported")
    count = 1
    for dim in shape:
        count *= int(dim)
    if count == 0:
        return {"shape": tuple(shape), "values": []}
    if descr.endswith("f8"):
        fmt = "d"
        size = 8
    elif descr.endswith("f4"):
        fmt = "f"
        size = 4
    elif descr.endswith("i8"):
        fmt = "q"
        size = 8
    elif descr.endswith("i4"):
        fmt = "i"
        size = 4
    elif descr.endswith("u8"):
        fmt = "Q"
        size = 8
    elif descr.endswith("u4"):
        fmt = "I"
        size = 4
    else:
        raise ValidationError(f"Unsupported npy dtype: {descr}")
    endian = ">" if descr.startswith(">") else "<"
    expected_bytes = count * size
    data = payload[offset:offset + expected_bytes]
    if len(data) != expected_bytes:
        raise ValidationError("Truncated npy payload")
    values = list(struct.unpack(endian + fmt * count, data))
    return {"shape": tuple(shape), "values": [float(value) for value in values]}


def _load_npz_numeric_arrays(path, keys=PREDICTION_KEYS):
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        for key in keys:
            name = f"{key}.npy"
            if name not in names:
                raise ValidationError(f"Missing prediction array: {key}")
            arrays[key] = _read_npy_bytes(archive.read(name))
    return arrays


def _mean_squared_error(prediction, target):
    if len(prediction) != len(target):
        raise ValidationError("Prediction and target lengths differ")
    if not prediction:
        raise ValidationError("Cannot validate MSE for empty predictions")
    return sum((pred - true) ** 2 for pred, true in zip(prediction, target)) / len(prediction)


def _assert_close(left, right, field, tolerance=1e-8):
    if abs(float(left) - float(right)) > tolerance:
        raise ValidationError(f"{field} mismatch: {left!r} != {right!r}")


def _checkpoint_has_gf_state(path):
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu")
        if all(key in checkpoint for key in ("g_state_dict", "f_state_dict")):
            return True
        state = checkpoint.get("state", {})
        return all(key in state for key in ("g", "f"))
    except Exception:
        pass
    with zipfile.ZipFile(path, "r") as archive:
        data_files = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not data_files:
            raise ValidationError(f"Checkpoint has no data.pkl: {path}")
        payload = b"".join(archive.read(name) for name in data_files)
    required = (b"g_state_dict", b"f_state_dict")
    if all(key in payload for key in required):
        return True
    nested = (b"state", b"g", b"f")
    return all(key in payload for key in nested)


def _assert_checkpoints(run_dir):
    for name in ("best_validation.pt", "final.pt"):
        path = os.path.join(run_dir, "checkpoints", name)
        if not _checkpoint_has_gf_state(path):
            raise ValidationError(f"Checkpoint lacks required g/f state: {path}")


def _stochastic_smoke_requires_batching_summary(config):
    variant = str(config.get("variant", ""))
    return variant.endswith("_s") or bool(config.get("require_multibatch_stochastic", False))


def _load_batching_rows(path):
    payload = _load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return payload["entries"]
    raise ValidationError("batching_summary.json must contain a list of selected-client rows")


def _assert_batching_summary(run_dir, config):
    if not _stochastic_smoke_requires_batching_summary(config):
        return
    path = os.path.join(run_dir, "batching_summary.json")
    if not os.path.exists(path):
        raise ValidationError("Missing batching_summary.json for stochastic smoke run")
    rows = _load_batching_rows(path)
    if not rows:
        raise ValidationError("batching_summary.json has no selected-client rows")

    variant = str(config.get("variant", ""))
    comm_round = _assert_int(config.get("comm_round"), "config.comm_round")
    client_num_per_round = _assert_int(
        config.get("client_num_per_round"),
        "config.client_num_per_round",
    )
    configured_batch_size = _assert_int(config.get("batch_size"), "config.batch_size")
    if configured_batch_size <= 0:
        raise ValidationError("Stochastic smoke run has non-positive batch_size")

    expected_row_count = comm_round * client_num_per_round
    if len(rows) != expected_row_count:
        raise ValidationError(
            f"batching_summary row count {len(rows)} != expected {expected_row_count}"
        )

    for index, row in enumerate(rows):
        prefix = f"batching_summary[{index}]"
        if row.get("variant") != variant:
            raise ValidationError(f"{prefix}.variant mismatch: {row.get('variant')!r} != {variant!r}")
        round_idx = _assert_int(row.get("round"), f"{prefix}.round")
        if round_idx < 0 or round_idx >= comm_round:
            raise ValidationError(f"{prefix}.round out of range: {round_idx}")
        client_id = _assert_int(row.get("client_id"), f"{prefix}.client_id")
        if client_id < 0:
            raise ValidationError(f"{prefix}.client_id is negative: {client_id}")
        sample_count = _assert_int(row.get("sample_count"), f"{prefix}.sample_count")
        if sample_count <= 0:
            raise ValidationError(f"{prefix}.sample_count must be positive: {sample_count}")
        row_batch_size = _assert_int(
            row.get("configured_batch_size"),
            f"{prefix}.configured_batch_size",
        )
        if row_batch_size != configured_batch_size:
            raise ValidationError(
                f"{prefix}.configured_batch_size mismatch: "
                f"{row_batch_size} != {configured_batch_size}"
            )
        num_batches = _assert_int(row.get("num_batches"), f"{prefix}.num_batches")
        expected_batches = int(math.ceil(sample_count * 1.0 / configured_batch_size))
        if num_batches != expected_batches:
            raise ValidationError(
                f"{prefix}.num_batches mismatch: {num_batches} != {expected_batches}"
            )
        if num_batches < 2:
            raise ValidationError(f"{prefix}.num_batches is less than 2: {num_batches}")
        first_batch_size = _assert_int(
            row.get("first_actual_batch_size"),
            f"{prefix}.first_actual_batch_size",
        )
        expected_first_batch_size = min(sample_count, configured_batch_size)
        if first_batch_size != expected_first_batch_size:
            raise ValidationError(
                f"{prefix}.first_actual_batch_size mismatch: "
                f"{first_batch_size} != {expected_first_batch_size}"
            )


def _assert_test_mse_by_round(run_dir, config, metrics, comm_round):
    log_test = bool(config.get("log_test_mse_by_round", False))
    if not log_test:
        return
    if not bool(config.get("test_mse_logged_by_round", False)):
        raise ValidationError("config.test_mse_logged_by_round must be true when log_test_mse_by_round=true")
    if bool(config.get("test_mse_used_for_selection", False)):
        raise ValidationError("config.test_mse_used_for_selection must be false")
    if str(config.get("selection_metric_source", "validation")).lower() != "validation":
        raise ValidationError("config.selection_metric_source must be validation")
    if not bool(metrics.get("test_mse_logged_by_round", False)):
        raise ValidationError("metrics.test_mse_logged_by_round must be true when log_test_mse_by_round=true")
    if bool(metrics.get("test_mse_used_for_selection", False)):
        raise ValidationError("metrics.test_mse_used_for_selection must be false")
    if str(metrics.get("selection_metric_source", "validation")).lower() != "validation":
        raise ValidationError("metrics.selection_metric_source must be validation")

    path = os.path.join(run_dir, "test_mse_by_round.csv")
    if not os.path.exists(path):
        raise ValidationError("Missing test_mse_by_round.csv when log_test_mse_by_round=true")
    rows = _load_csv_rows(path)
    if len(rows) != comm_round:
        raise ValidationError(f"test_mse_by_round row count {len(rows)} != comm_round {comm_round}")
    for index, row in enumerate(rows):
        prefix = f"test_mse_by_round[{index}]"
        round_idx = _assert_int(row.get("round"), f"{prefix}.round")
        if round_idx != index:
            raise ValidationError(f"{prefix}.round mismatch: {round_idx} != {index}")
        _assert_finite_number(row.get("test_mse"), f"{prefix}.test_mse")
        if str(row.get("finite")) != "True":
            raise ValidationError(f"{prefix}.finite flag is not True")
        if str(row.get("diverged")) != "False":
            raise ValidationError(f"{prefix}.diverged flag is not False")


def validate_run(
    run_dir,
    dataset=None,
    variant=None,
    seed=None,
    run_id=None,
    process_exit_status=0,
):
    if int(process_exit_status) != 0:
        raise ValidationError(f"Process exit status was {process_exit_status}")
    missing = [
        artifact for artifact in EXPECTED_ARTIFACTS
        if not os.path.exists(os.path.join(run_dir, artifact))
    ]
    if missing:
        raise ValidationError(f"Missing artifacts: {', '.join(missing)}")

    config = _load_json(os.path.join(run_dir, "effective_config.json"))
    metrics = _load_json(os.path.join(run_dir, "metrics.json"))
    rows = _load_csv_rows(os.path.join(run_dir, "mse_by_round.csv"))
    _assert_batching_summary(run_dir, config)

    if dataset is not None and config.get("dataset") != dataset:
        raise ValidationError(f"Dataset mismatch: {config.get('dataset')!r} != {dataset!r}")
    if variant is not None and config.get("variant") != variant:
        raise ValidationError(f"Variant mismatch: {config.get('variant')!r} != {variant!r}")
    if seed is not None and int(config.get("random_seed")) != int(seed):
        raise ValidationError(f"Seed mismatch: {config.get('random_seed')!r} != {seed!r}")
    if run_id is not None and str(config.get("run_id")) != str(run_id):
        raise ValidationError(f"Run ID mismatch: {config.get('run_id')!r} != {run_id!r}")

    comm_round = int(config["comm_round"])
    if len(rows) != comm_round:
        raise ValidationError(f"CSV row count {len(rows)} != comm_round {comm_round}")

    _assert_json_numbers_finite(metrics, "metrics")
    if bool(metrics.get("diverged", False)):
        raise ValidationError("metrics.diverged is true")
    _assert_test_mse_by_round(run_dir, config, metrics, comm_round)

    numeric_csv_fields = (
        "round",
        "train_mse",
        "val_mse",
        "gmm_train_objective",
        "gmm_val_objective",
        "gmm_eval",
    )
    parsed_rows = []
    for row in rows:
        parsed = {}
        for field in numeric_csv_fields:
            parsed[field] = _assert_finite_number(row[field], f"csv.{field}")
        if str(row.get("finite")) != "True":
            raise ValidationError("CSV finite flag is not True")
        if str(row.get("diverged")) != "False":
            raise ValidationError("CSV diverged flag is not False")
        parsed_rows.append(parsed)

    best_row = min(parsed_rows, key=lambda row: row["val_mse"])
    best_round = int(metrics["best_validation_round"])
    if best_round != int(best_row["round"]):
        raise ValidationError("best_validation_round does not match minimum validation-MSE row")
    _assert_close(metrics["best_validation_mse"], best_row["val_mse"], "best_validation_mse")
    final_row = parsed_rows[-1]
    _assert_close(metrics["final_validation_mse"], final_row["val_mse"], "final_validation_mse")

    arrays = _load_npz_numeric_arrays(os.path.join(run_dir, "predictions.npz"))
    shapes = {key: arrays[key]["shape"] for key in PREDICTION_KEYS}
    if len(set(shapes.values())) != 1:
        raise ValidationError(f"Prediction shapes differ: {shapes}")
    for key, array in arrays.items():
        for value in array["values"]:
            if not math.isfinite(value):
                raise ValidationError(f"Prediction array {key} contains non-finite values")
    best_mse = _mean_squared_error(
        arrays["best_validation_prediction"]["values"],
        arrays["true_g"]["values"],
    )
    _assert_close(metrics["test_mse_at_best_validation"], best_mse, "test_mse_at_best_validation")
    final_mse = _mean_squared_error(
        arrays["final_prediction"]["values"],
        arrays["true_g"]["values"],
    )
    _assert_close(metrics["final_test_mse"], final_mse, "final_test_mse")
    final_round = int(parsed_rows[-1]["round"])
    if best_round == final_round:
        if arrays["best_validation_prediction"]["values"] != arrays["final_prediction"]["values"]:
            raise ValidationError("Best and final predictions differ when final round is best")

    _assert_checkpoints(run_dir)
    return {
        "run_dir": run_dir,
        "dataset": config.get("dataset"),
        "variant": config.get("variant"),
        "comm_round": comm_round,
        "best_validation_round": best_round,
        "test_mse_at_best_validation": metrics["test_mse_at_best_validation"],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate an ABS smoke run directory.")
    parser.add_argument("run_dir")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--process-exit-status", type=int, default=0)
    args = parser.parse_args()
    result = validate_run(
        args.run_dir,
        dataset=args.dataset,
        variant=args.variant,
        seed=args.seed,
        run_id=args.run_id,
        process_exit_status=args.process_exit_status,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
