#!/usr/bin/env python3
"""Analyze and materialize the fast Step FedOGDA-S v5 stages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_fedogda_s_step_fast_v5 import FIELDS, make_row


ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_step_fast_v5"
EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
STAGE_A_MANIFEST = EXP_DIR / "stage_a_manifest.csv"
STAGE_B_MANIFEST = EXP_DIR / "stage_b_manifest.csv"
OUTPUT_ROOT = ROOT / "results" / "curve_fitting_tuning" / SCREEN_NAME
PLOT_ROOT = ROOT / "experiments" / "curve_fitting_plots"
CSV_DIR = PLOT_ROOT / "csv"
PNG_DIR = PLOT_ROOT / "png" / SCREEN_NAME
PDF_DIR = PLOT_ROOT / "pdf" / SCREEN_NAME
LOG_DIR = ROOT / "logs" / SCREEN_NAME

PREVIOUS_INCUMBENT_DIR = (
    ROOT
    / "results"
    / "curve_fitting_tuning"
    / "optimistic_curve_screen_v2"
    / "step"
    / "fedogda_s"
    / "seed_0"
    / "curvefit_step_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p005_cm15_lam0p1_slr1p5"
)
FEDGDA_REF_DIR = (
    ROOT
    / "results"
    / "curve_fitting_tuning"
    / "step_reference_repro_v1"
    / "step"
    / "fedgda_s"
    / "seed_0"
    / "curvefit_step_reference_repro_fedgda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p03_wd0p02_cm15_slr1p5"
)

INCUMBENT_CONFIG = {
    "learning_rate": 0.005,
    "critic_multiplier": 15.0,
    "objective_lambda_1": 0.1,
    "server_learning_rate": 1.5,
}
INCUMBENT_VAL = 0.010456710930982103
INCUMBENT_TEST = 0.010636225967950308
FEDGDA_REF_VAL = 0.006003955305925284
FEDGDA_REF_TEST = 0.006126142455462217
CHALLENGER_VAL_GATE = 0.0115


@dataclass(frozen=True)
class RunRow:
    manifest_row: dict[str, str]
    row: dict[str, Any]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def latest_pipeline_log() -> Path | None:
    logs = sorted(LOG_DIR.glob("pipeline_*.log"))
    return logs[-1] if logs else None


def format_elapsed(total_seconds: int) -> str:
    hours, rem = divmod(max(0, int(total_seconds)), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def execution_lines(elapsed_seconds: int | None = None) -> list[str]:
    lines = [
        "- GPU launch: `gpurun -g 2` with `--gpu-ids 0,1 --max-parallel 2`.",
        "- Thread caps: `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=4`.",
    ]
    if elapsed_seconds is not None:
        lines.append(f"- Elapsed wall-clock: about `{format_elapsed(elapsed_seconds)}`.")
        return lines
    log_path = latest_pipeline_log()
    if log_path is None:
        lines.append("- Elapsed wall-clock: unavailable; no pipeline log found.")
        return lines
    try:
        launch_time = datetime.strptime(log_path.stem.removeprefix("pipeline_"), "%Y%m%d_%H%M%S")
        finish_time = datetime.fromtimestamp(log_path.stat().st_mtime)
        total_seconds = max(0, int((finish_time - launch_time).total_seconds()))
        minutes, seconds = divmod(total_seconds, 60)
        lines.append(
            f"- Elapsed wall-clock: about `{minutes}m {seconds:02d}s` "
            f"from `{launch_time:%Y-%m-%d %H:%M:%S}` to `{finish_time:%Y-%m-%d %H:%M:%S}` local time."
        )
    except Exception:
        lines.append(f"- Elapsed wall-clock: unavailable; could not parse `{rel(log_path)}`.")
    lines.append(f"- Pipeline log: `{rel(log_path)}`.")
    return lines


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def to_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def pstdev_or_zero(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def curve_payload(run_dir: Path) -> dict[str, Any]:
    with np.load(run_dir / "predictions.npz") as data:
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        true_g = np.asarray(data["true_g"], dtype=float).reshape(-1)
        pred = np.asarray(data["best_validation_prediction"], dtype=float).reshape(-1)
    if not (x.size == true_g.size == pred.size):
        raise ValueError(f"shape mismatch in {run_dir / 'predictions.npz'}")
    if not (np.isfinite(x).all() and np.isfinite(true_g).all() and np.isfinite(pred).all()):
        raise ValueError(f"non-finite curve values in {run_dir / 'predictions.npz'}")
    order = np.argsort(x)
    x = x[order]
    true_g = true_g[order]
    pred = pred[order]
    err = pred - true_g
    true_amp = float(np.max(true_g) - np.min(true_g))
    pred_amp = float(np.max(pred) - np.min(pred))
    return {
        "x": x,
        "true_g": true_g,
        "pred": pred,
        "curve_mse": float(np.mean(err**2)),
        "curve_mae": float(np.mean(np.abs(err))),
        "curve_max_abs_error": float(np.max(np.abs(err))),
        "curve_corr": float(np.corrcoef(pred, true_g)[0, 1]) if pred.size > 1 else math.nan,
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_amp": pred_amp,
        "true_amp": true_amp,
        "amp_ratio": pred_amp / true_amp if true_amp > 0 else math.nan,
    }


def last_50_stability(mse_csv: Path) -> tuple[float, bool]:
    rows = read_csv(mse_csv)
    tail = rows[-50:]
    values = [to_float(row["val_mse"]) for row in tail]
    diverged = any(
        to_bool(row.get("diverged", "false")) or not to_bool(row.get("finite", "true"))
        for row in rows
    )
    return pstdev_or_zero(values), diverged


def required_artifact_errors(result_dir: Path) -> list[str]:
    required = [
        "metrics.json",
        "mse_by_round.csv",
        "predictions.npz",
        "checkpoints/best_validation.pt",
        "checkpoints/final.pt",
    ]
    return [item for item in required if not (result_dir / item).exists()]


def load_manifest_rows(manifest: Path) -> tuple[list[RunRow], list[dict[str, str]]]:
    loaded: list[RunRow] = []
    invalid: list[dict[str, str]] = []
    for manifest_row in read_csv(manifest):
        result_dir = ROOT / manifest_row["final_result_dir"]
        missing = required_artifact_errors(result_dir)
        if missing:
            invalid.append(
                {
                    "stage": manifest_row.get("stage", ""),
                    "run_id": manifest_row["run_id"],
                    "reason": "missing_artifacts:" + "|".join(missing),
                    "result_dir": rel(result_dir),
                }
            )
            continue
        try:
            metrics = read_json(result_dir / "metrics.json")
            curve = curve_payload(result_dir)
            last_50_std, history_diverged = last_50_stability(result_dir / "mse_by_round.csv")
            best_val = to_float(metrics["best_validation_mse"])
            final_val = to_float(metrics["final_validation_mse"])
            diverged = bool(metrics.get("diverged", False)) or history_diverged
            if diverged:
                raise ValueError("diverged=true")
            row = {
                "source": SCREEN_NAME,
                "dataset": "step",
                "stage": manifest_row["stage"],
                "method": manifest_row["method"],
                "run_id": manifest_row["run_id"],
                "seed": int(manifest_row["seed"]),
                "learning_rate": to_float(manifest_row["learning_rate"]),
                "critic_multiplier": to_float(manifest_row["critic_multiplier"]),
                "objective_lambda_1": to_float(manifest_row["objective_lambda_1"]),
                "server_learning_rate": to_float(manifest_row["server_learning_rate"]),
                "weight_decay": to_float(manifest_row["weight_decay"]),
                "epochs": int(manifest_row["epochs"]),
                "comm_round": int(manifest_row["comm_round"]),
                "best_validation_mse": best_val,
                "last_50_val_mse_std": last_50_std,
                "final_vs_best_validation_gap": final_val - best_val,
                "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
                "best_validation_round": int(metrics["best_validation_round"]),
                "final_validation_mse": final_val,
                "final_test_mse": to_float(metrics["final_test_mse"]),
                "diverged": False,
                "runtime_seconds": float(metrics.get("runtime_seconds", math.nan)),
                "result_dir": rel(result_dir),
                "_curve": curve,
            }
            row.update({key: value for key, value in curve.items() if key not in {"x", "true_g", "pred"}})
            loaded.append(RunRow(manifest_row=manifest_row, row=row))
        except Exception as exc:
            invalid.append(
                {
                    "stage": manifest_row.get("stage", ""),
                    "run_id": manifest_row["run_id"],
                    "reason": str(exc),
                    "result_dir": rel(result_dir),
                }
            )
    return loaded, invalid


def config_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["learning_rate"]),
        float(row["critic_multiplier"]),
        float(row["objective_lambda_1"]),
        float(row["server_learning_rate"]),
    )


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["best_validation_mse"]),
        float(row["last_50_val_mse_std"]),
        float(row["final_vs_best_validation_gap"]),
    )


def stability_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row["last_50_val_mse_std"]),
        float(row["final_vs_best_validation_gap"]),
    )


def is_incumbent(row: dict[str, Any]) -> bool:
    return all(abs(float(row[key]) - value) < 1e-12 for key, value in INCUMBENT_CONFIG.items())


def valid(items: list[RunRow]) -> list[RunRow]:
    return [
        item
        for item in items
        if not item.row["diverged"] and math.isfinite(float(item.row["best_validation_mse"]))
    ]


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "dataset",
        "stage",
        "method",
        "run_id",
        "seed",
        "learning_rate",
        "critic_multiplier",
        "objective_lambda_1",
        "server_learning_rate",
        "weight_decay",
        "epochs",
        "comm_round",
        "best_validation_mse",
        "last_50_val_mse_std",
        "final_vs_best_validation_gap",
        "test_mse_at_best_validation",
        "best_validation_round",
        "final_validation_mse",
        "final_test_mse",
        "curve_mse",
        "curve_mae",
        "curve_max_abs_error",
        "curve_corr",
        "amp_ratio",
        "pred_min",
        "pred_max",
        "diverged",
        "runtime_seconds",
        "result_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: public_row(row).get(field, "") for field in fields})


def write_invalid_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage", "run_id", "reason", "result_dir"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def challenger_qualifies(challenger: dict[str, Any], control: dict[str, Any]) -> tuple[bool, str]:
    if float(challenger["best_validation_mse"]) < float(control["best_validation_mse"]):
        return True, "beats_control_validation"
    if (
        float(challenger["best_validation_mse"]) <= CHALLENGER_VAL_GATE
        and stability_key(challenger) < stability_key(control)
    ):
        return True, "passes_val_gate_and_stability_tie_break"
    return False, "not_promoted"


def promoted_from_stage_a(stage_a: list[RunRow]) -> list[dict[str, Any]]:
    ranked = sorted(valid(stage_a), key=lambda item: selection_key(item.row))
    control = next((item for item in ranked if is_incumbent(item.row)), None)
    if control is None:
        return []
    challenger = next((item for item in ranked if not is_incumbent(item.row)), None)
    rows = [
        {
            "role": "incumbent_control",
            "promoted": True,
            "reason": "always_confirm_seeds_1_2",
            **public_row(control.row),
        }
    ]
    if challenger is not None:
        qualifies, reason = challenger_qualifies(challenger.row, control.row)
        rows.append(
            {
                "role": "best_non_incumbent_challenger",
                "promoted": qualifies,
                "reason": reason,
                **public_row(challenger.row),
            }
        )
    return rows


def materialize_stage_b(stage_a: list[RunRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    promoted = promoted_from_stage_a(stage_a)
    rows: list[dict[str, Any]] = []
    for item in promoted:
        if not item["promoted"]:
            continue
        for seed in (1, 2):
            rows.append(
                make_row(
                    stage="stage_b_confirm",
                    seed=seed,
                    learning_rate=float(item["learning_rate"]),
                    critic_multiplier=float(item["critic_multiplier"]),
                    objective_lambda_1=float(item["objective_lambda_1"]),
                    server_learning_rate=float(item["server_learning_rate"]),
                    comm_round=1500,
                    epochs=7,
                )
            )
    write_manifest(STAGE_B_MANIFEST, rows)
    return rows, promoted


def write_promoted_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "role",
        "promoted",
        "reason",
        "run_id",
        "seed",
        "learning_rate",
        "critic_multiplier",
        "objective_lambda_1",
        "server_learning_rate",
        "weight_decay",
        "best_validation_mse",
        "last_50_val_mse_std",
        "final_vs_best_validation_gap",
        "test_mse_at_best_validation",
        "best_validation_round",
        "curve_mse",
        "curve_mae",
        "curve_corr",
        "result_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: public_row(row).get(field, "") for field in fields})


def aggregate_confirm(items: list[RunRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float, float], list[RunRow]] = {}
    for item in valid(items):
        groups.setdefault(config_key(item.row), []).append(item)
    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        public = [item.row for item in group]
        seeds = sorted({int(row["seed"]) for row in public})
        if seeds != [0, 1, 2]:
            continue
        seed0 = next((item for item in group if item.row["seed"] == 0), group[0]).row
        rows.append(
            {
                "source": SCREEN_NAME,
                "dataset": "step",
                "stage": "stage_b_aggregate",
                "method": "fedogda_s",
                "run_id": "aggregate:" + seed0["run_id"],
                "seed": "|".join(str(seed) for seed in seeds),
                "seed_count": len(seeds),
                "learning_rate": key[0],
                "critic_multiplier": key[1],
                "objective_lambda_1": key[2],
                "server_learning_rate": key[3],
                "weight_decay": seed0["weight_decay"],
                "epochs": seed0["epochs"],
                "comm_round": seed0["comm_round"],
                "mean_best_validation_mse": mean_or_nan([row["best_validation_mse"] for row in public]),
                "std_best_validation_mse": pstdev_or_zero([row["best_validation_mse"] for row in public]),
                "mean_test_mse_at_best_validation": mean_or_nan(
                    [row["test_mse_at_best_validation"] for row in public]
                ),
                "std_test_mse_at_best_validation": pstdev_or_zero(
                    [row["test_mse_at_best_validation"] for row in public]
                ),
                "mean_last_50_val_mse_std": mean_or_nan([row["last_50_val_mse_std"] for row in public]),
                "mean_final_vs_best_validation_gap": mean_or_nan(
                    [row["final_vs_best_validation_gap"] for row in public]
                ),
                "mean_curve_mse": mean_or_nan([row["curve_mse"] for row in public]),
                "mean_curve_mae": mean_or_nan([row["curve_mae"] for row in public]),
                "mean_curve_corr": mean_or_nan([row["curve_corr"] for row in public]),
                "mean_amp_ratio": mean_or_nan([row["amp_ratio"] for row in public]),
                "representative_seed0_run_id": seed0["run_id"],
                "representative_seed0_result_dir": seed0["result_dir"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["mean_best_validation_mse"],
            row["mean_last_50_val_mse_std"],
            row["mean_final_vs_best_validation_gap"],
        ),
    )


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "dataset",
        "stage",
        "method",
        "run_id",
        "seed",
        "seed_count",
        "learning_rate",
        "critic_multiplier",
        "objective_lambda_1",
        "server_learning_rate",
        "weight_decay",
        "epochs",
        "comm_round",
        "mean_best_validation_mse",
        "std_best_validation_mse",
        "mean_test_mse_at_best_validation",
        "std_test_mse_at_best_validation",
        "mean_last_50_val_mse_std",
        "mean_final_vs_best_validation_gap",
        "mean_curve_mse",
        "mean_curve_mae",
        "mean_curve_corr",
        "mean_amp_ratio",
        "representative_seed0_run_id",
        "representative_seed0_result_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_reference(run_dir: Path, label: str) -> dict[str, Any] | None:
    if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.npz").exists():
        return None
    metrics = read_json(run_dir / "metrics.json")
    return {
        "label": label,
        "best_validation_mse": to_float(metrics["best_validation_mse"]),
        "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
        "final_test_mse": to_float(metrics["final_test_mse"]),
        "_curve": curve_payload(run_dir),
    }


def selected_seed0(selected_aggregate: dict[str, Any], items: list[RunRow]) -> RunRow | None:
    key = (
        float(selected_aggregate["learning_rate"]),
        float(selected_aggregate["critic_multiplier"]),
        float(selected_aggregate["objective_lambda_1"]),
        float(selected_aggregate["server_learning_rate"]),
    )
    return next(
        (
            item
            for item in items
            if config_key(item.row) == key and int(item.row["seed"]) == 0
        ),
        None,
    )


def plot_final(selected: RunRow | None, selected_aggregate: dict[str, Any] | None) -> list[Path]:
    if selected is None:
        return []
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PNG_DIR / f"{SCREEN_NAME}_best_step_optimistic_only.png",
        PDF_DIR / f"{SCREEN_NAME}_best_step_optimistic_only.pdf",
    ]
    row = selected.row
    curve = row["_curve"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(curve["x"], curve["true_g"], color="black", linewidth=2.6, label="Actual causal effect")
    fedgda = load_reference(FEDGDA_REF_DIR, "FedGDA-S reference")
    if fedgda is not None:
        ref_curve = fedgda["_curve"]
        ax.plot(
            ref_curve["x"],
            ref_curve["pred"],
            color="#15803d",
            linestyle="--",
            linewidth=2.1,
            label=f"FedGDA-S ref test={fedgda['test_mse_at_best_validation']:.4f}",
        )
    previous = load_reference(PREVIOUS_INCUMBENT_DIR, "Previous FedOGDA-S incumbent")
    if previous is not None:
        prev_curve = previous["_curve"]
        ax.plot(
            prev_curve["x"],
            prev_curve["pred"],
            color="#991b1b",
            linestyle="-.",
            linewidth=2.0,
            label=f"Previous FedOGDA-S test={previous['test_mse_at_best_validation']:.4f}",
        )
    if selected_aggregate is None:
        selected_label = (
            f"Selected v5 val={row['best_validation_mse']:.4f}, "
            f"test={row['test_mse_at_best_validation']:.4f}"
        )
    else:
        selected_label = (
            f"Selected v5 mean val={selected_aggregate['mean_best_validation_mse']:.4f}, "
            f"mean test={selected_aggregate['mean_test_mse_at_best_validation']:.4f}"
        )
    ax.plot(
        curve["x"],
        curve["pred"],
        color="#dc2626",
        linewidth=2.6,
        label=selected_label,
    )
    ax.set_title(
        "Step alpha=0.5: validation-selected FedOGDA-S v5\n"
        f"lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
        f"lambda={row['objective_lambda_1']:g}, server_lr={row['server_learning_rate']:g}, "
        f"T={row['comm_round']}, R={row['epochs']}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=7.2, loc="best")
    for output in outputs:
        fig.savefig(output, dpi=240 if output.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def write_summary(
    *,
    stage_a: list[RunRow],
    stage_a_invalid: list[dict[str, str]],
    stage_b: list[RunRow],
    stage_b_invalid: list[dict[str, str]],
    stage_b_rows: list[dict[str, str]],
    promoted: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    selected: RunRow | None,
    selected_aggregate: dict[str, Any] | None,
    outputs: list[Path],
    elapsed_seconds: int | None,
) -> Path:
    path = PLOT_ROOT / f"{SCREEN_NAME}_summary.md"
    lines = [
        f"# {SCREEN_NAME}",
        "",
        "Selection is validation-only. Test MSE and curve diagnostics are reported only after validation ranking.",
        "",
        "## Execution",
        "",
        *execution_lines(elapsed_seconds),
        "",
        "## Completion",
        "",
        f"- Stage A completed: `{len(stage_a)}`; missing/invalid: `{len(stage_a_invalid)}`.",
        f"- Stage B completed: `{len(stage_b)}`; missing/invalid: `{len(stage_b_invalid)}`.",
        f"- Stage B rows materialized: `{len(stage_b_rows)}`.",
        "",
        "## Selected",
        "",
    ]
    if selected is None or selected_aggregate is None:
        lines.append("- No complete 3-seed confirmation aggregate is available yet.")
    else:
        row = selected.row
        lines.extend(
            [
                (
                    f"- lr `{row['learning_rate']:g}`, cm `{row['critic_multiplier']:g}`, "
                    f"lambda `{row['objective_lambda_1']:g}`, "
                    f"server_lr `{row['server_learning_rate']:g}`, "
                    f"T `{row['comm_round']}`, R `{row['epochs']}`."
                ),
                f"- confirmation seeds `{selected_aggregate['seed']}`.",
                (
                    f"- mean validation MSE `{selected_aggregate['mean_best_validation_mse']:.9f}` "
                    f"+/- `{selected_aggregate['std_best_validation_mse']:.9f}`."
                ),
                (
                    f"- mean test@best `{selected_aggregate['mean_test_mse_at_best_validation']:.9f}` "
                    f"+/- `{selected_aggregate['std_test_mse_at_best_validation']:.9f}`."
                ),
                (
                    f"- mean curve MAE `{selected_aggregate['mean_curve_mae']:.9f}`, "
                    f"corr `{selected_aggregate['mean_curve_corr']:.9f}`, "
                    f"amp ratio `{selected_aggregate['mean_amp_ratio']:.9f}`."
                ),
                (
                    f"- mean last-50 val std `{selected_aggregate['mean_last_50_val_mse_std']:.9f}`, "
                    f"mean final-vs-best val gap `{selected_aggregate['mean_final_vs_best_validation_gap']:.9f}`."
                ),
                f"- plotted representative seed `{row['seed']}`: `{row['run_id']}`.",
            ]
        )
    lines.extend(["", "## Promotion", ""])
    if not promoted:
        lines.append("- No promotion decision is available yet.")
    else:
        for item in promoted:
            lines.append(
                f"- `{item['role']}` promoted=`{str(item['promoted']).lower()}` "
                f"reason=`{item['reason']}` val=`{float(item['best_validation_mse']):.9f}` "
                f"lr=`{float(item['learning_rate']):g}` cm=`{float(item['critic_multiplier']):g}` "
                f"lambda=`{float(item['objective_lambda_1']):g}` "
                f"server_lr=`{float(item['server_learning_rate']):g}`."
            )
    lines.extend(
        [
            "",
            "## FedGDA-S Reference",
            "",
            f"- FedGDA-S reference val `{FEDGDA_REF_VAL:.9f}`, test@best `{FEDGDA_REF_TEST:.9f}`.",
            f"- Previous FedOGDA-S incumbent val `{INCUMBENT_VAL:.9f}`, test@best `{INCUMBENT_TEST:.9f}`.",
        ]
    )
    if selected_aggregate is not None:
        beats = float(selected_aggregate["mean_test_mse_at_best_validation"]) < FEDGDA_REF_TEST
        lines.append(
            f"- FedOGDA-S beats FedGDA-S post-selection test reference: `{str(beats).lower()}`."
        )
    lines.extend(["", "## Outputs", ""])
    for output in outputs:
        lines.append(f"- `{rel(output)}`")
    lines.extend(
        [
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_all_candidates.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_stage_a_ranked.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_promoted_configs.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_confirmation_aggregate.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_final_selected.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_invalid_missing.csv')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze(materialize_next: bool, elapsed_seconds: int | None = None) -> None:
    stage_a, stage_a_invalid = load_manifest_rows(STAGE_A_MANIFEST)
    stage_b, stage_b_invalid = load_manifest_rows(STAGE_B_MANIFEST)
    all_items = stage_a + stage_b

    write_rows_csv(
        CSV_DIR / f"{SCREEN_NAME}_all_candidates.csv",
        [item.row for item in sorted(valid(all_items), key=lambda item: selection_key(item.row))],
    )
    write_rows_csv(
        CSV_DIR / f"{SCREEN_NAME}_stage_a_ranked.csv",
        [item.row for item in sorted(valid(stage_a), key=lambda item: selection_key(item.row))],
    )
    write_invalid_csv(CSV_DIR / f"{SCREEN_NAME}_invalid_missing.csv", stage_a_invalid + stage_b_invalid)

    promoted: list[dict[str, Any]] = promoted_from_stage_a(stage_a)
    stage_b_rows = read_csv(STAGE_B_MANIFEST)
    if materialize_next and not STAGE_B_MANIFEST.exists():
        expected_a = len(read_csv(STAGE_A_MANIFEST))
        if expected_a and len(stage_a) == expected_a and not stage_a_invalid:
            stage_b_rows, promoted = materialize_stage_b(stage_a)
    write_promoted_csv(CSV_DIR / f"{SCREEN_NAME}_promoted_configs.csv", promoted)

    aggregate_rows = aggregate_confirm(all_items)
    write_aggregate_csv(CSV_DIR / f"{SCREEN_NAME}_confirmation_aggregate.csv", aggregate_rows)
    selected_aggregate = aggregate_rows[0] if aggregate_rows else None
    selected = selected_seed0(selected_aggregate, all_items) if selected_aggregate is not None else None
    if selected_aggregate is None:
        write_aggregate_csv(CSV_DIR / f"{SCREEN_NAME}_final_selected.csv", [])
    else:
        write_aggregate_csv(CSV_DIR / f"{SCREEN_NAME}_final_selected.csv", [selected_aggregate])
    outputs = plot_final(selected, selected_aggregate)
    summary = write_summary(
        stage_a=stage_a,
        stage_a_invalid=stage_a_invalid,
        stage_b=stage_b,
        stage_b_invalid=stage_b_invalid,
        stage_b_rows=stage_b_rows,
        promoted=promoted,
        aggregate_rows=aggregate_rows,
        selected=selected,
        selected_aggregate=selected_aggregate,
        outputs=outputs,
        elapsed_seconds=elapsed_seconds,
    )
    print("Generated:")
    for output in outputs:
        print(f"  {rel(output)}")
    print(f"  {rel(summary)}")
    if selected_aggregate is None:
        print("Selected: pending 3-seed confirmation")
    else:
        print(
            "Selected: "
            f"lr={selected_aggregate['learning_rate']:g}, "
            f"cm={selected_aggregate['critic_multiplier']:g}, "
            f"lambda={selected_aggregate['objective_lambda_1']:g}, "
            f"slr={selected_aggregate['server_learning_rate']:g}, "
            f"mean_val={selected_aggregate['mean_best_validation_mse']:.9f}, "
            f"mean_test@best={selected_aggregate['mean_test_mse_at_best_validation']:.9f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-next", action="store_true")
    parser.add_argument("--elapsed-seconds", type=int)
    args = parser.parse_args()
    analyze(materialize_next=bool(args.materialize_next), elapsed_seconds=args.elapsed_seconds)


if __name__ == "__main__":
    main()
