#!/usr/bin/env python3
"""Analyze and materialize the fast Sine FedOGDA-S v4 stages."""

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

from prepare_fedogda_s_sine_fast_v4 import FIELDS, make_row


ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_sine_fast_v4"
EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
STAGE_A_MANIFEST = EXP_DIR / "stage_a_manifest.csv"
STAGE_B_MANIFEST = EXP_DIR / "stage_b_manifest.csv"
STAGE_C_MANIFEST = EXP_DIR / "stage_c_manifest.csv"
OUTPUT_ROOT = ROOT / "results" / "curve_fitting_tuning" / SCREEN_NAME
PLOT_ROOT = ROOT / "experiments" / "curve_fitting_plots"
CSV_DIR = PLOT_ROOT / "csv"
PNG_DIR = PLOT_ROOT / "png" / SCREEN_NAME
PDF_DIR = PLOT_ROOT / "pdf" / SCREEN_NAME
LOG_DIR = ROOT / "logs" / SCREEN_NAME
V3_CANDIDATES = CSV_DIR / "fedogda_s_focused_v3_screen_candidates.csv"

V2_FEDOGDA_REF = (
    ROOT
    / "results"
    / "curve_fitting_tuning"
    / "optimistic_curve_screen_v2"
    / "sin"
    / "fedogda_s"
    / "seed_0"
    / "curvefit_sin_fedogda_s_seed0_alpha1p0_T500_R3_batch256_glr0p005_cm10_lam0p03_slr1p5"
)

BASELINE_CONFIG = {
    "learning_rate": 0.01,
    "critic_multiplier": 8.0,
    "objective_lambda_1": 0.01,
    "server_learning_rate": 1.5,
}
BASELINE_VAL = 0.02964276923734608
BASELINE_TEST = 0.030104425463267824
BASELINE_CURVE_MAE = 0.13467988046524487
BASELINE_AMP_RATIO = 0.619


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


def execution_lines() -> list[str]:
    lines = [
        "- GPU launch: `gpurun -g 2` with `--gpu-ids 0,1 --max-parallel 2`.",
        "- Thread caps: `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=4`.",
    ]
    log_path = latest_pipeline_log()
    if log_path is None:
        lines.append("- Elapsed wall-clock: unavailable; no pipeline log found.")
        return lines
    try:
        launch_time = datetime.strptime(log_path.stem.removeprefix("pipeline_"), "%Y%m%d_%H%M%S")
        finish_time = datetime.fromtimestamp(log_path.stat().st_mtime)
        elapsed = finish_time - launch_time
        total_seconds = max(0, int(elapsed.total_seconds()))
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


def load_manifest_rows(manifest: Path) -> tuple[list[RunRow], list[str]]:
    loaded: list[RunRow] = []
    missing: list[str] = []
    for manifest_row in read_csv(manifest):
        result_dir = ROOT / manifest_row["final_result_dir"]
        if not (
            (result_dir / "metrics.json").exists()
            and (result_dir / "mse_by_round.csv").exists()
            and (result_dir / "predictions.npz").exists()
        ):
            missing.append(manifest_row["run_id"])
            continue
        try:
            metrics = read_json(result_dir / "metrics.json")
            curve = curve_payload(result_dir)
            last_50_std, history_diverged = last_50_stability(result_dir / "mse_by_round.csv")
            best_val = to_float(metrics["best_validation_mse"])
            final_val = to_float(metrics["final_validation_mse"])
            row = {
                "source": SCREEN_NAME,
                "dataset": "sin",
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
                "diverged": bool(metrics.get("diverged", False)) or history_diverged,
                "runtime_seconds": float(metrics.get("runtime_seconds", math.nan)),
                "result_dir": rel(result_dir),
                "_curve": curve,
            }
            row.update({key: value for key, value in curve.items() if key not in {"x", "true_g", "pred"}})
            loaded.append(RunRow(manifest_row=manifest_row, row=row))
        except Exception:
            missing.append(manifest_row["run_id"])
    return loaded, missing


def load_v3_sine() -> list[RunRow]:
    rows: list[RunRow] = []
    for source_row in read_csv(V3_CANDIDATES):
        if source_row.get("dataset") != "sin":
            continue
        if to_bool(source_row.get("diverged", "false")):
            continue
        run_dir = ROOT / source_row["result_dir"]
        try:
            curve = curve_payload(run_dir)
            row = {
                "source": "fedogda_s_focused_v3",
                "dataset": "sin",
                "stage": source_row.get("stage", "screen_sine"),
                "method": "fedogda_s",
                "run_id": source_row["run_id"],
                "seed": int(source_row["seed"]),
                "learning_rate": to_float(source_row["learning_rate"]),
                "critic_multiplier": to_float(source_row["critic_multiplier"]),
                "objective_lambda_1": to_float(source_row["objective_lambda_1"]),
                "server_learning_rate": to_float(source_row["server_learning_rate"]),
                "weight_decay": to_float(source_row["weight_decay"]),
                "epochs": int(source_row["epochs"]),
                "comm_round": int(source_row["comm_round"]),
                "best_validation_mse": to_float(source_row["best_validation_mse"]),
                "last_50_val_mse_std": to_float(source_row["last_50_val_mse_std"]),
                "final_vs_best_validation_gap": to_float(source_row["final_vs_best_validation_gap"]),
                "test_mse_at_best_validation": to_float(source_row["test_mse_at_best_validation"]),
                "best_validation_round": int(float(source_row["best_validation_round"])),
                "final_validation_mse": to_float(source_row["final_validation_mse"]),
                "final_test_mse": to_float(source_row["final_test_mse"]),
                "diverged": False,
                "runtime_seconds": to_float(source_row["runtime_seconds"]),
                "result_dir": source_row["result_dir"],
                "_curve": curve,
            }
            row.update({key: value for key, value in curve.items() if key not in {"x", "true_g", "pred"}})
            rows.append(RunRow(manifest_row={}, row=row))
        except Exception:
            continue
    return rows


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


def valid(items: list[RunRow]) -> list[RunRow]:
    return [
        item
        for item in items
        if not item.row["diverged"] and math.isfinite(float(item.row["best_validation_mse"]))
    ]


def dedupe_best_by_config(items: list[RunRow]) -> list[RunRow]:
    best: dict[tuple[float, float, float, float], RunRow] = {}
    for item in valid(items):
        key = config_key(item.row)
        if key not in best or selection_key(item.row) < selection_key(best[key].row):
            best[key] = item
    return sorted(best.values(), key=lambda item: selection_key(item.row))


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


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def materialize_stage_b(combined: list[RunRow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in dedupe_best_by_config(combined)[:3]:
        base = item.row
        for server_lr in (1.75, 2.0):
            rows.append(
                make_row(
                    stage="stage_b_server_lr_probe",
                    seed=0,
                    learning_rate=float(base["learning_rate"]),
                    critic_multiplier=float(base["critic_multiplier"]),
                    objective_lambda_1=float(base["objective_lambda_1"]),
                    server_learning_rate=server_lr,
                    comm_round=500,
                    epochs=3,
                )
            )
    write_manifest(STAGE_B_MANIFEST, rows)
    return rows


def is_baseline(row: dict[str, Any]) -> bool:
    return all(abs(float(row[key]) - value) < 1e-12 for key, value in BASELINE_CONFIG.items())


def challenger_clears_gate(row: dict[str, Any]) -> bool:
    return (
        float(row["best_validation_mse"]) < BASELINE_VAL
        and float(row["amp_ratio"]) >= BASELINE_AMP_RATIO
        and float(row["curve_mae"]) <= BASELINE_CURVE_MAE
    )


def materialize_stage_c(combined: list[RunRow]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    ranked = dedupe_best_by_config(combined)
    challenger = next((item for item in ranked if not is_baseline(item.row)), None)
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    baseline = next((item for item in ranked if is_baseline(item.row)), None)
    if baseline is None:
        raise SystemExit("Current v3 baseline was not found in combined candidates")
    configs.append(baseline.row)
    if challenger is not None and challenger_clears_gate(challenger.row):
        configs.append(challenger.row)
    elif challenger is not None:
        write_manifest(STAGE_C_MANIFEST, [])
        return rows, challenger.row
    for config in configs:
        for seed in (0, 1, 2):
            rows.append(
                make_row(
                    stage="stage_c_confirm",
                    seed=seed,
                    learning_rate=float(config["learning_rate"]),
                    critic_multiplier=float(config["critic_multiplier"]),
                    objective_lambda_1=float(config["objective_lambda_1"]),
                    server_learning_rate=float(config["server_learning_rate"]),
                    comm_round=1000,
                    epochs=3,
                )
            )
    write_manifest(STAGE_C_MANIFEST, rows)
    return rows, challenger.row if challenger is not None else None


def aggregate_confirm(items: list[RunRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float, float], list[RunRow]] = {}
    for item in valid(items):
        groups.setdefault(config_key(item.row), []).append(item)
    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        public = [item.row for item in group]
        seed0 = next((item for item in group if item.row["seed"] == 0), group[0]).row
        rows.append(
            {
                "source": SCREEN_NAME,
                "dataset": "sin",
                "stage": "stage_c_aggregate",
                "method": "fedogda_s",
                "run_id": "aggregate:" + seed0["run_id"],
                "seed": "|".join(str(row["seed"]) for row in sorted(public, key=lambda item: item["seed"])),
                "seed_count": len({row["seed"] for row in public}),
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
            -int(row["seed_count"] == 3),
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


def load_reference(run_dir: Path) -> dict[str, Any] | None:
    if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.npz").exists():
        return None
    metrics = read_json(run_dir / "metrics.json")
    return {
        "label": "Previous FedOGDA-S best",
        "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
        "_curve": curve_payload(run_dir),
    }


def plot_final(selected: RunRow, selected_aggregate: dict[str, Any] | None = None) -> list[Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PNG_DIR / f"{SCREEN_NAME}_best_optimistic_only.png",
        PDF_DIR / f"{SCREEN_NAME}_best_optimistic_only.pdf",
    ]
    row = selected.row
    curve = row["_curve"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(curve["x"], curve["true_g"], color="black", linewidth=2.6, label="Actual causal effect")
    ref = load_reference(V2_FEDOGDA_REF)
    if ref is not None:
        ref_curve = ref["_curve"]
        ax.plot(
            ref_curve["x"],
            ref_curve["pred"],
            color="#991b1b",
            linestyle="-.",
            linewidth=2.0,
            label=f"Previous FedOGDA-S best test={ref['test_mse_at_best_validation']:.4f}",
        )
    if selected_aggregate is None:
        selected_label = (
            f"Selected v4 val={row['best_validation_mse']:.4f}, "
            f"test={row['test_mse_at_best_validation']:.4f}"
        )
    else:
        selected_label = (
            f"Selected v4 mean val={selected_aggregate['mean_best_validation_mse']:.4f}, "
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
        "Sine alpha=1.0: best optimistic FedOGDA-S\n"
        f"lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
        f"lambda={row['objective_lambda_1']:g}, server_lr={row['server_learning_rate']:g}, "
        f"T={row['comm_round']}, R={row['epochs']}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=7.5, loc="best")
    for output in outputs:
        fig.savefig(output, dpi=240 if output.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def plot_heatmap(items: list[RunRow]) -> list[Path]:
    rows = [
        item.row
        for item in dedupe_best_by_config(items)
        if abs(float(item.row["server_learning_rate"]) - 1.5) < 1e-12
    ]
    if not rows:
        return []
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PNG_DIR / f"{SCREEN_NAME}_validation_heatmap.png",
        PDF_DIR / f"{SCREEN_NAME}_validation_heatmap.pdf",
    ]
    lrs = sorted({float(row["learning_rate"]) for row in rows})
    cms = sorted({float(row["critic_multiplier"]) for row in rows})
    matrix = np.full((len(cms), len(lrs)), np.nan)
    for row in rows:
        if abs(float(row["objective_lambda_1"]) - 0.01) > 1e-12:
            continue
        i = cms.index(float(row["critic_multiplier"]))
        j = lrs.index(float(row["learning_rate"]))
        matrix[i, j] = float(row["best_validation_mse"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(lrs)), [f"{value:g}" for value in lrs])
    ax.set_yticks(range(len(cms)), [f"{value:g}" for value in cms])
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("critic_multiplier")
    ax.set_title("Sine validation MSE, lambda=0.01, server_lr=1.5")
    for i in range(len(cms)):
        for j in range(len(lrs)):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, label="best_validation_mse")
    for output in outputs:
        fig.savefig(output, dpi=240 if output.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def write_summary(
    *,
    stage_a: list[RunRow],
    stage_a_missing: list[str],
    stage_b: list[RunRow],
    stage_b_missing: list[str],
    stage_c: list[RunRow],
    stage_c_missing: list[str],
    combined: list[RunRow],
    selected: RunRow,
    selected_aggregate: dict[str, Any] | None,
    challenger: dict[str, Any] | None,
    stage_c_rows: list[dict[str, Any]],
    outputs: list[Path],
) -> Path:
    path = PLOT_ROOT / f"{SCREEN_NAME}_summary.md"
    row = selected.row
    lines = [
        f"# {SCREEN_NAME}",
        "",
        "Selection is validation-only. Test MSE and curve diagnostics are reported only after validation ranking.",
        "",
        "## Execution",
        "",
        *execution_lines(),
        "",
        "## Completion",
        "",
        f"- Stage A completed: `{len(stage_a)}`; missing/invalid: `{len(stage_a_missing)}`.",
        f"- Stage B completed: `{len(stage_b)}`; missing/invalid: `{len(stage_b_missing)}`.",
        f"- Stage C completed: `{len(stage_c)}`; missing/invalid: `{len(stage_c_missing)}`.",
        f"- Stage C rows materialized: `{len(stage_c_rows)}`.",
        "",
        "## Selected",
        "",
    ]
    lines.append(
        f"- lr `{row['learning_rate']:g}`, cm `{row['critic_multiplier']:g}`, "
        f"lambda `{row['objective_lambda_1']:g}`, server_lr `{row['server_learning_rate']:g}`, "
        f"T `{row['comm_round']}`, R `{row['epochs']}`."
    )
    if selected_aggregate is None:
        lines.extend(
            [
                f"- seed `{row['seed']}` validation MSE `{row['best_validation_mse']:.9f}`.",
                f"- seed `{row['seed']}` test@best `{row['test_mse_at_best_validation']:.9f}`.",
                (
                    f"- seed `{row['seed']}` curve MAE `{row['curve_mae']:.9f}`, "
                    f"corr `{row['curve_corr']:.9f}`, amp ratio `{row['amp_ratio']:.9f}`."
                ),
            ]
        )
    else:
        lines.extend(
            [
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
                f"- plotted representative seed `{row['seed']}`: `{row['run_id']}`.",
            ]
        )
    lines.extend(["", "## Challenger Gate", ""])
    if challenger is None:
        lines.append("- No non-current challenger was available.")
    else:
        lines.append(
            f"- Best non-current challenger val `{challenger['best_validation_mse']:.9f}`, "
            f"curve MAE `{challenger['curve_mae']:.9f}`, amp ratio `{challenger['amp_ratio']:.9f}`."
        )
        lines.append(
            f"- Clears gate: `{str(challenger_clears_gate(challenger)).lower()}` "
            f"(requires val < `{BASELINE_VAL:.9f}`, MAE <= `{BASELINE_CURVE_MAE:.9f}`, "
            f"amp ratio >= `{BASELINE_AMP_RATIO:.3f}`)."
        )
    lines.extend(["", "## Outputs", ""])
    for output in outputs:
        lines.append(f"- `{rel(output)}`")
    lines.extend(
        [
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_all_candidates.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_stage_a_selected.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_stage_b_selected.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_confirmation_aggregate.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_final_selected.csv')}`",
            "",
            "## Baseline",
            "",
            f"- Current v3 baseline val `{BASELINE_VAL:.9f}`, test@best `{BASELINE_TEST:.9f}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def final_selection(
    combined: list[RunRow],
    stage_c: list[RunRow],
    aggregate_rows: list[dict[str, Any]],
) -> RunRow:
    if aggregate_rows:
        winner = aggregate_rows[0]
        key = (
            float(winner["learning_rate"]),
            float(winner["critic_multiplier"]),
            float(winner["objective_lambda_1"]),
            float(winner["server_learning_rate"]),
        )
        seed0 = next(
            (
                item
                for item in stage_c
                if config_key(item.row) == key and int(item.row["seed"]) == 0
            ),
            None,
        )
        if seed0 is not None:
            return seed0
    return dedupe_best_by_config(combined)[0]


def analyze(materialize_next: bool) -> None:
    stage_a, stage_a_missing = load_manifest_rows(STAGE_A_MANIFEST)
    stage_b, stage_b_missing = load_manifest_rows(STAGE_B_MANIFEST)
    stage_c, stage_c_missing = load_manifest_rows(STAGE_C_MANIFEST)
    current_v3 = load_v3_sine()
    combined = current_v3 + stage_a + stage_b

    all_candidates = [item.row for item in dedupe_best_by_config(combined)]
    write_rows_csv(CSV_DIR / f"{SCREEN_NAME}_all_candidates.csv", all_candidates)
    write_rows_csv(
        CSV_DIR / f"{SCREEN_NAME}_stage_a_selected.csv",
        [item.row for item in dedupe_best_by_config(current_v3 + stage_a)[:3]],
    )

    if materialize_next and len(stage_a) == len(read_csv(STAGE_A_MANIFEST)) and not STAGE_B_MANIFEST.exists():
        materialize_stage_b(current_v3 + stage_a)

    write_rows_csv(
        CSV_DIR / f"{SCREEN_NAME}_stage_b_selected.csv",
        [item.row for item in dedupe_best_by_config(combined)[:3]],
    )

    challenger: dict[str, Any] | None = None
    stage_c_rows = read_csv(STAGE_C_MANIFEST)
    if materialize_next and STAGE_B_MANIFEST.exists() and len(stage_b) == len(read_csv(STAGE_B_MANIFEST)) and not STAGE_C_MANIFEST.exists():
        stage_c_rows, challenger = materialize_stage_c(combined)
    elif STAGE_C_MANIFEST.exists():
        ranked = dedupe_best_by_config(combined)
        maybe = next((item for item in ranked if not is_baseline(item.row)), None)
        challenger = maybe.row if maybe is not None else None

    aggregate_rows = aggregate_confirm(stage_c)
    write_aggregate_csv(CSV_DIR / f"{SCREEN_NAME}_confirmation_aggregate.csv", aggregate_rows)
    selected = final_selection(combined, stage_c, aggregate_rows)
    selected_aggregate = aggregate_rows[0] if aggregate_rows else None
    if selected_aggregate is None:
        write_rows_csv(CSV_DIR / f"{SCREEN_NAME}_final_selected.csv", [selected.row])
    else:
        write_aggregate_csv(CSV_DIR / f"{SCREEN_NAME}_final_selected.csv", [selected_aggregate])
    outputs = plot_final(selected, selected_aggregate) + plot_heatmap(combined)
    summary = write_summary(
        stage_a=stage_a,
        stage_a_missing=stage_a_missing,
        stage_b=stage_b,
        stage_b_missing=stage_b_missing,
        stage_c=stage_c,
        stage_c_missing=stage_c_missing,
        combined=combined,
        selected=selected,
        selected_aggregate=selected_aggregate,
        challenger=challenger,
        stage_c_rows=stage_c_rows,
        outputs=outputs,
    )
    print("Generated:")
    for output in outputs:
        print(f"  {rel(output)}")
    print(f"  {rel(summary)}")
    print("Selected:")
    row = selected.row
    if selected_aggregate is None:
        print(
            f"  lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
            f"lambda={row['objective_lambda_1']:g}, slr={row['server_learning_rate']:g}, "
            f"val={row['best_validation_mse']:.9f}, test@best={row['test_mse_at_best_validation']:.9f}"
        )
    else:
        print(
            f"  lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
            f"lambda={row['objective_lambda_1']:g}, slr={row['server_learning_rate']:g}, "
            f"mean_val={selected_aggregate['mean_best_validation_mse']:.9f}, "
            f"mean_test@best={selected_aggregate['mean_test_mse_at_best_validation']:.9f}"
        )
    if challenger is not None:
        print(f"Challenger clears gate: {challenger_clears_gate(challenger)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-next", action="store_true")
    args = parser.parse_args()
    analyze(materialize_next=bool(args.materialize_next))


if __name__ == "__main__":
    main()
