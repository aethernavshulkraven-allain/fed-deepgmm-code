#!/usr/bin/env python3
"""Analyze and materialize FedOGDA-S focused v3 tuning stages.

The script never selects with Test MSE. Test and curve diagnostics are reported
only after validation ranking fixes the selected config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCREEN_NAME = "fedogda_s_focused_v3"
EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
SCREEN_MANIFEST = EXP_DIR / "manifest.csv"
CONFIRM_SEED0_MANIFEST = EXP_DIR / "confirm_seed0_manifest.csv"
CONFIRM_SEEDS_MANIFEST = EXP_DIR / "confirm_seeds_manifest.csv"
OUTPUT_ROOT = Path("results") / "curve_fitting_tuning" / SCREEN_NAME
PLOT_ROOT = ROOT / "experiments" / "curve_fitting_plots"
CSV_DIR = PLOT_ROOT / "csv"

REFERENCE_RUNS = {
    "sin": [
        (
            "FedGDA-S ref",
            ROOT
            / "results"
            / "rerun_protocol_v1"
            / "sin"
            / "fedgda_s"
            / "seed_0"
            / "rerun_protocol_v1_sin_fedgda_s_seed0_alpha1p0",
            "#2563eb",
            "--",
            1.6,
        ),
        (
            "FedOGDA-S v2 best",
            ROOT
            / "results"
            / "curve_fitting_tuning"
            / "optimistic_curve_screen_v2"
            / "sin"
            / "fedogda_s"
            / "seed_0"
            / "curvefit_sin_fedogda_s_seed0_alpha1p0_T500_R3_batch256_glr0p005_cm10_lam0p03_slr1p5",
            "#991b1b",
            "-.",
            1.6,
        ),
    ],
    "step": [
        (
            "FedGDA-S reference",
            ROOT
            / "results"
            / "curve_fitting_tuning"
            / "step_reference_repro_v1"
            / "step"
            / "fedgda_s"
            / "seed_0"
            / "curvefit_step_reference_repro_fedgda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p03_wd0p02_cm15_slr1p5",
            "#15803d",
            "--",
            1.6,
        ),
        (
            "FedOGDA-S v2 best",
            ROOT
            / "results"
            / "curve_fitting_tuning"
            / "optimistic_curve_screen_v2"
            / "step"
            / "fedogda_s"
            / "seed_0"
            / "curvefit_step_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p005_cm15_lam0p1_slr1p5",
            "#f97316",
            "-.",
            1.6,
        ),
    ],
}


@dataclass(frozen=True)
class LoadedRun:
    manifest_row: dict[str, str]
    row: dict[str, Any]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def token(value: Any) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def read_csv(path: Path) -> list[dict[str, str]]:
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
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def pstdev_or_zero(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def last_50_stability(mse_csv: Path) -> tuple[float, bool]:
    rows = read_csv(mse_csv)
    tail = rows[-50:]
    val_tail = [to_float(row["val_mse"]) for row in tail]
    diverged = any(
        to_bool(row.get("diverged", "false")) or not to_bool(row.get("finite", "true"))
        for row in rows
    )
    return pstdev_or_zero(val_tail), diverged


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
    corr = np.corrcoef(pred, true_g)[0, 1] if pred.size > 1 else math.nan
    return {
        "x": x,
        "true_g": true_g,
        "pred": pred,
        "curve_mse": float(np.mean(err**2)),
        "curve_mae": float(np.mean(np.abs(err))),
        "curve_max_abs_error": float(np.max(np.abs(err))),
        "curve_corr": float(corr),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


def load_manifest(manifest_path: Path) -> tuple[list[LoadedRun], list[str], list[dict[str, str]], list[str]]:
    manifest_rows = read_csv(manifest_path)
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    loaded: list[LoadedRun] = []
    missing: list[str] = []
    for manifest_row in manifest_rows:
        result_dir = ROOT / manifest_row["final_result_dir"]
        metrics_path = result_dir / "metrics.json"
        mse_path = result_dir / "mse_by_round.csv"
        predictions_path = result_dir / "predictions.npz"
        if not (metrics_path.exists() and mse_path.exists() and predictions_path.exists()):
            missing.append(manifest_row["run_id"])
            continue
        metrics = read_json(metrics_path)
        last_50_val_mse_std, history_diverged = last_50_stability(mse_path)
        curve = curve_payload(result_dir)
        best_validation_mse = to_float(metrics["best_validation_mse"])
        final_validation_mse = to_float(metrics["final_validation_mse"])
        row = {
            "run_id": manifest_row["run_id"],
            "stage": manifest_row.get("stage", ""),
            "dataset": manifest_row["dataset"],
            "method": manifest_row["method"],
            "seed": int(manifest_row["seed"]),
            "learning_rate": to_float(manifest_row["learning_rate"]),
            "critic_multiplier": to_float(manifest_row["critic_multiplier"]),
            "objective_lambda_1": to_float(manifest_row["objective_lambda_1"]),
            "server_learning_rate": to_float(manifest_row["server_learning_rate"]),
            "weight_decay": to_float(manifest_row["weight_decay"]),
            "epochs": int(manifest_row["epochs"]),
            "comm_round": int(manifest_row["comm_round"]),
            "best_validation_mse": best_validation_mse,
            "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
            "best_validation_round": int(metrics["best_validation_round"]),
            "final_validation_mse": final_validation_mse,
            "final_test_mse": to_float(metrics["final_test_mse"]),
            "final_vs_best_validation_gap": final_validation_mse - best_validation_mse,
            "last_50_val_mse_std": last_50_val_mse_std,
            "diverged": bool(metrics.get("diverged", False)) or history_diverged,
            "runtime_seconds": float(metrics.get("runtime_seconds", math.nan)),
            "result_dir": rel(result_dir),
        }
        row.update({key: value for key, value in curve.items() if key not in {"x", "true_g", "pred"}})
        row["_curve"] = curve
        loaded.append(LoadedRun(manifest_row=manifest_row, row=row))
    return loaded, missing, manifest_rows, fieldnames


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        row["best_validation_mse"],
        row["last_50_val_mse_std"],
        row["final_vs_best_validation_gap"],
    )


def valid_rows(loaded: list[LoadedRun]) -> list[LoadedRun]:
    return [
        item
        for item in loaded
        if not item.row["diverged"] and math.isfinite(float(item.row["best_validation_mse"]))
    ]


def top_by_dataset(loaded: list[LoadedRun], dataset: str, n: int) -> list[LoadedRun]:
    rows = [item for item in valid_rows(loaded) if item.row["dataset"] == dataset]
    return sorted(rows, key=lambda item: selection_key(item.row))[:n]


def config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["learning_rate"],
        row["critic_multiplier"],
        row["objective_lambda_1"],
        row["server_learning_rate"],
        row["epochs"],
        row["comm_round"],
    )


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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


def write_manifest(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_run_id(
    *,
    stage: str,
    dataset: str,
    seed: int,
    alpha: float,
    comm_round: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    critic_multiplier: float,
    objective_lambda_1: float,
    server_learning_rate: float,
) -> str:
    return (
        f"v3_{stage}_{dataset}_fedogda_s_seed{seed}_alpha{token(alpha)}"
        f"_T{comm_round}_R{epochs}_batch{batch_size}"
        f"_glr{token(learning_rate)}_cm{token(critic_multiplier)}"
        f"_lam{token(objective_lambda_1)}_slr{token(server_learning_rate)}"
    )


def confirm_row(template: dict[str, str], *, stage: str, seed: int, comm_round: int, epochs: int) -> dict[str, Any]:
    row = dict(template)
    dataset = row["dataset"]
    method = row["method"]
    alpha = to_float(row["partition_alpha"])
    batch_size = int(row["batch_size"])
    learning_rate = to_float(row["learning_rate"])
    critic_multiplier = to_float(row["critic_multiplier"])
    objective_lambda_1 = to_float(row["objective_lambda_1"])
    server_learning_rate = to_float(row["server_learning_rate"])
    rid = make_run_id(
        stage=stage,
        dataset=dataset,
        seed=seed,
        alpha=alpha,
        comm_round=comm_round,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        critic_multiplier=critic_multiplier,
        objective_lambda_1=objective_lambda_1,
        server_learning_rate=server_learning_rate,
    )
    row.update(
        {
            "run_id": rid,
            "stage": stage,
            "seed": seed,
            "comm_round": comm_round,
            "epochs": epochs,
            "output_root": str(OUTPUT_ROOT),
            "final_result_dir": str(OUTPUT_ROOT / dataset / method / f"seed_{seed}" / rid),
            "run_status": "not_started",
            "learning_rate_status": "fedogda_s_focused_v3_validation_selected",
            "weight_decay": 0.0,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
            "selected_without_test": True,
            "notes": (
                f"{SCREEN_NAME} {stage}; materialized from validation-only ranking. "
                "Test MSE remains post-selection only."
            ),
        }
    )
    return row


def materialize_confirm_seed0(
    loaded: list[LoadedRun],
    fieldnames: list[str],
    *,
    top_n: int,
) -> Path:
    rows: list[dict[str, Any]] = []
    for item in top_by_dataset(loaded, "sin", top_n):
        rows.append(confirm_row(item.manifest_row, stage="confirm_sine_seed0", seed=0, comm_round=1000, epochs=3))
    for item in top_by_dataset(loaded, "step", top_n):
        rows.append(confirm_row(item.manifest_row, stage="confirm_step_seed0", seed=0, comm_round=1500, epochs=7))
    write_manifest(CONFIRM_SEED0_MANIFEST, rows, fieldnames)
    return CONFIRM_SEED0_MANIFEST


def materialize_confirm_seeds(
    loaded: list[LoadedRun],
    fieldnames: list[str],
    *,
    top_n: int,
) -> Path:
    rows: list[dict[str, Any]] = []
    for dataset, stage in (("sin", "confirm_sine_seeds"), ("step", "confirm_step_seeds")):
        for item in top_by_dataset(loaded, dataset, top_n):
            for seed in (1, 2):
                rows.append(
                    confirm_row(
                        item.manifest_row,
                        stage=stage,
                        seed=seed,
                        comm_round=int(item.manifest_row["comm_round"]),
                        epochs=int(item.manifest_row["epochs"]),
                    )
                )
    write_manifest(CONFIRM_SEEDS_MANIFEST, rows, fieldnames)
    return CONFIRM_SEEDS_MANIFEST


def aggregate_final(seed0: list[LoadedRun], seeds: list[LoadedRun]) -> tuple[list[dict[str, Any]], dict[str, LoadedRun]]:
    groups: dict[tuple[Any, ...], list[LoadedRun]] = {}
    for item in valid_rows(seed0 + seeds):
        groups.setdefault(config_key(item.row), []).append(item)

    aggregate_rows: list[dict[str, Any]] = []
    selected_seed0: dict[str, LoadedRun] = {}
    for key, items in groups.items():
        dataset = key[0]
        rows = [item.row for item in items]
        seed_values = sorted({row["seed"] for row in rows})
        row0 = next((item for item in items if item.row["seed"] == 0), items[0])
        aggregate = {
            "dataset": dataset,
            "method": row0.row["method"],
            "stage": "final_aggregate",
            "run_id": "aggregate:" + row0.row["run_id"],
            "seed": "|".join(str(seed) for seed in seed_values),
            "learning_rate": row0.row["learning_rate"],
            "critic_multiplier": row0.row["critic_multiplier"],
            "objective_lambda_1": row0.row["objective_lambda_1"],
            "server_learning_rate": row0.row["server_learning_rate"],
            "weight_decay": row0.row["weight_decay"],
            "epochs": row0.row["epochs"],
            "comm_round": row0.row["comm_round"],
            "seed_count": len(seed_values),
            "mean_best_validation_mse": mean_or_nan([row["best_validation_mse"] for row in rows]),
            "std_best_validation_mse": pstdev_or_zero([row["best_validation_mse"] for row in rows]),
            "mean_test_mse_at_best_validation": mean_or_nan([row["test_mse_at_best_validation"] for row in rows]),
            "std_test_mse_at_best_validation": pstdev_or_zero([row["test_mse_at_best_validation"] for row in rows]),
            "mean_last_50_val_mse_std": mean_or_nan([row["last_50_val_mse_std"] for row in rows]),
            "mean_final_vs_best_validation_gap": mean_or_nan([row["final_vs_best_validation_gap"] for row in rows]),
            "representative_seed0_run_id": row0.row["run_id"],
            "representative_seed0_result_dir": row0.row["result_dir"],
        }
        aggregate_rows.append(aggregate)

    def aggregate_key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            row["mean_best_validation_mse"],
            row["mean_last_50_val_mse_std"],
            row["mean_final_vs_best_validation_gap"],
        )

    for dataset in ("sin", "step"):
        ranked = [row for row in aggregate_rows if row["dataset"] == dataset and row["seed_count"] == 3]
        if not ranked:
            ranked = [row for row in aggregate_rows if row["dataset"] == dataset]
        if not ranked:
            continue
        winner = sorted(ranked, key=aggregate_key)[0]
        match_key = (
            winner["dataset"],
            winner["learning_rate"],
            winner["critic_multiplier"],
            winner["objective_lambda_1"],
            winner["server_learning_rate"],
            winner["epochs"],
            winner["comm_round"],
        )
        selected_seed0[dataset] = next(
            (item for item in groups[match_key] if item.row["seed"] == 0),
            groups[match_key][0],
        )
    return aggregate_rows, selected_seed0


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "method",
        "stage",
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
        "representative_seed0_run_id",
        "representative_seed0_result_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["dataset"], item["mean_best_validation_mse"])):
            writer.writerow({field: row.get(field, "") for field in fields})


def load_reference(dataset: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for label, run_dir, color, linestyle, linewidth in REFERENCE_RUNS[dataset]:
        if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.npz").exists():
            continue
        metrics = read_json(run_dir / "metrics.json")
        references.append(
            {
                "label": label,
                "run_dir": rel(run_dir),
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
                "_curve": curve_payload(run_dir),
            }
        )
    return references


def plot_selected(selected: dict[str, LoadedRun], stage: str) -> list[Path]:
    png_dir = PLOT_ROOT / "png" / SCREEN_NAME
    pdf_dir = PLOT_ROOT / "pdf" / SCREEN_NAME
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        png_dir / f"{SCREEN_NAME}_{stage}_curves.png",
        pdf_dir / f"{SCREEN_NAME}_{stage}_curves.pdf",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    panels = [("sin", "Sine alpha=1.0"), ("step", "Step alpha=0.5")]
    for ax, (dataset, title) in zip(axes, panels):
        refs = load_reference(dataset)
        selected_item = selected.get(dataset)
        source = selected_item.row if selected_item is not None else refs[0]
        curve = source["_curve"]
        ax.plot(curve["x"], curve["true_g"], color="black", linewidth=2.3, label="Actual Causal Effect")
        for ref in refs:
            ref_curve = ref["_curve"]
            ax.plot(
                ref_curve["x"],
                ref_curve["pred"],
                color=ref["color"],
                linestyle=ref["linestyle"],
                linewidth=ref["linewidth"],
                label=f"{ref['label']} ({ref['test_mse_at_best_validation']:.4f})",
            )
        if selected_item is not None:
            row = selected_item.row
            selected_curve = row["_curve"]
            ax.plot(
                selected_curve["x"],
                selected_curve["pred"],
                color="#dc2626",
                linewidth=2.2,
                label=f"Selected v3 ({row['test_mse_at_best_validation']:.4f}; val {row['best_validation_mse']:.4f})",
            )
            subtitle = (
                f"lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
                f"lambda={row['objective_lambda_1']:g}, slr={row['server_learning_rate']:g}, "
                f"T={row['comm_round']}, R={row['epochs']}, seed={row['seed']}"
            )
        else:
            subtitle = "no selected run"
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.legend(fontsize=6.5)
    for output in outputs:
        fig.savefig(output, dpi=220 if output.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def write_summary(
    *,
    stage: str,
    loaded: list[LoadedRun],
    missing: list[str],
    selected: dict[str, LoadedRun],
    outputs: list[Path],
    csv_paths: list[Path],
    materialized: Path | None,
) -> Path:
    path = PLOT_ROOT / f"{SCREEN_NAME}_{stage}.md"
    lines = [
        f"# {SCREEN_NAME} {stage}",
        "",
        "Selection is validation-only: finite/non-diverged; lowest `best_validation_mse`; "
        "tie lower `last_50_val_mse_std`; tie lower `final_vs_best_validation_gap`.",
        "",
        "## Completion",
        "",
        f"- Completed candidate rows: {len(loaded)}.",
        f"- Missing/incomplete rows: {len(missing)}.",
        "",
        "## Outputs",
        "",
    ]
    for output in outputs + csv_paths:
        lines.append(f"- `{rel(output)}`")
    if materialized is not None:
        lines.append(f"- `{rel(materialized)}`")
    lines.extend(["", "## Selected", ""])
    for dataset in ("sin", "step"):
        item = selected.get(dataset)
        if item is None:
            lines.append(f"- {dataset}: no selected finite candidate.")
            continue
        row = item.row
        lines.append(
            f"- {dataset}: lr `{row['learning_rate']:g}`, cm `{row['critic_multiplier']:g}`, "
            f"lambda `{row['objective_lambda_1']:g}`, slr `{row['server_learning_rate']:g}`, "
            f"T `{row['comm_round']}`, R `{row['epochs']}`, seed `{row['seed']}`, "
            f"val `{row['best_validation_mse']:.9f}`, test@best `{row['test_mse_at_best_validation']:.9f}`, "
            f"corr `{row['curve_corr']:.6f}`."
        )
    if missing:
        lines.extend(["", "## Invalid Or Missing Rows", ""])
        for run_id in missing[:50]:
            lines.append(f"- `{run_id}`")
        if len(missing) > 50:
            lines.append(f"- ... {len(missing) - 50} more")
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze_single_manifest(
    *,
    stage: str,
    manifest_path: Path,
    allow_partial: bool,
    materialize_next: bool,
    top_screen: int,
    top_confirm: int,
) -> None:
    loaded, missing, _manifest_rows, fieldnames = load_manifest(manifest_path)
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} rows are incomplete; use --allow-partial to analyze anyway")

    candidate_rows = [item.row for item in valid_rows(loaded)]
    ranked_rows = sorted(candidate_rows, key=lambda row: (row["dataset"], selection_key(row)))
    candidate_path = CSV_DIR / f"{SCREEN_NAME}_{stage}_candidates.csv"
    write_rows_csv(candidate_path, ranked_rows)

    selected_items = {
        dataset: rows[0]
        for dataset in ("sin", "step")
        if (rows := top_by_dataset(loaded, dataset, 1))
    }
    selected_path = CSV_DIR / f"{SCREEN_NAME}_{stage}_selected.csv"
    write_rows_csv(selected_path, [item.row for item in selected_items.values()])

    materialized: Path | None = None
    if materialize_next:
        if stage == "screen":
            materialized = materialize_confirm_seed0(loaded, fieldnames, top_n=top_screen)
        elif stage == "confirm_seed0":
            materialized = materialize_confirm_seeds(loaded, fieldnames, top_n=top_confirm)
        else:
            raise SystemExit("--materialize-next is only supported for screen and confirm_seed0")

    outputs = plot_selected(selected_items, stage)
    md_path = write_summary(
        stage=stage,
        loaded=loaded,
        missing=missing,
        selected=selected_items,
        outputs=outputs,
        csv_paths=[candidate_path, selected_path],
        materialized=materialized,
    )

    print("Generated:")
    for path in outputs + [candidate_path, selected_path, md_path]:
        print(f"  {rel(path)}")
    if materialized is not None:
        print(f"  {rel(materialized)}")
    print("Selected:")
    for dataset in ("sin", "step"):
        item = selected_items.get(dataset)
        if item is None:
            print(f"  {dataset}: none")
        else:
            row = item.row
            print(
                f"  {dataset}: lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
                f"lambda={row['objective_lambda_1']:g}, slr={row['server_learning_rate']:g}, "
                f"val={row['best_validation_mse']:.9f}, test@best={row['test_mse_at_best_validation']:.9f}"
            )


def analyze_final(*, allow_partial: bool) -> None:
    seed0, missing_seed0, _rows0, _fields0 = load_manifest(CONFIRM_SEED0_MANIFEST)
    seeds, missing_seeds, _rows, _fields = load_manifest(CONFIRM_SEEDS_MANIFEST)
    missing = missing_seed0 + missing_seeds
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} rows are incomplete; use --allow-partial to analyze anyway")

    aggregate_rows, selected_seed0 = aggregate_final(seed0, seeds)
    aggregate_path = CSV_DIR / f"{SCREEN_NAME}_final_aggregate_candidates.csv"
    write_aggregate_csv(aggregate_path, aggregate_rows)
    selected_aggregate = [
        row
        for row in sorted(aggregate_rows, key=lambda item: (item["dataset"], item["mean_best_validation_mse"]))
        if row["dataset"] in selected_seed0
        and row["representative_seed0_run_id"] == selected_seed0[row["dataset"]].row["run_id"]
    ]
    selected_path = CSV_DIR / f"{SCREEN_NAME}_final_selected.csv"
    write_aggregate_csv(selected_path, selected_aggregate)
    outputs = plot_selected(selected_seed0, "final")
    md_path = write_summary(
        stage="final",
        loaded=seed0 + seeds,
        missing=missing,
        selected=selected_seed0,
        outputs=outputs,
        csv_paths=[aggregate_path, selected_path],
        materialized=None,
    )
    print("Generated:")
    for path in outputs + [aggregate_path, selected_path, md_path]:
        print(f"  {rel(path)}")
    print("Selected final configs:")
    for row in selected_aggregate:
        print(
            f"  {row['dataset']}: lr={row['learning_rate']:g}, cm={row['critic_multiplier']:g}, "
            f"lambda={row['objective_lambda_1']:g}, slr={row['server_learning_rate']:g}, "
            f"mean_val={row['mean_best_validation_mse']:.9f}, "
            f"mean_test@best={row['mean_test_mse_at_best_validation']:.9f}, seeds={row['seed']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "confirm_seed0", "final"], default="screen")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--materialize-next", action="store_true")
    parser.add_argument("--top-screen", type=int, default=6)
    parser.add_argument("--top-confirm", type=int, default=2)
    args = parser.parse_args()

    if args.stage == "screen":
        analyze_single_manifest(
            stage="screen",
            manifest_path=SCREEN_MANIFEST,
            allow_partial=bool(args.allow_partial),
            materialize_next=bool(args.materialize_next),
            top_screen=int(args.top_screen),
            top_confirm=int(args.top_confirm),
        )
    elif args.stage == "confirm_seed0":
        analyze_single_manifest(
            stage="confirm_seed0",
            manifest_path=CONFIRM_SEED0_MANIFEST,
            allow_partial=bool(args.allow_partial),
            materialize_next=bool(args.materialize_next),
            top_screen=int(args.top_screen),
            top_confirm=int(args.top_confirm),
        )
    else:
        analyze_final(allow_partial=bool(args.allow_partial))


if __name__ == "__main__":
    main()
