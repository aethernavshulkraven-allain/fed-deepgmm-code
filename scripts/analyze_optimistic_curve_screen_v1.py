#!/usr/bin/env python3
"""Analyze and plot the optimistic curve-fitting tuning screen.

Selection is validation-only. Test and curve diagnostics are reported only
after a candidate has been selected.
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
DEFAULT_SCREEN_NAME = "optimistic_curve_screen_v1"
DEFAULT_SCREEN_DIR = ROOT / "experiments" / "curve_fitting_tuning" / DEFAULT_SCREEN_NAME
PLOT_ROOT = ROOT / "experiments" / "curve_fitting_plots"
CSV_DIR = PLOT_ROOT / "csv"

STEP_OGDA_REFERENCE = (
    ROOT
    / "results"
    / "curve_fitting_tuning"
    / "step_reference_repro_v1"
    / "step"
    / "fedogda_s"
    / "seed_0"
    / "curvefit_step_reference_repro_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p01_wd0p02_cm15_slr1p5"
)


@dataclass(frozen=True)
class ReferenceCurve:
    panel: str
    label: str
    run_dir: Path
    color: str
    linestyle: Any
    linewidth: float


REFERENCES = [
    ReferenceCurve(
        panel="sine",
        label="FedGDA-S reference",
        run_dir=ROOT
        / "results"
        / "rerun_protocol_v1"
        / "sin"
        / "fedgda_s"
        / "seed_0"
        / "rerun_protocol_v1_sin_fedgda_s_seed0_alpha1p0",
        color="#2563eb",
        linestyle="--",
        linewidth=1.8,
    ),
    ReferenceCurve(
        panel="sine",
        label="Previous FedOGDA-D tuned",
        run_dir=ROOT
        / "results"
        / "sine_fedogda_tuning"
        / "stage_A2_from_A1_mini"
        / "sin"
        / "fedogda_d"
        / "seed_0"
        / "stage_A2_from_A1_mini_sin_fedogda_d_seed0_alpha1p0_R3_cm15_slr1.5_glr0p002",
        color="#991b1b",
        linestyle="-.",
        linewidth=1.5,
    ),
    ReferenceCurve(
        panel="step",
        label="FedGDA-S reference",
        run_dir=ROOT
        / "results"
        / "curve_fitting_tuning"
        / "step_reference_repro_v1"
        / "step"
        / "fedgda_s"
        / "seed_0"
        / "curvefit_step_reference_repro_fedgda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p03_wd0p02_cm15_slr1p5",
        color="#15803d",
        linestyle="--",
        linewidth=1.8,
    ),
    ReferenceCurve(
        panel="step",
        label="FedOGDA-S old ref",
        run_dir=STEP_OGDA_REFERENCE,
        color="#f97316",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.5,
    ),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def pstdev_or_zero(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


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
    predictions_path = run_dir / "predictions.npz"
    with np.load(predictions_path) as data:
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        true_g = np.asarray(data["true_g"], dtype=float).reshape(-1)
        pred = np.asarray(data["best_validation_prediction"], dtype=float).reshape(-1)
    if not (x.size == true_g.size == pred.size):
        raise ValueError(f"shape mismatch in {predictions_path}")
    if not (np.isfinite(x).all() and np.isfinite(true_g).all() and np.isfinite(pred).all()):
        raise ValueError(f"non-finite curve values in {predictions_path}")
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


def load_run(
    *,
    run_id: str,
    dataset: str,
    method: str,
    result_dir: Path,
    learning_rate: float,
    weight_decay: float,
    objective_lambda_1: float,
    epochs: int,
    comm_round: int,
    source: str,
) -> dict[str, Any] | None:
    metrics_path = result_dir / "metrics.json"
    mse_path = result_dir / "mse_by_round.csv"
    predictions_path = result_dir / "predictions.npz"
    if not (metrics_path.exists() and mse_path.exists() and predictions_path.exists()):
        return None
    metrics = read_json(metrics_path)
    last_50_val_mse_std, history_diverged = last_50_stability(mse_path)
    curve = curve_payload(result_dir)
    best_validation_mse = to_float(metrics["best_validation_mse"])
    final_validation_mse = to_float(metrics["final_validation_mse"])
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "method": method,
        "source": source,
        "seed": int(metrics.get("seed", 0)),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "objective_lambda_1": objective_lambda_1,
        "epochs": epochs,
        "comm_round": comm_round,
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
    return row


def load_screen_rows(screen_dir: Path, screen_name: str, allow_partial: bool) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for manifest_row in read_csv(screen_dir / "manifest.csv"):
        result_dir = ROOT / manifest_row["final_result_dir"]
        row = load_run(
            run_id=manifest_row["run_id"],
            dataset=manifest_row["dataset"],
            method=manifest_row["method"],
            result_dir=result_dir,
            learning_rate=to_float(manifest_row["learning_rate"]),
            weight_decay=to_float(manifest_row["weight_decay"]),
            objective_lambda_1=to_float(manifest_row.get("objective_lambda_1", 0.1)),
            epochs=int(manifest_row["epochs"]),
            comm_round=int(manifest_row["comm_round"]),
            source=screen_name,
        )
        if row is None:
            missing.append(manifest_row["run_id"])
            continue
        rows.append(row)
    if missing and not allow_partial:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest runs are incomplete; first missing: {preview}")
    return rows, missing


def load_step_reference_candidate() -> dict[str, Any]:
    row = load_run(
        run_id="curvefit_step_reference_repro_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p01_wd0p02_cm15_slr1p5",
        dataset="step",
        method="fedogda_s",
        result_dir=STEP_OGDA_REFERENCE,
        learning_rate=0.01,
        weight_decay=0.02,
        objective_lambda_1=0.1,
        epochs=7,
        comm_round=1500,
        source="step_reference_repro_v1_existing_reference",
    )
    if row is None:
        raise FileNotFoundError(STEP_OGDA_REFERENCE)
    return row


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        row["best_validation_mse"],
        row["last_50_val_mse_std"],
        row["final_vs_best_validation_gap"],
    )


def select_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for dataset in ("sin", "step"):
        candidates = [
            row
            for row in rows
            if row["dataset"] == dataset and not row["diverged"] and math.isfinite(row["best_validation_mse"])
        ]
        if candidates:
            selected[dataset] = sorted(candidates, key=selection_key)[0]
    return selected


def load_reference_curve(ref: ReferenceCurve) -> dict[str, Any]:
    curve = curve_payload(ref.run_dir)
    metrics = read_json(ref.run_dir / "metrics.json")
    return {
        "label": ref.label,
        "panel": ref.panel,
        "run_dir": rel(ref.run_dir),
        "color": ref.color,
        "linestyle": ref.linestyle,
        "linewidth": ref.linewidth,
        "test_mse_at_best_validation": to_float(metrics["test_mse_at_best_validation"]),
        "best_validation_mse": to_float(metrics["best_validation_mse"]),
        "_curve": curve,
    }


def plot(selected: dict[str, dict[str, Any]], screen_name: str) -> list[Path]:
    references = [load_reference_curve(ref) for ref in REFERENCES if ref.run_dir.exists()]
    png_dir = PLOT_ROOT / "png" / screen_name
    pdf_dir = PLOT_ROOT / "pdf" / screen_name
    outputs = [
        png_dir / f"{screen_name}_curves.png",
        pdf_dir / f"{screen_name}_curves.pdf",
    ]
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    panels = [("sin", "sine", "Sine alpha=1.0, seed 0"), ("step", "step", "Step alpha=0.5, seed 0")]
    for ax, (dataset, panel, title) in zip(axes, panels):
        panel_refs = [ref for ref in references if ref["panel"] == panel]
        selected_row = selected.get(dataset)
        curve_source = selected_row if selected_row is not None else panel_refs[0]
        curve = curve_source["_curve"]
        ax.plot(curve["x"], curve["true_g"], color="black", linewidth=2.3, label="Actual Causal Effect")

        for ref in panel_refs:
            ref_curve = ref["_curve"]
            label = f"{ref['label']} ({ref['test_mse_at_best_validation']:.4f})"
            ax.plot(
                ref_curve["x"],
                ref_curve["pred"],
                color=ref["color"],
                linestyle=ref["linestyle"],
                linewidth=ref["linewidth"],
                label=label,
            )

        if selected_row is not None:
            selected_curve = selected_row["_curve"]
            label = (
                f"Selected FedOGDA-S ({selected_row['test_mse_at_best_validation']:.4f}; "
                f"val {selected_row['best_validation_mse']:.4f})"
            )
            ax.plot(
                selected_curve["x"],
                selected_curve["pred"],
                color="#dc2626",
                linestyle="-",
                linewidth=2.2,
                label=label,
            )
            subtitle = (
                f"selected lr={selected_row['learning_rate']:g}, wd={selected_row['weight_decay']:g}, "
                f"lambda={selected_row['objective_lambda_1']:g}, R={selected_row['epochs']}"
            )
        else:
            subtitle = "no completed selected run yet"

        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.legend(fontsize=6.7)

    for path in outputs:
        fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_csvs(rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]], screen_name: str) -> tuple[Path, Path]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = CSV_DIR / f"{screen_name}_candidates.csv"
    selected_path = CSV_DIR / f"{screen_name}_selected.csv"
    fields = [
        "dataset",
        "method",
        "source",
        "run_id",
        "seed",
        "learning_rate",
        "weight_decay",
        "objective_lambda_1",
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
    with candidate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["dataset"], selection_key(item))):
            writer.writerow({field: public_row(row).get(field, "") for field in fields})
    with selected_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in ("sin", "step"):
            if dataset in selected:
                writer.writerow({field: public_row(selected[dataset]).get(field, "") for field in fields})
    return candidate_path, selected_path


def write_markdown(
    *,
    screen_name: str,
    rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    missing: list[str],
    outputs: list[Path],
    candidate_path: Path,
    selected_path: Path,
    expected_sin: int,
    expected_step: int,
) -> Path:
    path = PLOT_ROOT / f"{screen_name}.md"
    completed_by_dataset = {
        dataset: sum(row["dataset"] == dataset and row["source"] == screen_name for row in rows)
        for dataset in ("sin", "step")
    }
    lines = [
        f"# {screen_name}",
        "",
        "Selection rule: finite/non-diverged candidates only; lowest `best_validation_mse`; "
        "tie-break lower `last_50_val_mse_std`; tie-break lower `final_vs_best_validation_gap`.",
        "",
        "Test MSE and curve diagnostics below are post-selection readouts.",
        "",
        "## Outputs",
        "",
        *[f"- `{rel(output)}`" for output in outputs],
        f"- `{rel(candidate_path)}`",
        f"- `{rel(selected_path)}`",
        "",
        "## Completion",
        "",
        f"- Sine screen rows completed: {completed_by_dataset['sin']}/{expected_sin}.",
        f"- Step screen rows completed: {completed_by_dataset['step']}/{expected_step}, plus the existing Step OGDA reference candidate.",
        f"- Missing/incomplete manifest rows: {len(missing)}.",
        "",
        "## Validation-Selected Readout",
        "",
    ]
    for dataset in ("sin", "step"):
        row = selected.get(dataset)
        if row is None:
            lines.append(f"- {dataset}: no completed non-diverged candidate yet.")
            continue
        lines.append(
            f"- {dataset}: lr `{row['learning_rate']:g}`, wd `{row['weight_decay']:g}`, "
            f"lambda `{row['objective_lambda_1']:g}`, R `{row['epochs']}`, "
            f"best val `{row['best_validation_mse']:.9f}`, "
            f"test@best `{row['test_mse_at_best_validation']:.9f}`, "
            f"curve corr `{row['curve_corr']:.6f}`."
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The Sine panel includes the old tuned deterministic OGDA line for visual context; it is not part of this screen's stochastic OGDA-S selection pool.",
            "- The Step selection pool includes the existing reproduced FedOGDA-S reference row (`lr=0.01`, `wd=0.02`) because it was intentionally not relaunched.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true", help="Analyze completed rows without requiring the full screen.")
    parser.add_argument("--screen-name", default=DEFAULT_SCREEN_NAME)
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    parser.add_argument("--expected-sin", type=int, default=12)
    parser.add_argument("--expected-step", type=int, default=8)
    args = parser.parse_args()

    screen_dir = args.screen_dir
    if not screen_dir.is_absolute():
        screen_dir = ROOT / screen_dir
    screen_rows, missing = load_screen_rows(
        screen_dir=screen_dir,
        screen_name=args.screen_name,
        allow_partial=args.allow_partial,
    )
    all_rows = screen_rows + [load_step_reference_candidate()]
    selected = select_rows(all_rows)
    outputs = plot(selected, args.screen_name)
    candidate_path, selected_path = write_csvs(all_rows, selected, args.screen_name)
    md_path = write_markdown(
        screen_name=args.screen_name,
        rows=all_rows,
        selected=selected,
        missing=missing,
        outputs=outputs,
        candidate_path=candidate_path,
        selected_path=selected_path,
        expected_sin=args.expected_sin,
        expected_step=args.expected_step,
    )

    print("Generated:")
    for output in outputs:
        print(f"  {rel(output)}")
    print(f"  {rel(candidate_path)}")
    print(f"  {rel(selected_path)}")
    print(f"  {rel(md_path)}")
    print("Selected:")
    for dataset in ("sin", "step"):
        row = selected.get(dataset)
        if row is None:
            print(f"  {dataset}: none")
        else:
            print(
                f"  {dataset}: lr={row['learning_rate']:g}, wd={row['weight_decay']:g}, "
                f"lambda={row['objective_lambda_1']:g}, R={row['epochs']}, "
                f"val={row['best_validation_mse']:.9f}, "
                f"test@best={row['test_mse_at_best_validation']:.9f}"
            )


if __name__ == "__main__":
    main()
