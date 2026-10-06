#!/usr/bin/env python3
"""Generate focused Sine/Step curve plots from validation-selected artifacts.

This is reporting-only: it reads completed result artifacts and writes plots
under experiments/curve_fitting_plots/.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "curve_fitting_plots"
PNG_DIR = OUT / "png" / "better_sine_step"
PDF_DIR = OUT / "pdf" / "better_sine_step"
CSV_DIR = OUT / "csv"


@dataclass(frozen=True)
class RunSpec:
    panel: str
    label: str
    source: str
    run_dir: Path
    seed: int
    method: str
    notes: str = ""


SINE_FEDGDA = [
    RunSpec(
        panel="sine",
        label="FedDeepGMM-GDA",
        source="paired FedGDA-D baseline, alpha=1.0",
        run_dir=ROOT
        / "results"
        / "rerun_protocol_v1"
        / "sin"
        / "fedgda_d"
        / f"seed_{seed}"
        / f"rerun_protocol_v1_sin_fedgda_d_seed{seed}_alpha1p0",
        seed=seed,
        method="fedgda_d",
    )
    for seed in (0, 1, 2)
]

SINE_FEDOGDA = [
    RunSpec(
        panel="sine",
        label="FedDeepGMM-OGDA-D tuned",
        source="validation-locked Sine A2-lite FedOGDA-D, alpha=1.0",
        run_dir=ROOT
        / "results"
        / "sine_fedogda_tuning"
        / "stage_A2_from_A1_mini"
        / "sin"
        / "fedogda_d"
        / f"seed_{seed}"
        / f"stage_A2_from_A1_mini_sin_fedogda_d_seed{seed}_alpha1p0_R3_cm15_slr1.5_glr0p002",
        seed=seed,
        method="fedogda_d",
    )
    for seed in (0, 1, 2)
]

STEP_RUNS = [
    RunSpec(
        panel="step",
        label="FedDeepGMM-SGDA",
        source="Step reference-recipe reproduction, minibatch",
        run_dir=ROOT
        / "results"
        / "curve_fitting_tuning"
        / "step_reference_repro_v1"
        / "step"
        / "fedgda_s"
        / "seed_0"
        / "curvefit_step_reference_repro_fedgda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p03_wd0p02_cm15_slr1p5",
        seed=0,
        method="fedgda_s",
        notes="lowest best_validation_mse in the 4-row Step reproduction",
    ),
    RunSpec(
        panel="step",
        label="FedDeepGMM-OGDA-S",
        source="Step reference-recipe reproduction, minibatch",
        run_dir=ROOT
        / "results"
        / "curve_fitting_tuning"
        / "step_reference_repro_v1"
        / "step"
        / "fedogda_s"
        / "seed_0"
        / "curvefit_step_reference_repro_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p01_wd0p02_cm15_slr1p5",
        seed=0,
        method="fedogda_s",
    ),
    RunSpec(
        panel="step",
        label="FedDeepGMM-GDA partial-FB",
        source="Step reference-recipe reproduction, partial-client full batch",
        run_dir=ROOT
        / "results"
        / "curve_fitting_tuning"
        / "step_reference_repro_v1"
        / "step"
        / "fedgda_d"
        / "seed_0"
        / "curvefit_step_reference_repro_fedgda_d_partialfb_seed0_alpha0p5_T1500_R7_batch0_glr0p03_wd0p02_cm15_slr1p5",
        seed=0,
        method="fedgda_d",
        notes="partial-client full-batch row; not standard full-participation D",
    ),
    RunSpec(
        panel="step",
        label="FedDeepGMM-OGDA-D partial-FB",
        source="Step reference-recipe reproduction, partial-client full batch",
        run_dir=ROOT
        / "results"
        / "curve_fitting_tuning"
        / "step_reference_repro_v1"
        / "step"
        / "fedogda_d"
        / "seed_0"
        / "curvefit_step_reference_repro_fedogda_d_partialfb_seed0_alpha0p5_T1500_R7_batch0_glr0p01_wd0p02_cm15_slr1p5",
        seed=0,
        method="fedogda_d",
        notes="partial-client full-batch row; not standard full-participation D",
    ),
]


STYLE = {
    "Actual Causal Effect": {"color": "black", "linestyle": "-", "linewidth": 2.4},
    "FedDeepGMM-GDA": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.9},
    "FedDeepGMM-OGDA-D tuned": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.0},
    "FedDeepGMM-SGDA": {"color": "#2ca02c", "linestyle": ":", "linewidth": 2.2},
    "FedDeepGMM-OGDA-S": {"color": "#ff7f0e", "linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.9},
    "FedDeepGMM-GDA partial-FB": {"color": "#2563eb", "linestyle": "--", "linewidth": 1.6},
    "FedDeepGMM-OGDA-D partial-FB": {"color": "#b91c1c", "linestyle": "-.", "linewidth": 1.7},
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def load_curve(spec: RunSpec) -> dict[str, Any]:
    pred_path = spec.run_dir / "predictions.npz"
    metrics_path = spec.run_dir / "metrics.json"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path, allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        true_g = np.asarray(data["true_g"], dtype=float).reshape(-1)
        pred = np.asarray(data["best_validation_prediction"], dtype=float).reshape(-1)
    if not (x.size == true_g.size == pred.size):
        raise ValueError(f"shape mismatch in {pred_path}")
    if not (np.isfinite(x).all() and np.isfinite(true_g).all() and np.isfinite(pred).all()):
        raise ValueError(f"non-finite curve values in {pred_path}")
    order = np.argsort(x)
    x = x[order]
    true_g = true_g[order]
    pred = pred[order]
    metrics = read_json(metrics_path)
    diff = pred - true_g
    return {
        "spec": spec,
        "x": x,
        "true_g": true_g,
        "pred": pred,
        "curve_mse": float(np.mean(diff**2)),
        "curve_mae": float(np.mean(np.abs(diff))),
        "curve_max_abs_error": float(np.max(np.abs(diff))),
        "best_validation_mse": metrics.get("best_validation_mse", ""),
        "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation", ""),
        "best_validation_round": metrics.get("best_validation_round", ""),
        "final_test_mse": metrics.get("final_test_mse", ""),
        "run_dir": rel(spec.run_dir),
    }


def curves_align(curves: list[dict[str, Any]]) -> bool:
    if not curves:
        return False
    ref_x = curves[0]["x"]
    ref_true = curves[0]["true_g"]
    for curve in curves[1:]:
        if curve["x"].shape != ref_x.shape:
            return False
        if not np.allclose(curve["x"], ref_x, rtol=0, atol=1e-12):
            return False
        if not np.allclose(curve["true_g"], ref_true, rtol=0, atol=1e-12):
            return False
    return True


def mean_curve(curves: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not curves_align(curves):
        raise ValueError(f"{label} curves do not share x/true_g grids")
    ref = curves[0]
    pred = np.vstack([curve["pred"] for curve in curves]).mean(axis=0)
    diff = pred - ref["true_g"]
    return {
        "label": label,
        "x": ref["x"],
        "true_g": ref["true_g"],
        "pred": pred,
        "curve_mse": float(np.mean(diff**2)),
        "curve_mae": float(np.mean(np.abs(diff))),
        "curve_max_abs_error": float(np.max(np.abs(diff))),
        "mean_test_mse_at_best_validation": float(
            np.mean([float(curve["test_mse_at_best_validation"]) for curve in curves])
        ),
        "mean_best_validation_mse": float(np.mean([float(curve["best_validation_mse"]) for curve in curves])),
    }


def write_metrics(rows: list[dict[str, Any]], sine_means: list[dict[str, Any]]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_DIR / "better_sine_step_curve_metrics.csv"
    fields = [
        "panel",
        "label",
        "source",
        "method",
        "seed",
        "curve_mse",
        "curve_mae",
        "curve_max_abs_error",
        "best_validation_mse",
        "test_mse_at_best_validation",
        "best_validation_round",
        "final_test_mse",
        "run_dir",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            spec: RunSpec = row["spec"]
            writer.writerow(
                {
                    "panel": spec.panel,
                    "label": spec.label,
                    "source": spec.source,
                    "method": spec.method,
                    "seed": spec.seed,
                    "curve_mse": row["curve_mse"],
                    "curve_mae": row["curve_mae"],
                    "curve_max_abs_error": row["curve_max_abs_error"],
                    "best_validation_mse": row["best_validation_mse"],
                    "test_mse_at_best_validation": row["test_mse_at_best_validation"],
                    "best_validation_round": row["best_validation_round"],
                    "final_test_mse": row["final_test_mse"],
                    "run_dir": row["run_dir"],
                    "notes": spec.notes,
                }
            )
        for row in sine_means:
            writer.writerow(
                {
                    "panel": "sine",
                    "label": row["label"],
                    "source": "mean across seeds 0,1,2",
                    "method": "aggregate",
                    "seed": "mean",
                    "curve_mse": row["curve_mse"],
                    "curve_mae": row["curve_mae"],
                    "curve_max_abs_error": row["curve_max_abs_error"],
                    "best_validation_mse": row["mean_best_validation_mse"],
                    "test_mse_at_best_validation": row["mean_test_mse_at_best_validation"],
                    "best_validation_round": "",
                    "final_test_mse": "",
                    "run_dir": "",
                    "notes": "aggregate row",
                }
            )


def plot_outputs(sine_gda: dict[str, Any], sine_ogda: dict[str, Any], step_curves: list[dict[str, Any]]) -> list[Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PNG_DIR / "better_sine_step_curves.png",
        PDF_DIR / "better_sine_step_curves.pdf",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        sine_gda["x"],
        sine_gda["true_g"],
        label="Actual Causal Effect",
        **STYLE["Actual Causal Effect"],
    )
    for curve in (sine_gda, sine_ogda):
        ax.plot(curve["x"], curve["pred"], label=curve["label"], **STYLE[curve["label"]])
    ax.set_title(
        "Sine alpha=1.0, mean of seeds 0-2\n"
        f"test@best: OGDA-D {sine_ogda['mean_test_mse_at_best_validation']:.4f} vs "
        f"GDA-D {sine_gda['mean_test_mse_at_best_validation']:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(fontsize=7)

    if not curves_align(step_curves):
        raise ValueError("Step reproduction curves do not share x/true_g grids")
    ax = axes[1]
    ref = step_curves[0]
    ax.plot(
        ref["x"],
        ref["true_g"],
        label="Actual Causal Effect",
        **STYLE["Actual Causal Effect"],
    )
    for curve in step_curves:
        label = curve["spec"].label
        test_value = curve["test_mse_at_best_validation"]
        display = f"{label} ({float(test_value):.4f})" if test_value != "" else label
        ax.plot(curve["x"], curve["pred"], label=display, **STYLE[label])
    ax.set_title(
        "Step alpha=0.5, seed 0 reference-recipe reproduction\n"
        "legend values are test@best after validation-only selection",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(fontsize=6.5)

    for path in outputs:
        fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def write_markdown(outputs: list[Path], rows: list[dict[str, Any]], sine_means: list[dict[str, Any]]) -> Path:
    path = OUT / "better_sine_step_curves.md"
    step_sorted = sorted(
        [row for row in rows if row["spec"].panel == "step"],
        key=lambda row: float(row["best_validation_mse"]),
    )
    lines = [
        "# Better Sine/Step Curves",
        "",
        "Generated from saved `best_validation_prediction` arrays. Test MSE values are post-selection readouts.",
        "",
        "## Outputs",
        "",
        *[f"- `{rel(output)}`" for output in outputs],
        f"- `experiments/curve_fitting_plots/csv/better_sine_step_curve_metrics.csv`",
        "",
        "## Completion Status",
        "",
        "- Sine A2-lite: seed 0 plus continuation seeds 1 and 2 are available; continuation manifest is 2/2 passed.",
        "- Step reference-recipe reproduction: 4/4 rows passed.",
        "",
        "## Sine Mean Readout",
        "",
    ]
    for row in sine_means:
        lines.append(
            f"- {row['label']}: mean test@best `{row['mean_test_mse_at_best_validation']:.9f}`, "
            f"curve MSE `{row['curve_mse']:.9f}`."
        )
    lines.extend(["", "## Step Seed-0 Ranking By Validation", ""])
    for row in step_sorted:
        spec: RunSpec = row["spec"]
        lines.append(
            f"- {spec.label}: best val `{float(row['best_validation_mse']):.9f}`, "
            f"test@best `{float(row['test_mse_at_best_validation']):.9f}`, "
            f"best round `{row['best_validation_round']}`."
        )
    lines.extend(
        [
            "",
            "Caveat: the Step partial-FB rows use partial clients, so they are labeled separately from standard full-participation deterministic rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    sine_gda_curves = [load_curve(spec) for spec in SINE_FEDGDA]
    sine_ogda_curves = [load_curve(spec) for spec in SINE_FEDOGDA]
    step_curves = [load_curve(spec) for spec in STEP_RUNS]
    sine_gda = mean_curve(sine_gda_curves, "FedDeepGMM-GDA")
    sine_ogda = mean_curve(sine_ogda_curves, "FedDeepGMM-OGDA-D tuned")
    all_rows = sine_gda_curves + sine_ogda_curves + step_curves
    sine_means = [sine_gda, sine_ogda]
    outputs = plot_outputs(sine_gda, sine_ogda, step_curves)
    write_metrics(all_rows, sine_means)
    md_path = write_markdown(outputs, all_rows, sine_means)
    print("Generated:")
    for output in outputs:
        print(f"  {rel(output)}")
    print(f"  {rel(md_path)}")
    print("  experiments/curve_fitting_plots/csv/better_sine_step_curve_metrics.csv")


if __name__ == "__main__":
    main()
