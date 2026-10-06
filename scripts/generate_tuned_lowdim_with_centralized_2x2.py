#!/usr/bin/env python3
"""Generate a tuned low-dimensional 2x2 curve summary with centralized overlays.

This is reporting-only. It reads completed `predictions.npz` and `metrics.json`
artifacts and writes plot/report files under `experiments/curve_fitting_plots/`.
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
PNG_DIR = OUT / "png" / "coauthor_summary"
PDF_DIR = OUT / "pdf" / "coauthor_summary"
CSV_DIR = OUT / "csv"

PLOT_STEM = "lowdim_tuned_with_centralized_summary_2x2_paper_v1"

DATASET_LABEL = {
    "abs": "Absolute",
    "step": "Step",
    "linear": "Linear",
    "sin": "Sine",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Each learned curve has both a distinct line style and marker so the figure
# remains readable in grayscale and for readers with color-vision deficiencies.
# Markers are applied sparsely in `plot`.
STYLE = {
    "true": {
        "color": "#111111",
        "linestyle": "-",
        "linewidth": 2.25,
        "zorder": 8,
    },
    "fedgda_d": {
        "color": "#0072B2",
        "linestyle": "--",
        "linewidth": 1.65,
        "marker": "o",
    },
    "fedogda_d": {
        "color": "#D55E00",
        "linestyle": "-.",
        "linewidth": 1.75,
        "marker": "s",
    },
    "fedgda_s": {
        "color": "#009E73",
        "linestyle": "--",
        "linewidth": 1.65,
        "marker": "^",
    },
    "fedogda_s": {
        "color": "#E69F00",
        "linestyle": "-.",
        "linewidth": 1.75,
        "marker": "D",
    },
    "central_gda": {
        "color": "#CC79A7",
        "linestyle": ":",
        "linewidth": 1.45,
        "marker": "v",
    },
    "central_sgda": {
        "color": "#6A3D9A",
        "linestyle": (0, (3, 1, 1, 1)),
        "linewidth": 1.45,
        "marker": "P",
    },
    "central_oadam": {
        "color": "#56B4E9",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.55,
        "marker": "X",
    },
}

LEGEND_ORDER = [
    "Actual causal effect",
    "DeepGMM-OAdam",
    "DeepGMM-GDA",
    "DeepGMM-SGDA",
    "FedDeepGMM-GDA",
    "FedDeepGMM-SGDA",
    "FedDeepGMM-OGDA-D",
    "FedDeepGMM-OGDA-S",
]


@dataclass(frozen=True)
class Curve:
    dataset: str
    label: str
    style_key: str
    x: np.ndarray
    true_g: np.ndarray
    pred: np.ndarray
    source: str
    method: str
    seed: str
    run_dirs: tuple[str, ...]
    best_validation_mse: float
    test_mse_at_best_validation: float
    curve_mse: float
    curve_mae: float


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def alpha_slug(alpha: str) -> str:
    return alpha.replace(".", "p")


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def metric_mean(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else float("nan")


def load_curve(
    *,
    run_dir: Path,
    dataset: str,
    label: str,
    style_key: str,
    source: str,
    method: str,
    seed: int | str,
) -> Curve:
    pred_path = run_dir / "predictions.npz"
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
    metrics = read_json(run_dir / "metrics.json")
    diff = pred - true_g
    return Curve(
        dataset=dataset,
        label=label,
        style_key=style_key,
        x=x,
        true_g=true_g,
        pred=pred,
        source=source,
        method=method,
        seed=str(seed),
        run_dirs=(rel(run_dir),),
        best_validation_mse=finite_float(metrics.get("best_validation_mse")),
        test_mse_at_best_validation=finite_float(metrics.get("test_mse_at_best_validation")),
        curve_mse=float(np.mean(diff**2)),
        curve_mae=float(np.mean(np.abs(diff))),
    )


def curves_align(curves: list[Curve]) -> bool:
    if not curves:
        return False
    ref = curves[0]
    for curve in curves[1:]:
        if curve.x.shape != ref.x.shape or curve.true_g.shape != ref.true_g.shape:
            return False
        if not np.allclose(curve.x, ref.x, rtol=0, atol=1e-12):
            return False
        if not np.allclose(curve.true_g, ref.true_g, rtol=0, atol=1e-12):
            return False
    return True


def mean_curve(curves: list[Curve], *, label: str, style_key: str, source: str, method: str) -> Curve:
    if not curves_align(curves):
        raise ValueError(f"{label} curves do not share x/true_g grids")
    ref = curves[0]
    pred = np.vstack([curve.pred for curve in curves]).mean(axis=0)
    diff = pred - ref.true_g
    return Curve(
        dataset=ref.dataset,
        label=label,
        style_key=style_key,
        x=ref.x,
        true_g=ref.true_g,
        pred=pred,
        source=source,
        method=method,
        seed="mean:" + "|".join(curve.seed for curve in curves),
        run_dirs=tuple(run_dir for curve in curves for run_dir in curve.run_dirs),
        best_validation_mse=metric_mean([curve.best_validation_mse for curve in curves]),
        test_mse_at_best_validation=metric_mean([curve.test_mse_at_best_validation for curve in curves]),
        curve_mse=float(np.mean(diff**2)),
        curve_mae=float(np.mean(np.abs(diff))),
    )


def base_federated_mean(dataset: str, method: str, alpha: str, label: str, style_key: str) -> Curve:
    curves = []
    for seed in (0, 1, 2):
        run_dir = (
            ROOT
            / "results"
            / "rerun_protocol_v1"
            / dataset
            / method
            / f"seed_{seed}"
            / f"rerun_protocol_v1_{dataset}_{method}_seed{seed}_alpha{alpha_slug(alpha)}"
        )
        curves.append(
            load_curve(
                run_dir=run_dir,
                dataset=dataset,
                label=label,
                style_key=style_key,
                source=f"base_sweep alpha={alpha}",
                method=method,
                seed=seed,
            )
        )
    return mean_curve(
        curves,
        label=label,
        style_key=style_key,
        source=f"mean base_sweep alpha={alpha}, seeds 0-2",
        method=method,
    )


def centralized_mean(dataset: str, method: str, label: str, style_key: str) -> Curve:
    if method in {"gda", "sgda"}:
        path = ROOT / "experiments" / "centralized_baselines" / "centralized_c5_final_gda_sgda_tuned_results.csv"
        source = "centralized C5 validation-tuned GDA/SGDA"
        rows = [row for row in read_csv(path) if row["dataset"] == dataset and row["method"] == method]
        dir_key = "output_dir"
    else:
        path = ROOT / "experiments" / "centralized_baselines" / "centralized_c3_full_run_results.csv"
        source = "centralized C3 OAdam"
        rows = [
            row
            for row in read_csv(path)
            if row["dataset"] == dataset
            and row["method"] == method
            and row.get("status") == "completed_valid"
            and row.get("validation_status") == "pass"
        ]
        dir_key = "output_dir"
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if len(rows) != 3:
        raise ValueError(f"expected 3 centralized {dataset}/{method} rows, found {len(rows)}")
    curves = [
        load_curve(
            run_dir=ROOT / row[dir_key],
            dataset=dataset,
            label=label,
            style_key=style_key,
            source=source,
            method=method,
            seed=row["seed"],
        )
        for row in rows
    ]
    return mean_curve(curves, label=label, style_key=style_key, source=f"{source}, mean seeds 0-2", method=method)


def sine_tuned_fedogda_s_mean() -> Curve:
    final = read_csv(CSV_DIR / "fedogda_s_sine_fast_v4_final_selected.csv")[0]
    rows = []
    for row in read_csv(ROOT / "experiments" / "curve_fitting_tuning" / "fedogda_s_sine_fast_v4" / "stage_c_manifest.csv"):
        if (
            row["method"] == "fedogda_s"
            and row["dataset"] == "sin"
            and float(row["learning_rate"]) == float(final["learning_rate"])
            and float(row["critic_multiplier"]) == float(final["critic_multiplier"])
            and float(row["objective_lambda_1"]) == float(final["objective_lambda_1"])
            and float(row["server_learning_rate"]) == float(final["server_learning_rate"])
        ):
            rows.append(row)
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [0, 1, 2]:
        raise ValueError("Sine v4 final config does not have seed 0,1,2 Stage C rows")
    curves = [
        load_curve(
            run_dir=ROOT / row["final_result_dir"],
            dataset="sin",
            label="FedDeepGMM-OGDA-S",
            style_key="fedogda_s",
            source="fedogda_s_sine_fast_v4 final validation-selected config",
            method="fedogda_s",
            seed=row["seed"],
        )
        for row in rows
    ]
    return mean_curve(
        curves,
        label="FedDeepGMM-OGDA-S",
        style_key="fedogda_s",
        source="fedogda_s_sine_fast_v4 final config, mean seeds 0-2",
        method="fedogda_s",
    )


def step_tuned_fedogda_s_mean() -> Curve:
    final = read_csv(CSV_DIR / "fedogda_s_step_fast_v5_final_selected.csv")[0]
    rows = []
    for row in read_csv(CSV_DIR / "fedogda_s_step_fast_v5_all_candidates.csv"):
        if (
            row["method"] == "fedogda_s"
            and row["dataset"] == "step"
            and float(row["learning_rate"]) == float(final["learning_rate"])
            and float(row["critic_multiplier"]) == float(final["critic_multiplier"])
            and float(row["objective_lambda_1"]) == float(final["objective_lambda_1"])
            and float(row["server_learning_rate"]) == float(final["server_learning_rate"])
        ):
            rows.append(row)
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [0, 1, 2]:
        raise ValueError("Step v5 final config does not have seed 0,1,2 rows")
    curves = [
        load_curve(
            run_dir=ROOT / row["result_dir"],
            dataset="step",
            label="FedDeepGMM-OGDA-S",
            style_key="fedogda_s",
            source="fedogda_s_step_fast_v5 final validation-selected config",
            method="fedogda_s",
            seed=row["seed"],
        )
        for row in rows
    ]
    return mean_curve(
        curves,
        label="FedDeepGMM-OGDA-S",
        style_key="fedogda_s",
        source="fedogda_s_step_fast_v5 final config, mean seeds 0-2",
        method="fedogda_s",
    )


def step_fedgda_s_reference() -> Curve:
    run_dir = (
        ROOT
        / "results"
        / "curve_fitting_tuning"
        / "step_reference_repro_v1"
        / "step"
        / "fedgda_s"
        / "seed_0"
        / "curvefit_step_reference_repro_fedgda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p03_wd0p02_cm15_slr1p5"
    )
    return load_curve(
        run_dir=run_dir,
        dataset="step",
        label="FedDeepGMM-SGDA",
        style_key="fedgda_s",
        source="step_reference_repro_v1 FedGDA-S reference, seed 0",
        method="fedgda_s",
        seed=0,
    )


def panel_curves() -> dict[str, dict[str, Any]]:
    return {
        "abs": {
            "title": r"(a) Absolute ($\alpha=1.0$)",
            "curves": [
                base_federated_mean("abs", "fedgda_d", "1.0", "FedDeepGMM-GDA", "fedgda_d"),
                base_federated_mean("abs", "fedogda_d", "1.0", "FedDeepGMM-OGDA-D", "fedogda_d"),
            ],
        },
        "step": {
            "title": r"(b) Step ($\alpha=0.5$)",
            "curves": [
                step_fedgda_s_reference(),
                step_tuned_fedogda_s_mean(),
            ],
        },
        "linear": {
            "title": r"(c) Linear ($\alpha=0.1$)",
            "curves": [
                base_federated_mean("linear", "fedgda_d", "0.1", "FedDeepGMM-GDA", "fedgda_d"),
                base_federated_mean("linear", "fedogda_d", "0.1", "FedDeepGMM-OGDA-D", "fedogda_d"),
            ],
        },
        "sin": {
            "title": r"(d) Sine ($\alpha=1.0$)",
            "curves": [
                base_federated_mean("sin", "fedgda_s", "1.0", "FedDeepGMM-SGDA", "fedgda_s"),
                sine_tuned_fedogda_s_mean(),
            ],
        },
    }


def add_centralized_curves(panels: dict[str, dict[str, Any]]) -> None:
    for dataset, panel in panels.items():
        panel["curves"].extend(
            [
                centralized_mean(dataset, "gda", "DeepGMM-GDA", "central_gda"),
                centralized_mean(dataset, "sgda", "DeepGMM-SGDA", "central_sgda"),
                centralized_mean(dataset, "oadam", "DeepGMM-OAdam", "central_oadam"),
            ]
        )


def write_metrics_csv(panels: dict[str, dict[str, Any]]) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_DIR / f"{PLOT_STEM}_curve_metrics.csv"
    fields = [
        "dataset",
        "panel_title",
        "label",
        "source",
        "method",
        "seed",
        "best_validation_mse",
        "test_mse_at_best_validation",
        "curve_mse",
        "curve_mae",
        "run_dirs",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, panel in panels.items():
            for curve in panel["curves"]:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "panel_title": panel["title"],
                        "label": curve.label,
                        "source": curve.source,
                        "method": curve.method,
                        "seed": curve.seed,
                        "best_validation_mse": curve.best_validation_mse,
                        "test_mse_at_best_validation": curve.test_mse_at_best_validation,
                        "curve_mse": curve.curve_mse,
                        "curve_mae": curve.curve_mae,
                        "run_dirs": "|".join(curve.run_dirs),
                    }
                )
    return path


def plot(panels: dict[str, dict[str, Any]]) -> list[Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [PNG_DIR / f"{PLOT_STEM}.png", PDF_DIR / f"{PLOT_STEM}.pdf"]

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.75))
    fig.subplots_adjust(left=0.085, right=0.992, top=0.965, bottom=0.205, wspace=0.24, hspace=0.34)
    legend_entries: dict[str, Any] = {}

    order = ["abs", "step", "linear", "sin"]
    for ax, dataset in zip(axes.reshape(-1), order):
        panel = panels[dataset]
        curves: list[Curve] = panel["curves"]
        true_curve = curves[0]
        true_line, = ax.plot(
            true_curve.x,
            true_curve.true_g,
            label="Actual causal effect",
            **STYLE["true"],
        )
        legend_entries.setdefault("Actual causal effect", true_line)

        for curve in curves:
            style = dict(STYLE[curve.style_key])
            if style.get("marker"):
                style.update(
                    {
                        "markevery": max(1, curve.x.size // 11),
                        "markersize": 3.5,
                        "markerfacecolor": "white",
                        "markeredgewidth": 0.75,
                        "zorder": 4,
                    }
                )
            line, = ax.plot(curve.x, curve.pred, label=curve.label, **style)
            legend_entries.setdefault(curve.label, line)

        ax.set_title(panel["title"], fontsize=9.5, fontweight="semibold", pad=4)
        ax.set_xlabel(r"$x$", fontsize=9)
        ax.set_ylabel(r"$g(x)$", fontsize=9)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=7.5,
            length=3.2,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=1.8,
        )
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.42, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.75)

    legend_labels = [label for label in LEGEND_ORDER if label in legend_entries]
    fig.legend(
        [legend_entries[label] for label in legend_labels],
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=7.15,
        handlelength=2.7,
        columnspacing=1.05,
        handletextpad=0.5,
    )

    for output in outputs:
        fig.savefig(
            output,
            dpi=400 if output.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)
    return outputs


def write_markdown(outputs: list[Path], metrics_csv: Path, panels: dict[str, dict[str, Any]]) -> Path:
    path = OUT / f"{PLOT_STEM}.md"
    lines = [
        f"# {PLOT_STEM}",
        "",
        "Unified 2x2 low-dimensional curve summary generated from saved `best_validation_prediction` arrays.",
        "Federated tuned/final selections are validation-selected; test MSE values below are post-selection readouts.",
        "Paper v1 changes presentation only; underlying run directories, predictions, and scientific metrics are unchanged.",
        "",
        "## Outputs",
        "",
        *[f"- `{rel(output)}`" for output in outputs],
        f"- `{rel(metrics_csv)}`",
        "",
        "## Panel Sources",
        "",
    ]
    for dataset in ["abs", "step", "linear", "sin"]:
        panel = panels[dataset]
        lines.append(f"### {panel['title']}")
        for curve in panel["curves"]:
            lines.append(
                f"- {curve.label}: test@best `{curve.test_mse_at_best_validation:.9f}`, "
                f"curve MAE `{curve.curve_mae:.9f}`; {curve.source}."
            )
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    panels = panel_curves()
    add_centralized_curves(panels)
    metrics_csv = write_metrics_csv(panels)
    outputs = plot(panels)
    summary = write_markdown(outputs, metrics_csv, panels)
    print("Generated:")
    for output in outputs:
        print(f"  {rel(output)}")
    print(f"  {rel(metrics_csv)}")
    print(f"  {rel(summary)}")


if __name__ == "__main__":
    main()
