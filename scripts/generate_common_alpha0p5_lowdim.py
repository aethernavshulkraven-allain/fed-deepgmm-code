#!/usr/bin/env python3
"""Render the common-alpha low-dimensional paper candidate.

All federated panels use Dirichlet concentration alpha=0.5 and aggregate
seeds 0, 1, and 2.  FedOGDA-S uses validation-selected tuning artifacts for
Absolute, Linear, and Step.  The Sine panel uses the validation-selected,
three-seed confirmed FedGDA-S and FedOGDA-S artifacts from the paired v1
sweep.

This script is reporting-only: it reads completed predictions/metrics and
writes new plot, CSV, and Markdown files without modifying run artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import generate_tuned_lowdim_with_centralized_2x2 as base


ALPHA = "0.5"
PLOT_STEM = "lowdim_common_alpha0p5_tuned_sine_paired_v1"
SELECTED_SINE_RUNS = (
    base.ROOT
    / "experiments"
    / "curve_fitting_plots"
    / "csv"
    / "sine_alpha0p5_paired_v1_selected_seed_runs.csv"
)

STYLE_BY_LABEL: dict[str, dict[str, Any]] = {
    r"True structural function $g_0$": {
        "color": "#111111",
        "linewidth": 2.05,
        "linestyle": "-",
        "zorder": 9,
    },
    "DeepGMM-OAdam": {
        "color": "#E69F00",
        "linewidth": 1.35,
        "linestyle": (0, (5, 2)),
        "marker": "X",
    },
    "DeepGMM-SGDA": {
        "color": "#009E73",
        "linewidth": 1.3,
        "linestyle": (0, (2, 1)),
        "marker": "P",
    },
    "DeepGMM-GDA": {
        "color": "#6A3D9A",
        "linewidth": 1.3,
        "linestyle": (0, (4, 1, 1, 1)),
        "marker": "v",
    },
    "FedDeepGMM-SGDA": {
        "color": "#0072B2",
        "linewidth": 1.6,
        "linestyle": "--",
        "marker": "o",
    },
    "FedDeepGMM-OGDA-S": {
        "color": "#D55E00",
        "linewidth": 1.75,
        "linestyle": "-.",
        "marker": "s",
    },
}

LEGEND_ORDER = [
    r"True structural function $g_0$",
    "DeepGMM-OAdam",
    "DeepGMM-SGDA",
    "DeepGMM-GDA",
    "FedDeepGMM-SGDA",
    "FedDeepGMM-OGDA-S",
]

PANEL_LABELS = {
    "abs": r"(a) Absolute",
    "step": r"(b) Step",
    "linear": r"(c) Linear",
    "sin": r"(d) Sine",
}


def pilot_tuned_fedogda_s_mean(dataset: str) -> base.Curve:
    rows = [
        row
        for row in base.read_csv(
            base.ROOT
            / "experiments"
            / "rerun_protocol_v1"
            / "tuning_fedogda_s"
            / "pilot_alpha0p5"
            / "selected_seed_metrics.csv"
        )
        if row["dataset"] == dataset
    ]
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [0, 1, 2]:
        raise ValueError(f"expected tuned {dataset} FedOGDA-S seeds 0,1,2")

    curves = [
        base.load_curve(
            run_dir=base.ROOT / row["result_dir"],
            dataset=dataset,
            label="FedDeepGMM-OGDA-S",
            style_key="fedogda_s",
            source=(
                "fedogda_s pilot alpha=0.5, validation-selected "
                "critic multiplier/weight decay"
            ),
            method="fedogda_s",
            seed=row["seed"],
        )
        for row in rows
    ]
    return base.mean_curve(
        curves,
        label="FedDeepGMM-OGDA-S",
        style_key="fedogda_s",
        source=(
            "fedogda_s pilot alpha=0.5, validation-selected config, "
            "mean seeds 0-2"
        ),
        method="fedogda_s",
    )


def confirmed_sine_mean(
    method: str,
    label: str,
    style_key: str,
) -> base.Curve:
    rows = [
        row
        for row in base.read_csv(SELECTED_SINE_RUNS)
        if row["method"] == method
    ]
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [0, 1, 2]:
        raise ValueError(f"expected selected Sine {method} seeds 0,1,2")
    if any(row["diverged"].lower() == "true" for row in rows):
        raise ValueError(f"selected Sine {method} contains a diverged run")

    curves = [
        base.load_curve(
            run_dir=base.ROOT / row["result_dir"],
            dataset="sin",
            label=label,
            style_key=style_key,
            source=(
                "Sine alpha=0.5 paired v1, validation-selected configuration, "
                "1000-round confirmation"
            ),
            method=method,
            seed=row["seed"],
        )
        for row in rows
    ]
    return base.mean_curve(
        curves,
        label=label,
        style_key=style_key,
        source=(
            "Sine alpha=0.5 paired v1, validation-selected configuration, "
            "mean confirmed seeds 0-2"
        ),
        method=method,
    )


def common_alpha_panels() -> dict[str, dict[str, Any]]:
    tuned_ogda = {
        "abs": pilot_tuned_fedogda_s_mean("abs"),
        "step": base.step_tuned_fedogda_s_mean(),
        "linear": pilot_tuned_fedogda_s_mean("linear"),
        "sin": confirmed_sine_mean(
            "fedogda_s",
            "FedDeepGMM-OGDA-S",
            "fedogda_s",
        ),
    }

    panels: dict[str, dict[str, Any]] = {}
    for dataset in ["abs", "step", "linear", "sin"]:
        fedgda_s = (
            confirmed_sine_mean(
                "fedgda_s",
                "FedDeepGMM-SGDA",
                "fedgda_s",
            )
            if dataset == "sin"
            else base.base_federated_mean(
                dataset,
                "fedgda_s",
                ALPHA,
                "FedDeepGMM-SGDA",
                "fedgda_s",
            )
        )
        panels[dataset] = {
            "title": rf"{PANEL_LABELS[dataset]} ($\alpha={ALPHA}$)",
            "curves": [
                fedgda_s,
                tuned_ogda[dataset],
            ],
        }
    base.add_centralized_curves(panels)
    return panels


def line_style(label: str, x_size: int) -> dict[str, Any]:
    style = dict(STYLE_BY_LABEL[label])
    if style.get("marker"):
        style.update(
            {
                "markevery": max(1, x_size // 9),
                "markersize": 3.0,
                "markerfacecolor": "white",
                "markeredgewidth": 0.7,
                "zorder": 4,
            }
        )
    return style


def plot(panels: dict[str, dict[str, Any]]) -> list[Path]:
    base.PNG_DIR.mkdir(parents=True, exist_ok=True)
    base.PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        base.PNG_DIR / f"{PLOT_STEM}.png",
        base.PDF_DIR / f"{PLOT_STEM}.pdf",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.95))
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.72,
        bottom=0.23,
        wspace=0.28,
    )

    legend_entries: dict[str, Line2D] = {}
    for ax, dataset in zip(axes, ["abs", "step", "linear", "sin"]):
        curves: list[base.Curve] = panels[dataset]["curves"]
        reference = curves[0]
        true_label = r"True structural function $g_0$"
        true_line, = ax.plot(
            reference.x,
            reference.true_g,
            label=true_label,
            **line_style(true_label, reference.x.size),
        )
        legend_entries.setdefault(true_label, true_line)

        for curve in curves:
            line, = ax.plot(
                curve.x,
                curve.pred,
                label=curve.label,
                **line_style(curve.label, curve.x.size),
            )
            legend_entries.setdefault(curve.label, line)

        ax.set_xlabel(r"$x$", fontsize=8.2, labelpad=1.5)
        if dataset == "abs":
            ax.set_ylabel(r"$g(x)$", fontsize=8.2, labelpad=1.5)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=6.6,
            length=2.6,
            width=0.65,
        )
        ax.set_axisbelow(True)
        ax.grid(True, color="#C9C9C9", linewidth=0.48, alpha=0.72)
        for spine in ax.spines.values():
            spine.set_color("#3F3F3F")
            spine.set_linewidth(0.68)

        ax.text(
            0.5,
            -0.25,
            PANEL_LABELS[dataset],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.9,
            fontweight="semibold",
        )

    for label in LEGEND_ORDER:
        legend_entries.setdefault(
            label,
            Line2D([], [], label=label, **line_style(label, 100)),
        )

    legend = fig.legend(
        [legend_entries[label] for label in LEGEND_ORDER],
        LEGEND_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=6.35,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#8A8A8A",
        borderpad=0.38,
        labelspacing=0.34,
        handlelength=2.65,
        handletextpad=0.48,
        columnspacing=1.05,
    )
    legend.get_frame().set_linewidth(0.55)

    fig.text(
        0.5,
        0.755,
        r"Common Dirichlet concentration $\alpha=0.5$; mean over seeds 0--2",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#333333",
    )

    for output in outputs:
        fig.savefig(
            output,
            dpi=450 if output.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)
    return outputs


def write_report(
    panels: dict[str, dict[str, Any]],
    outputs: list[Path],
    metrics_csv: Path,
) -> Path:
    path = base.OUT / f"{PLOT_STEM}.md"
    lines = [
        f"# {PLOT_STEM}",
        "",
        "Common-alpha paper candidate generated from validation-selected artifacts.",
        "All learned curves use saved `best_validation_prediction` arrays.",
        "",
        "## Outputs",
        "",
        *[f"- `{base.rel(output)}`" for output in outputs],
        f"- `{base.rel(metrics_csv)}`",
        "",
        "## Protocol",
        "",
        "- Common federated Dirichlet concentration: `alpha=0.5`.",
        "- Every plotted method curve is the pointwise mean over seeds `0,1,2`.",
        "- FedDeepGMM-SGDA uses the completed alpha=0.5 base sweep for Absolute, Step, and Linear; Sine uses the paired confirmation.",
        (
            "- FedDeepGMM-OGDA-S is validation-tuned for Absolute and Linear "
            "(alpha=0.5 pilot) and Step (v5 final)."
        ),
        (
            "- Sine FedDeepGMM-SGDA and FedDeepGMM-OGDA-S use matched "
            "validation-driven screening/refinement budgets and independent "
            "1000-round confirmations over seeds 0,1,2."
        ),
        "- Centralized methods do not use the federated partition alpha.",
        "",
        "## Publication status",
        "",
        (
            "**Paper candidate with scope caveats.** The common-alpha presentation "
            "and Sine tuning are internally consistent. Comparative claims across "
            "all four scenarios should still disclose that the non-Sine federated "
            "methods do not all have the same matched tuning budget, and the "
            "synthetic DGP remains reproducible but not independently certified as "
            "paper-aligned."
        ),
        "",
        "## Panel metrics",
        "",
    ]
    for dataset in ["abs", "step", "linear", "sin"]:
        lines.append(f"### {panels[dataset]['title']}")
        for curve in panels[dataset]["curves"]:
            lines.append(
                f"- {curve.label}: validation MSE `{curve.best_validation_mse:.9f}`, "
                f"test@best `{curve.test_mse_at_best_validation:.9f}`, "
                f"curve MAE `{curve.curve_mae:.9f}`; {curve.source}."
            )
        lines.append("")

    lines.extend(
        [
            "## Draft caption",
            "",
            (
                "`Validation-checkpoint estimates of the structural function "
                "$g_0$ in four low-dimensional scenarios. Federated experiments "
                "use a common Dirichlet concentration $\\alpha=0.5$; curves are "
                "pointwise means over seeds 0--2.`"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    panels = common_alpha_panels()
    base.PLOT_STEM = PLOT_STEM
    metrics_csv = base.write_metrics_csv(panels)
    outputs = plot(panels)
    report = write_report(panels, outputs, metrics_csv)

    print("Generated:")
    for output in outputs:
        print(f"  {base.rel(output)}")
    print(f"  {base.rel(metrics_csv)}")
    print(f"  {base.rel(report)}")


if __name__ == "__main__":
    main()
