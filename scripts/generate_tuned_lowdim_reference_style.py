#!/usr/bin/env python3
"""Render the audited low-dimensional curves in a compact paper style.

This reporting-only renderer reuses the validation-selected curves and metric
loading logic from ``generate_tuned_lowdim_with_centralized_2x2.py``. It
changes presentation only and writes a new output bundle without replacing the
audited source figure or paper-v1 figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import generate_tuned_lowdim_with_centralized_2x2 as base


PLOT_STEM = "lowdim_tuned_with_centralized_summary_1x4_paper_v2"

STYLE_BY_LABEL: dict[str, dict[str, Any]] = {
    r"True structural function $g_0$": {
        "color": "#1f77b4",
        "linewidth": 1.65,
        "linestyle": "-",
        "zorder": 8,
    },
    "DeepGMM-OAdam": {
        "color": "#ff7f0e",
        "linewidth": 1.25,
        "linestyle": "-",
    },
    "DeepGMM-SGDA": {
        "color": "#2ca02c",
        "linewidth": 1.25,
        "linestyle": "-",
    },
    "FedDeepGMM-SGDA": {
        "color": "#d62728",
        "linewidth": 1.25,
        "linestyle": "-",
    },
    "DeepGMM-GDA": {
        "color": "#9467bd",
        "linewidth": 1.25,
        "linestyle": "-",
    },
    "FedDeepGMM-GDA": {
        "color": "#8c564b",
        "linewidth": 1.25,
        "linestyle": "-",
    },
    "FedDeepGMM-OGDA-D": {
        "color": "#e377c2",
        "linewidth": 1.35,
        "linestyle": "-",
    },
    "FedDeepGMM-OGDA-S": {
        "color": "#17becf",
        "linewidth": 1.35,
        "linestyle": "-",
    },
}

LEGEND_ORDER = [
    r"True structural function $g_0$",
    "DeepGMM-OAdam",
    "DeepGMM-SGDA",
    "FedDeepGMM-SGDA",
    "DeepGMM-GDA",
    "FedDeepGMM-GDA",
    "FedDeepGMM-OGDA-D",
    "FedDeepGMM-OGDA-S",
]

PANEL_LABELS = {
    "abs": r"(a) Absolute",
    "step": r"(b) Step",
    "linear": r"(c) Linear",
    "sin": r"(d) Sine",
}

PANEL_SETTINGS = {
    "abs": r"$\alpha=1.0$",
    "step": r"$\alpha=0.5$",
    "linear": r"$\alpha=0.1$",
    "sin": r"$\alpha=1.0$",
}


def plot(panels: dict[str, dict[str, Any]]) -> list[Path]:
    base.PNG_DIR.mkdir(parents=True, exist_ok=True)
    base.PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        base.PNG_DIR / f"{PLOT_STEM}.png",
        base.PDF_DIR / f"{PLOT_STEM}.pdf",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.62))
    fig.subplots_adjust(
        left=0.025,
        right=0.995,
        top=0.73,
        bottom=0.19,
        wspace=0.24,
    )

    legend_entries: dict[str, Line2D] = {}
    for ax, dataset in zip(axes, ["abs", "step", "linear", "sin"]):
        curves = panels[dataset]["curves"]
        true_curve = curves[0]
        true_label = r"True structural function $g_0$"
        true_line, = ax.plot(
            true_curve.x,
            true_curve.true_g,
            label=true_label,
            **STYLE_BY_LABEL[true_label],
        )
        legend_entries.setdefault(true_label, true_line)

        for curve in curves:
            line, = ax.plot(
                curve.x,
                curve.pred,
                label=curve.label,
                **STYLE_BY_LABEL[curve.label],
            )
            legend_entries.setdefault(curve.label, line)

        ax.set_axisbelow(True)
        ax.grid(True, color="#B8B8B8", linewidth=0.55, alpha=0.68)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.65)

        ax.text(
            0.5,
            -0.17,
            PANEL_LABELS[dataset],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.2,
            fontweight="bold",
        )

    # Use proxies for methods that only appear in the deterministic or
    # stochastic panels so the compact legend remains globally complete.
    for label in LEGEND_ORDER:
        legend_entries.setdefault(label, Line2D([], [], label=label, **STYLE_BY_LABEL[label]))

    legend = fig.legend(
        [legend_entries[label] for label in LEGEND_ORDER],
        LEGEND_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        fontsize=6.05,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#9A9A9A",
        borderpad=0.32,
        labelspacing=0.28,
        handlelength=2.2,
        handletextpad=0.42,
        columnspacing=0.9,
    )
    legend.get_frame().set_linewidth(0.55)

    for output in outputs:
        fig.savefig(
            output,
            dpi=400 if output.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)
    return outputs


def write_report(outputs: list[Path], metrics_csv: Path) -> Path:
    path = base.OUT / f"{PLOT_STEM}.md"
    lines = [
        f"# {PLOT_STEM}",
        "",
        "Reference-inspired compact paper rendering of the audited low-dimensional curves.",
        "Presentation changed only; run directories, predictions, and scientific metrics are unchanged.",
        "",
        "## Outputs",
        "",
        *[f"- `{base.rel(output)}`" for output in outputs],
        f"- `{base.rel(metrics_csv)}`",
        "",
        "## Panel settings",
        "",
        *[
            f"- {PANEL_LABELS[dataset]}: {PANEL_SETTINGS[dataset]}"
            for dataset in ["abs", "step", "linear", "sin"]
        ],
        "",
        "## Suggested LaTeX caption",
        "",
        (
            "`Validation-selected estimates $\\hat{g}$ compared with the true "
            "structural function $g_0$ in the low-dimensional scenarios.`"
        ),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    panels = base.panel_curves()
    base.add_centralized_curves(panels)

    base.PLOT_STEM = PLOT_STEM
    metrics_csv = base.write_metrics_csv(panels)
    outputs = plot(panels)
    report = write_report(outputs, metrics_csv)

    print("Generated:")
    for output in outputs:
        print(f"  {base.rel(output)}")
    print(f"  {base.rel(metrics_csv)}")
    print(f"  {base.rel(report)}")


if __name__ == "__main__":
    main()
