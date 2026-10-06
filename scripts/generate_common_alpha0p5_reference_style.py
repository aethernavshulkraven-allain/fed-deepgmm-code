#!/usr/bin/env python3
"""Render the tuned common-alpha curves in the paper's 1x4 visual style.

This reporting-only renderer keeps the validation-selected, three-seed Sine
curves from the paired v1 sweep and adds the available alpha=0.5 FedGDA and
FedOGDA-D means so the figure contains the same eight legend entries as the
paper reference. Existing figures and run artifacts are not overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import generate_common_alpha0p5_lowdim as common


base = common.base
PLOT_STEM = "lowdim_common_alpha0p5_tuned_sine_reference_style_v2"

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

STYLE_BY_LABEL: dict[str, dict[str, Any]] = {
    "Actual Causal Effect": {
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

# Matplotlib fills this four-column legend down columns, producing the same
# two-row ordering as the paper reference.
LEGEND_ORDER = [
    "Actual Causal Effect",
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


def paper_panels() -> dict[str, dict[str, Any]]:
    panels = common.common_alpha_panels()
    for dataset in ["abs", "step", "linear", "sin"]:
        panels[dataset]["curves"].extend(
            [
                base.base_federated_mean(
                    dataset,
                    "fedgda_d",
                    common.ALPHA,
                    "FedDeepGMM-GDA",
                    "fedgda_d",
                ),
                base.base_federated_mean(
                    dataset,
                    "fedogda_d",
                    common.ALPHA,
                    "FedDeepGMM-OGDA-D",
                    "fedogda_d",
                ),
            ]
        )
    return panels


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
        curves: list[base.Curve] = panels[dataset]["curves"]
        reference = curves[0]
        true_label = "Actual Causal Effect"
        true_line, = ax.plot(
            reference.x,
            reference.true_g,
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

    for label in LEGEND_ORDER:
        legend_entries.setdefault(
            label,
            Line2D([], [], label=label, **STYLE_BY_LABEL[label]),
        )

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
            facecolor="white",
        )
    plt.close(fig)
    return outputs


def write_report(outputs: list[Path], metrics_csv: Path) -> Path:
    path = base.OUT / f"{PLOT_STEM}.md"
    lines = [
        f"# {PLOT_STEM}",
        "",
        "Paper-reference rendering of the tuned common-alpha low-dimensional curves.",
        "Presentation changed only; validation-selected predictions and metrics are unchanged.",
        "",
        "## Protocol",
        "",
        "- All federated curves use `alpha=0.5` and are means over seeds `0,1,2`.",
        "- Sine FedDeepGMM-SGDA and FedDeepGMM-OGDA-S use the paired v1 confirmed winners.",
        "- The other curve sources and selection provenance are recorded in the metrics CSV.",
        "",
        "## Outputs",
        "",
        *[f"- `{base.rel(output)}`" for output in outputs],
        f"- `{base.rel(metrics_csv)}`",
        "",
        "## Suggested caption",
        "",
        (
            "`Validation-selected estimates of the causal response function compared "
            "with the true effect in four low-dimensional scenarios. All federated "
            "curves use a common Dirichlet concentration $\\alpha=0.5$ and show "
            "pointwise means over seeds 0--2.`"
        ),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    panels = paper_panels()
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
