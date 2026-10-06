#!/usr/bin/env python3
"""Plot completed FedOGDA-S v3 screen candidates around selected settings.

This is a post-stop visualization: it uses only completed, finite screen rows
and keeps selection validation-only. Test/curve numbers are labels after the
validation-selected settings are fixed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_fedogda_s_focused_v3 import (  # noqa: E402
    SCREEN_MANIFEST,
    SCREEN_NAME,
    PLOT_ROOT,
    load_manifest,
    load_reference,
    selection_key,
    top_by_dataset,
    valid_rows,
)


def fmt_config(row: dict[str, object]) -> str:
    return (
        f"lr={float(row['learning_rate']):g}, "
        f"cm={float(row['critic_multiplier']):g}, "
        f"lam={float(row['objective_lambda_1']):g}, "
        f"slr={float(row['server_learning_rate']):g}"
    )


def main() -> int:
    loaded, missing, _, _ = load_manifest(SCREEN_MANIFEST)
    valid = valid_rows(loaded)
    selected = {
        dataset: top_by_dataset(loaded, dataset, 1)[0]
        for dataset in ("sin", "step")
        if top_by_dataset(loaded, dataset, 1)
    }

    png_dir = PLOT_ROOT / "png" / SCREEN_NAME
    pdf_dir = PLOT_ROOT / "pdf" / SCREEN_NAME
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{SCREEN_NAME}_screen_candidate_bundle.png"
    pdf_path = pdf_dir / f"{SCREEN_NAME}_screen_candidate_bundle.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2), constrained_layout=True)
    panels = [("sin", "Sine alpha=1.0"), ("step", "Step alpha=0.5 proxy")]
    top_colors = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#f97316", "#0891b2"]

    for ax, (dataset, title) in zip(axes, panels):
        dataset_items = sorted(
            [item for item in valid if item.row["dataset"] == dataset],
            key=lambda item: selection_key(item.row),
        )
        if not dataset_items:
            ax.set_title(f"{title}\nno completed candidates")
            continue

        true_curve = dataset_items[0].row["_curve"]
        ax.plot(
            true_curve["x"],
            true_curve["true_g"],
            color="black",
            linewidth=2.5,
            label="Actual causal effect",
            zorder=8,
        )

        for item in dataset_items:
            curve = item.row["_curve"]
            ax.plot(
                curve["x"],
                curve["pred"],
                color="#9ca3af",
                linewidth=0.75,
                alpha=0.18,
                zorder=1,
            )

        for ref in load_reference(dataset):
            curve = ref["_curve"]
            ax.plot(
                curve["x"],
                curve["pred"],
                color=ref["color"],
                linestyle=ref["linestyle"],
                linewidth=1.8,
                alpha=0.9,
                label=f"{ref['label']} test={ref['test_mse_at_best_validation']:.4f}",
                zorder=4,
            )

        top_items = dataset_items[:6]
        for rank, item in enumerate(top_items, start=1):
            row = item.row
            curve = row["_curve"]
            color = top_colors[rank - 1]
            linewidth = 2.6 if rank == 1 else 1.45
            alpha = 1.0 if rank == 1 else 0.78
            label = (
                f"#{rank} v3 val={row['best_validation_mse']:.4f}, "
                f"test={row['test_mse_at_best_validation']:.4f}, {fmt_config(row)}"
            )
            ax.plot(
                curve["x"],
                curve["pred"],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=7 - rank * 0.1,
            )

        winner = selected[dataset].row
        ax.set_title(
            f"{title}\ncompleted={len(dataset_items)}, missing={sum(1 for name in missing if dataset in name)}, "
            f"selected: {fmt_config(winner)}",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.legend(fontsize=6.15, loc="best", frameon=True)

    fig.suptitle(
        "FedOGDA-S focused v3 partial screen: all completed candidates in gray, top validation candidates highlighted",
        fontsize=12,
    )
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(png_path.relative_to(ROOT))
    print(pdf_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
