#!/usr/bin/env python3
"""Plot only the best optimistic FedOGDA-S candidate from v3 tuning.

The figure is intentionally sparse: actual curve, previous FedOGDA-S reference,
and the validation-selected v3 FedOGDA-S candidate. It omits the gray candidate
cloud and non-optimistic baselines.
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
    PLOT_ROOT,
    SCREEN_MANIFEST,
    SCREEN_NAME,
    curve_payload,
    load_manifest,
    read_json,
    to_float,
    top_by_dataset,
)


FEDOGDA_REFS = {
    "sin": (
        "Previous FedOGDA-S best",
        ROOT
        / "results"
        / "curve_fitting_tuning"
        / "optimistic_curve_screen_v2"
        / "sin"
        / "fedogda_s"
        / "seed_0"
        / "curvefit_sin_fedogda_s_seed0_alpha1p0_T500_R3_batch256_glr0p005_cm10_lam0p03_slr1p5",
        "#991b1b",
    ),
    "step": (
        "Previous FedOGDA-S best",
        ROOT
        / "results"
        / "curve_fitting_tuning"
        / "optimistic_curve_screen_v2"
        / "step"
        / "fedogda_s"
        / "seed_0"
        / "curvefit_step_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p005_cm15_lam0p1_slr1p5",
        "#f97316",
    ),
}


def load_fedogda_reference(dataset: str) -> dict[str, object] | None:
    label, run_dir, color = FEDOGDA_REFS[dataset]
    if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.npz").exists():
        return None
    metrics = read_json(run_dir / "metrics.json")
    return {
        "label": label,
        "color": color,
        "test": to_float(metrics["test_mse_at_best_validation"]),
        "curve": curve_payload(run_dir),
    }


def fmt_settings(row: dict[str, object]) -> str:
    return (
        f"lr={float(row['learning_rate']):g}, "
        f"cm={float(row['critic_multiplier']):g}, "
        f"lambda={float(row['objective_lambda_1']):g}, "
        f"server_lr={float(row['server_learning_rate']):g}"
    )


def main() -> int:
    loaded, missing, _, _ = load_manifest(SCREEN_MANIFEST)
    selected = {
        dataset: top_by_dataset(loaded, dataset, 1)[0]
        for dataset in ("sin", "step")
        if top_by_dataset(loaded, dataset, 1)
    }

    png_dir = PLOT_ROOT / "png" / SCREEN_NAME
    pdf_dir = PLOT_ROOT / "pdf" / SCREEN_NAME
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{SCREEN_NAME}_best_optimistic_only.png"
    pdf_path = pdf_dir / f"{SCREEN_NAME}_best_optimistic_only.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    panels = [("sin", "Sine alpha=1.0"), ("step", "Step alpha=0.5 proxy")]

    for ax, (dataset, title) in zip(axes, panels):
        item = selected[dataset]
        row = item.row
        curve = row["_curve"]

        ax.plot(
            curve["x"],
            curve["true_g"],
            color="black",
            linewidth=2.6,
            label="Actual causal effect",
            zorder=4,
        )

        ref = load_fedogda_reference(dataset)
        if ref is not None:
            ref_curve = ref["curve"]
            ax.plot(
                ref_curve["x"],
                ref_curve["pred"],
                color=ref["color"],
                linestyle="-.",
                linewidth=2.0,
                label=f"{ref['label']} test={ref['test']:.4f}",
                zorder=3,
            )

        ax.plot(
            curve["x"],
            curve["pred"],
            color="#dc2626",
            linewidth=2.6,
            label=(
                f"Selected v3 FedOGDA-S val={row['best_validation_mse']:.4f}, "
                f"test={row['test_mse_at_best_validation']:.4f}"
            ),
            zorder=5,
        )

        missing_count = sum(1 for name in missing if dataset in name)
        ax.set_title(
            f"{title}\n{fmt_settings(row)}\nT={int(row['comm_round'])}, R={int(row['epochs'])}; "
            f"missing screen rows={missing_count}",
            fontsize=9.4,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.legend(fontsize=7.2, loc="best", frameon=True)

    fig.suptitle("Best optimistic FedOGDA-S candidate only", fontsize=12)
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(png_path.relative_to(ROOT))
    print(pdf_path.relative_to(ROOT))
    for dataset, item in selected.items():
        row = item.row
        print(
            dataset,
            fmt_settings(row),
            f"val={row['best_validation_mse']:.9g}",
            f"test={row['test_mse_at_best_validation']:.9g}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
