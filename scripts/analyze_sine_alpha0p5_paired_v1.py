#!/usr/bin/env python3
"""Analyze and materialize the matched Sine alpha=0.5 tuning stages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_fedogda_s_sine_fast_v4 import FIELDS
from prepare_sine_alpha0p5_paired_v1 import METHOD_INFO, ROOT, SCREEN_NAME, make_row


EXP_DIR = ROOT / "experiments" / "curve_fitting_tuning" / SCREEN_NAME
OUTPUT_ROOT = ROOT / "results" / "curve_fitting_tuning" / SCREEN_NAME
PLOT_ROOT = ROOT / "experiments" / "curve_fitting_plots"
CSV_DIR = PLOT_ROOT / "csv"
PNG_DIR = PLOT_ROOT / "png" / SCREEN_NAME
PDF_DIR = PLOT_ROOT / "pdf" / SCREEN_NAME

STAGE_A_MANIFEST = EXP_DIR / "stage_a_manifest.csv"
STAGE_B_MANIFEST = EXP_DIR / "stage_b_manifest.csv"
STAGE_C_MANIFEST = EXP_DIR / "stage_c_manifest.csv"

BASELINE = {
    "fedgda_s": {
        "mean_best_validation_mse": 0.07630442559801574,
        "mean_test_mse_at_best_validation": 0.07803413463060639,
    },
    "fedogda_s": {
        "mean_best_validation_mse": 0.08360137117039597,
        "mean_test_mse_at_best_validation": 0.08614024481590275,
    },
}

RUN_FIELDS = [
    "stage",
    "method",
    "run_id",
    "seed",
    "learning_rate",
    "critic_multiplier",
    "objective_lambda_1",
    "server_learning_rate",
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
    "curve_corr",
    "amp_ratio",
    "diverged",
    "runtime_seconds",
    "result_dir",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def curve_payload(run_dir: Path) -> dict[str, Any]:
    with np.load(run_dir / "predictions.npz") as data:
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        true_g = np.asarray(data["true_g"], dtype=float).reshape(-1)
        pred = np.asarray(data["best_validation_prediction"], dtype=float).reshape(-1)
    order = np.argsort(x)
    x, true_g, pred = x[order], true_g[order], pred[order]
    if not (
        x.size == true_g.size == pred.size
        and np.isfinite(x).all()
        and np.isfinite(true_g).all()
        and np.isfinite(pred).all()
    ):
        raise ValueError(f"invalid curve artifact: {run_dir}")
    error = pred - true_g
    true_amp = float(np.ptp(true_g))
    return {
        "_x": x,
        "_true_g": true_g,
        "_pred": pred,
        "curve_mse": float(np.mean(error**2)),
        "curve_mae": float(np.mean(np.abs(error))),
        "curve_corr": float(np.corrcoef(pred, true_g)[0, 1]),
        "amp_ratio": float(np.ptp(pred)) / true_amp if true_amp else math.nan,
    }


def history_payload(path: Path) -> tuple[float, bool]:
    rows = read_csv(path)
    values = [float(row["val_mse"]) for row in rows[-50:]]
    diverged = any(
        to_bool(row.get("diverged", "false"))
        or not to_bool(row.get("finite", "true"))
        for row in rows
    )
    return statistics.pstdev(values) if len(values) > 1 else 0.0, diverged


def load_runs(manifest: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for manifest_row in read_csv(manifest):
        run_dir = ROOT / manifest_row["final_result_dir"]
        needed = [
            run_dir / "metrics.json",
            run_dir / "mse_by_round.csv",
            run_dir / "predictions.npz",
        ]
        if not all(path.exists() for path in needed):
            missing.append(manifest_row["run_id"])
            continue
        try:
            metrics = read_json(run_dir / "metrics.json")
            curve = curve_payload(run_dir)
            last_50_std, history_diverged = history_payload(run_dir / "mse_by_round.csv")
            best_val = float(metrics["best_validation_mse"])
            final_val = float(metrics["final_validation_mse"])
            row = {
                "stage": manifest_row["stage"],
                "method": manifest_row["method"],
                "run_id": manifest_row["run_id"],
                "seed": int(manifest_row["seed"]),
                "learning_rate": float(manifest_row["learning_rate"]),
                "critic_multiplier": float(manifest_row["critic_multiplier"]),
                "objective_lambda_1": float(manifest_row["objective_lambda_1"]),
                "server_learning_rate": float(manifest_row["server_learning_rate"]),
                "comm_round": int(manifest_row["comm_round"]),
                "best_validation_mse": best_val,
                "last_50_val_mse_std": last_50_std,
                "final_vs_best_validation_gap": final_val - best_val,
                "test_mse_at_best_validation": float(metrics["test_mse_at_best_validation"]),
                "best_validation_round": int(metrics["best_validation_round"]),
                "final_validation_mse": final_val,
                "final_test_mse": float(metrics["final_test_mse"]),
                "diverged": bool(metrics.get("diverged", False)) or history_diverged,
                "runtime_seconds": float(metrics.get("runtime_seconds", math.nan)),
                "result_dir": rel(run_dir),
                "_manifest": manifest_row,
            }
            row.update(curve)
            if not math.isfinite(best_val):
                raise ValueError("non-finite validation MSE")
            runs.append(row)
        except Exception:
            missing.append(manifest_row["run_id"])
    return runs, missing


def config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["method"],
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


def valid(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in runs
        if not row["diverged"] and math.isfinite(float(row["best_validation_mse"]))
    ]


def dedupe_ranked(runs: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in valid(runs):
        if row["method"] != method:
            continue
        key = config_key(row)
        if key not in best or selection_key(row) < selection_key(best[key]):
            best[key] = row
    return sorted(best.values(), key=selection_key)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, FIELDS)


def materialize_stage_b(stage_a: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_INFO:
        for candidate in dedupe_ranked(stage_a, method)[:2]:
            for server_lr in (1.25, 1.75, 2.0):
                if abs(server_lr - float(candidate["server_learning_rate"])) < 1e-12:
                    continue
                rows.append(
                    make_row(
                        method=method,
                        stage="stage_b_server_lr",
                        seed=0,
                        learning_rate=float(candidate["learning_rate"]),
                        critic_multiplier=float(candidate["critic_multiplier"]),
                        objective_lambda_1=float(candidate["objective_lambda_1"]),
                        server_learning_rate=server_lr,
                        comm_round=500,
                    )
                )
    write_manifest(STAGE_B_MANIFEST, rows)
    return rows


def materialize_stage_c(
    stage_a: list[dict[str, Any]],
    stage_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combined = stage_a + stage_b
    for method in METHOD_INFO:
        for candidate in dedupe_ranked(combined, method)[:2]:
            for seed in (0, 1, 2):
                rows.append(
                    make_row(
                        method=method,
                        stage="stage_c_confirm",
                        seed=seed,
                        learning_rate=float(candidate["learning_rate"]),
                        critic_multiplier=float(candidate["critic_multiplier"]),
                        objective_lambda_1=float(candidate["objective_lambda_1"]),
                        server_learning_rate=float(candidate["server_learning_rate"]),
                        comm_round=1000,
                    )
                )
    write_manifest(STAGE_C_MANIFEST, rows)
    return rows


def aggregate_confirmation(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in valid(runs):
        groups[config_key(row)].append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in groups.items():
        seeds = sorted({int(row["seed"]) for row in group})
        mean = statistics.fmean
        aggregates.append(
            {
                "method": key[0],
                "learning_rate": key[1],
                "critic_multiplier": key[2],
                "objective_lambda_1": key[3],
                "server_learning_rate": key[4],
                "seed_values": "|".join(map(str, seeds)),
                "seed_count": len(seeds),
                "diverged_count": sum(bool(row["diverged"]) for row in group),
                "mean_best_validation_mse": mean([row["best_validation_mse"] for row in group]),
                "std_best_validation_mse": statistics.pstdev(
                    [row["best_validation_mse"] for row in group]
                ),
                "mean_last_50_val_mse_std": mean(
                    [row["last_50_val_mse_std"] for row in group]
                ),
                "mean_final_vs_best_validation_gap": mean(
                    [row["final_vs_best_validation_gap"] for row in group]
                ),
                "mean_test_mse_at_best_validation": mean(
                    [row["test_mse_at_best_validation"] for row in group]
                ),
                "std_test_mse_at_best_validation": statistics.pstdev(
                    [row["test_mse_at_best_validation"] for row in group]
                ),
                "mean_curve_mae": mean([row["curve_mae"] for row in group]),
                "mean_curve_corr": mean([row["curve_corr"] for row in group]),
                "mean_amp_ratio": mean([row["amp_ratio"] for row in group]),
                "_runs": group,
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            row["method"],
            -int(row["seed_count"] == 3 and row["diverged_count"] == 0),
            row["mean_best_validation_mse"],
            row["mean_last_50_val_mse_std"],
            row["mean_final_vs_best_validation_gap"],
        ),
    )


AGG_FIELDS = [
    "method",
    "learning_rate",
    "critic_multiplier",
    "objective_lambda_1",
    "server_learning_rate",
    "seed_values",
    "seed_count",
    "diverged_count",
    "mean_best_validation_mse",
    "std_best_validation_mse",
    "mean_last_50_val_mse_std",
    "mean_final_vs_best_validation_gap",
    "mean_test_mse_at_best_validation",
    "std_test_mse_at_best_validation",
    "mean_curve_mae",
    "mean_curve_corr",
    "mean_amp_ratio",
]


def winners(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for method in METHOD_INFO:
        eligible = [
            row
            for row in aggregates
            if row["method"] == method
            and row["seed_count"] == 3
            and row["diverged_count"] == 0
        ]
        eligible.sort(
            key=lambda row: (
                row["mean_best_validation_mse"],
                row["mean_last_50_val_mse_std"],
                row["mean_final_vs_best_validation_gap"],
            )
        )
        if not eligible:
            raise SystemExit(f"no eligible confirmed configuration for {method}")
        chosen = dict(eligible[0])
        chosen["selection_rule"] = (
            "lowest mean_best_validation_mse across seeds 0-2; tie lower "
            "mean_last_50_val_mse_std; tie lower mean_final_vs_best_validation_gap"
        )
        selected.append(chosen)
    return selected


def mean_curve(runs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = runs[0]
    for row in runs[1:]:
        if not np.allclose(row["_x"], reference["_x"], rtol=0, atol=1e-12):
            raise ValueError("curve x grids do not align")
        if not np.allclose(row["_true_g"], reference["_true_g"], rtol=0, atol=1e-12):
            raise ValueError("true structural functions do not align")
    return (
        reference["_x"],
        reference["_true_g"],
        np.vstack([row["_pred"] for row in runs]).mean(axis=0),
    )


def plot_selected(selected: list[dict[str, Any]]) -> list[Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PNG_DIR / f"{SCREEN_NAME}_selected_curves.png",
        PDF_DIR / f"{SCREEN_NAME}_selected_curves.pdf",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {"fedgda_s": "#0072B2", "fedogda_s": "#D55E00"}
    styles = {"fedgda_s": "--", "fedogda_s": "-."}
    x, true_g, _ = mean_curve(selected[0]["_runs"])
    ax.plot(x, true_g, color="#111111", linewidth=2.4, label=r"True $g_0$")
    for row in selected:
        curve_x, _, pred = mean_curve(row["_runs"])
        label = (
            f"{METHOD_INFO[row['method']][0]} "
            f"(val={row['mean_best_validation_mse']:.4f}, "
            f"test={row['mean_test_mse_at_best_validation']:.4f})"
        )
        ax.plot(
            curve_x,
            pred,
            color=colors[row["method"]],
            linestyle=styles[row["method"]],
            linewidth=2.0,
            label=label,
        )
    ax.set_title(r"Sine, common $\alpha=0.5$: validation-selected confirmations")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$g(x)$")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    for output in outputs:
        fig.savefig(output, dpi=300 if output.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def plot_dynamics(selected: list[dict[str, Any]]) -> list[Path]:
    outputs = [
        PNG_DIR / f"{SCREEN_NAME}_selected_validation_dynamics.png",
        PDF_DIR / f"{SCREEN_NAME}_selected_validation_dynamics.pdf",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    colors = {"fedgda_s": "#0072B2", "fedogda_s": "#D55E00"}
    for selected_row in selected:
        histories = []
        for run in selected_row["_runs"]:
            history = read_csv(ROOT / run["result_dir"] / "mse_by_round.csv")
            histories.append(np.asarray([float(row["val_mse"]) for row in history]))
        limit = min(len(history) for history in histories)
        matrix = np.vstack([history[:limit] for history in histories])
        rounds = np.arange(limit)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        label = METHOD_INFO[selected_row["method"]][0]
        ax.plot(rounds, mean, color=colors[selected_row["method"]], linewidth=1.5, label=label)
        ax.fill_between(rounds, mean - std, mean + std, color=colors[selected_row["method"]], alpha=0.16)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation MSE")
    ax.set_title(r"Sine $\alpha=0.5$: validation dynamics (mean $\pm$ SD, seeds 0--2)")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    for output in outputs:
        fig.savefig(output, dpi=300 if output.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def write_summary(
    *,
    stage_a: list[dict[str, Any]],
    stage_b: list[dict[str, Any]],
    stage_c: list[dict[str, Any]],
    missing: dict[str, list[str]],
    selected: list[dict[str, Any]],
    outputs: list[Path],
) -> Path:
    path = PLOT_ROOT / f"{SCREEN_NAME}_summary.md"
    lines = [
        f"# {SCREEN_NAME}",
        "",
        "All configuration and checkpoint selection is validation-only.",
        "Test and curve metrics are read only after the confirmed validation winner is fixed.",
        "",
        "## Completion",
        "",
        f"- Stage A: `{len(stage_a)}` valid; `{len(missing['stage_a'])}` missing/invalid.",
        f"- Stage B: `{len(stage_b)}` valid; `{len(missing['stage_b'])}` missing/invalid.",
        f"- Stage C: `{len(stage_c)}` valid; `{len(missing['stage_c'])}` missing/invalid.",
        "",
        "## Selected configurations",
        "",
    ]
    for row in selected:
        baseline = BASELINE[row["method"]]
        val_reduction = 1.0 - row["mean_best_validation_mse"] / baseline["mean_best_validation_mse"]
        test_reduction = (
            1.0
            - row["mean_test_mse_at_best_validation"]
            / baseline["mean_test_mse_at_best_validation"]
        )
        lines.extend(
            [
                f"### {METHOD_INFO[row['method']][0]}",
                "",
                (
                    f"- lr `{row['learning_rate']:g}`, cm `{row['critic_multiplier']:g}`, "
                    f"lambda `{row['objective_lambda_1']:g}`, "
                    f"server lr `{row['server_learning_rate']:g}`."
                ),
                (
                    f"- Mean validation MSE `{row['mean_best_validation_mse']:.9f}` "
                    f"(reduction vs existing preset `{val_reduction:.1%}`)."
                ),
                (
                    f"- Post-selection mean test@best `{row['mean_test_mse_at_best_validation']:.9f}` "
                    f"+/- `{row['std_test_mse_at_best_validation']:.9f}` "
                    f"(reduction vs existing preset `{test_reduction:.1%}`)."
                ),
                (
                    f"- Mean curve MAE `{row['mean_curve_mae']:.9f}`, "
                    f"correlation `{row['mean_curve_corr']:.9f}`, "
                    f"amplitude ratio `{row['mean_amp_ratio']:.9f}`."
                ),
                "",
            ]
        )
    lines.extend(["## Outputs", ""])
    lines.extend(f"- `{rel(output)}`" for output in outputs)
    lines.extend(
        [
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_all_runs.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_confirmation_aggregates.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_final_selected.csv')}`",
            f"- `{rel(CSV_DIR / f'{SCREEN_NAME}_selected_seed_runs.csv')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def analyze(action: str) -> None:
    stage_a, missing_a = load_runs(STAGE_A_MANIFEST)
    stage_b, missing_b = load_runs(STAGE_B_MANIFEST)
    stage_c, missing_c = load_runs(STAGE_C_MANIFEST)
    all_runs = stage_a + stage_b + stage_c
    write_csv(CSV_DIR / f"{SCREEN_NAME}_all_runs.csv", all_runs, RUN_FIELDS)
    write_csv(
        CSV_DIR / f"{SCREEN_NAME}_stage_a_ranked.csv",
        [row for method in METHOD_INFO for row in dedupe_ranked(stage_a, method)],
        RUN_FIELDS,
    )

    if action == "stage-a":
        rows = materialize_stage_b(stage_a)
        print(json.dumps({"stage_a_valid": len(stage_a), "stage_b_rows": len(rows)}, indent=2))
        return

    write_csv(
        CSV_DIR / f"{SCREEN_NAME}_stage_b_ranked.csv",
        [row for method in METHOD_INFO for row in dedupe_ranked(stage_a + stage_b, method)],
        RUN_FIELDS,
    )
    if action == "stage-b":
        rows = materialize_stage_c(stage_a, stage_b)
        print(
            json.dumps(
                {
                    "stage_a_valid": len(stage_a),
                    "stage_b_valid": len(stage_b),
                    "stage_c_rows": len(rows),
                },
                indent=2,
            )
        )
        return

    aggregates = aggregate_confirmation(stage_c)
    write_csv(
        CSV_DIR / f"{SCREEN_NAME}_confirmation_aggregates.csv",
        aggregates,
        AGG_FIELDS,
    )
    selected = winners(aggregates)
    write_csv(
        CSV_DIR / f"{SCREEN_NAME}_final_selected.csv",
        selected,
        AGG_FIELDS + ["selection_rule"],
    )
    selected_runs = [run for row in selected for run in row["_runs"]]
    write_csv(
        CSV_DIR / f"{SCREEN_NAME}_selected_seed_runs.csv",
        sorted(selected_runs, key=lambda row: (row["method"], row["seed"])),
        RUN_FIELDS,
    )
    outputs = plot_selected(selected) + plot_dynamics(selected)
    summary = write_summary(
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        missing={"stage_a": missing_a, "stage_b": missing_b, "stage_c": missing_c},
        selected=selected,
        outputs=outputs,
    )
    print(
        json.dumps(
            {
                "selected": [
                    {
                        key: row[key]
                        for key in [
                            "method",
                            "learning_rate",
                            "critic_multiplier",
                            "objective_lambda_1",
                            "server_learning_rate",
                            "mean_best_validation_mse",
                            "mean_test_mse_at_best_validation",
                        ]
                    }
                    for row in selected
                ],
                "summary": rel(summary),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stage-a", "stage-b", "final"])
    args = parser.parse_args()
    analyze(args.action)


if __name__ == "__main__":
    main()
