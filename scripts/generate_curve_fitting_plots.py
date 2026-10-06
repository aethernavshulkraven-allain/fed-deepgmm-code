#!/usr/bin/env python3
"""Generate low-dimensional curve-fitting plots from existing artifacts.

This script is intentionally reporting-only. It does not launch training and
does not modify result artifacts. It overwrites only files under
experiments/curve_fitting_plots/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "curve_fitting_plots"
PNG = OUT / "png"
PDF = OUT / "pdf"
CSV = OUT / "csv"

DATASETS = ["abs", "step", "linear", "sin"]
DATASET_LABEL = {"abs": "Absolute", "step": "Step", "linear": "Linear", "sin": "Sine"}
FED_METHODS = ["fedgda_d", "fedogda_d", "fedgda_s", "fedogda_s"]
METHOD_LABEL = {
    "fedgda_d": "FedGDA-D",
    "fedogda_d": "FedOGDA-D",
    "fedgda_s": "FedGDA-S",
    "fedogda_s": "FedOGDA-S",
    "gda": "DeepGMM-GDA",
    "sgda": "DeepGMM-SGDA",
    "oadam": "DeepGMM-OAdam",
}
LINE_STYLE = {
    "true g": {"color": "black", "linestyle": "-", "linewidth": 2.2},
    "FedGDA-D": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.8},
    "FedOGDA-D": {"color": "#d62728", "linestyle": "-.", "linewidth": 1.8},
    "FedGDA-S": {"color": "#2ca02c", "linestyle": ":", "linewidth": 2.0},
    "FedOGDA-S": {"color": "#ff7f0e", "linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.8},
    "DeepGMM-GDA": {"color": "#9467bd", "linestyle": "--", "linewidth": 1.7},
    "DeepGMM-SGDA": {"color": "#8c564b", "linestyle": ":", "linewidth": 2.0},
    "DeepGMM-OAdam": {"color": "#17becf", "linestyle": "-.", "linewidth": 1.8},
    "Tuned FedOGDA-D": {"color": "#d62728", "linestyle": "-.", "linewidth": 1.8},
}


@dataclass
class Artifact:
    source_family: str
    dataset: str
    method: str
    method_label: str
    alpha: str
    seed: int | str
    result_dir: str
    prediction_path: str
    metrics_path: str
    config_path: str
    notes: str = ""


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for base in [PNG, PDF, CSV]:
        base.mkdir(parents=True, exist_ok=True)
    for sub in [
        "main_pairwise",
        "main_pairwise_aggregate",
        "all_methods_original",
        "all_methods_original_aggregate",
        "tuned_sine_a2_lite",
        "coauthor_summary",
        "centralized",
        "centralized_aggregate",
        "fedogda_s_tuning_pilot",
        "fedogda_s_tuning_pilot_aggregate",
    ]:
        (PNG / sub).mkdir(parents=True, exist_ok=True)
        (PDF / sub).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def alpha_slug(alpha: Any) -> str:
    text = str(alpha)
    return text.replace(".", "p").replace("-", "m")


def parse_alpha(value: Any) -> float | None:
    try:
        if str(value).lower() == "nan":
            return None
        return float(value)
    except Exception:
        return None


def discover_base_sweep() -> list[Artifact]:
    manifest = pd.read_csv(ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv")
    rows = manifest[
        (manifest["training_scope"] == "federated")
        & (manifest["dataset"].isin(DATASETS))
        & (manifest["method"].isin(FED_METHODS))
    ]
    artifacts: list[Artifact] = []
    for _, row in rows.iterrows():
        result_dir = ROOT / str(row["final_result_dir"])
        artifacts.append(
            Artifact(
                source_family="base_sweep",
                dataset=str(row["dataset"]),
                method=str(row["method"]),
                method_label=METHOD_LABEL[str(row["method"])],
                alpha=str(row["alpha"]),
                seed=int(row["seed"]),
                result_dir=rel(result_dir),
                prediction_path=rel(result_dir / "predictions.npz"),
                metrics_path=rel(result_dir / "metrics.json"),
                config_path=rel(result_dir / "effective_config.json"),
                notes="original low-dimensional federated base sweep",
            )
        )
    return artifacts


def discover_tuned_sine() -> list[Artifact]:
    path = ROOT / "experiments" / "sine_fedogda_tuning" / "a2_lite_fedogda_d_seed_metrics.csv"
    if not path.exists():
        return []
    rows = pd.read_csv(path)
    artifacts: list[Artifact] = []
    for _, row in rows.iterrows():
        result_dir = ROOT / str(row["result_dir"])
        artifacts.append(
            Artifact(
                source_family="tuned_sine_a2_lite",
                dataset="sin",
                method="fedogda_d",
                method_label="Tuned FedOGDA-D",
                alpha="1.0",
                seed=int(row["seed"]),
                result_dir=rel(result_dir),
                prediction_path=rel(result_dir / "predictions.npz"),
                metrics_path=rel(result_dir / "metrics.json"),
                config_path=rel(result_dir / "effective_config.json"),
                notes="validation-locked deterministic Sine A2-lite FedOGDA-D",
            )
        )
    return artifacts


def discover_pilot() -> list[Artifact]:
    path = (
        ROOT
        / "experiments"
        / "rerun_protocol_v1"
        / "tuning_fedogda_s"
        / "pilot_alpha0p5"
        / "stability_metrics.csv"
    )
    if not path.exists():
        return []
    selected_path = path.parent / "selected_seed_metrics.csv"
    selected_ids: set[str] = set()
    if selected_path.exists():
        selected_ids = set(pd.read_csv(selected_path)["run_id"].astype(str))
    artifacts: list[Artifact] = []
    for _, row in pd.read_csv(path).iterrows():
        result_dir = ROOT / str(row["result_dir"])
        is_selected = str(row["run_id"]) in selected_ids
        artifacts.append(
            Artifact(
                source_family="fedogda_s_tuning_pilot_selected"
                if is_selected
                else "fedogda_s_tuning_pilot_all_configs",
                dataset=str(row["dataset"]),
                method="fedogda_s",
                method_label="FedOGDA-S",
                alpha=str(row["alpha"]),
                seed=int(row["seed"]),
                result_dir=rel(result_dir),
                prediction_path=rel(result_dir / "predictions.npz"),
                metrics_path=rel(result_dir / "metrics.json"),
                config_path=rel(result_dir / "effective_config.json"),
                notes=(
                    f"FedOGDA-S pilot cm={row['critic_multiplier']} wd={row['weight_decay']}"
                    + ("; selected by validation" if is_selected else "")
                ),
            )
        )
    return artifacts


def discover_centralized() -> list[Artifact]:
    artifacts: list[Artifact] = []
    c5_path = ROOT / "experiments" / "centralized_baselines" / "centralized_c5_final_gda_sgda_tuned_results.csv"
    if c5_path.exists():
        for _, row in pd.read_csv(c5_path).iterrows():
            result_dir = ROOT / str(row["output_dir"])
            method = str(row["method"])
            artifacts.append(
                Artifact(
                    source_family="centralized_c5_tuned_gda_sgda",
                    dataset=str(row["dataset"]),
                    method=method,
                    method_label=METHOD_LABEL[method],
                    alpha="na",
                    seed=int(row["seed"]),
                    result_dir=rel(result_dir),
                    prediction_path=rel(result_dir / "predictions.npz"),
                    metrics_path=rel(result_dir / "metrics.json"),
                    config_path=rel(result_dir / "effective_config.json"),
                    notes="C5 validation-tuned centralized GDA/SGDA",
                )
            )
    c3_path = ROOT / "experiments" / "centralized_baselines" / "centralized_c3_full_run_results.csv"
    if c3_path.exists():
        c3 = pd.read_csv(c3_path)
        for _, row in c3[c3["method"] == "oadam"].iterrows():
            result_dir = ROOT / str(row["output_dir"])
            artifacts.append(
                Artifact(
                    source_family="centralized_c3_oadam",
                    dataset=str(row["dataset"]),
                    method="oadam",
                    method_label=METHOD_LABEL["oadam"],
                    alpha="na",
                    seed=int(row["seed"]),
                    result_dir=rel(result_dir),
                    prediction_path=rel(result_dir / "predictions.npz"),
                    metrics_path=rel(result_dir / "metrics.json"),
                    config_path=rel(result_dir / "effective_config.json"),
                    notes="C3 full centralized OAdam",
                )
            )
    return artifacts


def pick_key(keys: list[str], choices: list[str]) -> str | None:
    for choice in choices:
        if choice in keys:
            return choice
    return None


def load_curve(artifact: Artifact) -> dict[str, Any]:
    prediction_path = ROOT / artifact.prediction_path
    metrics = read_json(ROOT / artifact.metrics_path)
    config = read_json(ROOT / artifact.config_path)
    base = {
        **asdict(artifact),
        "included": False,
        "skip_reason": "",
        "prediction_key_used": "",
        "final_prediction_key_used": "",
        "pred_finite": False,
        "x_shape": "",
        "true_g_shape": "",
        "pred_shape": "",
        "curve_mse": np.nan,
        "curve_mae": np.nan,
        "curve_max_abs_error": np.nan,
        "best_vs_final_max_abs_diff": np.nan,
        "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation", np.nan),
        "best_validation_mse": metrics.get("best_validation_mse", np.nan),
        "best_validation_round": metrics.get("best_validation_round", np.nan),
        "final_test_mse": metrics.get("final_test_mse", metrics.get("test_mse_final", np.nan)),
        "selection_metric_source": metrics.get("selection_metric_source", config.get("selection_metric_source", "")),
        "test_mse_used_for_selection": metrics.get("test_mse_used_for_selection", config.get("test_mse_used_for_selection", "")),
    }
    if not prediction_path.exists():
        base["skip_reason"] = "predictions.npz missing"
        return base
    try:
        data = np.load(prediction_path, allow_pickle=True)
        keys = list(data.files)
        x_key = "x" if "x" in keys else None
        true_key = "true_g" if "true_g" in keys else None
        pred_key = pick_key(keys, ["best_validation_prediction", "best_prediction", "pred_best"])
        final_key = pick_key(keys, ["final_prediction", "pred_final"])
        if pred_key is None:
            fallback = [
                key
                for key in keys
                if key not in {"x", "true_g", "algorithm", "variant", "seed", "run_id"}
                and np.asarray(data[key]).ndim > 0
            ]
            pred_key = fallback[0] if fallback else None
        if x_key is None or true_key is None or pred_key is None:
            base["skip_reason"] = f"missing required keys; keys={keys}"
            return base
        x = np.asarray(data[x_key], dtype=float).reshape(-1)
        true = np.asarray(data[true_key], dtype=float).reshape(-1)
        pred = np.asarray(data[pred_key], dtype=float).reshape(-1)
        base["x_shape"] = str(tuple(np.asarray(data[x_key]).shape))
        base["true_g_shape"] = str(tuple(np.asarray(data[true_key]).shape))
        base["pred_shape"] = str(tuple(np.asarray(data[pred_key]).shape))
        base["prediction_key_used"] = pred_key
        if not (x.size == true.size == pred.size):
            base["skip_reason"] = f"shape mismatch x={x.size} true={true.size} pred={pred.size}"
            return base
        finite = bool(np.isfinite(x).all() and np.isfinite(true).all() and np.isfinite(pred).all())
        base["pred_finite"] = finite
        if not finite:
            base["skip_reason"] = "non-finite x/true/pred"
            return base
        diff = pred - true
        base["curve_mse"] = float(np.mean(diff**2))
        base["curve_mae"] = float(np.mean(np.abs(diff)))
        base["curve_max_abs_error"] = float(np.max(np.abs(diff)))
        if final_key:
            final_pred = np.asarray(data[final_key], dtype=float).reshape(-1)
            base["final_prediction_key_used"] = final_key
            if final_pred.size == pred.size and np.isfinite(final_pred).all():
                base["best_vs_final_max_abs_diff"] = float(np.max(np.abs(pred - final_pred)))
        order = np.argsort(x)
        base["x"] = x[order]
        base["true_g"] = true[order]
        base["pred"] = pred[order]
        base["included"] = True
        return base
    except Exception as exc:  # pragma: no cover - reporting script robustness
        base["skip_reason"] = f"{type(exc).__name__}: {exc}"
        return base


def arrays_align(curves: list[dict[str, Any]]) -> bool:
    if not curves:
        return False
    ref_x = curves[0]["x"]
    ref_true = curves[0]["true_g"]
    for curve in curves[1:]:
        if curve["x"].shape != ref_x.shape or curve["true_g"].shape != ref_true.shape:
            return False
        if not np.allclose(curve["x"], ref_x, rtol=0, atol=1e-12):
            return False
        if not np.allclose(curve["true_g"], ref_true, rtol=0, atol=1e-12):
            return False
    return True


def plot_curves(
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    true_curve: tuple[np.ndarray, np.ndarray],
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    for path in [png_path.parent, pdf_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    x_true, y_true = true_curve
    ax.plot(x_true, y_true, label="true g", **LINE_STYLE["true g"])
    for label, x, y in curves:
        style = LINE_STYLE.get(label, {"linewidth": 1.6})
        ax.plot(x, y, label=label, **style)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(fontsize=8)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_aggregate(
    method_curves: dict[str, list[dict[str, Any]]],
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> tuple[bool, str]:
    all_curves = [curve for curves in method_curves.values() for curve in curves]
    if not arrays_align(all_curves):
        return False, "x/true_g grids do not align across seeds/methods"
    ref = all_curves[0]
    curves_to_plot: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, curves in method_curves.items():
        preds = np.vstack([curve["pred"] for curve in curves])
        mean_pred = preds.mean(axis=0)
        curves_to_plot.append((label, ref["x"], mean_pred))
    plot_curves(curves_to_plot, (ref["x"], ref["true_g"]), title, png_path, pdf_path)
    return True, ""


def save_plot_index(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(CSV / "curve_fit_plot_index.csv", index=False)
    return df


def png_pdf(subdir: str, name: str) -> tuple[Path, Path]:
    return PNG / subdir / f"{name}.png", PDF / subdir / f"{name}.pdf"


def make_main_pairwise(curves: pd.DataFrame, curve_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = curves[curves["source_family"] == "base_sweep"]
    for dataset in DATASETS:
        for alpha in ["0.1", "0.5", "1.0"]:
            alpha_rows = []
            for seed in [0, 1, 2]:
                g = base[
                    (base["dataset"] == dataset)
                    & (base["method"] == "fedgda_d")
                    & (base["alpha"].astype(str) == alpha)
                    & (base["seed"].astype(str) == str(seed))
                ]
                o = base[
                    (base["dataset"] == dataset)
                    & (base["method"] == "fedogda_d")
                    & (base["alpha"].astype(str) == alpha)
                    & (base["seed"].astype(str) == str(seed))
                ]
                if g.empty or o.empty:
                    continue
                cg = curve_map[g.iloc[0]["artifact_id"]]
                co = curve_map[o.iloc[0]["artifact_id"]]
                if not arrays_align([cg, co]):
                    rows.append(
                        {
                            "plot_family": "main_pairwise",
                            "dataset": dataset,
                            "alpha": alpha,
                            "seed": seed,
                            "status": "skipped",
                            "reason": "paired x/true_g grids do not align",
                        }
                    )
                    continue
                name = f"{dataset}_alpha{alpha_slug(alpha)}_seed{seed}_fedgda_d_vs_fedogda_d"
                png_path, pdf_path = png_pdf("main_pairwise", name)
                plot_curves(
                    [
                        ("FedGDA-D", cg["x"], cg["pred"]),
                        ("FedOGDA-D", co["x"], co["pred"]),
                    ],
                    (cg["x"], cg["true_g"]),
                    f"{DATASET_LABEL[dataset]} alpha={alpha} seed={seed}: FedGDA-D vs FedOGDA-D",
                    png_path,
                    pdf_path,
                )
                rows.append(
                    {
                        "plot_family": "main_pairwise",
                        "dataset": dataset,
                        "alpha": alpha,
                        "seed": seed,
                        "status": "created",
                        "png_path": rel(png_path),
                        "pdf_path": rel(pdf_path),
                        "methods": "FedGDA-D|FedOGDA-D",
                    }
                )
                alpha_rows.extend([cg, co])
            method_curves = {
                "FedGDA-D": [
                    curve_map[row["artifact_id"]]
                    for _, row in base[
                        (base["dataset"] == dataset)
                        & (base["method"] == "fedgda_d")
                        & (base["alpha"].astype(str) == alpha)
                    ].iterrows()
                ],
                "FedOGDA-D": [
                    curve_map[row["artifact_id"]]
                    for _, row in base[
                        (base["dataset"] == dataset)
                        & (base["method"] == "fedogda_d")
                        & (base["alpha"].astype(str) == alpha)
                    ].iterrows()
                ],
            }
            if all(len(v) == 3 for v in method_curves.values()):
                name = f"{dataset}_alpha{alpha_slug(alpha)}_fedgda_d_vs_fedogda_d_mean"
                png_path, pdf_path = png_pdf("main_pairwise_aggregate", name)
                ok, reason = plot_aggregate(
                    method_curves,
                    f"{DATASET_LABEL[dataset]} alpha={alpha}: mean FedGDA-D vs FedOGDA-D",
                    png_path,
                    pdf_path,
                )
                rows.append(
                    {
                        "plot_family": "main_pairwise_aggregate",
                        "dataset": dataset,
                        "alpha": alpha,
                        "seed": "aggregate",
                        "status": "created" if ok else "skipped",
                        "reason": reason,
                        "png_path": rel(png_path) if ok else "",
                        "pdf_path": rel(pdf_path) if ok else "",
                        "methods": "FedGDA-D|FedOGDA-D",
                    }
                )
    return rows


def make_all_methods_original(curves: pd.DataFrame, curve_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = curves[curves["source_family"] == "base_sweep"]
    for dataset in DATASETS:
        for alpha in ["0.1", "0.5", "1.0"]:
            for seed in [0, 1, 2]:
                selected = []
                labels = []
                for method in FED_METHODS:
                    sub = base[
                        (base["dataset"] == dataset)
                        & (base["method"] == method)
                        & (base["alpha"].astype(str) == alpha)
                        & (base["seed"].astype(str) == str(seed))
                    ]
                    if not sub.empty:
                        selected.append(curve_map[sub.iloc[0]["artifact_id"]])
                        labels.append(METHOD_LABEL[method])
                if len(selected) < 2:
                    continue
                if not arrays_align(selected):
                    rows.append({"plot_family": "all_methods_original", "dataset": dataset, "alpha": alpha, "seed": seed, "status": "skipped", "reason": "x/true_g grids do not align"})
                    continue
                name = f"{dataset}_alpha{alpha_slug(alpha)}_seed{seed}_all_federated_methods"
                png_path, pdf_path = png_pdf("all_methods_original", name)
                plot_curves(
                    [(label, curve["x"], curve["pred"]) for label, curve in zip(labels, selected)],
                    (selected[0]["x"], selected[0]["true_g"]),
                    f"{DATASET_LABEL[dataset]} alpha={alpha} seed={seed}: all federated methods",
                    png_path,
                    pdf_path,
                )
                rows.append({"plot_family": "all_methods_original", "dataset": dataset, "alpha": alpha, "seed": seed, "status": "created", "png_path": rel(png_path), "pdf_path": rel(pdf_path), "methods": "|".join(labels)})
            method_curves = {}
            for method in FED_METHODS:
                label = METHOD_LABEL[method]
                sub = base[(base["dataset"] == dataset) & (base["method"] == method) & (base["alpha"].astype(str) == alpha)]
                method_curves[label] = [curve_map[row["artifact_id"]] for _, row in sub.iterrows()]
            if all(len(v) == 3 for v in method_curves.values()):
                name = f"{dataset}_alpha{alpha_slug(alpha)}_all_federated_methods_mean"
                png_path, pdf_path = png_pdf("all_methods_original_aggregate", name)
                ok, reason = plot_aggregate(method_curves, f"{DATASET_LABEL[dataset]} alpha={alpha}: mean all federated methods", png_path, pdf_path)
                rows.append({"plot_family": "all_methods_original_aggregate", "dataset": dataset, "alpha": alpha, "seed": "aggregate", "status": "created" if ok else "skipped", "reason": reason, "png_path": rel(png_path) if ok else "", "pdf_path": rel(pdf_path) if ok else "", "methods": "|".join(method_curves)})
    return rows


def make_tuned_sine(curves: pd.DataFrame, curve_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tuned = curves[curves["source_family"] == "tuned_sine_a2_lite"]
    base = curves[curves["source_family"] == "base_sweep"]
    method_curves = {"FedGDA-D": [], "Tuned FedOGDA-D": []}
    for seed in [0, 1, 2]:
        g = base[(base["dataset"] == "sin") & (base["method"] == "fedgda_d") & (base["alpha"].astype(str) == "1.0") & (base["seed"].astype(str) == str(seed))]
        o = tuned[tuned["seed"].astype(str) == str(seed)]
        if g.empty or o.empty:
            continue
        cg = curve_map[g.iloc[0]["artifact_id"]]
        co = curve_map[o.iloc[0]["artifact_id"]]
        if not arrays_align([cg, co]):
            rows.append({"plot_family": "tuned_sine_a2_lite", "dataset": "sin", "alpha": "1.0", "seed": seed, "status": "skipped", "reason": "paired x/true_g grids do not align"})
            continue
        name = f"sine_a2_lite_seed{seed}_fedgda_d_vs_fedogda_d"
        png_path, pdf_path = png_pdf("tuned_sine_a2_lite", name)
        plot_curves(
            [("FedGDA-D", cg["x"], cg["pred"]), ("Tuned FedOGDA-D", co["x"], co["pred"])],
            (cg["x"], cg["true_g"]),
            f"Sine tuned A2-lite seed={seed}: FedGDA-D vs tuned FedOGDA-D",
            png_path,
            pdf_path,
        )
        rows.append({"plot_family": "tuned_sine_a2_lite", "dataset": "sin", "alpha": "1.0", "seed": seed, "status": "created", "png_path": rel(png_path), "pdf_path": rel(pdf_path), "methods": "FedGDA-D|Tuned FedOGDA-D"})
        method_curves["FedGDA-D"].append(cg)
        method_curves["Tuned FedOGDA-D"].append(co)
    if all(len(v) == 3 for v in method_curves.values()):
        png_path, pdf_path = png_pdf("tuned_sine_a2_lite", "sine_a2_lite_all_seeds_mean")
        ok, reason = plot_aggregate(method_curves, "Sine tuned A2-lite: mean across seeds", png_path, pdf_path)
        rows.append({"plot_family": "tuned_sine_a2_lite_aggregate", "dataset": "sin", "alpha": "1.0", "seed": "aggregate", "status": "created" if ok else "skipped", "reason": reason, "png_path": rel(png_path) if ok else "", "pdf_path": rel(pdf_path) if ok else "", "methods": "FedGDA-D|Tuned FedOGDA-D"})
    return rows


def select_summary_settings(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = metrics[metrics["source_family"] == "base_sweep"]
    for dataset in ["abs", "step", "linear"]:
        rows = []
        for alpha in ["0.1", "0.5", "1.0"]:
            g = base[(base["dataset"] == dataset) & (base["method"] == "fedgda_d") & (base["alpha"].astype(str) == alpha)]
            o = base[(base["dataset"] == dataset) & (base["method"] == "fedogda_d") & (base["alpha"].astype(str) == alpha)]
            if len(g) == 3 and len(o) == 3:
                rows.append(
                    {
                        "alpha": alpha,
                        "validation_improvement": float(g["best_validation_mse"].mean() - o["best_validation_mse"].mean()),
                    }
                )
        best = max(rows, key=lambda row: row["validation_improvement"])
        out[dataset] = {"alpha": best["alpha"], "source": "base_sweep", "selection_rule": "largest mean best-validation-MSE improvement across seeds"}
    out["sin"] = {"alpha": "1.0", "source": "tuned_sine_a2_lite", "selection_rule": "pre-locked tuned Sine A2-lite validation-only recipe"}
    return out


def make_coauthor_summary(curves: pd.DataFrame, curve_map: dict[str, dict[str, Any]], metrics: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    settings = select_summary_settings(metrics)
    panel_items = []
    for dataset, setting in settings.items():
        alpha = setting["alpha"]
        if dataset == "sin":
            gsub = curves[(curves["source_family"] == "base_sweep") & (curves["dataset"] == "sin") & (curves["method"] == "fedgda_d") & (curves["alpha"].astype(str) == "1.0")]
            osub = curves[curves["source_family"] == "tuned_sine_a2_lite"]
            labels = ["FedGDA-D", "Tuned FedOGDA-D"]
        else:
            gsub = curves[(curves["source_family"] == "base_sweep") & (curves["dataset"] == dataset) & (curves["method"] == "fedgda_d") & (curves["alpha"].astype(str) == alpha)]
            osub = curves[(curves["source_family"] == "base_sweep") & (curves["dataset"] == dataset) & (curves["method"] == "fedogda_d") & (curves["alpha"].astype(str) == alpha)]
            labels = ["FedGDA-D", "FedOGDA-D"]
        method_curves = {
            labels[0]: [curve_map[row["artifact_id"]] for _, row in gsub.iterrows()],
            labels[1]: [curve_map[row["artifact_id"]] for _, row in osub.iterrows()],
        }
        name = f"{dataset}_summary_curve"
        png_path, pdf_path = png_pdf("coauthor_summary", name)
        ok, reason = plot_aggregate(method_curves, f"{DATASET_LABEL[dataset]} summary curve ({setting['source']}, alpha={alpha})", png_path, pdf_path)
        if not ok:
            # Fall back to seed 0, explicitly labeled.
            g0 = method_curves[labels[0]][0]
            o0 = method_curves[labels[1]][0]
            if arrays_align([g0, o0]):
                plot_curves(
                    [(labels[0], g0["x"], g0["pred"]), (labels[1], o0["x"], o0["pred"])],
                    (g0["x"], g0["true_g"]),
                    f"{DATASET_LABEL[dataset]} summary curve seed=0 ({setting['source']}, alpha={alpha})",
                    png_path,
                    pdf_path,
                )
                ok = True
                reason = "aggregate skipped; used seed 0 because grids did not align"
                panel_items.append((dataset, labels, g0, o0, setting, reason))
        else:
            ref = method_curves[labels[0]][0]
            p1 = np.vstack([c["pred"] for c in method_curves[labels[0]]]).mean(axis=0)
            p2 = np.vstack([c["pred"] for c in method_curves[labels[1]]]).mean(axis=0)
            panel_items.append((dataset, labels, {"x": ref["x"], "true_g": ref["true_g"], "pred": p1}, {"x": ref["x"], "true_g": ref["true_g"], "pred": p2}, setting, "aggregate mean"))
        rows.append({"plot_family": "coauthor_summary", "dataset": dataset, "alpha": alpha, "seed": "aggregate_or_seed0", "status": "created" if ok else "skipped", "reason": reason, "png_path": rel(png_path) if ok else "", "pdf_path": rel(pdf_path) if ok else "", "methods": "|".join(labels), "selection_rule": setting["selection_rule"]})
    if len(panel_items) == 4:
        for extension, base in [("png", PNG), ("pdf", PDF)]:
            out = base / "coauthor_summary" / f"lowdim_deterministic_summary_2x2.{extension}"
            fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
            for ax, (dataset, labels, gcurve, ocurve, setting, note) in zip(axes.reshape(-1), panel_items):
                ax.plot(gcurve["x"], gcurve["true_g"], label="true g", **LINE_STYLE["true g"])
                ax.plot(gcurve["x"], gcurve["pred"], label=labels[0], **LINE_STYLE.get(labels[0], {}))
                ax.plot(ocurve["x"], ocurve["pred"], label=labels[1], **LINE_STYLE.get(labels[1], LINE_STYLE.get("FedOGDA-D", {})))
                suffix = " tuned A2-lite" if dataset == "sin" else f" alpha={setting['alpha']}"
                ax.set_title(f"{DATASET_LABEL[dataset]}{suffix}", fontsize=10)
                ax.set_xlabel("x")
                ax.set_ylabel("g(x)")
                ax.grid(True, linewidth=0.3, alpha=0.35)
                ax.legend(fontsize=7)
            fig.savefig(out, dpi=220 if extension == "png" else None)
            plt.close(fig)
        rows.append({"plot_family": "coauthor_summary_2x2", "dataset": "lowdim", "alpha": "selected_by_validation", "seed": "aggregate_or_seed0", "status": "created", "png_path": rel(PNG / "coauthor_summary" / "lowdim_deterministic_summary_2x2.png"), "pdf_path": rel(PDF / "coauthor_summary" / "lowdim_deterministic_summary_2x2.pdf"), "methods": "FedGDA-D|FedOGDA-D/Tuned FedOGDA-D", "selection_rule": "per-function validation-only summary setting"})
    return rows, settings


def make_centralized(curves: pd.DataFrame, curve_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    central = curves[curves["source_family"].isin(["centralized_c5_tuned_gda_sgda", "centralized_c3_oadam"])]
    for dataset in DATASETS:
        # Seed 0 per-function plot.
        selected = []
        labels = []
        for method in ["gda", "sgda", "oadam"]:
            sub = central[(central["dataset"] == dataset) & (central["method"] == method) & (central["seed"].astype(str) == "0")]
            if not sub.empty:
                selected.append(curve_map[sub.iloc[0]["artifact_id"]])
                labels.append(METHOD_LABEL[method])
        if len(selected) == 3 and arrays_align(selected):
            name = f"{dataset}_centralized_seed0_gda_sgda_oadam"
            png_path, pdf_path = png_pdf("centralized", name)
            plot_curves(
                [(label, curve["x"], curve["pred"]) for label, curve in zip(labels, selected)],
                (selected[0]["x"], selected[0]["true_g"]),
                f"{DATASET_LABEL[dataset]} centralized seed=0: GDA vs SGDA vs OAdam",
                png_path,
                pdf_path,
            )
            rows.append({"plot_family": "centralized", "dataset": dataset, "alpha": "na", "seed": 0, "status": "created", "png_path": rel(png_path), "pdf_path": rel(pdf_path), "methods": "|".join(labels)})
        method_curves = {}
        for method in ["gda", "sgda", "oadam"]:
            label = METHOD_LABEL[method]
            sub = central[(central["dataset"] == dataset) & (central["method"] == method)]
            method_curves[label] = [curve_map[row["artifact_id"]] for _, row in sub.iterrows()]
        if all(len(v) == 3 for v in method_curves.values()):
            name = f"{dataset}_centralized_gda_sgda_oadam_mean"
            png_path, pdf_path = png_pdf("centralized_aggregate", name)
            ok, reason = plot_aggregate(method_curves, f"{DATASET_LABEL[dataset]} centralized: mean GDA/SGDA/OAdam", png_path, pdf_path)
            rows.append({"plot_family": "centralized_aggregate", "dataset": dataset, "alpha": "na", "seed": "aggregate", "status": "created" if ok else "skipped", "reason": reason, "png_path": rel(png_path) if ok else "", "pdf_path": rel(pdf_path) if ok else "", "methods": "|".join(method_curves)})
    return rows


def make_pilot_plots(curves: pd.DataFrame, curve_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = curves[curves["source_family"] == "fedogda_s_tuning_pilot_selected"]
    base = curves[curves["source_family"] == "base_sweep"]
    for dataset in sorted(selected["dataset"].unique()):
        method_curves = {"FedGDA-S": [], "FedOGDA-S": []}
        for seed in [0, 1, 2]:
            g = base[(base["dataset"] == dataset) & (base["method"] == "fedgda_s") & (base["alpha"].astype(str) == "0.5") & (base["seed"].astype(str) == str(seed))]
            o = selected[(selected["dataset"] == dataset) & (selected["seed"].astype(str) == str(seed))]
            if g.empty or o.empty:
                continue
            cg = curve_map[g.iloc[0]["artifact_id"]]
            co = curve_map[o.iloc[0]["artifact_id"]]
            if not arrays_align([cg, co]):
                rows.append({"plot_family": "fedogda_s_tuning_pilot", "dataset": dataset, "alpha": "0.5", "seed": seed, "status": "skipped", "reason": "paired x/true_g grids do not align"})
                continue
            name = f"{dataset}_alpha0p5_seed{seed}_fedgda_s_vs_tuned_fedogda_s"
            png_path, pdf_path = png_pdf("fedogda_s_tuning_pilot", name)
            plot_curves(
                [("FedGDA-S", cg["x"], cg["pred"]), ("FedOGDA-S", co["x"], co["pred"])],
                (cg["x"], cg["true_g"]),
                f"{DATASET_LABEL[dataset]} alpha=0.5 seed={seed}: FedGDA-S vs tuned FedOGDA-S",
                png_path,
                pdf_path,
            )
            rows.append({"plot_family": "fedogda_s_tuning_pilot", "dataset": dataset, "alpha": "0.5", "seed": seed, "status": "created", "png_path": rel(png_path), "pdf_path": rel(pdf_path), "methods": "FedGDA-S|FedOGDA-S"})
            method_curves["FedGDA-S"].append(cg)
            method_curves["FedOGDA-S"].append(co)
        if all(len(v) == 3 for v in method_curves.values()):
            name = f"{dataset}_alpha0p5_fedgda_s_vs_tuned_fedogda_s_mean"
            png_path, pdf_path = png_pdf("fedogda_s_tuning_pilot_aggregate", name)
            ok, reason = plot_aggregate(method_curves, f"{DATASET_LABEL[dataset]} alpha=0.5 stochastic pilot mean", png_path, pdf_path)
            rows.append({"plot_family": "fedogda_s_tuning_pilot_aggregate", "dataset": dataset, "alpha": "0.5", "seed": "aggregate", "status": "created" if ok else "skipped", "reason": reason, "png_path": rel(png_path) if ok else "", "pdf_path": rel(pdf_path) if ok else "", "methods": "FedGDA-S|FedOGDA-S"})
    return rows


def pairwise_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = metrics[metrics["source_family"] == "base_sweep"]
    for dataset in DATASETS:
        for alpha in ["0.1", "0.5", "1.0"]:
            for seed in [0, 1, 2]:
                g = base[(base["dataset"] == dataset) & (base["method"] == "fedgda_d") & (base["alpha"].astype(str) == alpha) & (base["seed"].astype(str) == str(seed))]
                o = base[(base["dataset"] == dataset) & (base["method"] == "fedogda_d") & (base["alpha"].astype(str) == alpha) & (base["seed"].astype(str) == str(seed))]
                if g.empty or o.empty:
                    continue
                g, o = g.iloc[0], o.iloc[0]
                rows.append(
                    {
                        "source_family": "base_sweep",
                        "dataset": dataset,
                        "function": DATASET_LABEL[dataset],
                        "alpha": alpha,
                        "seed": seed,
                        "fedgda_curve_mse": g["curve_mse"],
                        "fedogda_curve_mse": o["curve_mse"],
                        "curve_mse_gap_fedogda_minus_fedgda": o["curve_mse"] - g["curve_mse"],
                        "fedgda_curve_mae": g["curve_mae"],
                        "fedogda_curve_mae": o["curve_mae"],
                        "curve_mae_gap_fedogda_minus_fedgda": o["curve_mae"] - g["curve_mae"],
                        "fedgda_curve_max_abs_error": g["curve_max_abs_error"],
                        "fedogda_curve_max_abs_error": o["curve_max_abs_error"],
                        "curve_max_abs_gap_fedogda_minus_fedgda": o["curve_max_abs_error"] - g["curve_max_abs_error"],
                        "fedgda_test_mse_at_best_validation": g["test_mse_at_best_validation"],
                        "fedogda_test_mse_at_best_validation": o["test_mse_at_best_validation"],
                        "test_mse_gap_fedogda_minus_fedgda": o["test_mse_at_best_validation"] - g["test_mse_at_best_validation"],
                        "winner_by_curve_mse": "FedOGDA-D" if o["curve_mse"] < g["curve_mse"] else "FedGDA-D",
                        "winner_by_test_mse": "FedOGDA-D" if o["test_mse_at_best_validation"] < g["test_mse_at_best_validation"] else "FedGDA-D",
                    }
                )
    tuned = metrics[metrics["source_family"] == "tuned_sine_a2_lite"]
    for seed in [0, 1, 2]:
        g = base[(base["dataset"] == "sin") & (base["method"] == "fedgda_d") & (base["alpha"].astype(str) == "1.0") & (base["seed"].astype(str) == str(seed))]
        o = tuned[tuned["seed"].astype(str) == str(seed)]
        if g.empty or o.empty:
            continue
        g, o = g.iloc[0], o.iloc[0]
        rows.append(
            {
                "source_family": "tuned_sine_a2_lite",
                "dataset": "sin",
                "function": "Sine",
                "alpha": "1.0",
                "seed": seed,
                "fedgda_curve_mse": g["curve_mse"],
                "fedogda_curve_mse": o["curve_mse"],
                "curve_mse_gap_fedogda_minus_fedgda": o["curve_mse"] - g["curve_mse"],
                "fedgda_curve_mae": g["curve_mae"],
                "fedogda_curve_mae": o["curve_mae"],
                "curve_mae_gap_fedogda_minus_fedgda": o["curve_mae"] - g["curve_mae"],
                "fedgda_curve_max_abs_error": g["curve_max_abs_error"],
                "fedogda_curve_max_abs_error": o["curve_max_abs_error"],
                "curve_max_abs_gap_fedogda_minus_fedgda": o["curve_max_abs_error"] - g["curve_max_abs_error"],
                "fedgda_test_mse_at_best_validation": g["test_mse_at_best_validation"],
                "fedogda_test_mse_at_best_validation": o["test_mse_at_best_validation"],
                "test_mse_gap_fedogda_minus_fedgda": o["test_mse_at_best_validation"] - g["test_mse_at_best_validation"],
                "winner_by_curve_mse": "FedOGDA-D" if o["curve_mse"] < g["curve_mse"] else "FedGDA-D",
                "winner_by_test_mse": "FedOGDA-D" if o["test_mse_at_best_validation"] < g["test_mse_at_best_validation"] else "FedGDA-D",
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if math.isnan(val) else f"{val:.6g}")
            else:
                vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    plot_index: pd.DataFrame,
    summary_settings: dict[str, dict[str, Any]],
) -> None:
    included = int(inventory["included"].sum())
    skipped = int((~inventory["included"].astype(bool)).sum())
    created = plot_index[plot_index["status"] == "created"]
    skipped_plots = plot_index[plot_index["status"] == "skipped"]
    agg_metrics = (
        metrics[metrics["included"]]
        .groupby(["source_family", "dataset", "method_label", "alpha"], dropna=False)
        .agg(
            runs=("artifact_id", "count"),
            mean_curve_mse=("curve_mse", "mean"),
            mean_curve_mae=("curve_mae", "mean"),
            mean_curve_max_abs_error=("curve_max_abs_error", "mean"),
            mean_test_mse_at_best_validation=("test_mse_at_best_validation", "mean"),
        )
        .reset_index()
    )
    base_pair_summary = (
        pairwise.groupby(["source_family", "function"], dropna=False)
        .agg(
            pairs=("seed", "count"),
            fedogda_curve_mse_wins=("winner_by_curve_mse", lambda s: int((s == "FedOGDA-D").sum())),
            fedogda_test_mse_wins=("winner_by_test_mse", lambda s: int((s == "FedOGDA-D").sum())),
            mean_curve_mse_gap=("curve_mse_gap_fedogda_minus_fedgda", "mean"),
            mean_test_mse_gap=("test_mse_gap_fedogda_minus_fedgda", "mean"),
        )
        .reset_index()
    )
    recommended_rows = []

    def add_recommended(plot_family: str, dataset: str, alpha: str | None = None) -> None:
        sub = created[(created["plot_family"] == plot_family) & (created["dataset"] == dataset)]
        if alpha is not None:
            sub = sub[sub["alpha"].astype(str) == str(alpha)]
        if not sub.empty:
            recommended_rows.append(sub.iloc[0].to_dict())

    add_recommended("coauthor_summary_2x2", "lowdim")
    add_recommended("tuned_sine_a2_lite_aggregate", "sin")
    for dataset in ["abs", "step", "linear"]:
        setting = summary_settings.get(dataset, {})
        add_recommended("main_pairwise_aggregate", dataset, setting.get("alpha"))
    # Keep centralized recommendations compact: one easy function, one linear
    # function, and Sine because it is central to the current coauthor question.
    for dataset in ["abs", "linear", "sin"]:
        add_recommended("centralized_aggregate", dataset)
    recommended = pd.DataFrame(recommended_rows)
    summary_rows = []
    for dataset, setting in summary_settings.items():
        summary_rows.append({"function": DATASET_LABEL[dataset], **setting})
    report = f"""# Curve-Fitting Plot Report

Generated: {datetime.now().isoformat(timespec='seconds')}

## 1. Executive Summary

- Prediction artifacts considered: {len(inventory)}
- Prediction artifacts used in metrics: {included}
- Prediction artifacts skipped: {skipped}
- Plot files created: {len(created) * 2} ({len(created)} PNG + {len(created)} PDF)
- Plot requests skipped: {len(skipped_plots)}{'; skipped aggregate plots indicate non-identical x/true_g grids across seeds/methods.' if len(skipped_plots) else '.'}
- Main deterministic FedOGDA-D vs FedGDA-D plots were generated for the original low-dimensional sweep. Tuned Sine A2-lite plots are separate.
- Centralized plots use reportable centralized outputs: C5 tuned GDA/SGDA plus C3 OAdam. Tiny smoke is not included.
- High-dimensional FEMNIST/CIFAR is not applicable for curve-fitting here because final validated low-dimensional-style `x`/`true_g` prediction artifacts are not complete.

## 2. Metric Direction and Selection Policy

Lower curve MSE, MAE, and max absolute error are better. The prediction used is validation-selected when a validation-selected key is available, with preference order `best_validation_prediction`, `best_prediction`, then `pred_best`. Test MSE and visual appearance were not used for selection. Tuned Sine A2-lite is kept separate from the original Sine sweep.

## 3. Artifact Inventory

Full inventory CSV: `experiments/curve_fitting_plots/csv/curve_fit_artifact_inventory.csv`

Compact inventory by family:

{md_table(inventory.groupby(['source_family', 'included'], dropna=False).size().reset_index(name='count'))}

Skipped artifact reasons:

{md_table(inventory[~inventory['included'].astype(bool)].groupby(['skip_reason'], dropna=False).size().reset_index(name='count'))}

## 4. Curve-Fit Metrics

All-run metric CSV: `experiments/curve_fitting_plots/csv/curve_fit_metrics_all_runs.csv`

Aggregate metric sample:

{md_table(agg_metrics, max_rows=30)}

FedOGDA-D vs FedGDA-D pairwise summary:

{md_table(base_pair_summary)}

Pairwise CSV: `experiments/curve_fitting_plots/csv/curve_fit_pairwise_fedogda_vs_fedgda.csv`

## 5. Plot Index

Plot index CSV: `experiments/curve_fitting_plots/csv/curve_fit_plot_index.csv`

Created plot counts:

{md_table(created.groupby('plot_family').size().reset_index(name='created_plots'))}

Skipped plot counts:

{md_table(skipped_plots.groupby(['plot_family', 'reason']).size().reset_index(name='skipped_plots'))}

## 6. Recommended Plots

{md_table(recommended[['plot_family', 'dataset', 'alpha', 'seed', 'png_path', 'pdf_path', 'methods']], max_rows=8)}

Why these are useful:

- `lowdim_deterministic_summary_2x2` is the cleanest four-function visual summary.
- Tuned Sine A2-lite all-seed mean directly supports the scoped Sine claim.
- Absolute/Step/Linear aggregate pairwise plots show the original deterministic FedOGDA-D vs FedGDA-D behavior without mixing in tuning extensions.
- Centralized aggregate plots show the reportable DeepGMM centralized baselines after C5 tuning for GDA/SGDA and C3 OAdam.

Summary-plot selection rule:

{md_table(pd.DataFrame(summary_rows))}

## 7. Caveats

- Legacy base sweep lacks per-round Test MSE; curve plots use saved prediction artifacts and scalar metrics, not reconstructed last-50 Test MSE.
- Curve plots use saved sorted test-point predictions from `predictions.npz`; they are not dense-grid checkpoint re-evaluations unless the saved artifact itself is dense.
- Do not overclaim visual superiority if numeric metrics disagree. Use the CSV metrics as the authoritative numeric readout.
- Tiny centralized smoke runs are excluded from paper-facing plots.
- High-dimensional FEMNIST/CIFAR should be summarized with MSE bars/tables after validated runs exist, not with low-dimensional curve-fitting plots.
"""
    (OUT / "curve_fitting_report.md").write_text(report)


def main() -> None:
    ensure_dirs()
    artifacts = []
    artifacts.extend(discover_base_sweep())
    artifacts.extend(discover_tuned_sine())
    artifacts.extend(discover_pilot())
    artifacts.extend(discover_centralized())

    loaded = []
    curve_map: dict[str, dict[str, Any]] = {}
    for idx, artifact in enumerate(artifacts):
        curve = load_curve(artifact)
        curve["artifact_id"] = f"a{idx:04d}"
        loaded.append(curve)
        if curve["included"]:
            curve_map[curve["artifact_id"]] = curve

    inventory_cols = [
        "artifact_id",
        "source_family",
        "dataset",
        "method",
        "method_label",
        "alpha",
        "seed",
        "prediction_path",
        "result_dir",
        "prediction_key_used",
        "final_prediction_key_used",
        "pred_finite",
        "x_shape",
        "true_g_shape",
        "pred_shape",
        "included",
        "skip_reason",
        "notes",
    ]
    inventory = pd.DataFrame([{k: v for k, v in row.items() if k not in {"x", "true_g", "pred"}} for row in loaded])
    for col in inventory_cols:
        if col not in inventory.columns:
            inventory[col] = ""
    inventory[inventory_cols].to_csv(CSV / "curve_fit_artifact_inventory.csv", index=False)

    metrics_cols = [
        "artifact_id",
        "source_family",
        "dataset",
        "method",
        "method_label",
        "alpha",
        "seed",
        "prediction_key_used",
        "curve_mse",
        "curve_mae",
        "curve_max_abs_error",
        "best_vs_final_max_abs_diff",
        "best_validation_mse",
        "best_validation_round",
        "test_mse_at_best_validation",
        "final_test_mse",
        "selection_metric_source",
        "test_mse_used_for_selection",
        "included",
        "prediction_path",
        "result_dir",
    ]
    metrics = pd.DataFrame([{k: v for k, v in row.items() if k not in {"x", "true_g", "pred"}} for row in loaded])
    metrics[metrics_cols].to_csv(CSV / "curve_fit_metrics_all_runs.csv", index=False)

    included = metrics[metrics["included"].astype(bool)].copy()
    plot_rows: list[dict[str, Any]] = []
    plot_rows.extend(make_main_pairwise(included, curve_map))
    plot_rows.extend(make_all_methods_original(included, curve_map))
    plot_rows.extend(make_tuned_sine(included, curve_map))
    summary_rows, summary_settings = make_coauthor_summary(included, curve_map, included)
    plot_rows.extend(summary_rows)
    plot_rows.extend(make_centralized(included, curve_map))
    plot_rows.extend(make_pilot_plots(included, curve_map))
    plot_index = save_plot_index(plot_rows)

    pairwise = pairwise_metrics(included)
    pairwise.to_csv(CSV / "curve_fit_pairwise_fedogda_vs_fedgda.csv", index=False)

    write_report(inventory, metrics, pairwise, plot_index, summary_settings)

    created = int((plot_index["status"] == "created").sum())
    skipped = int((plot_index["status"] == "skipped").sum())
    print(f"Prediction artifacts considered: {len(inventory)}")
    print(f"Prediction artifacts included: {int(inventory['included'].sum())}")
    print(f"Plot requests created: {created} ({created * 2} files)")
    print(f"Plot requests skipped: {skipped}")
    print(f"Report: {rel(OUT / 'curve_fitting_report.md')}")


if __name__ == "__main__":
    main()
