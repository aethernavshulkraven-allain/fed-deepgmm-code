#!/usr/bin/env python3
"""Summarize high-dimensional stochastic GPU utilization profiling artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = REPO_ROOT / "results" / "_profiling" / "highdim_stochastic_gpu_util"
PROTOCOL_DIR = REPO_ROOT / "experiments" / "highdim_coauthor_protocol_v1"
REPORT_PATH = PROTOCOL_DIR / "stochastic_gpu_util_investigation.md"
SUMMARY_CSV = PROTOCOL_DIR / "stochastic_gpu_util_profile_summary.csv"
PRODUCTION_ROOTS = (
    REPO_ROOT / "results" / "rerun_protocol_v1_real_images_abs_alpha0p5",
    REPO_ROOT / "results" / "rerun_protocol_v1_real_images_abs_alpha0p1",
    REPO_ROOT / "results" / "rerun_protocol_v1_real_images_abs_alpha1",
)
FINAL_MANIFESTS = tuple(sorted(PROTOCOL_DIR.glob("alpha*/final_manifest_stochastic.csv")))
PRIMARY_CPU_PHASES = (
    "device_init",
    "data_load",
    "model_create",
    "runner_init",
    "runner_run",
)
PROJECTED_FINAL_ROUNDS = 1500


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 2) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{digits}f}"


def fmt_min(seconds: Any, digits: int = 1) -> str:
    numeric = safe_float(seconds)
    if numeric is None:
        return "NA"
    return f"{numeric / 60:.{digits}f}"


def fmt_hours(seconds: Any, digits: int = 1) -> str:
    numeric = safe_float(seconds)
    if numeric is None:
        return "NA"
    return f"{numeric / 3600:.{digits}f}"


def pct_fraction(value: Any) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric * 100:.1f}%"


def pct(value: Any) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.1f}%"


def phase_share(row: dict[str, Any], key: str) -> str:
    wall = safe_float(row.get("wall_seconds"))
    value = safe_float(row.get(key))
    if not wall or value is None:
        return "NA"
    return f"{value:.1f}s ({value / wall * 100:.1f}%)"


def read_runtime_profile(summary_path: Path) -> tuple[dict[str, float], dict[str, int], list[str]]:
    runtime_path = summary_path.with_name("runtime_profile.csv")
    phase_cpu = defaultdict(float)
    phase_counts = Counter()
    reg_loop_details: list[str] = []
    if not runtime_path.exists():
        return {}, {}, reg_loop_details

    with runtime_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            phase = row.get("phase") or ""
            phase_counts[phase] += 1
            phase_cpu[phase] += safe_float(row.get("process_cpu_seconds")) or 0.0
            if phase == "trainer_reg_epoch_loop_shape" and row.get("detail"):
                reg_loop_details.append(row["detail"])
    return dict(phase_cpu), dict(phase_counts), reg_loop_details


def derive_cpu_total(summary: dict[str, Any], runtime_cpu: dict[str, float]) -> float | None:
    explicit = safe_float(summary.get("process_cpu_seconds_total"))
    if explicit is not None:
        return explicit
    total = sum(runtime_cpu.get(phase, 0.0) for phase in PRIMARY_CPU_PHASES)
    return total if total > 0 else None


def phase_from_path(summary_path: Path) -> str:
    try:
        return summary_path.relative_to(PROFILE_ROOT).parts[0]
    except ValueError:
        return "unknown"


def projected_1500_seconds(row: dict[str, Any]) -> float | None:
    rounds = safe_float(row.get("completed_rounds")) or safe_float(row.get("rounds"))
    wall = safe_float(row.get("wall_seconds"))
    round_total = safe_float(row.get("round_total"))
    if not rounds or wall is None or round_total is None:
        return None
    fixed = max(0.0, wall - round_total)
    return fixed + round_total / rounds * PROJECTED_FINAL_ROUNDS


def read_profile_rows(profile_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(profile_root.glob("**/profile_summary.json")):
        summary = load_json(summary_path)
        env_path = summary_path.with_name("profile_environment.json")
        env = load_json(env_path) if env_path.exists() else {}
        metrics_path = summary_path.with_name("metrics.json")
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        config = env.get("effective_config", {})
        phases = summary.get("phase_totals_seconds", {})
        phase_cpu_summary = summary.get("phase_process_cpu_seconds", {})
        runtime_cpu, runtime_counts, reg_loop_details = read_runtime_profile(summary_path)
        cpu_total = derive_cpu_total(summary, runtime_cpu)
        wall = safe_float(summary.get("wall_clock_profile_seconds"))
        completed_rounds = (
            safe_float(summary.get("completed_rounds"))
            or safe_float(summary.get("phase_counts", {}).get("round_total"))
            or safe_float(config.get("comm_round"))
        )
        method = config.get("variant") or config.get("method") or "unknown"
        row = {
            "phase": phase_from_path(summary_path),
            "profile_dir": str(summary_path.parent.relative_to(REPO_ROOT)),
            "run_id": config.get("run_id", summary_path.parent.name),
            "dataset": config.get("dataset", "unknown"),
            "method": method,
            "seed": config.get("random_seed", ""),
            "rounds": int(config.get("comm_round", 0) or 0),
            "completed_rounds": int(completed_rounds or 0),
            "wall_seconds": wall,
            "seconds_per_round": wall / completed_rounds if wall and completed_rounds else None,
            "training_loop_seconds_per_round": (
                safe_float(phases.get("round_total")) / completed_rounds
                if phases.get("round_total") is not None and completed_rounds
                else None
            ),
            "rounds_per_second_wall": completed_rounds / wall if wall and completed_rounds else None,
            "rounds_per_second_training_loop": (
                completed_rounds / safe_float(phases.get("round_total"))
                if safe_float(phases.get("round_total")) and completed_rounds
                else None
            ),
            "cpu_seconds_total": cpu_total,
            "cpu_util_pct": cpu_total / wall * 100 if cpu_total is not None and wall else None,
            "gpu_avg_pct": summary.get("gpu_utilization_avg_pct"),
            "gpu_max_pct": summary.get("gpu_utilization_max_pct"),
            "gpu_mem_avg_mb": summary.get("gpu_memory_used_avg_mb"),
            "gpu_mem_max_mb": summary.get("gpu_memory_used_max_mb"),
            "gpu_power_avg_w": summary.get("gpu_power_avg_w"),
            "gpu_power_max_w": summary.get("gpu_power_max_w"),
            "gpu_samples": summary.get("gpu_telemetry_sample_count"),
            "explained_fraction": summary.get("top_level_explained_fraction_of_wall_clock"),
            "data_load": phases.get("data_load"),
            "model_selection": phases.get("model_selection"),
            "runner_init": phases.get("runner_init"),
            "runner_run": phases.get("runner_run"),
            "training_total": phases.get("training_total"),
            "round_total": phases.get("round_total"),
            "client_train_gmm": phases.get("client_train_gmm"),
            "client_train_reg": phases.get("client_train_reg"),
            "client_train_reg_skipped": phases.get("client_train_reg_skipped"),
            "aggregate_gmm": phases.get("aggregate_gmm"),
            "aggregate_reg": phases.get("aggregate_reg"),
            "aggregate_reg_skipped": phases.get("aggregate_reg_skipped"),
            "eval_global_model": phases.get("eval_global_model"),
            "write_mse_by_round": phases.get("write_mse_by_round"),
            "client_collect_state": phases.get("client_collect_state"),
            "set_global_params": phases.get("set_global_params"),
            "trainer_reg_local_training": phases.get("trainer_reg_local_training"),
            "trainer_gmm_local_training": phases.get("trainer_gmm_local_training"),
            "trainer_set_reg_params": phases.get("trainer_set_reg_params"),
            "phase_cpu_runner_init": phase_cpu_summary.get("runner_init") or runtime_cpu.get("runner_init"),
            "phase_cpu_runner_run": phase_cpu_summary.get("runner_run") or runtime_cpu.get("runner_run"),
            "top_phases": summary.get("top_phases_by_total_seconds", [])[:8],
            "cuda_visible_devices": env.get("env", {}).get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": env.get("env", {}).get("NVIDIA_VISIBLE_DEVICES"),
            "skip_aux_reg": env.get("env", {}).get("FEDGMM_PROFILE_SKIP_AUX_REG"),
            "skip_model_selection": config.get("skip_model_selection"),
            "skip_gmm_eval": config.get("skip_gmm_eval"),
            "gmm_eval_proxy": config.get("gmm_eval_proxy"),
            "auxiliary_regression": config.get("auxiliary_regression"),
            "auxiliary_regression_epochs": config.get("auxiliary_regression_epochs"),
            "append_round_csv": config.get("append_round_csv"),
            "periodic_checkpoint_interval": config.get("periodic_checkpoint_interval"),
            "dataloader_num_workers": config.get("dataloader_num_workers"),
            "dataloader_pin_memory": config.get("dataloader_pin_memory"),
            "cuda_available": env.get("torch", {}).get("cuda_available"),
            "cuda_device_count": env.get("torch", {}).get("cuda_device_count"),
            "cuda_current_device": env.get("torch", {}).get("cuda_current_device"),
            "cuda_current_device_name": env.get("torch", {}).get("cuda_current_device_name"),
            "cuda_device_names": env.get("torch", {}).get("cuda_device_names"),
            "reg_loop_details": sorted(set(reg_loop_details)),
            "runtime_phase_counts": runtime_counts,
            "best_validation_mse": metrics.get("best_validation_mse"),
            "best_validation_round": metrics.get("best_validation_round"),
            "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation"),
            "final_validation_mse": metrics.get("final_validation_mse"),
            "final_test_mse": metrics.get("final_test_mse"),
            "diverged": metrics.get("diverged"),
        }
        row["projected_1500_seconds"] = projected_1500_seconds(row)
        rows.append(row)
    return rows


def read_production_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in PRODUCTION_ROOTS:
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("**/metrics.json")):
            config_path = metrics_path.with_name("effective_config.json")
            if not config_path.exists():
                continue
            metrics = load_json(metrics_path)
            config = load_json(config_path)
            if config.get("mode") != "stochastic" or int(config.get("comm_round", 0)) != 1500:
                continue
            rows.append({
                "run_id": config.get("run_id"),
                "dataset": config.get("dataset"),
                "method": config.get("variant"),
                "seed": config.get("random_seed"),
                "alpha": config.get("partition_alpha"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "best_validation_round": metrics.get("best_validation_round"),
                "test_mse_at_best_validation": metrics.get("test_mse_at_best_validation"),
                "path": str(metrics_path.parent.relative_to(REPO_ROOT)),
            })
    return rows


def read_plan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in FINAL_MANIFESTS:
        with manifest_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                final_dir = REPO_ROOT / row["final_result_dir"]
                metrics_path = final_dir / "metrics.json"
                mse_path = final_dir / "mse_by_round.csv"
                completed = metrics_path.exists()
                partial_rounds = 0
                if mse_path.exists():
                    with mse_path.open("r") as mse_handle:
                        partial_rounds = max(0, sum(1 for _ in mse_handle) - 1)
                status = "completed" if completed else ("partial_or_running" if partial_rounds else "pending")
                rows.append({
                    "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "seed": int(row["seed"]),
                    "alpha": float(row["alpha"]),
                    "rounds": int(row["comm_round"]),
                    "output_root": row["output_root"],
                    "final_result_dir": row["final_result_dir"],
                    "status": status,
                    "partial_rounds": partial_rounds,
                })
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "run_id",
        "dataset",
        "method",
        "seed",
        "completed_rounds",
        "wall_seconds",
        "seconds_per_round",
        "training_loop_seconds_per_round",
        "rounds_per_second_wall",
        "rounds_per_second_training_loop",
        "cpu_seconds_total",
        "cpu_util_pct",
        "gpu_avg_pct",
        "gpu_max_pct",
        "gpu_mem_avg_mb",
        "gpu_mem_max_mb",
        "gpu_power_avg_w",
        "gpu_power_max_w",
        "explained_fraction",
        "data_load",
        "model_selection",
        "round_total",
        "client_train_gmm",
        "client_train_reg",
        "client_train_reg_skipped",
        "aggregate_reg",
        "aggregate_reg_skipped",
        "eval_global_model",
        "write_mse_by_round",
        "projected_1500_seconds",
        "best_validation_mse",
        "best_validation_round",
        "test_mse_at_best_validation",
        "final_validation_mse",
        "final_test_mse",
        "diverged",
        "cuda_visible_devices",
        "nvidia_visible_devices",
        "cuda_available",
        "cuda_device_count",
        "cuda_current_device",
        "cuda_current_device_name",
        "skip_aux_reg",
        "skip_model_selection",
        "skip_gmm_eval",
        "gmm_eval_proxy",
        "auxiliary_regression",
        "auxiliary_regression_epochs",
        "append_round_csv",
        "periodic_checkpoint_interval",
        "dataloader_num_workers",
        "dataloader_pin_memory",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def production_summary(plan_rows: list[dict[str, Any]], production_rows: list[dict[str, Any]]) -> list[str]:
    if not plan_rows:
        return ["No stochastic final manifests were found."]

    by_status = Counter(row["status"] for row in plan_rows)
    runtimes = [
        safe_float(row.get("runtime_seconds"))
        for row in production_rows
        if safe_float(row.get("runtime_seconds")) is not None
    ]
    runtimes = [value for value in runtimes if value is not None]
    lines = [
        f"Planned stochastic final runs from manifests: `{len(plan_rows)}`.",
        f"Artifact status: completed `{by_status['completed']}`, partial/running `{by_status['partial_or_running']}`, pending `{by_status['pending']}`.",
    ]
    if runtimes:
        lines.append(
            "Completed 1500-round runtime seconds: "
            f"min `{min(runtimes):.1f}`, median `{statistics.median(runtimes):.1f}`, max `{max(runtimes):.1f}`."
        )
    lines.append("")
    lines.append("| alpha | dataset | method | completed | partial/running | pending |")
    lines.append("|---:|---|---|---:|---:|---:|")
    grouped: dict[tuple[float, str, str], Counter] = defaultdict(Counter)
    for row in plan_rows:
        grouped[(row["alpha"], row["dataset"], row["method"])][row["status"]] += 1
    for (alpha, dataset, method), counts in sorted(grouped.items()):
        lines.append(
            f"| {alpha:g} | {dataset} | {method} | {counts['completed']} | "
            f"{counts['partial_or_running']} | {counts['pending']} |"
        )
    return lines


def profile_summary(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [
            "No profiling summaries found yet under `results/_profiling/highdim_stochastic_gpu_util/`.",
            "Run the baseline matrix with:",
            "",
            "```bash",
            "gpurun -g 1 python scripts/run_highdim_stochastic_gpu_profile.py --phase baseline50 --rounds 50",
            "```",
        ]

    lines = [
        f"Profiling summaries found: `{len(rows)}`.",
        "",
        "### Runtime, GPU, CPU",
        "",
        "| phase | dataset | method | rounds | wall min | wall sec/round | loop sec/round | wall rounds/sec | avg GPU | max GPU | CPU util | max mem GB | explained |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['dataset']} | {row['method']} | {row['completed_rounds']} | "
            f"{fmt_min(row['wall_seconds'])} | {fmt(row['seconds_per_round'], 3)} | "
            f"{fmt(row['training_loop_seconds_per_round'], 3)} | {fmt(row['rounds_per_second_wall'], 4)} | "
            f"{pct(row['gpu_avg_pct'])} | {pct(row['gpu_max_pct'])} | {pct(row['cpu_util_pct'])} | "
            f"{fmt((safe_float(row.get('gpu_mem_max_mb')) or 0.0) / 1024, 1)} | "
            f"{pct_fraction(row['explained_fraction'])} |"
        )

    lines.extend([
        "",
        "### Phase Breakdown",
        "",
        "| phase | dataset | method | setup/model selection | round loop | GMM train | aux reg train | eval | state collect | reg aggregate | CSV write |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['dataset']} | {row['method']} | "
            f"{phase_share(row, 'model_selection')} | {phase_share(row, 'round_total')} | "
            f"{phase_share(row, 'client_train_gmm')} | {phase_share(row, 'client_train_reg')} | "
            f"{phase_share(row, 'eval_global_model')} | {phase_share(row, 'client_collect_state')} | "
            f"{phase_share(row, 'aggregate_reg')} | {phase_share(row, 'write_mse_by_round')} |"
        )

    lines.extend([
        "",
        "### CUDA Visibility",
        "",
        "| phase | dataset | method | cuda available | current device | current device name | CUDA_VISIBLE_DEVICES | NVIDIA_VISIBLE_DEVICES | torch device count |",
        "|---|---|---|---:|---:|---|---|---|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['dataset']} | {row['method']} | {row['cuda_available']} | "
            f"{row['cuda_current_device']} | {row['cuda_current_device_name']} | "
            f"{row['cuda_visible_devices']} | {row['nvidia_visible_devices']} | {row['cuda_device_count']} |"
        )

    details = sorted({detail for row in rows for detail in row.get("reg_loop_details", [])})
    if details:
        lines.extend([
            "",
            "### Nested Epoch Evidence",
            "",
            "The auxiliary regression trainer recorded this loop shape:",
        ])
        for detail in details:
            lines.append(f"- `{detail}`")
    return lines


def baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("phase") == "baseline50"
        and str(row.get("skip_aux_reg")).strip() not in {"1", "true", "True"}
    ]


def queue_projection(rows: list[dict[str, Any]], plan_rows: list[dict[str, Any]]) -> list[str]:
    base = baseline_rows(rows)
    if not base:
        return ["Queue projection unavailable until baseline profile summaries exist."]

    one_seed_alpha_cifar = sum(
        safe_float(row.get("projected_1500_seconds")) or 0.0
        for row in base
    )
    cifar_multiplier = 15  # 5 seeds x 3 alphas for the six CIFAR representative rows.
    cifar_total = one_seed_alpha_cifar * cifar_multiplier
    full_total_if_femnist_similar = cifar_total * 2
    model_selection_total = sum((safe_float(row.get("model_selection")) or 0.0) for row in base) * cifar_multiplier
    aux_reg_per_round_total = sum(
        ((safe_float(row.get("client_train_reg")) or 0.0) + (safe_float(row.get("aggregate_reg")) or 0.0))
        / max(1, int(row.get("completed_rounds") or 1))
        * PROJECTED_FINAL_ROUNDS
        for row in base
    ) * cifar_multiplier
    planned_count = len(plan_rows)
    completed_count = sum(1 for row in plan_rows if row["status"] == "completed")
    remaining_fraction = (planned_count - completed_count) / planned_count if planned_count else 1.0
    full_remaining = full_total_if_femnist_similar * remaining_fraction
    skip_model_selection = max(0.0, full_total_if_femnist_similar - model_selection_total * 2)
    skip_aux_reg = max(0.0, full_total_if_femnist_similar - aux_reg_per_round_total * 2)

    lines = [
        "Projection assumptions: baseline CIFAR profiles are scaled as `fixed setup + 1500 * measured round-loop/50`; FEMNIST is shown only as a same-cost placeholder until FEMNIST profiles are run.",
        "",
        "| estimate | 1 GPU | 2 GPUs | note |",
        "|---|---:|---:|---|",
        (
            f"| CIFAR 90-run subset, current path | {fmt_hours(cifar_total)} h | "
            f"{fmt_hours(cifar_total / 2)} h | 3 CIFAR scenarios x 2 methods x 5 seeds x 3 alphas |"
        ),
        (
            f"| Full 180-run plan if FEMNIST similar | {fmt_hours(full_total_if_femnist_similar)} h | "
            f"{fmt_hours(full_total_if_femnist_similar / 2)} h | rough; FEMNIST needs its own profile |"
        ),
        (
            f"| Remaining full plan at current completion fraction | {fmt_hours(full_remaining)} h | "
            f"{fmt_hours(full_remaining / 2)} h | uses manifest completed count |"
        ),
        (
            f"| Full plan, bypass model selection after validation | {fmt_hours(skip_model_selection)} h | "
            f"{fmt_hours(skip_model_selection / 2)} h | requires equivalence/sign-off that final configs are already fixed |"
        ),
        (
            f"| Full plan, remove measured aux-reg train+reg aggregate | {fmt_hours(skip_aux_reg)} h | "
            f"{fmt_hours(skip_aux_reg / 2)} h | lower-bound estimate; requires equivalence validation |"
        ),
    ]
    return lines


def metric_delta(base: dict[str, Any] | None, row: dict[str, Any], key: str) -> str:
    base_value = safe_float(base.get(key)) if base else None
    row_value = safe_float(row.get(key))
    if base_value is None or row_value is None:
        return "NA"
    return f"{base_value:.4f} -> {row_value:.4f} ({row_value - base_value:+.4f})"


def diagnostic_comparisons(rows: list[dict[str, Any]]) -> list[str]:
    base_by_key = {
        (row["dataset"], row["method"]): row
        for row in baseline_rows(rows)
    }
    diagnostics = [
        row
        for row in rows
        if row.get("phase") != "baseline50"
    ]
    if not diagnostics:
        return ["No diagnostic optimized profiles have completed yet."]

    lines = [
        "| diagnostic phase | dataset | method | baseline wall min | diagnostic wall min | baseline loop sec/round | diagnostic loop sec/round | best val MSE | test MSE at best val | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in diagnostics:
        base = base_by_key.get((row["dataset"], row["method"]))
        lines.append(
            f"| {row['phase']} | {row['dataset']} | {row['method']} | "
            f"{fmt_min(base.get('wall_seconds') if base else None)} | {fmt_min(row['wall_seconds'])} | "
            f"{fmt(base.get('training_loop_seconds_per_round') if base else None, 3)} | "
            f"{fmt(row['training_loop_seconds_per_round'], 3)} | "
            f"{metric_delta(base, row, 'best_validation_mse')} | "
            f"{metric_delta(base, row, 'test_mse_at_best_validation')} | "
            "diagnostic-only; compare metrics across seeds before adopting |"
        )
    return lines


def recommendations(rows: list[dict[str, Any]]) -> list[str]:
    base = baseline_rows(rows)
    avg_gpu = [
        safe_float(row.get("gpu_avg_pct"))
        for row in base
        if safe_float(row.get("gpu_avg_pct")) is not None
    ]
    model_selection = [
        safe_float(row.get("model_selection"))
        for row in base
        if safe_float(row.get("model_selection")) is not None
    ]
    reg_train = [
        safe_float(row.get("client_train_reg"))
        for row in base
        if safe_float(row.get("client_train_reg")) is not None
    ]
    lines = [
        "Ranked recommendation list:",
        "",
        "1. Implemented safe operational fix: run one independent manifest worker per available broker GPU, with disjoint output roots and `--resume-skip-completed`.",
        "2. Implemented protocol-preserving code cleanup: append round CSV rows by default, avoid repeated eval-target CPU copies, avoid unused eval-history state copies, keep auxiliary-regression state on device by default, refresh global state explicitly after aggregation, and make DataLoader workers/pinned memory configurable.",
        "3. Implemented bug fix requiring coauthor awareness: auxiliary regression now treats `auxiliary_regression_epochs=epochs` as exactly that many local passes instead of the previous nested `epochs x epochs` loop.",
        "4. Requires equivalence validation: disable auxiliary regression entirely for final GMM runs if 3-seed diagnostics confirm `g`/`f` validation and test metrics are unchanged.",
        "5. Requires equivalence validation/sign-off: bypass repeated model-selection setup in final runs if the validation-selected architecture/hyperparameters are already fixed; this uses validation MSE as the internal per-round proxy because critic history is absent.",
        "6. Professor sign-off required: fewer validation rounds, fewer communication rounds, changed client sampling, changed batch semantics, or precision changes such as float32/TF32.",
    ]
    if avg_gpu:
        lines.append("")
        lines.append(
            f"Baseline average GPU utilization range is `{min(avg_gpu):.1f}%` to `{max(avg_gpu):.1f}%`, so there is clear underfill/headroom on H100."
        )
    if model_selection:
        lines.append(
            f"Baseline model-selection cost ranges from `{min(model_selection):.1f}s` to `{max(model_selection):.1f}s` per run before federated rounds start."
        )
    if reg_train:
        lines.append(
            f"Baseline auxiliary regression training cost ranges from `{min(reg_train):.1f}s` to `{max(reg_train):.1f}s` per 50-round profile."
        )
    return lines


def implemented_usage() -> list[str]:
    return [
        "Implemented runtime controls now flow through `scripts/run_manifest.py` and are recorded in `effective_config.json`/`metrics.json`.",
        "",
        "Profile the fast-path candidate without touching production outputs:",
        "",
        "```bash",
        "gpurun -g 1 python scripts/run_highdim_stochastic_gpu_profile.py --phase optimized50 --rounds 50 --datasets cifar10_x --methods fedgda_s --disable-aux-reg --skip-model-selection --disable-periodic-checkpoints",
        "```",
        "",
        "Launch production only after equivalence/sign-off, using explicit flags so generated YAMLs show the protocol changes:",
        "",
        "```bash",
        "python scripts/run_manifest.py --manifest experiments/highdim_coauthor_protocol_v1/alpha1/final_manifest_stochastic.csv --config-dir experiments/highdim_coauthor_protocol_v1/alpha1/generated_configs_final_fast --output-root results/rerun_protocol_v1_real_images_abs_alpha1_fast --gpu-ids 0,1 --max-parallel 2 --resume-skip-completed --disable-auxiliary-regression --skip-model-selection --override-periodic-checkpoint-interval 0 --results-json experiments/highdim_coauthor_protocol_v1/alpha1/final_stochastic_fast_launcher_results.json",
        "```",
    ]


def main() -> int:
    profile_rows = read_profile_rows(PROFILE_ROOT)
    production_rows = read_production_rows()
    plan_rows = read_plan_rows()
    write_summary_csv(profile_rows, SUMMARY_CSV)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stochastic GPU Utilization Investigation",
        "",
        f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`.",
        "",
        "Scope: high-dimensional stochastic final-run path for `FedGDA-S` and `FedOGDA-S`. This report is validation-safe: it summarizes runtime/profiling artifacts only and does not use Test MSE for hyperparameter selection.",
        "",
        "## Current Production Timing",
        "",
        *production_summary(plan_rows, production_rows),
        "",
        "## Profiling Matrix",
        "",
        *profile_summary(profile_rows),
        "",
        "## Queue-Time Estimates",
        "",
        *queue_projection(profile_rows, plan_rows),
        "",
        "## Diagnostic Comparisons",
        "",
        *diagnostic_comparisons(profile_rows),
        "",
        "## Recommendations",
        "",
        *recommendations(profile_rows),
        "",
        "## Implemented Usage",
        "",
        *implemented_usage(),
        "",
        "## Artifact Locations",
        "",
        f"- Profile root: `{PROFILE_ROOT.relative_to(REPO_ROOT)}`",
        f"- Summary CSV: `{SUMMARY_CSV.relative_to(REPO_ROOT)}`",
        f"- Report: `{REPORT_PATH.relative_to(REPO_ROOT)}`",
    ]
    with REPORT_PATH.open("w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
