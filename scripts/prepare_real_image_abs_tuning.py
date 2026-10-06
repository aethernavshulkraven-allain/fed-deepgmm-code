#!/usr/bin/env python3
"""Prepare the validation-only high-dimensional fixed-abs tuning manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
SOURCE_MANIFEST = PROTOCOL_ROOT / "manifest.csv"
TUNING_ROOT = PROTOCOL_ROOT / "tuning"
CSV_PATH = TUNING_ROOT / "manifest.csv"
JSON_PATH = TUNING_ROOT / "manifest.json"
SUMMARY_PATH = TUNING_ROOT / "tuning_protocol.json"
README_PATH = TUNING_ROOT / "README.md"
OUTPUT_ROOT = "results/rerun_protocol_v1_real_images_abs_alpha0p5_tuning"
TUNING_SEED = 0
TUNING_ROUNDS = 150

# Paired FedGDA/FedOGDA methods receive the same four-candidate budget within
# each stochasticity regime. Values bracket the repository's documented
# high-dimensional settings without using Test MSE.
GRIDS = {
    "deterministic": {
        "learning_rate": (0.001, 0.003),
        "weight_decay": (0.001, 0.01),
    },
    "stochastic": {
        "learning_rate": (0.003, 0.01),
        "weight_decay": (0.05, 0.1),
    },
}


def token(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


def load_source_rows() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def generate_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, source_rows = load_source_rows()
    rows: list[dict[str, str]] = []
    for source in source_rows:
        if int(source["seed"]) != TUNING_SEED:
            continue
        regime = "deterministic" if source["method"].endswith("_d") else "stochastic"
        grid = GRIDS[regime]
        for learning_rate in grid["learning_rate"]:
            for weight_decay in grid["weight_decay"]:
                row = dict(source)
                suffix = f"lr{token(learning_rate)}_wd{token(weight_decay)}"
                run_id = f"tune_{source['dataset']}_{source['method']}_seed{TUNING_SEED}_{suffix}"
                row.update(
                    {
                        "run_id": run_id,
                        "protocol_version": "rerun_protocol_v1_real_images_abs_alpha0p5_tuning_v1",
                        "run_group": "real_images_abs_alpha0p5_validation_tuning",
                        "output_root": OUTPUT_ROOT,
                        "final_result_dir": (
                            f"{OUTPUT_ROOT}/{source['dataset']}/{source['method']}/"
                            f"seed_{TUNING_SEED}/{run_id}"
                        ),
                        "implementation_status": "ready_pending_smoke",
                        "run_status": "not_started",
                        "comm_round": str(TUNING_ROUNDS),
                        "learning_rate": str(learning_rate),
                        "learning_rate_status": "validation_tuning_candidate",
                        "weight_decay": str(weight_decay),
                        "notes": (
                            "Validation-only fixed-abs high-dimensional tuning candidate; "
                            "Test MSE must not be inspected until this dataset+method config is selected."
                        ),
                    }
                )
                rows.append(row)
    return fieldnames, rows


def write_readme(path: Path, count: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# Fixed-abs high-dimensional tuning protocol",
                "",
                f"Candidates: `{count}` (6 scenarios × 4 methods × 4 candidates × seed 0).",
                f"Tuning budget: `{TUNING_ROUNDS}` communication rounds per candidate.",
                "",
                "Selection is performed separately for every scenario and method.",
                "Only validation metrics may rank candidates. Candidates with numerical divergence are excluded.",
                "Primary key: lowest `best_validation_mse`; tie-break by lower last-50 validation MSE standard deviation,",
                "then lower final-versus-best validation gap. Test MSE is reported only after the candidate is fixed.",
                "",
                "FedGDA and FedOGDA receive identical candidate counts and grids within each stochasticity regime.",
                "The final five-seed runs use the selected candidate with the original 500-round budget.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    fieldnames, rows = generate_rows()
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    JSON_PATH.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "protocol_version": "rerun_protocol_v1_real_images_abs_alpha0p5_tuning_v1",
        "g_function": "abs",
        "alpha": 0.5,
        "tuning_seed": TUNING_SEED,
        "tuning_rounds": TUNING_ROUNDS,
        "candidate_count": len(rows),
        "candidate_count_per_dataset_method": 4,
        "grids": GRIDS,
        "selection_metric_source": "validation_only",
        "selection_rule": (
            "exclude_diverged; min best_validation_mse; tie min last_50_val_mse_std; "
            "tie min final_vs_best_validation_gap"
        ),
        "test_mse_used_for_selection": False,
        "output_root": OUTPUT_ROOT,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(README_PATH, len(rows))
    print(json.dumps({
        "candidate_count": len(rows),
        "manifest": str(CSV_PATH.relative_to(REPO_ROOT)),
        "summary": str(SUMMARY_PATH.relative_to(REPO_ROOT)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
