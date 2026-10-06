#!/usr/bin/env python3
"""Prepare the post-gate deterministic multi-seed validation campaign."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_learning_gate_20260802/gate_manifest.csv"
)
CAMPAIGN_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_multiseed_validation_20260803"
)
OUTPUT_ROOT = REPO_ROOT / "results/highdim_deterministic_multiseed_validation_20260803"
SEEDS = (1, 2)
CANDIDATES = (
    ("fedgda_d", 0.03, 3.0, "best_clean_gate_candidate"),
    ("fedogda_d", 0.03, 1.0, "best_fully_finite_gate_candidate"),
    ("fedogda_d", 0.01, 3.0, "lower_lr_stability_fallback"),
)


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def source_row(
    rows: list[dict[str, str]], method: str, learning_rate: float, critic_multiplier: float
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and float(row["learning_rate"]) == learning_rate
        and float(row["critic_multiplier"]) == critic_multiplier
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source row for {method}/lr={learning_rate}/cm={critic_multiplier}, "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    fieldnames, source_rows = load_source()
    rows: list[dict[str, str]] = []
    for method, learning_rate, critic_multiplier, role in CANDIDATES:
        source = source_row(source_rows, method, learning_rate, critic_multiplier)
        for seed in SEEDS:
            row = dict(source)
            run_id = (
                f"det_multiseed_femnist_z_{method}_seed{seed}_alpha0p5"
                f"_lr{token(learning_rate)}_cm{token(critic_multiplier)}"
            )
            row.update(
                {
                    "run_id": run_id,
                    "protocol_version": "highdim_deterministic_multiseed_validation_v1",
                    "run_group": "highdim_deterministic_multiseed_validation_20260803",
                    "seed": str(seed),
                    "output_root": str(OUTPUT_ROOT),
                    "final_result_dir": str(
                        OUTPUT_ROOT / "femnist_z" / method / f"seed_{seed}" / run_id
                    ),
                    "implementation_status": "ready",
                    "run_status": "not_started",
                    "preflight_required": "False",
                    "preflight_status": "not_required",
                    "learning_rate_status": role,
                    "notes": (
                        "Validation-only post-gate seed confirmation; numerical stability is a "
                        "required acceptance criterion and Test MSE must not select a candidate."
                    ),
                }
            )
            rows.append(row)

    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CAMPAIGN_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "campaign": "highdim_deterministic_multiseed_validation_20260803",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "output_root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
        "dataset": "femnist_z",
        "alpha": 0.5,
        "seeds": list(SEEDS),
        "candidates": [
            {
                "method": method,
                "learning_rate": learning_rate,
                "critic_multiplier": critic_multiplier,
                "role": role,
            }
            for method, learning_rate, critic_multiplier, role in CANDIDATES
        ],
        "runs": len(rows),
        "communication_rounds": 150,
        "local_epochs": 3,
        "clients_total": 1000,
        "clients_per_round": 1000,
        "batch_size": 0,
        "auxiliary_regression": False,
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
    }
    with (CAMPAIGN_DIR / "setup_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
