#!/usr/bin/env python3
"""Create a launch-candidate manifest with explicit LR/WD values.

The canonical rerun protocol manifest records the intended run grid.  This
script creates a separate candidate manifest that can be consumed by
``scripts/run_manifest.py`` without mutating the canonical protocol file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "manifest.csv"
OUTPUT_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1" / "launch_candidate"
OUTPUT_CSV = OUTPUT_DIR / "manifest.csv"
OUTPUT_JSON = OUTPUT_DIR / "manifest.json"


METHOD_DEFAULTS = {
    # Tiny full-gradient smoke passed with these conservative values.
    "fedgda_d": {
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "critic_multiplier": 10.0,
        "learning_rate_status": "candidate_default_from_full_gradient_smoke_pending_validation_tuning",
    },
    "fedogda_d": {
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "critic_multiplier": 10.0,
        "learning_rate_status": "candidate_default_pending_fedogda_d_smoke_or_validation_tuning",
    },
    # Prior repo-config stochastic ABS five-seed runs used these values.
    "fedgda_s": {
        "learning_rate": 0.003,
        "weight_decay": 0.03,
        "critic_multiplier": 10.0,
        "learning_rate_status": "candidate_default_from_prior_repo_config_pending_validation_tuning",
    },
    "fedogda_s": {
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "critic_multiplier": 10.0,
        "learning_rate_status": "candidate_default_from_prior_repo_config_pending_validation_tuning",
    },
}


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def prepare_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        if row.get("training_scope") == "federated":
            defaults = METHOD_DEFAULTS[row["method"]]
            row["learning_rate"] = str(defaults["learning_rate"])
            row["weight_decay"] = str(defaults["weight_decay"])
            row["critic_multiplier"] = str(defaults["critic_multiplier"])
            row["learning_rate_status"] = defaults["learning_rate_status"]
            row["implementation_status"] = "launch_candidate_pending_small_gpu_pilot"
            row["run_status"] = "not_started"
            if row.get("preflight_required") == "True":
                row["preflight_status"] = "passed_full_gradient_preflight_batch"
        out.append(row)
    return out


def main() -> int:
    fieldnames, rows = load_rows(INPUT_CSV)
    prepared = prepare_rows(rows)
    write_csv(OUTPUT_CSV, fieldnames, prepared)
    write_json(OUTPUT_JSON, prepared)
    counts = {
        "total": len(prepared),
        "federated_launch_candidates": sum(row["training_scope"] == "federated" for row in prepared),
        "centralized_still_blocked": sum(row["training_scope"] == "centralized" for row in prepared),
        "csv": str(OUTPUT_CSV.relative_to(REPO_ROOT)),
        "json": str(OUTPUT_JSON.relative_to(REPO_ROOT)),
        "method_defaults": METHOD_DEFAULTS,
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
