#!/usr/bin/env python3
"""Fill the 120-run final manifest from validation-selected tuning configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
BASE_MANIFEST = PROTOCOL_DIR / "manifest.csv"
SELECTED = PROTOCOL_DIR / "tuning" / "selected_configs.csv"
OUTPUT_CSV = PROTOCOL_DIR / "final_manifest.csv"
OUTPUT_JSON = PROTOCOL_DIR / "final_manifest.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def materialize(
    base_rows: list[dict[str, str]], selected_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    if len(base_rows) != 120:
        raise ValueError(f"expected 120 base rows, found {len(base_rows)}")
    if len(selected_rows) != 24:
        raise ValueError(f"expected 24 selected configs, found {len(selected_rows)}")
    selected = {(row["dataset"], row["method"]): row for row in selected_rows}
    if len(selected) != 24:
        raise ValueError("selected configs do not contain 24 unique dataset+method pairs")
    output: list[dict[str, str]] = []
    for original in base_rows:
        row = dict(original)
        key = (row["dataset"], row["method"])
        if key not in selected:
            raise ValueError(f"missing selected config for {key}")
        choice = selected[key]
        row["learning_rate"] = choice["learning_rate"]
        row["weight_decay"] = choice["weight_decay"]
        row["critic_multiplier"] = choice["critic_multiplier"]
        row["learning_rate_status"] = "selected_by_validation_tuning"
        row["implementation_status"] = "ready"
        row["preflight_status"] = "passed" if row["preflight_required"].lower() == "true" else "not_required"
        row["notes"] = (
            row["notes"]
            + f" Validation-selected tuning source: {choice['run_id']}."
        )
        output.append(row)
    return output


def main() -> int:
    if not SELECTED.exists():
        raise SystemExit(
            "selected_configs.csv is missing; complete tuning and run "
            "scripts/analyze_real_image_abs_tuning.py first"
        )
    base_rows = read_csv(BASE_MANIFEST)
    selected_rows = read_csv(SELECTED)
    rows = materialize(base_rows, selected_rows)
    fieldnames = list(base_rows[0])
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT_JSON.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"rows": len(rows), "csv": str(OUTPUT_CSV.relative_to(REPO_ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
