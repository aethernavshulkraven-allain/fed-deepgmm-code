#!/usr/bin/env python3
"""Derive the Study A v2 campaign contract from the tested v1 validator schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    REPO_ROOT / "experiments" / "eicu_study_a_validation" / "default_contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.base).open() as handle:
        contract = json.load(handle)
    contract["contract_version"] = "2.1.0-study-a-v2-offhours"
    contract["study"]["name"] = "eICU Study A v2 off-hours"
    contract["study"]["claim"] = "semi_synthetic_benchmark_no_clinical_claim"
    contract["global_rules"]["dataset"] = args.dataset
    contract["scenario_rules"]["required_metadata_fields"] += [
        "certification_passed",
        "client_id_column",
        "wardid_used_as_client",
    ]
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
