#!/usr/bin/env python3
"""Require exact GMM equivalence before disabling auxiliary regression."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


METHODS = ("fedgda_d", "fedogda_d")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_curve(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def state_equal(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
    left_state = left[key]
    right_state = right[key]
    if left_state.keys() != right_state.keys():
        return False
    return all(torch.equal(left_state[name], right_state[name]) for name in left_state)


def run_dir(root: Path, method: str, suffix: str) -> Path:
    run_id = f"det_gate_equiv_femnist_z_{method}_seed0_alpha0p5_{suffix}_r10"
    return root / "femnist_z" / method / "seed_0" / run_id


def compare_method(root: Path, method: str) -> dict[str, Any]:
    on_dir = run_dir(root, method, "auxon")
    off_dir = run_dir(root, method, "auxoff")
    on_curve = load_curve(on_dir / "mse_by_round.csv")
    off_curve = load_curve(off_dir / "mse_by_round.csv")
    on_final = torch.load(on_dir / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    off_final = torch.load(off_dir / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    on_best = torch.load(
        on_dir / "checkpoints/best_validation.pt", map_location="cpu", weights_only=False
    )
    off_best = torch.load(
        off_dir / "checkpoints/best_validation.pt", map_location="cpu", weights_only=False
    )
    on_metrics = load_json(on_dir / "metrics.json")
    off_metrics = load_json(off_dir / "metrics.json")

    metric_keys = (
        "best_validation_mse",
        "best_validation_round",
        "test_mse_at_best_validation",
        "final_validation_mse",
        "final_test_mse",
        "diverged",
    )
    metric_match = all(on_metrics.get(key) == off_metrics.get(key) for key in metric_keys)
    checks = {
        "curve_exact": on_curve == off_curve,
        "final_g_exact": state_equal(on_final, off_final, "g_state_dict"),
        "final_f_exact": state_equal(on_final, off_final, "f_state_dict"),
        "best_g_exact": state_equal(on_best, off_best, "g_state_dict"),
        "best_f_exact": state_equal(on_best, off_best, "f_state_dict"),
        "selection_metrics_exact": metric_match,
    }
    return {
        "method": method,
        "passed": all(checks.values()),
        "checks": checks,
        "aux_on_runtime_seconds": on_metrics.get("runtime_seconds"),
        "aux_off_runtime_seconds": off_metrics.get("runtime_seconds"),
        "aux_off_projected_150_round_seconds": (
            float(off_metrics["runtime_seconds"]) * 15.0
            if off_metrics.get("runtime_seconds") is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    methods = [compare_method(args.root, method) for method in METHODS]
    projected_gpu_seconds = sum(
        float(item["aux_off_projected_150_round_seconds"]) * 12 for item in methods
    )
    report = {
        "passed": all(item["passed"] for item in methods),
        "methods": methods,
        "projected_gate_gpu_hours_conservative": projected_gpu_seconds / 3600.0,
        "projected_two_gpu_wall_hours_conservative": projected_gpu_seconds / 7200.0,
        "projection_note": "Linear 10-to-150 round projection includes setup 15 times and is conservative.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
