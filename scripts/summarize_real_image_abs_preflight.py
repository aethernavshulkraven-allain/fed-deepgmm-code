#!/usr/bin/env python3
"""Consolidate fixed-abs data, deterministic, and stochastic preflight evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "rerun_protocol_v1_real_images_abs_alpha0p5"
PREFLIGHT_DIR = PROTOCOL_DIR / "preflight"


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def main() -> int:
    certification = load(PROTOCOL_DIR / "data_certification.json")
    deterministic = []
    for path in sorted(PREFLIGHT_DIR.glob("full_gradient_*/summary.json")):
        rows = load(path)
        deterministic.append({
            "name": path.parent.name,
            "runs": len(rows),
            "passed": sum(row["status"] == "passed" for row in rows),
            "seeds": sorted(int(row["seed"]) for row in rows),
            "sample_count_sums": sorted({int(row["sample_count_sum"]) for row in rows}),
        })
    stochastic = []
    for path in sorted(PREFLIGHT_DIR.glob("*_s_results.json")):
        rows = load(path)
        for row in rows:
            run_dir = Path(row["run_dir"])
            prediction_path = run_dir / "predictions.npz"
            sample_shapes = {}
            if prediction_path.exists():
                with np.load(prediction_path) as predictions:
                    sample_shapes = {
                        key: list(predictions[key].shape)
                        for key in ("x", "true_g", "best_validation_prediction", "final_prediction")
                    }
            stochastic.append({
                "name": path.stem,
                "run_id": row["run_id"],
                "status": row["status"],
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "prediction_shapes": sample_shapes,
            })
    summary = {
        "g_function": "abs",
        "alpha": 0.5,
        "data_certified": bool(certification.get("core_invariants_pass")),
        "data_scenarios": len(certification.get("datasets", [])),
        "data_split_isolation_pass": all(
            row.get("split_isolation_pass") for row in certification.get("datasets", [])
        ),
        "deterministic_preflights": deterministic,
        "deterministic_runs_passed": sum(item["passed"] for item in deterministic),
        "deterministic_runs_expected": 12,
        "stochastic_smokes": stochastic,
        "stochastic_smokes_passed": sum(item["status"] == "passed" for item in stochastic),
        "stochastic_smokes_expected": 4,
    }
    summary["preflight_pass"] = (
        summary["data_certified"]
        and summary["data_scenarios"] == 6
        and summary["data_split_isolation_pass"]
        and summary["deterministic_runs_passed"] == summary["deterministic_runs_expected"]
        and summary["stochastic_smokes_passed"] == summary["stochastic_smokes_expected"]
    )
    with (PREFLIGHT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    lines = [
        "# High-dimensional preflight summary",
        "",
        f"- Fixed response function: `abs`",
        f"- Certified data scenarios: `{summary['data_scenarios']}/6`",
        f"- Exact image split isolation: `{'pass' if summary['data_split_isolation_pass'] else 'fail'}`",
        f"- Deterministic full-gradient checks: `{summary['deterministic_runs_passed']}/12`",
        f"- Stochastic end-to-end smokes: `{summary['stochastic_smokes_passed']}/4`",
        f"- Overall preflight: `{'pass' if summary['preflight_pass'] else 'fail'}`",
        "",
        "The deterministic checks cover both image families, both deterministic optimizers, and all three seeds. ",
        "The stochastic smokes cover both image families and both stochastic optimizers through artifact writing.",
    ]
    (PREFLIGHT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["preflight_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
