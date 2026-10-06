"""Apply the frozen tuning selection rule to a completed Study A tuning stage.

Selection order, per protocol_v1.md S8.1, applied per (g0, method) across its
LR/server-LR candidates:

    1. run must complete without divergence
    2. lowest (equal-client, for eICU) validation structural MSE
    3. (equal-client) validation moment violation *at that selected
       checkpoint* as the first tie-breaker -- not the best moment violation
       seen at any round, which could come from a different, worse-MSE round
    4. final-minus-best (equal-client) validation MSE gap as the second
       tie-breaker -- not raw final validation MSE, which does not measure
       degradation from the selected checkpoint

Reads ``equal_client_validation_moment_violation_at_best_validation`` and
``final_vs_best_validation_gap`` when present (eICU runs, once
fedavg_api.py's Gate 1 fields are written) and falls back to
``best_moment_violation``/``final_validation_mse`` for older metrics.json
files that predate those fields, so this still works against
non-Study-A/legacy runs.

The test set is never read by this script.

Usage:
    python scripts/select_eicu_study_a_tuning.py \\
        --manifest experiments/eicu_study_a/tuning_manifest.json \\
        --out experiments/eicu_study_a/selected_hyperparameters.json
"""

import argparse
import json
import os


def load_manifest(path):
    with open(path) as handle:
        return json.load(handle)


def run_dir_for(row):
    return os.path.join(
        row["output_root"],
        row["dataset"],
        row["method"],
        f"seed_{int(row['seed'])}",
        row["run_id"],
    )


def load_metrics(row):
    path = os.path.join(run_dir_for(row), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def moment_violation_at_selected_checkpoint(metrics):
    value = metrics.get("equal_client_validation_moment_violation_at_best_validation")
    if value is None:
        value = metrics["best_moment_violation"]
    return float(value)


def final_vs_best_validation_gap(metrics):
    value = metrics.get("final_vs_best_validation_gap")
    if value is None:
        value = metrics["final_validation_mse"]
    return float(value)


def selection_key(metrics):
    """Lower is better on every component -- a plain ascending sort applies
    the full 3-level tie-break in one comparison.
    """
    return (
        float(metrics["best_validation_mse"]),
        moment_violation_at_selected_checkpoint(metrics),
        final_vs_best_validation_gap(metrics),
    )


def select_candidates(rows):
    """Groups by (g0, method), returns {"g0:method": {...selection detail...}}."""
    groups = {}
    for row in rows:
        groups.setdefault((row["g0"], row["method"]), []).append(row)

    selected = {}
    report = {}
    for (g0, method), candidates in sorted(groups.items()):
        key = f"{g0}:{method}"
        evaluated = []
        for row in candidates:
            metrics = load_metrics(row)
            if metrics is None:
                evaluated.append({"row": row, "status": "missing_metrics"})
                continue
            if bool(metrics.get("diverged", False)):
                evaluated.append({"row": row, "status": "diverged", "metrics": metrics})
                continue
            evaluated.append({"row": row, "status": "eligible", "metrics": metrics})

        eligible = [e for e in evaluated if e["status"] == "eligible"]
        report[key] = {
            "n_candidates": len(candidates),
            "n_eligible": len(eligible),
            "n_diverged": sum(1 for e in evaluated if e["status"] == "diverged"),
            "n_missing": sum(1 for e in evaluated if e["status"] == "missing_metrics"),
        }
        if not eligible:
            report[key]["selected"] = None
            continue

        best = min(eligible, key=lambda e: selection_key(e["metrics"]))
        selected[key] = {
            "learning_rate": float(best["row"]["learning_rate"]),
            "server_learning_rate": float(best["row"]["server_learning_rate"]),
            "run_id": best["row"]["run_id"],
            "best_validation_mse": float(best["metrics"]["best_validation_mse"]),
            "best_moment_violation": float(best["metrics"]["best_moment_violation"]),
            "final_validation_mse": float(best["metrics"]["final_validation_mse"]),
            "moment_violation_at_best_validation": moment_violation_at_selected_checkpoint(best["metrics"]),
            "final_vs_best_validation_gap": final_vs_best_validation_gap(best["metrics"]),
        }
        report[key]["selected"] = selected[key]

    return selected, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_manifest(args.manifest)
    selected, report = select_candidates(rows)

    incomplete = {k: v for k, v in report.items() if not v.get("selected")}
    if incomplete:
        print("WARNING: no eligible candidate for:")
        for key, detail in incomplete.items():
            print(f"  {key}: {detail}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(selected, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report_path = os.path.splitext(args.out)[0] + "_report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    print(f"wrote {report_path}")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
