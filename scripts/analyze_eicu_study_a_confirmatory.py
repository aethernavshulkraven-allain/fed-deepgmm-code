"""Aggregate the Study A confirmatory campaign into the primary comparison.

The primary question this answers:

    Does FedOGDA provide lower final-iterate error and less degradation from
    the best checkpoint than FedGDA under natural hospital heterogeneity?

Reports, per (g0, method): mean/std/median across the 5 confirmatory seeds of
best-validation test MSE, final-iterate test MSE, and the gap between them
(how much a method degrades if you had used the final iterate instead of the
validation-selected checkpoint) -- plus paired FedOGDA-minus-FedGDA
differences at matched seeds, and divergence counts.

Deliberately reports effect sizes and per-seed spread rather than a
significance test: five paired seeds is not enough repetitions to support a
strong significance claim, so the emphasis is on consistency of direction
across seeds, not a p-value.

Usage:
    python scripts/analyze_eicu_study_a_confirmatory.py \\
        --manifest experiments/eicu_study_a/confirmatory_manifest.json \\
        --out experiments/eicu_study_a/confirmatory_report
"""

import argparse
import json
import os
import statistics


def load_manifest(path):
    with open(path) as handle:
        return json.load(handle)


def run_dir_for(row):
    return os.path.join(
        row["output_root"], row["dataset"], row["method"], f"seed_{int(row['seed'])}", row["run_id"],
    )


def load_metrics(row):
    path = os.path.join(run_dir_for(row), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def collect(rows):
    """One record per row, joined with its metrics (None if not yet run)."""
    records = []
    for row in rows:
        metrics = load_metrics(row)
        records.append({"row": row, "metrics": metrics})
    return records


def summarize_group(records):
    """mean/std/median/individual values + divergence count for one (g0, method)."""
    complete = [r for r in records if r["metrics"] is not None]
    diverged = [r for r in complete if bool(r["metrics"].get("diverged", False))]
    clean = [r for r in complete if not bool(r["metrics"].get("diverged", False))]

    def series(key):
        return [float(r["metrics"][key]) for r in clean]

    out = {
        "n_seeds_expected": len(records),
        "n_seeds_complete": len(complete),
        "n_diverged": len(diverged),
        "seeds": [int(r["row"]["seed"]) for r in clean],
    }
    for key in (
        "test_mse_at_best_validation",
        "final_test_mse",
        "best_validation_mse",
        "final_validation_mse",
        "best_moment_violation",
        "final_moment_violation",
    ):
        values = series(key)
        if not values:
            out[key] = None
            continue
        out[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "values": values,
        }

    if clean:
        degradation = [
            float(r["metrics"]["final_test_mse"]) - float(r["metrics"]["test_mse_at_best_validation"])
            for r in clean
        ]
        out["final_vs_best_degradation"] = {
            "mean": statistics.mean(degradation),
            "std": statistics.pstdev(degradation) if len(degradation) > 1 else 0.0,
            "values": degradation,
        }
    else:
        out["final_vs_best_degradation"] = None

    return out


def paired_differences(gda_records, ogda_records, key="test_mse_at_best_validation"):
    """FedOGDA - FedGDA at each matched seed (both must be clean, non-diverged)."""
    gda_by_seed = {
        int(r["row"]["seed"]): r["metrics"]
        for r in gda_records
        if r["metrics"] is not None and not r["metrics"].get("diverged", False)
    }
    ogda_by_seed = {
        int(r["row"]["seed"]): r["metrics"]
        for r in ogda_records
        if r["metrics"] is not None and not r["metrics"].get("diverged", False)
    }
    shared_seeds = sorted(set(gda_by_seed) & set(ogda_by_seed))
    diffs = [float(ogda_by_seed[s][key]) - float(gda_by_seed[s][key]) for s in shared_seeds]
    return {
        "seeds": shared_seeds,
        "differences": diffs,
        "mean_difference": statistics.mean(diffs) if diffs else None,
        "n_ogda_better": sum(1 for d in diffs if d < 0),
        "n_gda_better": sum(1 for d in diffs if d > 0),
    }


def render_report(summary, pairwise):
    lines = []
    add = lines.append
    add("# Study A confirmatory results\n")
    add(
        "Primary question: does FedOGDA give lower final-iterate error and less "
        "best-vs-final degradation than FedGDA under natural hospital "
        "heterogeneity? Reported as effect sizes and per-seed consistency, "
        "not a significance test -- five paired seeds does not support one.\n"
    )

    for g0 in sorted(summary):
        add(f"## g0 = `{g0}`\n")
        add("| method | seeds ok | diverged | test MSE @ best-val (mean±std) | final test MSE (mean±std) | final-vs-best degradation |")
        add("|---|---|---|---|---|---|")
        for method in sorted(summary[g0]):
            s = summary[g0][method]
            best = s.get("test_mse_at_best_validation")
            final = s.get("final_test_mse")
            deg = s.get("final_vs_best_degradation")
            add(
                f"| {method} | {s['n_seeds_complete']}/{s['n_seeds_expected']} | {s['n_diverged']} | "
                f"{best['mean']:.4g}±{best['std']:.2g} | " if best else f"| {method} | - | - | n/a | "
            )
            if best:
                lines[-1] += (
                    f"{final['mean']:.4g}±{final['std']:.2g} | "
                    f"{deg['mean']:+.4g}±{deg['std']:.2g} |"
                )
            else:
                lines[-1] += "n/a | n/a |"
        add("")

        if g0 in pairwise:
            pw = pairwise[g0]
            add(f"### Paired FedOGDA - FedGDA (test MSE @ best-validation), g0=`{g0}`\n")
            add(f"- matched seeds: {pw['seeds']}")
            add(f"- per-seed differences: {[round(d, 4) for d in pw['differences']]}")
            add(f"- mean difference: {pw['mean_difference']}")
            add(f"- FedOGDA better in {pw['n_ogda_better']}/{len(pw['differences'])} seeds")
            add("")

    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_manifest(args.manifest)
    records = collect(rows)

    by_g0_method = {}
    for record in records:
        g0 = record["row"]["g0"]
        method = record["row"]["method"]
        by_g0_method.setdefault(g0, {}).setdefault(method, []).append(record)

    summary = {
        g0: {method: summarize_group(recs) for method, recs in methods.items()}
        for g0, methods in by_g0_method.items()
    }

    pairwise = {}
    for g0, methods in by_g0_method.items():
        gda_key = next((m for m in methods if m.startswith("fedgda")), None)
        ogda_key = next((m for m in methods if m.startswith("fedogda")), None)
        if gda_key and ogda_key:
            pairwise[g0] = paired_differences(methods[gda_key], methods[ogda_key])

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump({"summary": summary, "pairwise": pairwise}, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")

    report = render_report(summary, pairwise)
    report_path = os.path.join(args.out, "README.md")
    with open(report_path, "w") as handle:
        handle.write(report + "\n")

    print(report)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
