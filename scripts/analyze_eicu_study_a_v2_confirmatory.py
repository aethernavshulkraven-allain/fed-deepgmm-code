#!/usr/bin/env python3
"""Pre-registered confirmatory analysis for Study A v2 (eicu_study_a_v2_offhours).

This is written and unit-tested BEFORE any v2 confirmatory result exists (see
``experiments/eicu_study_a_v2_offhours_demo_20260727/status.md``: the 30-row
primary federated matrix is not materialized). That ordering is the entire
point: the decision rule below is fixed in code now, while there is nothing
to peek at, so it cannot be chosen -- consciously or not -- after seeing
which rule makes FedOGDA look better.

PRE-DECLARED DECISION RULE (fixed here; not a CLI flag; do not edit this
module to change the rule after v2 results exist without saying so loudly)
------------------------------------------------------------------------

Endpoints, from protocol_v2.md's "Tuning and final matrix" section:

* Primary: equal-client Test structural MSE at the validation-selected
  checkpoint (``PRIMARY_ENDPOINT_KEYS``: ``equal_client_test_mse_at_best_validation``,
  falling back to the compatibility field ``test_mse_at_best_validation`` --
  ``metric_policy.md`` requires that compatibility field to itself carry
  equal-client semantics for Study A, so this fallback is not a silent swap
  to sample-weighted).
* Secondary (reported, not used for the verdict): sample-weighted MSE,
  held-out moments, effect (ATE) error, individual-effect MAE, per-hospital
  distribution summaries, final-vs-best degradation, oscillation
  (``eicu_common.final_iterate_stability``), divergence, runtime.

FedOGDA-minus-FedGDA differences on the primary endpoint are paired by
(scenario seed, optimizer seed): pairs are (101,1101)...(105,1105), giving
exactly 5 paired differences per g0 in {linear, interaction, mlp}, 15 pooled
across all three. A negative difference means FedOGDA had lower (better)
Test MSE.

With n=5 per g0, there is no room for p-value theater, so this module NEVER
computes or reports a significance test. Instead, a scope (one g0, or the
15-pair pooled set) is called "FedOGDA favored" / "FedGDA favored" only when
BOTH of the following hold on that scope's differences, computed by
``classify_verdict``:

  (a) sign consistency strictly above a bare majority --
      ``SIGN_CONSISTENCY_MIN_PER_G0`` = 4 of 5 for a single g0,
      ``SIGN_CONSISTENCY_MIN_POOLED`` = 12 of 15 pooled; and
  (b) the mean paired difference has the same sign as that majority
      direction (so one large outlier cannot flip the sign consistency
      into a misleading mean, or vice versa).

Any scope that does not clear this bar -- a tie, a bare 3/5 majority,
disagreement in direction across g0 functions, or too few clean (both
methods non-diverged) pairs -- is reported as ``"inconclusive"``. This
module can and should report FedOGDA losing; see
``scripts/analyze_sine_a2_lite.py``'s "does not improve ... in these pairs"
/ "not supported" for the tone a negative or inconclusive result is
reported in here.

Test-blindness
--------------

This script is a *reporting* tool: every run's checkpoint and configuration
was already fixed, without reading any Test field, by
``scripts/select_eicu_study_a_v2_tuning.py`` (which records
``"test_fields_read": []``) before any of these runs launched. This script
reads Test-derived metrics.json fields, but only to report them -- it never
feeds a Test value back into a selection decision, and it does not touch
``mse_by_round.csv``'s validation columns for anything except the stability
diagnostic (also validation-only: ``primary_val_mse``, never a Test column).
The emitted summary JSON records exactly which Test-derived fields were read,
under ``test_fields_read_for_final_reporting_only``, to make this auditable
the same way the tuning selector's ``test_fields_read: []`` is.

Usage
-----

    python scripts/analyze_eicu_study_a_v2_confirmatory.py \\
        --manifest experiments/eicu_study_a_v2_offhours_demo_20260727/final_manifest.json \\
        --out experiments/eicu_study_a_v2_offhours_demo_20260727/confirmatory_report

The manifest is filtered to ``role == "confirmatory"`` rows (the 30-row
primary federated FedGDA/FedOGDA matrix); centralized and
aggregation-ablation rows in a full 105-row manifest are ignored by this
script (they are out of scope for this primary comparison).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eicu_common import final_iterate_stability  # noqa: E402

G0_VARIANTS = ("linear", "interaction", "mlp")
FEDERATED_METHODS = ("fedgda_s", "fedogda_s")
CONFIRMATORY_SEED_PAIRS = (
    (101, 1101),
    (102, 1102),
    (103, 1103),
    (104, 1104),
    (105, 1105),
)

# Primary endpoint: try the equal-client field first; only fall back to the
# compatibility field if absent, never the reverse (metric_policy.md).
PRIMARY_ENDPOINT_KEYS = (
    "equal_client_test_mse_at_best_validation",
    "test_mse_at_best_validation",
)

# Secondary metrics.json fields reported verbatim (all Test-derived / final
# reporting only -- see module docstring "Test-blindness").
SECONDARY_METRIC_KEYS = (
    "sample_weighted_test_mse_at_best_validation",
    "equal_client_test_moment_violation_at_best_validation",
    "sample_weighted_test_moment_violation_at_best_validation",
    "equal_client_final_test_structural_mse",
    "sample_weighted_final_test_structural_mse",
    "final_test_mse",
    "final_vs_best_test_gap",
    "final_vs_best_validation_gap",
    "runtime_seconds",
)

# Effect-error / individual-effect-MAE fields. The federated trainer writes
# these as None in metrics.json (it has no scenario-specific counterfactual
# machinery -- see fedavg_api.py's comment on this), so this script falls
# back to the sibling per-client checkpoint evaluation artifact that
# scripts/analyze_eicu_study_a_checkpoint.py produces, if present.
EFFECT_METRIC_KEYS = (
    "equal_client_absolute_ate_error_at_best_validation",
    "sample_weighted_absolute_ate_error_at_best_validation",
    "equal_client_individual_effect_mae_at_best_validation",
    "sample_weighted_individual_effect_mae_at_best_validation",
)

# Pre-declared decision-rule constants. NOT CLI flags -- see module
# docstring. Do not add a --sign-threshold style argument to this script.
SIGN_CONSISTENCY_MIN_PER_G0 = 4  # of 5 paired seeds
SIGN_CONSISTENCY_MIN_POOLED = 12  # of 15 pooled pairs

# Stability diagnostic: equal-client validation MSE is the actual checkpoint
# selector for Study A (primary_val_mse == equal_client_val_mse whenever
# client_id is available, per fedavg_api.py) -- metric_policy.md's
# "Stability metrics" section wants equal-client validation MSE, not pooled.
STABILITY_METRIC_COLUMN = "primary_val_mse"


def run_dir_for(row: dict[str, Any]) -> str:
    seed = row.get("optimizer_seed", row.get("seed"))
    return os.path.join(
        str(row["output_root"]), str(row["dataset"]), str(row["method"]),
        f"seed_{int(seed)}", str(row["run_id"]),
    )


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def load_metrics(row: dict[str, Any]) -> dict[str, Any] | None:
    return load_json(os.path.join(run_dir_for(row), "metrics.json"))


def load_per_client_eval_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    """Sibling artifact from analyze_eicu_study_a_checkpoint.py, if present.

    Default output location for that script is alongside the checkpoint,
    i.e. ``<run_dir>/checkpoints/per_client_eval_best_validation_test_summary.json``.
    Absent is normal (that script is invoked separately, per-checkpoint) --
    this function returns None rather than raising, and callers must treat
    effect/individual-effect metrics as unavailable, not zero.
    """
    path = os.path.join(
        run_dir_for(row), "checkpoints",
        "per_client_eval_best_validation_test_summary.json",
    )
    return load_json(path)


def primary_endpoint_value(metrics: dict[str, Any]) -> tuple[float, str]:
    """Returns (value, which_key_was_used); raises if neither key is present."""
    for key in PRIMARY_ENDPOINT_KEYS:
        if metrics.get(key) is not None:
            return float(metrics[key]), key
    raise KeyError(
        f"metrics.json has neither of {PRIMARY_ENDPOINT_KEYS!r} -- "
        "cannot compute the primary endpoint for this run"
    )


def effect_metrics(metrics: dict[str, Any], per_client_eval_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Effect (ATE) error + individual-effect MAE, with provenance of the source."""
    out: dict[str, Any] = {key: metrics.get(key) for key in EFFECT_METRIC_KEYS}
    if all(out[key] is not None for key in EFFECT_METRIC_KEYS):
        out["source"] = "metrics_json"
        return out
    if per_client_eval_summary is not None:
        aggregates = per_client_eval_summary.get("aggregates", {})
        ate = aggregates.get("absolute_ate_error", {})
        ite = aggregates.get("individual_effect_mae", {})
        out["equal_client_absolute_ate_error_at_best_validation"] = ate.get("equal_client")
        out["sample_weighted_absolute_ate_error_at_best_validation"] = ate.get("sample_weighted")
        out["equal_client_individual_effect_mae_at_best_validation"] = ite.get("equal_client")
        out["sample_weighted_individual_effect_mae_at_best_validation"] = ite.get("sample_weighted")
        out["source"] = "per_client_checkpoint_eval"
        return out
    out["source"] = "unavailable"
    return out


def per_hospital_distribution(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Distribution across hospitals of structural MSE / moment violation for one run.

    Prefers analyze_eicu_study_a_checkpoint.py's richer per-client summary
    (which also has ATE / individual-effect MAE distributions); falls back
    to the plain per_client_metrics.csv artifact every run writes.
    """
    summary = load_per_client_eval_summary(row)
    if summary is not None:
        return {"source": "per_client_checkpoint_eval", "aggregates": summary.get("aggregates", {})}
    artifact = metrics.get("per_client_metrics_artifact")
    if artifact and os.path.exists(artifact):
        import csv

        with open(artifact, newline="") as handle:
            rows = list(csv.DictReader(handle))
        mse_values = [float(r["structural_mse"]) for r in rows if r.get("structural_mse")]
        if not mse_values:
            return None
        return {
            "source": "per_client_metrics_csv",
            "n_clients": len(rows),
            "structural_mse": {
                "min": min(mse_values),
                "median": statistics.median(mse_values),
                "max": max(mse_values),
                "mean": statistics.fmean(mse_values),
            },
        }
    return None


def stability_for_run(row: dict[str, Any]) -> dict[str, Any] | None:
    mse_csv = os.path.join(run_dir_for(row), "mse_by_round.csv")
    if not os.path.exists(mse_csv):
        return None
    return final_iterate_stability(mse_csv, metric_column=STABILITY_METRIC_COLUMN)


def collect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per confirmatory row: raw row + metrics (None if not yet run) + diagnostics."""
    records = []
    for row in rows:
        metrics = load_metrics(row)
        record: dict[str, Any] = {"row": row, "metrics": metrics}
        if metrics is not None:
            per_client_eval_summary = load_per_client_eval_summary(row)
            record["effect_metrics"] = effect_metrics(metrics, per_client_eval_summary)
            record["per_hospital_distribution"] = per_hospital_distribution(row, metrics)
            record["stability"] = stability_for_run(row)
        records.append(record)
    return records


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in records if r["metrics"] is not None]
    diverged = [r for r in complete if bool(r["metrics"].get("diverged", False))]
    clean = [r for r in complete if not bool(r["metrics"].get("diverged", False))]

    out: dict[str, Any] = {
        "n_seeds_expected": len(records),
        "n_seeds_complete": len(complete),
        "n_diverged": len(diverged),
        "diverged_seed_pairs": [
            {
                "scenario_seed": r["row"].get("scenario_seed"),
                "optimizer_seed": r["row"].get("optimizer_seed", r["row"].get("seed")),
            }
            for r in diverged
        ],
        "seeds": sorted(
            int(r["row"].get("optimizer_seed", r["row"]["seed"])) for r in clean
        ),
    }

    if not clean:
        out["primary"] = None
        out["secondary"] = {}
        out["effect"] = None
        out["stability"] = None
        return out

    primary_values = []
    primary_key_used = None
    for r in clean:
        value, key_used = primary_endpoint_value(r["metrics"])
        primary_values.append(value)
        primary_key_used = key_used
    out["primary"] = {
        "metric_key": primary_key_used,
        "mean": statistics.mean(primary_values),
        "std": statistics.pstdev(primary_values) if len(primary_values) > 1 else 0.0,
        "median": statistics.median(primary_values),
        "min": min(primary_values),
        "max": max(primary_values),
        "values": primary_values,
    }

    secondary: dict[str, Any] = {}
    for key in SECONDARY_METRIC_KEYS:
        values = [float(r["metrics"][key]) for r in clean if r["metrics"].get(key) is not None]
        if not values:
            secondary[key] = None
            continue
        secondary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "n": len(values),
        }
    out["secondary"] = secondary

    effect_available = [r for r in clean if r["effect_metrics"]["source"] != "unavailable"]
    if effect_available:
        effect_summary = {}
        for key in EFFECT_METRIC_KEYS:
            values = [
                r["effect_metrics"][key] for r in effect_available
                if r["effect_metrics"].get(key) is not None
            ]
            effect_summary[key] = statistics.mean(values) if values else None
        effect_summary["n_available"] = len(effect_available)
        effect_summary["n_expected"] = len(clean)
        out["effect"] = effect_summary
    else:
        out["effect"] = {"n_available": 0, "n_expected": len(clean), "note": "unavailable for all seeds"}

    stability_records = [r["stability"] for r in clean if r.get("stability") is not None]
    if stability_records:
        tail_stds = [s["tail_std"] for s in stability_records if s["tail_std"] is not None]
        out["stability"] = {
            "n_with_stability_data": len(stability_records),
            "n_stability_diverged": sum(1 for s in stability_records if s["diverged"]),
            "mean_tail_std": statistics.mean(tail_stds) if tail_stds else None,
            "metric_column": STABILITY_METRIC_COLUMN,
        }
    else:
        out["stability"] = None

    return out


def classify_verdict(differences: list[float], min_sign_consistency: int) -> dict[str, Any]:
    """Apply the pre-declared decision rule (see module docstring) to one scope's
    FedOGDA-minus-FedGDA paired differences. Lower is better (structural MSE)."""
    n = len(differences)
    n_ogda_better = sum(1 for d in differences if d < 0)
    n_gda_better = sum(1 for d in differences if d > 0)
    n_tied = n - n_ogda_better - n_gda_better
    mean_diff = statistics.mean(differences) if differences else None

    verdict = "inconclusive"
    if n > 0 and mean_diff is not None:
        if n_ogda_better >= min_sign_consistency and mean_diff < 0:
            verdict = "fedogda_favored"
        elif n_gda_better >= min_sign_consistency and mean_diff > 0:
            verdict = "fedgda_favored"

    return {
        "n_pairs": n,
        "n_fedogda_better": n_ogda_better,
        "n_fedgda_better": n_gda_better,
        "n_tied": n_tied,
        "min_sign_consistency_required": min_sign_consistency,
        "mean_difference": mean_diff,
        "median_difference": statistics.median(differences) if differences else None,
        "differences": differences,
        "verdict": verdict,
    }


def paired_primary_differences(
    gda_records: list[dict[str, Any]], ogda_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """FedOGDA-minus-FedGDA primary-endpoint differences at matched (scenario, optimizer) seeds.

    Both sides of a pair must be complete and non-diverged, matching v1's
    analyze_eicu_study_a_confirmatory.py precedent.
    """

    def by_seed(records):
        out = {}
        for r in records:
            if r["metrics"] is None or bool(r["metrics"].get("diverged", False)):
                continue
            seed = int(r["row"].get("optimizer_seed", r["row"]["seed"]))
            out[seed] = r["metrics"]
        return out

    gda_by_seed = by_seed(gda_records)
    ogda_by_seed = by_seed(ogda_records)
    shared_seeds = sorted(set(gda_by_seed) & set(ogda_by_seed))
    differences = []
    for seed in shared_seeds:
        gda_value, _ = primary_endpoint_value(gda_by_seed[seed])
        ogda_value, _ = primary_endpoint_value(ogda_by_seed[seed])
        differences.append(ogda_value - gda_value)
    return {"seeds": shared_seeds, "differences": differences}


def run_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confirmatory_rows = [r for r in rows if r.get("role", "confirmatory") == "confirmatory"]
    by_g0_method: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in confirmatory_rows:
        by_g0_method.setdefault(row["g0"], {}).setdefault(row["method"], []).append(row)

    records_by_g0_method = {
        g0: {method: collect(recs) for method, recs in methods.items()}
        for g0, methods in by_g0_method.items()
    }

    summary = {
        g0: {method: summarize_group(recs) for method, recs in methods.items()}
        for g0, methods in records_by_g0_method.items()
    }

    per_g0_pairs: dict[str, dict[str, Any]] = {}
    pooled_differences: list[float] = []
    for g0 in G0_VARIANTS:
        methods = records_by_g0_method.get(g0, {})
        gda_key = next((m for m in methods if m.startswith("fedgda")), None)
        ogda_key = next((m for m in methods if m.startswith("fedogda")), None)
        if not (gda_key and ogda_key):
            continue
        paired = paired_primary_differences(methods[gda_key], methods[ogda_key])
        verdict = classify_verdict(paired["differences"], SIGN_CONSISTENCY_MIN_PER_G0)
        per_g0_pairs[g0] = {**paired, **verdict}
        pooled_differences.extend(paired["differences"])

    pooled_verdict = classify_verdict(pooled_differences, SIGN_CONSISTENCY_MIN_POOLED)

    return {
        "study": "eicu_study_a_v2_offhours",
        "primary_endpoint_keys_in_priority_order": list(PRIMARY_ENDPOINT_KEYS),
        "decision_rule": {
            "description": (
                "A scope (one g0, or the 15-pair pooled set) is 'favored' for "
                "a method only if that method is better in >= "
                f"{SIGN_CONSISTENCY_MIN_PER_G0}/5 paired seeds for a single "
                f"g0 (or >= {SIGN_CONSISTENCY_MIN_POOLED}/15 pooled) AND the "
                "mean paired difference agrees in sign. No p-value is "
                "computed. Pre-declared in this module's docstring; not "
                "settable via CLI flag."
            ),
            "sign_consistency_min_per_g0": SIGN_CONSISTENCY_MIN_PER_G0,
            "sign_consistency_min_pooled": SIGN_CONSISTENCY_MIN_POOLED,
        },
        "summary": summary,
        "pairwise_primary_endpoint": {"per_g0": per_g0_pairs, "pooled": pooled_verdict},
        "test_fields_read_for_final_reporting_only": list(PRIMARY_ENDPOINT_KEYS)
        + list(SECONDARY_METRIC_KEYS)
        + list(EFFECT_METRIC_KEYS),
        "test_fields_used_for_selection": [],
    }


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Study A v2 confirmatory results\n")
    add(
        "Primary endpoint: equal-client Test structural MSE at the "
        "validation-selected checkpoint. Paired FedOGDA-minus-FedGDA "
        "differences by (scenario seed, optimizer seed); 5 pairs per g0, "
        "15 pooled. Decision rule is pre-declared in the module docstring "
        "and reproduced below -- no p-value is computed; n=5 does not "
        "support one.\n"
    )
    add(f"> {result['decision_rule']['description']}\n")

    for g0 in sorted(result["summary"]):
        add(f"## g0 = `{g0}`\n")
        add("| method | seeds ok | diverged | primary Test MSE @ best-val (mean +/- std) |")
        add("|---|---|---|---|")
        for method in sorted(result["summary"][g0]):
            s = result["summary"][g0][method]
            primary = s.get("primary")
            mean_std = f"{_fmt(primary['mean'])} +/- {_fmt(primary['std'], '.2g')}" if primary else "n/a"
            add(f"| {method} | {s['n_seeds_complete']}/{s['n_seeds_expected']} | {s['n_diverged']} | {mean_std} |")
        add("")

        pair = result["pairwise_primary_endpoint"]["per_g0"].get(g0)
        if pair:
            add(f"### Paired FedOGDA - FedGDA, g0=`{g0}`\n")
            add(f"- matched seeds: {pair['seeds']}")
            add(f"- per-seed differences: {[round(d, 4) for d in pair['differences']]}")
            add(f"- mean difference: {_fmt(pair['mean_difference'])}")
            add(
                f"- FedOGDA better in {pair['n_fedogda_better']}/{pair['n_pairs']} "
                f"seeds (needs >= {pair['min_sign_consistency_required']} for a verdict)"
            )
            add(f"- **verdict: {pair['verdict']}**")
            add("")

    pooled = result["pairwise_primary_endpoint"]["pooled"]
    add("## Pooled (15 pairs across all g0)\n")
    add(f"- FedOGDA better in {pooled['n_fedogda_better']}/{pooled['n_pairs']} pooled pairs")
    add(f"- mean difference: {_fmt(pooled['mean_difference'])}")
    add(f"- **pooled verdict: {pooled['verdict']}**\n")

    if pooled["verdict"] == "inconclusive":
        add(
            "No pooled or unanimous per-g0 verdict clears the pre-declared "
            "sign-consistency bar. This is reported as inconclusive, not "
            "rounded up to a win for either method -- see "
            "`scripts/analyze_sine_a2_lite.py` for the precedent of "
            "reporting an unsupported subclaim plainly rather than "
            "reframing it.\n"
        )

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.manifest) as handle:
        rows = json.load(handle)
    result = run_analysis(rows)

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")

    csv_path = os.path.join(args.out, "per_group_primary_endpoint.csv")
    import csv as csv_module

    with open(csv_path, "w", newline="") as handle:
        writer = csv_module.DictWriter(
            handle,
            fieldnames=[
                "g0", "method", "n_seeds_complete", "n_seeds_expected",
                "n_diverged", "primary_mean", "primary_std", "primary_median",
            ],
        )
        writer.writeheader()
        for g0 in sorted(result["summary"]):
            for method in sorted(result["summary"][g0]):
                s = result["summary"][g0][method]
                primary = s.get("primary") or {}
                writer.writerow({
                    "g0": g0,
                    "method": method,
                    "n_seeds_complete": s["n_seeds_complete"],
                    "n_seeds_expected": s["n_seeds_expected"],
                    "n_diverged": s["n_diverged"],
                    "primary_mean": primary.get("mean", ""),
                    "primary_std": primary.get("std", ""),
                    "primary_median": primary.get("median", ""),
                })

    report = render_report(result)
    report_path = os.path.join(args.out, "README.md")
    with open(report_path, "w") as handle:
        handle.write(report + "\n")

    print(report)
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
