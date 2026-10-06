"""Instrument diagnostics on a built eICU cohort.

Runs the checks that must pass before any causal estimate is reported: relevance
(partial F / partial R^2), overlap, covariate balance across instrument quantiles,
and the naive-OLS vs 2SLS comparison.

Two cautions are wired into the output rather than left to the reader:

* The gap between naive OLS and 2SLS illustrates confounding by indication. It is
  **not** evidence that the IV estimate is correct.
* When the audit gate reports ``insufficient_data``, every number here is labelled a
  pipeline artefact, because a cohort with no eligible client cannot support an
  estimate no matter how the arithmetic comes out.

Usage:
    python scripts/analyze_eicu_iv_diagnostics.py --cohort experiments/eicu_v1_demo/cohort.csv
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eicu_common import REPO_ROOT, continuous_covariate_columns  # noqa: E402
from eicu_instrument import (  # noqa: E402
    PREFERENCE_HOSPITAL,
    PREFERENCE_WARD,
    build_instrument,
    structural_instrument_variation,
)
from eicu_iv_diagnostics import (  # noqa: E402
    first_stage_diagnostics,
    ols,
    overlap_by_quantile,
    standardized_mean_differences,
    two_stage_least_squares,
)


def design_matrix(cohort, columns):
    """Numeric covariate matrix with median imputation and missingness indicators.

    Medians come from the frame passed in; the training pipeline recomputes them
    from training rows only. Missingness in eICU is partly a statement about which
    interfaces a hospital connected, so the indicator carries real information.
    """
    import numpy as np

    blocks, names = [], []
    for col in columns:
        values = cohort[col].astype("float64")
        missing = values.isna()
        if missing.all():
            continue
        blocks.append(values.fillna(values.median()).to_numpy())
        names.append(col)
        if missing.any():
            blocks.append(missing.astype("float64").to_numpy())
            names.append(col + "_missing")

    if not blocks:
        return np.empty((len(cohort), 0)), []
    return np.column_stack(blocks), names


def run_diagnostics(cohort, construction, client_col, seed=0):
    import numpy as np

    z, _, _ = build_instrument(
        cohort, construction=construction, client_col=client_col, seed=seed
    )
    frame = cohort.assign(z=z.values)

    treatment = frame["treatment"].to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy(dtype=float)
    instrument = frame["z"].to_numpy(dtype=float)

    covariate_cols = continuous_covariate_columns(frame)
    covariates, covariate_names = design_matrix(frame, covariate_cols)

    unit_col = "wardid" if construction == PREFERENCE_WARD else "hospitalid"
    structural = structural_instrument_variation(frame, client_col, unit_col, "z")

    naive = ols(
        np.column_stack([treatment, covariates]) if covariates.size else treatment,
        outcome,
    )
    iv = two_stage_least_squares(
        instrument,
        treatment,
        covariates=covariates if covariates.size else None,
        outcome=outcome,
    )

    return {
        "construction": construction,
        "client_col": client_col,
        "n": int(len(frame)),
        "n_clients": int(frame[client_col].nunique()),
        "treatment_rate": float(treatment.mean()),
        "mortality_rate": float(outcome.mean()),
        "first_stage": first_stage_diagnostics(
            instrument, treatment, covariates=covariates if covariates.size else None
        ),
        "overlap": overlap_by_quantile(instrument, treatment, n_bins=5),
        "balance": standardized_mean_differences(
            covariates, instrument, names=covariate_names
        )[:15]
        if covariates.size
        else [],
        "naive_ols_effect": float(naive["coef"][1]),
        "naive_ols_stderr": float(naive["stderr"][1]),
        "iv_effect": iv["effect"],
        "iv_stderr": iv["effect_stderr"],
        "clients_with_structural_variation": int((structural > 0.01).sum()),
        "n_covariates": len(covariate_names),
    }


def render(results, gate):
    lines = []
    add = lines.append

    add("# eICU instrument diagnostics\n")
    if gate and gate.get("construction") == "insufficient_data":
        add(
            "> **Pipeline artefact.** The Stage-1 audit reports "
            "`insufficient_data` for this release: no client meets the "
            "pre-registered eligibility thresholds. The numbers below verify that "
            "the code runs end to end. They are not estimates."
        )
        add("")

    for result in results:
        fs = result["first_stage"]
        add(f"## Construction: `{result['construction']}`\n")
        add(f"- rows: {result['n']}, clients: {result['n_clients']}")
        add(f"- covariates in the design: {result['n_covariates']}")
        add(
            f"- clients with structural instrument variation: "
            f"{result['clients_with_structural_variation']}"
        )
        add(f"- treatment rate: {result['treatment_rate']:.3f}")
        add(f"- mortality rate: {result['mortality_rate']:.3f}")
        add("")

        add("### Relevance (first stage)\n")
        add("| quantity | value |")
        add("|---|---|")
        add(f"| instrument coefficient | {fs['instrument_coef']:.4f} |")
        add(f"| robust SE | {fs['instrument_stderr']:.4f} |")
        add(f"| partial F | {fs['partial_f']:.2f} |")
        add(f"| partial R^2 | {fs['partial_r2']:.4f} |")
        warn = "yes" if fs["weak_instrument_warning"] else "no"
        add(f"| weak-instrument warning (F < 10) | {warn} |")
        add("")

        if result["overlap"]:
            add("### Overlap: treatment rate by instrument quintile\n")
            add("| bin | Z range | n | treatment rate |")
            add("|---|---|---|---|")
            for row in result["overlap"]:
                add(
                    f"| {row['bin']} | [{row['z_low']:.3f}, {row['z_high']:.3f}] | "
                    f"{row['n']} | {row['treatment_rate']:.3f} |"
                )
            add("")

        if result["balance"]:
            add("### Covariate balance (top / bottom instrument quintile)\n")
            add("| covariate | mean low Z | mean high Z | SMD |")
            add("|---|---|---|---|")
            for row in result["balance"]:
                add(
                    f"| `{row['covariate']}` | {row['mean_low']:.3f} | "
                    f"{row['mean_high']:.3f} | {row['smd']:+.3f} |"
                )
            add("")
            add(
                "Large |SMD| indicates the instrument may encode patient "
                "composition or unit specialisation rather than practice style — "
                "the principal exclusion-restriction threat here."
            )
            add("")

        add("### Effect estimates\n")
        add("| estimator | effect on in-hospital mortality | robust SE |")
        add("|---|---|---|")
        add(
            f"| naive OLS (confounded) | {result['naive_ols_effect']:+.4f} | "
            f"{result['naive_ols_stderr']:.4f} |"
        )
        add(f"| 2SLS | {result['iv_effect']:+.4f} | {result['iv_stderr']:.4f} |")
        add("")
        add(
            "The gap between the two illustrates confounding by indication. It is "
            "not evidence that the IV estimate is correct."
        )
        add("")

    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort",
        default=os.path.join(REPO_ROOT, "experiments", "eicu_v1_demo", "cohort.csv"),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    import pandas as pd

    args = parse_args(argv)
    cohort = pd.read_csv(args.cohort)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.cohort))
    os.makedirs(out_dir, exist_ok=True)

    gate_path = os.path.join(out_dir, "construction_decision.json")
    gate = None
    if os.path.exists(gate_path):
        with open(gate_path) as handle:
            gate = json.load(handle)

    results = [run_diagnostics(cohort, PREFERENCE_WARD, "hospitalid", seed=args.seed)]

    grouping = [c for c in ("region", "teachingstatus", "numbedscategory") if c in cohort]
    if grouping:
        grouped = cohort.copy()
        grouped["client_group"] = (
            grouped[grouping].astype("string").fillna("NA").agg(" | ".join, axis=1)
        )
        results.append(
            run_diagnostics(
                grouped, PREFERENCE_HOSPITAL, "client_group", seed=args.seed
            )
        )

    report = render(results, gate)
    report_path = os.path.join(out_dir, "iv_diagnostics.md")
    with open(report_path, "w") as handle:
        handle.write(report + "\n")

    json_path = os.path.join(out_dir, "iv_diagnostics.json")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")

    print(report)
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
