"""Standalone post-hoc per-client evaluation for a Study A checkpoint.

Deliberately does **not** go through the FedML data-loading pipeline (client_id
is dropped before it reaches the global train/val/test objects used at
training time -- see the alignment review this closes). Instead it loads a
trained checkpoint and the scenario `.npz` directly, reconstructs the g/f
networks from the checkpoint's own `effective_config`, and evaluates them on
one split, grouped by the `client_id` array that the scenario already carries.

Reports, per client and in aggregate:
    * structural MSE against the known g0
    * absolute ATE error (|mean predicted effect - mean true effect|, taken
      *after* averaging -- positive/negative per-row errors can cancel)
    * individual-effect MAE (mean absolute per-row effect error -- cannot
      cancel; metric_policy.md is explicit that this is NOT the same
      quantity as ATE error and the two must not be substituted)
    * held-out moment violation
    * equal-client vs. sample-weighted aggregates of all of the above

Usage:
    python scripts/analyze_eicu_study_a_checkpoint.py \\
        --checkpoint results/.../checkpoints/best_validation.pt \\
        --scenario fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth/linear_seed0.npz \\
        --split test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_ROOT = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from models.mlp_model import MLPModel  # noqa: E402

from experiment_utils import moment_violation as _moment_violation_scalar  # noqa: E402

# Matches model_hub.py's eicu* branch exactly -- must stay in sync with it, since
# a checkpoint's state dict is only loadable into a model of the same shape.
EICU_HIDDEN_WIDTHS = [64, 64]


def _hidden_widths(effective_config):
    value = effective_config.get("hidden_widths", EICU_HIDDEN_WIDTHS)
    if isinstance(value, (list, tuple)):
        widths = [int(item) for item in value]
    else:
        widths = [
            int(item.strip())
            for item in str(value).strip("[]").split(",")
            if item.strip()
        ]
    return widths or list(EICU_HIDDEN_WIDTHS)


def _activation(effective_config):
    value = str(effective_config.get("model_activation", "leaky_relu")).lower()
    choices = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
    if value not in choices:
        raise ValueError(f"unsupported checkpoint model_activation {value!r}")
    return choices[value]


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("g_state_dict", "f_state_dict", "effective_config"):
        if key not in checkpoint:
            raise ValueError(f"{path} is missing {key!r}; not a Study A checkpoint?")
    return checkpoint


def build_models(effective_config):
    input_dim_g = int(effective_config.get("input_dim_g", 0))
    input_dim_f = int(effective_config.get("input_dim_f", 0))
    if input_dim_g <= 0 or input_dim_f <= 0:
        raise ValueError(
            "checkpoint's effective_config has no usable input_dim_g/input_dim_f "
            f"(got {input_dim_g}, {input_dim_f}); was it trained before this field "
            "was added?"
        )
    widths = _hidden_widths(effective_config)
    activation = _activation(effective_config)
    g = MLPModel(input_dim=input_dim_g, layer_widths=widths, activation=activation).double()
    f = MLPModel(input_dim=input_dim_f, layer_widths=widths, activation=activation).double()
    return g, f


def load_scenario_split(scenario_path, split):
    data = np.load(scenario_path)
    required = ("x", "z", "y", "g", "client_id", "g0_treated", "g0_control", "true_effect")
    missing = [key for key in required if f"{split}_{key}" not in data.files]
    if missing:
        raise ValueError(
            f"scenario {scenario_path} split {split!r} is missing arrays: {missing} "
            "-- was it generated before scenario certification was added?"
        )
    return {key: data[f"{split}_{key}"] for key in required}


def load_metadata(scenario_path):
    meta_path = os.path.splitext(scenario_path)[0] + "_metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no sibling metadata file at {meta_path}")
    with open(meta_path) as handle:
        return json.load(handle)


def evaluate(g, f, split_data):
    """Per-sample predictions and errors, with no gradient tracking."""
    x = torch.as_tensor(split_data["x"]).double()
    z = torch.as_tensor(split_data["z"]).double()
    y = torch.as_tensor(split_data["y"]).double().squeeze(-1)
    true_g = split_data["g"].squeeze(-1)

    n_covariates = x.shape[1] - 1
    w = x[:, 1:]
    x_treated = torch.cat([torch.ones(len(w), 1, dtype=torch.float64), w], dim=1)
    x_control = torch.cat([torch.zeros(len(w), 1, dtype=torch.float64), w], dim=1)

    with torch.no_grad():
        g_pred = torch.squeeze(g(x)).numpy()
        f_pred = torch.squeeze(f(z)).numpy()
        pred_treated = torch.squeeze(g(x_treated)).numpy()
        pred_control = torch.squeeze(g(x_control)).numpy()

    y_np = y.numpy()
    squared_error = (g_pred - true_g) ** 2
    pred_effect = pred_treated - pred_control
    true_effect = split_data["true_effect"].squeeze(-1)
    # Per-row individual-effect error -- absolute value taken per row before
    # any averaging, so it cannot benefit from cross-row sign cancellation.
    individual_effect_error = pred_effect - true_effect

    return {
        "g_pred": g_pred,
        "f_pred": f_pred,
        "y": y_np,
        "squared_error": squared_error,
        "pred_effect": pred_effect,
        "true_effect": true_effect,
        "individual_effect_error": individual_effect_error,
        "client_id": split_data["client_id"],
    }


def per_client_metrics(evald, client_code_to_hospital):
    """One row per client: MSE, true/predicted ATE, individual-effect MAE,
    moment violation, sample count.

    ``per_client_abs_ate_error`` (|pred_ate - true_ate| for THIS client) is
    reported only as a per-client distribution statistic -- the campaign
    aggregate ATE error is computed differently (see ``aggregate_ate_error``):
    metric_policy.md requires averaging predicted and true ATEs *across*
    clients first and taking one absolute value at the end, not averaging
    each client's already-absolute error.
    """
    rows = []
    for code in np.unique(evald["client_id"]):
        mask = evald["client_id"] == code
        hospital_id = client_code_to_hospital.get(str(int(code)), client_code_to_hospital.get(int(code)))
        true_ate = float(evald["true_effect"][mask].mean())
        pred_ate = float(evald["pred_effect"][mask].mean())
        rows.append(
            {
                "client_code": int(code),
                "hospital_id": hospital_id,
                "n": int(mask.sum()),
                "mse": float(evald["squared_error"][mask].mean()),
                "true_ate": true_ate,
                "pred_ate": pred_ate,
                "per_client_abs_ate_error": abs(pred_ate - true_ate),
                "individual_effect_mae": float(np.abs(evald["individual_effect_error"][mask]).mean()),
                "moment_violation": _moment_violation_scalar(
                    evald["f_pred"][mask], evald["y"][mask], evald["g_pred"][mask]
                ),
            }
        )
    return sorted(rows, key=lambda r: r["client_code"])


def aggregate(rows, key):
    """Equal-client (mean of per-client values) vs sample-weighted (n-weighted).

    Valid for metrics that are already per-client scalars whose equal-client
    and sample-weighted aggregates are each other's straightforward mean --
    MSE, individual-effect MAE, and moment violation. NOT valid for ATE error:
    see ``aggregate_ate_error``.
    """
    values = np.array([r[key] for r in rows])
    counts = np.array([r["n"] for r in rows])
    equal_client = float(values.mean())
    sample_weighted = float(np.average(values, weights=counts))
    return {
        "equal_client": equal_client,
        "sample_weighted": sample_weighted,
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "worst": float(values.max()),
    }


def aggregate_ate_error(rows):
    """Absolute ATE error, per metric_policy.md's exact (non-substitutable)
    definition: average predicted and true client ATEs *across clients*
    first, then take one absolute difference -- not the mean of each
    client's already-absolute error (that quantity is reported separately,
    below, purely as a per-client distribution diagnostic).
    """
    pred_ate = np.array([r["pred_ate"] for r in rows])
    true_ate = np.array([r["true_ate"] for r in rows])
    counts = np.array([r["n"] for r in rows])
    equal_client = abs(float(pred_ate.mean()) - float(true_ate.mean()))
    sample_weighted = abs(
        float(np.average(pred_ate, weights=counts)) - float(np.average(true_ate, weights=counts))
    )
    per_client_abs = np.array([r["per_client_abs_ate_error"] for r in rows])
    return {
        "equal_client": equal_client,
        "sample_weighted": sample_weighted,
        "median": float(np.median(per_client_abs)),
        "p90": float(np.percentile(per_client_abs, 90)),
        "worst": float(per_client_abs.max()),
    }


def render_report(checkpoint_path, scenario_path, split, rows, checkpoint):
    lines = []
    add = lines.append
    add("# Study A per-client checkpoint evaluation\n")
    add(f"- checkpoint: `{checkpoint_path}`")
    add(f"- checkpoint_type: `{checkpoint.get('checkpoint_type')}`, round: {checkpoint.get('round')}")
    add(f"- scenario: `{scenario_path}`")
    add(f"- split: `{split}`")
    add(f"- clients evaluated: {len(rows)}\n")

    add("## Aggregate metrics\n")
    add("| metric | equal-client | sample-weighted | median | p90 | worst client |")
    add("|---|---|---|---|---|---|")
    for key, label in (("mse", "structural MSE"), ("individual_effect_mae", "individual-effect MAE"), ("moment_violation", "moment violation")):
        agg = aggregate(rows, key)
        add(
            f"| {label} | {agg['equal_client']:.4g} | {agg['sample_weighted']:.4g} | "
            f"{agg['median']:.4g} | {agg['p90']:.4g} | {agg['worst']:.4g} |"
        )
    ate_agg = aggregate_ate_error(rows)
    add(
        f"| absolute ATE error | {ate_agg['equal_client']:.4g} | {ate_agg['sample_weighted']:.4g} | "
        f"{ate_agg['median']:.4g} | {ate_agg['p90']:.4g} | {ate_agg['worst']:.4g} |"
    )
    add("")
    add(
        "Absolute ATE error is not individual-effect MAE: the former can be small "
        "because positive and negative per-client errors cancel; the latter cannot "
        "(metric_policy.md).\n"
    )

    add("## Per-client detail\n")
    add("| hospital_id | n | MSE | true ATE | pred ATE | \\|ATE error\\| | individual-effect MAE | moment violation |")
    add("|---|---|---|---|---|---|---|---|")
    for row in rows:
        add(
            f"| {row['hospital_id']} | {row['n']} | {row['mse']:.4g} | "
            f"{row['true_ate']:.4g} | {row['pred_ate']:.4g} | {row['per_client_abs_ate_error']:.4g} | "
            f"{row['individual_effect_mae']:.4g} | {row['moment_violation']:.4g} |"
        )
    add("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--out", default=None, help="defaults to alongside the checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    checkpoint = load_checkpoint(args.checkpoint)
    g, f = build_models(checkpoint["effective_config"])
    g.load_state_dict(checkpoint["g_state_dict"])
    f.load_state_dict(checkpoint["f_state_dict"])
    g.eval()
    f.eval()

    split_data = load_scenario_split(args.scenario, args.split)
    metadata = load_metadata(args.scenario)
    client_code_to_hospital = metadata.get("client_code_to_hospital", {})

    evald = evaluate(g, f, split_data)
    rows = per_client_metrics(evald, client_code_to_hospital)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_type = checkpoint.get("checkpoint_type", "checkpoint")

    csv_path = os.path.join(out_dir, f"per_client_eval_{checkpoint_type}_{args.split}.csv")
    with open(csv_path, "w") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row[k]) for k in fieldnames) + "\n")

    report = render_report(args.checkpoint, args.scenario, args.split, rows, checkpoint)
    report_path = os.path.join(out_dir, f"per_client_eval_{checkpoint_type}_{args.split}.md")
    with open(report_path, "w") as handle:
        handle.write(report + "\n")

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "scenario": os.path.abspath(args.scenario),
        "split": args.split,
        "checkpoint_type": checkpoint_type,
        "round": checkpoint.get("round"),
        "n_clients": len(rows),
        "aggregates": {
            **{key: aggregate(rows, key) for key in ("mse", "individual_effect_mae", "moment_violation")},
            "absolute_ate_error": aggregate_ate_error(rows),
        },
    }
    summary_path = os.path.join(out_dir, f"per_client_eval_{checkpoint_type}_{args.split}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(report)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
