#!/usr/bin/env python3
"""500-round, 3-seed (0,1,2), alpha=0.5 adjudication manifest for the
Psi-vs-MSE winner discrepancy found 2026-08-18.

For each cell in the given cell set, takes the union of {Psi rank-1, Psi
rank-2, existing MSE-selected winner}, de-duplicates identical (lr, cm)
pairs, and generates new-run rows only for candidates that don't already
have 500-round/alpha-0.5 data -- the existing MSE-winner's runs in
deterministic_finals_20260813 are reused, not repeated.

Two invocations needed (per the decided plan, kept as separate manifests
since they answer different questions):
  --cells x       -> femnist_x, cifar10_x (both methods): is Psi usable here at all?
  --cells signal  -> femnist_z, femnist_xz, cifar10_z, cifar10_xz (both methods):
                     does the screen-stage Psi gap survive 500 rounds / 3 seeds?

Promotion rule (per the decided plan, applied when scoring the results of
this manifest, not by this script): promote by median validation Psi across
the 3 seeds, with non-finite-event type and late-round stability reported
alongside -- not by seed-0-only or by the 150-round screen value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_deterministic.csv"
)
SCREEN_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
FINALS_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
EXISTING_FINALS_ROOT = REPO_ROOT / "results/highdim_deterministic_finals_20260813"

X_CELLS = [("femnist_x", "fedgda_d"), ("femnist_x", "fedogda_d"),
           ("cifar10_x", "fedgda_d"), ("cifar10_x", "fedogda_d")]
SIGNAL_CELLS = [("femnist_z", "fedgda_d"), ("femnist_z", "fedogda_d"),
                ("femnist_xz", "fedgda_d"), ("femnist_xz", "fedogda_d"),
                ("cifar10_z", "fedgda_d"), ("cifar10_z", "fedogda_d"),
                ("cifar10_xz", "fedgda_d"), ("cifar10_xz", "fedogda_d")]

SEEDS = (0, 1, 2)
ALPHA = 0.5
COMM_ROUND = 500
CLIENT_COUNT = 10

# Ground truth, mirrored from run_manifest.METHOD_TO_OPTIMIZER /
# experiment_utils.get_effective_config()'s algorithm map. See
# deterministic_screen_20260813/QUARANTINE_20260819_mislabeled_fedogda_expand2/
# for the incident this exists to prevent: reference_row() matched on
# dataset only, so every generated row silently inherited
# client_optimizer/method_label from whichever source row for that dataset
# came first (always fedgda_d/sgd), regardless of the intended method.
METHOD_TO_OPTIMIZER = {"fedgda_d": "sgd", "fedogda_d": "ogda"}
METHOD_LABEL = {"fedgda_d": "FedGDA-D", "fedogda_d": "FedOGDA-D"}

EXTRA_FIELDS = (
    "auxiliary_regression", "auxiliary_regression_epochs", "objective_lambda_1",
    "append_round_csv", "periodic_checkpoint_interval", "log_test_mse_by_round",
    "test_mse_used_for_selection", "selection_metric_source", "objective_mode",
    "aggregation_weighting",
)


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source():
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows, dataset, method):
    matches = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
    if not matches:
        raise RuntimeError(f"No source row for dataset={dataset}, method={method}")
    return matches[0]


def common_row(source, protocol_version, run_group):
    row = dict(source)
    row.update({
        "protocol_version": protocol_version,
        "run_group": run_group,
        "training_scope": "federated",
        "alpha": f"{ALPHA:g}",
        "partition_alpha": f"{ALPHA:g}",
        "client_num_in_total": str(CLIENT_COUNT),
        "client_num_per_round": str(CLIENT_COUNT),
        "comm_round": str(COMM_ROUND),
        "epochs": "3",
        "batch_size": "0",
        "weight_decay": "0.001",
        "server_learning_rate": "1.5",
        "gradient_clip_norm": "1.0",
        "objective_lambda_1": "0.1",
        "run_status": "not_started",
        "implementation_status": "ready",
        "preflight_required": "False",
        "preflight_status": "not_required",
        "auxiliary_regression": "False",
        "auxiliary_regression_epochs": "0",
        "append_round_csv": "True",
        "periodic_checkpoint_interval": "0",
        "log_test_mse_by_round": "False",
        "test_mse_used_for_selection": "False",
        "selection_metric_source": "validation",
        "objective_mode": "legacy",
        "aggregation_weighting": "sample_size",
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", choices=["x", "signal"], required=True)
    parser.add_argument("--psi-rescore", default=str(SCREEN_DIR / "psi_rescore.json"))
    parser.add_argument("--campaign-dir", default=str(REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260818"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results/highdim_psi_adjudication_20260818"))
    parser.add_argument("--protocol-version", default="highdim_psi_adjudication_v1")
    parser.add_argument("--run-group", default="highdim_psi_adjudication_20260818")
    parser.add_argument("--manifest-name", default=None, help="Override output filename stem (default: adjudication_<cells>)")
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    output_root = Path(args.output_root)

    cell_list = X_CELLS if args.cells == "x" else SIGNAL_CELLS
    psi = json.load(open(args.psi_rescore))
    mse_winners = json.load(open(FINALS_DIR / "frozen_winners.json"))
    source_fields, source_rows = load_source()
    fieldnames = source_fields + [f for f in EXTRA_FIELDS if f not in source_fields]

    rows = []
    plan = []
    for ds, method in cell_list:
        key = f"{ds}|{method}"
        entry = psi[key]
        mse_w = mse_winners[f"{ds}/{method}"]
        candidates = {}  # (lr, cm) -> set of source labels
        for label, cand in (("psi_rank1", entry["rank_1"]), ("psi_rank2", entry["rank_2"])):
            if cand is None:
                continue
            candidates.setdefault((cand["lr"], cand["cm"]), set()).add(label)
        candidates.setdefault((mse_w["lr"], mse_w["cm"]), set()).add("mse_winner")

        cell_plan = {"dataset": ds, "method": method, "candidates": []}
        for (lr, cm), labels in sorted(candidates.items()):
            is_existing_mse_winner = (lr, cm) == (mse_w["lr"], mse_w["cm"])
            cell_plan["candidates"].append({
                "lr": lr, "cm": cm, "labels": sorted(labels),
                "reused_from_finals": is_existing_mse_winner,
            })
            if is_existing_mse_winner:
                continue  # already have 500rd/alpha0.5/seeds0-2 data in deterministic_finals_20260813
            source = reference_row(source_rows, ds, method)
            for seed in SEEDS:
                row = common_row(source, args.protocol_version, args.run_group)
                run_id = (
                    f"det_adjudicate_{ds}_{method}_seed{seed}_alpha0p5"
                    f"_lr{token(lr)}_cm{token(cm)}"
                )
                row.update({
                    "run_id": run_id, "dataset": ds, "method": method, "seed": str(seed),
                    "client_optimizer": METHOD_TO_OPTIMIZER[method],
                    "method_label": METHOD_LABEL[method],
                    "learning_rate": f"{lr:g}", "learning_rate_status": "psi_adjudication_candidate",
                    "critic_multiplier": f"{cm:g}",
                    "output_root": str(output_root),
                    "final_result_dir": str(output_root / ds / method / f"seed_{seed}" / run_id),
                    "notes": (
                        f"Psi-vs-MSE adjudication candidate ({','.join(sorted(labels))}). "
                        f"500 rounds, alpha=0.5. Promote by cross-seed median of "
                        f"last-50-round mean Psi, not seed-0-only or best-round. "
                        f"See PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md."
                    ),
                })
                rows.append(row)
        plan.append(cell_plan)

    manifest_name = args.manifest_name or f"adjudication_{args.cells}_manifest"
    manifest_path = campaign_dir / f"{manifest_name}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "campaign": f"{args.run_group}_{args.cells}",
        "psi_rescore_source": str(args.psi_rescore),
        "new_runs": len(rows),
        "existing_finals_root_reused": str(EXISTING_FINALS_ROOT),
        "plan": plan,
    }
    with (campaign_dir / f"{manifest_name.replace('_manifest', '')}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
