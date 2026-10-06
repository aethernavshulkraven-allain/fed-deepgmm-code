#!/usr/bin/env python3
"""Corrected rerun of the ten mislabeled FedOGDA-D boundary-expansion-2
candidates (2026-08-19).

Root cause: prepare_highdim_deterministic_screen_expand2_20260818.py's
reference_row() matched the hyperparameter-defaults template by dataset
only. The first row per dataset in the source manifest is always the
fedgda_d/sgd row, so every fedogda_d-targeted row in that script's output
(screen_expand2_manifest.csv) silently inherited client_optimizer="sgd"
and method_label="FedGDA-D" from that template, while its own "method"
column correctly said "fedogda_d". All ten of those rows launched, ran the
full 150-round screen protocol to completion, and wrote genuine
FedGDA-D (SGD) results into the fedgda_d output tree under fedogda_d-named
run_ids -- see QUARANTINE_20260819_mislabeled_fedogda_expand2/ for the
quarantined artifacts and INVALIDATION_NOTE_20260819.md for the full
incident record. reference_row() has since been fixed in that script (and
in prepare_highdim_psi_adjudication_20260818.py, which had the identical
bug and had not yet launched) to match on (dataset, method) and to set
client_optimizer/method_label explicitly rather than trusting the copied
template.

This script does NOT reuse the misleading old run_ids or overwrite their
(quarantined) outputs -- it regenerates only the ten fedogda_d candidates,
under new run_ids (protocol_version bumped to
"highdim_deterministic_screen_expand2_corrected_v1") that cannot collide
with anything already on disk. The seven fedgda_d rows from the original
expand2 manifest were correct throughout (client_optimizer was already
"sgd" as intended) and are NOT regenerated here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_deterministic.csv"
)
CAMPAIGN_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813"
)
OUTPUT_ROOT = REPO_ROOT / "results/highdim_deterministic_screen_20260813"
PSI_RESCORE_PATH = CAMPAIGN_DIR / "psi_rescore.json"
ORIGINAL_EXPAND2_MANIFEST = CAMPAIGN_DIR / "screen_expand2_manifest.csv"

COMM_ROUND = 150
CLIENT_COUNT = 10
TARGET_METHOD = "fedogda_d"

METHOD_TO_OPTIMIZER = {"fedgda_d": "sgd", "fedogda_d": "ogda"}
METHOD_LABEL = {"fedgda_d": "FedGDA-D", "fedogda_d": "FedOGDA-D"}

EXTRA_FIELDS = (
    "auxiliary_regression",
    "auxiliary_regression_epochs",
    "objective_lambda_1",
    "append_round_csv",
    "periodic_checkpoint_interval",
    "log_test_mse_by_round",
    "test_mse_used_for_selection",
    "selection_metric_source",
    "objective_mode",
    "aggregation_weighting",
)


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reference_row(rows: list[dict[str, str]], dataset: str, method: str) -> dict[str, str]:
    matches = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
    if not matches:
        raise RuntimeError(f"No source row for dataset={dataset}, method={method}")
    return matches[0]


def common_row(source: dict[str, str]) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "protocol_version": "highdim_deterministic_screen_expand2_corrected_v1",
            "run_group": "highdim_deterministic_screen_20260813",
            "training_scope": "federated",
            "seed": "0",
            "alpha": "0.5",
            "partition_alpha": "0.5",
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
        }
    )
    return row


def main() -> int:
    with ORIGINAL_EXPAND2_MANIFEST.open(newline="") as handle:
        original_rows = list(csv.DictReader(handle))
    mislabeled = [r for r in original_rows if r["method"] == TARGET_METHOD]

    _, source_rows = load_source()
    source_fields, _ = load_source()
    fieldnames = source_fields + [f for f in EXTRA_FIELDS if f not in source_fields]

    rows: list[dict[str, str]] = []
    corrected_pairs = []
    for old in mislabeled:
        dataset = old["dataset"]
        lr = float(old["learning_rate"])
        cm = float(old["critic_multiplier"])
        source = reference_row(source_rows, dataset, TARGET_METHOD)
        row = common_row(source)
        run_id = (
            f"det_screen_expand2corr_{dataset}_{TARGET_METHOD}_seed0_alpha0p5"
            f"_lr{token(lr)}_cm{token(cm)}"
        )
        row.update(
            {
                "run_id": run_id,
                "dataset": dataset,
                "method": TARGET_METHOD,
                "client_optimizer": METHOD_TO_OPTIMIZER[TARGET_METHOD],
                "method_label": METHOD_LABEL[TARGET_METHOD],
                "learning_rate": f"{lr:g}",
                "learning_rate_status": "screen_boundary_expansion2_candidate_corrected",
                "critic_multiplier": f"{cm:g}",
                "output_root": str(OUTPUT_ROOT),
                "final_result_dir": str(OUTPUT_ROOT / dataset / TARGET_METHOD / "seed_0" / run_id),
                "notes": (
                    f"Corrected rerun of mislabeled run {old['run_id']} "
                    "(2026-08-18 template-lookup bug: client_optimizer was silently "
                    "'sgd' instead of 'ogda'). Original mislabeled artifacts quarantined, "
                    "not reused. See INVALIDATION_NOTE_20260819.md."
                ),
            }
        )
        rows.append(row)
        corrected_pairs.append({"old_run_id": old["run_id"], "new_run_id": run_id, "dataset": dataset, "lr": lr, "cm": cm})

    manifest_path = CAMPAIGN_DIR / "screen_expand2_corrected_v1_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "campaign": "highdim_deterministic_screen_20260813_expand2_corrected_v1",
        "corrects": "screen_expand2_manifest.csv fedogda_d rows (client_optimizer template-lookup bug)",
        "runs": len(rows),
        "corrected_pairs": corrected_pairs,
    }
    with (CAMPAIGN_DIR / "screen_expand2_corrected_v1_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
