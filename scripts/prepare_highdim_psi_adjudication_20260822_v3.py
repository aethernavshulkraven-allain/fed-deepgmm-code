#!/usr/bin/env python3
"""Materialize the preserved, superseded v3 adjudication packet."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260822_v3"
)
RESULT_ROOT = "results/highdim_psi_adjudication_20260822_v3"
DIAGNOSTIC_RESULT_ROOT = "results/highdim_bn_buffer_diagnostic_20260822_v3"


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_source_templates() -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    templates: dict[tuple[str, str], dict[str, str]] = {}
    fieldnames: list[str] | None = None
    for cells in ("signal", "x"):
        path = SOURCE_DIR / f"adjudication_{cells}_manifest.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                templates.setdefault((row["dataset"], row["method"]), row)
    if fieldnames is None:
        raise ValueError("source manifests have no header")
    if "server_buffer_policy" not in fieldnames:
        fieldnames.append("server_buffer_policy")
    return fieldnames, templates


def build_row(
    template: dict[str, str],
    *,
    dataset: str,
    method: str,
    seed: int,
    lr: float,
    cm: float,
    labels: list[str],
) -> dict[str, str]:
    row = dict(template)
    run_id = (
        f"det_adjudicate_v3_{dataset}_{method}_seed{seed}_alpha0p5_"
        f"lr{token(lr)}_cm{token(cm)}"
    )
    row.update({
        "run_id": run_id,
        "protocol_version": "highdim_psi_adjudication_v3",
        "run_group": "highdim_psi_adjudication_20260822_v3",
        "dataset": dataset,
        "method": method,
        "seed": str(seed),
        "output_root": RESULT_ROOT,
        "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}",
        "run_status": "not_started",
        "preflight_required": "True",
        "preflight_status": "pending_bn_buffer_diagnostic",
        "comm_round": "500",
        "learning_rate": f"{lr:g}",
        "critic_multiplier": f"{cm:g}",
        "server_buffer_policy": "direct_client_aggregate",
        "scientific_status": "superseded_pre_fix_selected_shortlist",
        "notes": (
            f"Post-buffer-fix v3 candidate ({','.join(labels)}). All candidates and all "
            "seeds rerun from initialization; no v2/finals model states are reused."
        ),
    })
    return row


def write_campaign(
    cells: str,
    fieldnames: list[str],
    templates: dict[tuple[str, str], dict[str, str]],
) -> list[Path]:
    source_summary_path = SOURCE_DIR / f"adjudication_{cells}_summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    plan = deepcopy(source_summary["plan"])
    rows = []
    for cell in plan:
        dataset, method = cell["dataset"], cell["method"]
        template = templates[(dataset, method)]
        for candidate in cell["candidates"]:
            candidate["previously_reused_from_finals"] = bool(
                candidate["reused_from_finals"]
            )
            candidate["reused_from_finals"] = False
            for seed in (0, 1, 2):
                rows.append(build_row(
                    template,
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    lr=float(candidate["lr"]),
                    cm=float(candidate["cm"]),
                    labels=list(candidate["labels"]),
                ))

    manifest_path = OUTPUT_DIR / f"adjudication_{cells}_manifest.csv"
    campaign_fieldnames = list(fieldnames)
    if "scientific_status" not in campaign_fieldnames:
        campaign_fieldnames.append("scientific_status")
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=campaign_fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "campaign": "highdim_psi_adjudication_20260822_v3",
        "source_plan": str(source_summary_path.relative_to(REPO_ROOT)),
        "model_state_reuse": False,
        "new_runs": len(rows),
        "server_buffer_policy": "direct_client_aggregate",
        "plan": plan,
    }
    summary_path = OUTPUT_DIR / f"adjudication_{cells}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return [manifest_path, summary_path]


def write_diagnostic(
    fieldnames: list[str],
    templates: dict[tuple[str, str], dict[str, str]],
) -> Path:
    dataset, method, seed, lr, cm = "femnist_z", "fedogda_d", 1, 0.001, 10.0
    row = build_row(
        templates[(dataset, method)],
        dataset=dataset,
        method=method,
        seed=seed,
        lr=lr,
        cm=cm,
        labels=["bn_buffer_diagnostic"],
    )
    run_id = "bn_buffer_diagnostic_v3_femnist_z_fedogda_d_seed1_lr0p001_cm10"
    row.update({
        "run_id": run_id,
        "protocol_version": "highdim_bn_buffer_diagnostic_v3",
        "run_group": "highdim_bn_buffer_diagnostic_20260822_v3",
        "output_root": DIAGNOSTIC_RESULT_ROOT,
        "final_result_dir": f"{DIAGNOSTIC_RESULT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}",
        "comm_round": "120",
        "preflight_required": "False",
        "preflight_status": "diagnostic",
        "scientific_status": "eligible",
        "notes": (
            "Reproduction configuration for the v2 seed with isolated nonfinite critic "
            "rounds. Must finish cleanly before either v3 adjudication stage starts."
        ),
    })
    path = OUTPUT_DIR / "bn_buffer_diagnostic_manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    return path


def write_hashes(paths: list[Path]) -> None:
    records = []
    for path in sorted(paths):
        records.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (OUTPUT_DIR / "generated_artifact_hashes.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames, templates = load_source_templates()
    generated = []
    for cells in ("signal", "x"):
        generated.extend(write_campaign(cells, fieldnames, templates))
    generated.append(write_diagnostic(fieldnames, templates))
    generated.extend([
        OUTPUT_DIR / "CORRECTION_ADDENDUM_20260822.md",
        REPO_ROOT / "fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py",
        REPO_ROOT / "fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py",
        REPO_ROOT / "scripts/check_manifest_stage_complete.py",
        REPO_ROOT / "scripts/certify_highdim_bn_diagnostic_20260822.py",
        REPO_ROOT / "scripts/launch_highdim_bn_buffer_diagnostic_20260822_v3.sh",
        REPO_ROOT / "scripts/launch_highdim_psi_adjudication_20260822_v3.sh",
        REPO_ROOT / "scripts/run_manifest.py",
        REPO_ROOT / "scripts/score_highdim_adjudication_20260819.py",
        REPO_ROOT / "scripts/verify_protocol_hashes.py",
        OUTPUT_DIR / "bn_buffer_diagnostic_certification.json",
        OUTPUT_DIR / "diagnostic_launch_hashes.json",
        Path(__file__),
    ])
    write_hashes(generated)
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
        "signal_runs": sum(1 for _ in csv.DictReader(
            (OUTPUT_DIR / "adjudication_signal_manifest.csv").open()
        )),
        "x_runs": sum(1 for _ in csv.DictReader(
            (OUTPUT_DIR / "adjudication_x_manifest.csv").open()
        )),
        "diagnostic_runs": 1,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
