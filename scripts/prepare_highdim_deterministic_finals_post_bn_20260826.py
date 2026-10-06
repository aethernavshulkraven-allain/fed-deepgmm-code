#!/usr/bin/env python3
"""Prepare the 3-alpha x 5-seed final deterministic evidence (closeout plan
Phase 6 / SS9.2), under the exact-reuse accounting frozen in
PROTOCOL_DECISION_ADDENDUM_20260826.md SS6:

  V4 winners at alpha=0.5, seeds 0-2 (reused)........... 36
  alpha=0.1 stability runs, seed 0 (reused)............. 12
  New alpha=0.1 runs, seeds 1-4.......................... 48
  New alpha=0.5 runs, seeds 3-4.......................... 24
  New alpha=1.0 runs, seeds 0-4.......................... 60
  ------------------------------------------------------------
  Total final evidence (6 datasets x 2 methods x 3 alphas x 5 seeds) = 180
  New runs required after V4 + stability............... 132

Produces two separate artifacts, deliberately not one 180-row manifest:

- finals_launch_manifest.csv: only the 132 genuinely new rows, built from a
  real screen-manifest template row per cell (same technique
  prepare_highdim_deterministic_stability_alpha0p1_20260826.py uses) so every
  column run_manifest.py's build_config() requires without a default is
  present. This is the only file ever handed to run_manifest.py.
- finals_evidence_ledger.json: the full 180-trajectory accounting for
  aggregate_highdim_deterministic_finals_post_bn_20260826.py. Reused entries
  carry their *real, original* run_id and final_result_dir verbatim -- never
  a new alias -- because run_manifest.py's validate_artifacts() requires the
  on-disk effective_config.json's run_id to equal the row's run_id exactly;
  aliasing a reused row's run_id while pointing at the original directory
  would make every reused row fail that check. Reused rows are never passed
  to run_manifest.py at all -- there is nothing to launch or resume-skip for
  them, they already exist.

Consumes the same v4_winners.json contract as the stability preparer, plus
that stage's stability_results.json and stability_manifest.csv. Fails
closed -- rather than silently skipping or improvising -- if any cell's
stability check required the frozen alpha=0.1 retune escape hatch: retuning
a cell is deliberate, per-cell work (closeout plan SS9.3), not something
this preparer performs automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from highdim_protocol_hash_closure_20260822 import (  # noqa: E402
    CORE_DATASET_FILES,
    CORE_PROTOCOL_DOCS,
    CORE_SOURCES,
)

PROTOCOL_ROOT = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1"
DEFAULT_SCREEN_MANIFEST = (
    PROTOCOL_ROOT / "deterministic_screen_post_bn_20260822" / "screen_manifest.csv"
)
DEFAULT_OUTPUT_DIR = PROTOCOL_ROOT / "deterministic_finals_post_bn_20260826"
RESULT_ROOT = "results/highdim_deterministic_finals_post_bn_20260826"
METHOD_OPTIMIZERS = {"fedgda_d": "sgd", "fedogda_d": "ogda"}
ALPHA_TOKENS = {0.1: "alpha0p1", 0.5: "alpha0p5", 1.0: "alpha1"}


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _new_row(template, dataset, method, seed, alpha, lr, cm):
    run_id = (
        f"det_finals_postbn_{dataset}_{method}_seed{seed}_{ALPHA_TOKENS[alpha]}_"
        f"lr{_token(lr)}_cm{_token(cm)}"
    )
    row = dict(template)
    row.update({
        "run_id": run_id,
        "protocol_version": "highdim_deterministic_finals_post_bn_v1",
        "run_group": "highdim_deterministic_finals_post_bn_20260826",
        "seed": str(seed),
        "alpha": f"{alpha:g}",
        "partition_alpha": f"{alpha:g}",
        "learning_rate": f"{lr:g}",
        "critic_multiplier": f"{cm:g}",
        "comm_round": "500",
        "output_root": RESULT_ROOT,
        "final_result_dir": f"{RESULT_ROOT}/{dataset}/{method}/seed_{seed}/{run_id}",
        "run_status": "not_started",
        "preflight_required": "True",
        "preflight_status": "bn_buffer_diagnostic_certified",
        "server_buffer_policy": "direct_client_aggregate",
        # Every dataset in this campaign is a femnist_*/cifar10_* image
        # scenario; skip the ~10 GiB-scale full test-tensor write for every
        # new final-matrix run (closeout plan Phase 1 SS4.4).
        "compact_predictions_only": "True",
        # The template row's own source_manifest/source_run_id describe
        # that (arbitrarily-picked) screen row's ancestry, not this new
        # final-matrix row's -- this row's real lineage is the V4/stability
        # ledger entries in finals_evidence_ledger.json, not a single field.
        "source_manifest": "",
        "source_run_id": "",
        "notes": (
            f"Final-matrix trajectory for {dataset}/{method} at alpha={alpha:g}, "
            f"seed={seed} (lr={lr:g}, cm={cm:g}); fresh initialization."
        ),
    })
    return row


FINALS_COMM_ROUND = 500  # every finals-related trajectory (new or reused) is 500 rounds


def _ledger_entry(*, dataset, method, seed, alpha, lr, cm, run_id, final_result_dir, reused, source_stage):
    return {
        "run_id": run_id,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "alpha": alpha,
        # This campaign's "alpha" IS the Dirichlet partition concentration
        # (closeout review finding: a preparer that relabels alpha without
        # moving partition_alpha with it silently launches at the wrong
        # alpha) -- recording it explicitly here, rather than only as
        # "alpha", lets the aggregator cross-check it against the run's real
        # effective_config.json instead of trusting the ledger's own label.
        "partition_alpha": alpha,
        "learning_rate": lr,
        "critic_multiplier": cm,
        "comm_round": FINALS_COMM_ROUND,
        "client_optimizer": METHOD_OPTIMIZERS[method],
        "final_result_dir": final_result_dir,
        "reused": reused,
        "source_stage": source_stage,
    }


def prepare(winners_path: Path, stability_results_path: Path, stability_manifest_path: Path,
            screen_manifest_path: Path, output_dir: Path,
            retune_results_path: Path | None = None) -> dict:
    winners = _load_json(winners_path)
    if winners.get("status") != "complete":
        raise ValueError("V4 winners are absent or incomplete")
    if list(winners.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("V4 winners must cover exactly seeds 0, 1, 2")
    v4_cells = winners.get("cells")
    if not isinstance(v4_cells, dict) or len(v4_cells) != 12:
        raise ValueError("V4 winners must contain exactly 12 cells")

    stability = _load_json(stability_results_path)
    if stability.get("status") != "complete":
        raise ValueError("stability results are absent or incomplete")
    stability_cells = stability.get("cells")
    if not isinstance(stability_cells, dict) or len(stability_cells) != 12:
        raise ValueError("stability results must contain exactly 12 cells")
    retune_required = sorted(
        name for name, cell in stability_cells.items() if cell.get("outcome") != "pass"
    )
    retune_winners: dict[str, dict] = {}
    if retune_required:
        if retune_results_path is None:
            raise ValueError(
                "cannot generate the final matrix: these cells require the frozen "
                "per-cell alpha=0.1 retune escape hatch first (closeout plan SS9.1/SS9.3: "
                "Screen -> Rank -> Confirm -> Promote, starting from "
                f"scripts/prepare_highdim_stability_retune_alpha0p1_20260827.py): {retune_required}"
            )
        retune_results = _load_json(retune_results_path)
        if retune_results.get("status") != "complete":
            raise ValueError("alpha=0.1 retune results are absent or incomplete")
        if retune_results.get("stage") != "promote":
            raise ValueError(
                f"--retune-results must be the Promote stage's own output "
                f"(scripts/score_highdim_stability_retune_promote_alpha0p1_20260827.py), "
                f"got stage={retune_results.get('stage')!r} -- the Screen stage's raw "
                "output alone (SS9.1) is never eligible to promote a winner"
            )
        retune_cells = retune_results.get("cells")
        if not isinstance(retune_cells, dict):
            raise ValueError("alpha=0.1 retune results must contain a cells object")
        missing_retunes = sorted(set(retune_required) - set(retune_cells))
        if missing_retunes:
            raise ValueError(f"alpha=0.1 retune results are missing these cells: {missing_retunes}")
        retune_winners = {name: retune_cells[name] for name in retune_required}

    with stability_manifest_path.open(newline="") as handle:
        stability_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    stability_run_id_by_cell = {}
    for cell_name, cell in stability_cells.items():
        run_id = cell.get("run_id")
        if run_id not in stability_rows:
            raise ValueError(f"{cell_name}: stability result run_id {run_id!r} not found in stability manifest")
        stability_run_id_by_cell[cell_name] = run_id

    with screen_manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        templates: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            templates.setdefault((row["dataset"], row["method"]), row)
    if "compact_predictions_only" not in fieldnames:
        fieldnames.append("compact_predictions_only")

    output_dir.mkdir(parents=True, exist_ok=True)
    new_rows = []
    ledger = []
    for cell_name in sorted(v4_cells):
        cell = v4_cells[cell_name]
        dataset, method = cell["dataset"], cell["method"]
        if method not in METHOD_OPTIMIZERS:
            raise ValueError(f"{cell_name}: unsupported method {method!r}")
        template = templates.get((dataset, method))
        if template is None:
            raise ValueError(f"{cell_name}: no screen-manifest template row for this cell")
        winner = cell["winner"]
        lr, cm = float(winner["lr"]), float(winner["cm"])
        run_ids_by_seed = winner["run_ids"]
        if set(run_ids_by_seed) != {"0", "1", "2"}:
            raise ValueError(f"{cell_name}: winner must carry run_ids for seeds 0, 1, 2")
        if cell_name not in stability_run_id_by_cell:
            raise ValueError(f"{cell_name}: present in V4 winners but not in stability results")

        # alpha=0.5, seeds 0-2: reused from V4 verbatim (36 entries). Real
        # run_id/final_result_dir, never aliased -- see module docstring.
        for seed_str, source_run_id in sorted(run_ids_by_seed.items()):
            seed = int(seed_str)
            ledger.append(_ledger_entry(
                dataset=dataset, method=method, seed=seed, alpha=0.5, lr=lr, cm=cm,
                run_id=source_run_id,
                final_result_dir=(
                    f"results/highdim_psi_adjudication_post_bn_v4/{dataset}/{method}/"
                    f"seed_{seed}/{source_run_id}"
                ),
                reused=True, source_stage="psi_adjudication_post_bn_v4",
            ))
        if cell_name in retune_winners:
            # SS9.3: a retuned cell's original failed stability run does not
            # count as a final winner trajectory -- the newly selected
            # alpha=0.1 winner runs across ALL required final seeds (0-4),
            # none reused. alpha=0.5/alpha=1.0 are untouched -- retuning is
            # alpha=0.1-specific.
            retune_lr = float(retune_winners[cell_name]["winner"]["lr"])
            retune_cm = float(retune_winners[cell_name]["winner"]["cm"])
            new_specs = (
                [(seed, 0.1, retune_lr, retune_cm) for seed in (0, 1, 2, 3, 4)]
                + [(seed, 0.5, lr, cm) for seed in (3, 4)]
                + [(seed, 1.0, lr, cm) for seed in (0, 1, 2, 3, 4)]
            )
        else:
            # alpha=0.1, seed 0: reused from the stability stage verbatim.
            stability_run_id = stability_run_id_by_cell[cell_name]
            ledger.append(_ledger_entry(
                dataset=dataset, method=method, seed=0, alpha=0.1, lr=lr, cm=cm,
                run_id=stability_run_id,
                final_result_dir=stability_rows[stability_run_id]["final_result_dir"],
                reused=True, source_stage="deterministic_stability_alpha0p1_20260826",
            ))
            new_specs = (
                [(seed, 0.1, lr, cm) for seed in (1, 2, 3, 4)]
                + [(seed, 0.5, lr, cm) for seed in (3, 4)]
                + [(seed, 1.0, lr, cm) for seed in (0, 1, 2, 3, 4)]
            )
        # New rows: 12 per non-retuned cell (48+24+60=132 total across 12
        # cells), 15 per retuned cell (its alpha=0.1 slice grows from 4 new
        # + 1 reused to 5 new).
        for seed, alpha, row_lr, row_cm in new_specs:
            row = _new_row(template, dataset, method, seed, alpha, row_lr, row_cm)
            new_rows.append(row)
            ledger.append(_ledger_entry(
                dataset=dataset, method=method, seed=seed, alpha=alpha, lr=row_lr, cm=row_cm,
                run_id=row["run_id"], final_result_dir=row["final_result_dir"],
                reused=False, source_stage="",
            ))

    launch_manifest_path = output_dir / "finals_launch_manifest.csv"
    with launch_manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(new_rows)

    ledger_run_ids = [entry["run_id"] for entry in ledger]
    if len(set(ledger_run_ids)) != len(ledger_run_ids):
        duplicates = sorted({run_id for run_id in ledger_run_ids if ledger_run_ids.count(run_id) > 1})
        raise ValueError(f"finals evidence ledger has duplicate run_ids: {duplicates}")

    ledger_path = output_dir / "finals_evidence_ledger.json"
    ledger_path.write_text(json.dumps({
        "status": "complete",
        "total_trajectories": len(ledger),
        "reused_trajectories": sum(1 for entry in ledger if entry["reused"]),
        "new_trajectories": sum(1 for entry in ledger if not entry["reused"]),
        "trajectories": ledger,
    }, indent=2, sort_keys=True) + "\n")

    summary_path = output_dir / "finals_summary.json"
    summary_path.write_text(json.dumps({
        "campaign": "highdim_deterministic_finals_post_bn_20260826",
        "total_trajectories": len(ledger),
        "reused_trajectories": sum(1 for entry in ledger if entry["reused"]),
        "new_trajectories": len(new_rows),
        "server_buffer_policy": "direct_client_aggregate",
        "source_v4_winners": _relative_or_str(winners_path),
        "source_stability_results": _relative_or_str(stability_results_path),
        "source_screen_manifest": _relative_or_str(screen_manifest_path),
    }, indent=2, sort_keys=True) + "\n")

    hashed_paths = [
        launch_manifest_path, ledger_path, summary_path,
        winners_path, stability_results_path, stability_manifest_path, screen_manifest_path,
        *([retune_results_path] if retune_results_path is not None else []),
        *(REPO_ROOT / source for source in CORE_SOURCES),
        *(REPO_ROOT / dataset for dataset in CORE_DATASET_FILES),
        *(REPO_ROOT / doc for doc in CORE_PROTOCOL_DOCS),
        REPO_ROOT / "scripts/launch_highdim_deterministic_finals_post_bn_20260826.sh",
        REPO_ROOT / "scripts/aggregate_highdim_deterministic_finals_post_bn_20260826.py",
        Path(__file__),
    ]
    (output_dir / "generated_artifact_hashes.json").write_text(json.dumps([
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in sorted(hashed_paths)
    ], indent=2, sort_keys=True) + "\n")

    return {
        "launch_manifest": str(launch_manifest_path),
        "ledger": str(ledger_path),
        "total": len(ledger),
        "reused": sum(1 for entry in ledger if entry["reused"]),
        "new": len(new_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v4-winners", type=Path, required=True)
    parser.add_argument("--stability-results", type=Path, required=True)
    parser.add_argument("--stability-manifest", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, default=DEFAULT_SCREEN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--retune-results", type=Path, default=None,
        help="scripts/score_highdim_stability_retune_promote_alpha0p1_20260827.py's "
        "Promote-stage output (after Screen->Rank->Confirm); required only if any "
        "cell's stability outcome is not 'pass'.",
    )
    args = parser.parse_args()
    try:
        result = prepare(
            args.v4_winners.resolve(), args.stability_results.resolve(),
            args.stability_manifest.resolve(), args.screen_manifest.resolve(),
            args.output_dir.resolve(),
            args.retune_results.resolve() if args.retune_results else None,
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"FINALS PREPARATION BLOCKED: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
