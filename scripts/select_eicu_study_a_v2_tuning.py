#!/usr/bin/env python3
"""Select Study A v2 hyperparameters using the exact validation-only rule.

For each (g0, federated method), choose the completed non-diverged candidate
with the lowest equal-client validation structural MSE. Break an exact tie
only with equal-client validation moment violation at that checkpoint. No Test
field is read.

Two additional, pre-run correctness guards live here:

* ``at_grid_edge``: the tuning grid is the learning-rate x server-learning-rate
  product given in protocol_v2.json ("tuning"). If the winner for a (g0,
  method) group sits on the edge of the explored range for a parameter,
  "FedOGDA beats FedGDA" may just mean one method wanted a learning rate
  outside the grid. Every group's winner is checked against the grid values
  *actually present in that group's own candidates* (not a hardcoded
  constant), and the result is recorded per group in the emitted selection
  JSON plus printed loudly to stderr -- see ``grid_edge_warnings``.
* Horizon mismatch: an LR selected at ``TUNING_ROUNDS`` is not verified to
  still win at ``FINAL_ROUNDS``. Both are read from protocol_v2.json
  ("training"). Amendment 1 (2026-07-27) set them equal precisely so this risk
  is zero by construction, and ``horizon_mismatch_warning`` is empty whenever
  they agree. Should a future amendment separate them again, the warning
  returns and ``build_horizon_confirmation_manifest`` /
  ``--emit-horizon-confirmation-manifest`` can *generate* (never launch) a
  small validation-only manifest that re-runs each selected config at
  ``FINAL_ROUNDS``. This module only builds machinery, it never launches
  anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "experiments" / "eicu_study_a_v2_offhours" / "protocol_v2.json"
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Read the pre-registered protocol that defines the grid and horizon."""
    with path.open() as fh:
        return json.load(fh)


_PROTOCOL = load_protocol()

# Derived from protocol_v2.json rather than duplicated, so that widening the
# grid there is enough to widen the completeness check here.
LOCAL_LR_MULTIPLIERS = tuple(float(m) for m in _PROTOCOL["tuning"]["local_lr_multipliers"])
SERVER_LEARNING_RATES = tuple(
    float(s) for s in _PROTOCOL["tuning"]["server_learning_rates"]
)
EXPECTED_CANDIDATES_PER_GROUP = len(LOCAL_LR_MULTIPLIERS) * len(SERVER_LEARNING_RATES)

# protocol_v2.json "training".tuning_rounds / .final_rounds. Amendment 1 set
# these equal; the horizon-confirmation machinery below stays in place because
# a future amendment could separate them again, but it only warns when they
# actually differ.
TUNING_ROUNDS = int(_PROTOCOL["training"]["tuning_rounds"])
FINAL_ROUNDS = int(_PROTOCOL["training"]["final_rounds"])

# protocol_v2.json "tuning".scenario_seed / .optimizer_seed -- the single
# scenario/optimizer pair every tuning candidate (and therefore every
# horizon-confirmation row) is drawn from.
TUNING_SCENARIO_SEED = 11
TUNING_OPTIMIZER_SEED = 1011
TUNING_SEED_PAIR_ID = "tuning_01"

HORIZON_CONFIRMATION_ROLE = "horizon_confirmation"
G0_VARIANTS = ("linear", "interaction", "mlp")
FEDERATED_METHODS = ("fedgda_s", "fedogda_s")


def horizon_mismatch_warning(
    tuning_rounds: int = TUNING_ROUNDS, final_rounds: int = FINAL_ROUNDS
) -> str:
    """Warn that a pick made at one budget is unverified at another.

    Returns the empty string when the two budgets agree: a selection made at
    the same horizon the campaign reports at carries no such risk, and emitting
    a warning anyway would train readers to ignore it.
    """
    if tuning_rounds == final_rounds:
        return ""
    return (
        f"selected using rounds={tuning_rounds}; NOT verified to "
        f"remain optimal at the final campaign's rounds={final_rounds}. "
        "Run this module with --emit-horizon-confirmation-manifest to "
        "generate (never launch) a validation-only re-run of this "
        f"exact (learning_rate, server_learning_rate) at "
        f"rounds={final_rounds} before trusting this pick for the "
        "final campaign."
    )


def run_dir(row: dict[str, Any]) -> Path:
    return (
        Path(row["output_root"])
        / row["dataset"]
        / row["method"]
        / f"seed_{int(row['optimizer_seed'])}"
        / row["run_id"]
    )


def grid_edge_flags(
    candidates: list[dict[str, Any]], winner_row: dict[str, Any]
) -> tuple[dict[str, str], list[float], list[float]]:
    """Which of the winner's tuned parameters sit at the edge of this group's grid.

    Uses the learning_rate / server_learning_rate values actually present
    among ``candidates`` (all rows configured for this (g0, method) group,
    diverged or not -- a divergent candidate is still an explored grid
    point), not a hardcoded constant, so this stays correct even if the grid
    changes shape.

    A parameter with only one distinct explored value can't have an
    "edge" (there is nothing to be at the edge of), so it is skipped --
    this only ever happens if a manifest is malformed, since every real
    tuning group explores the full grid. Note that with only 2 explored
    server_learning_rate values, EVERY winner is at an edge on that
    parameter -- that is correct and intentional, not a bug: with two grid
    points there is no interior point to distinguish "the middle won" from
    "the edge won", so the honest thing is to always flag it.

    Returns ``(edge_parameters, learning_rate_grid, server_learning_rate_grid)``
    where ``edge_parameters`` maps parameter name -> ``"low"`` or ``"high"``
    for parameters at an edge (absent if not at an edge), and the two grid
    lists are the sorted distinct values actually observed, for audit.
    """
    lr_values = sorted({float(c["learning_rate"]) for c in candidates})
    slr_values = sorted({float(c["server_learning_rate"]) for c in candidates})
    winner_lr = float(winner_row["learning_rate"])
    winner_slr = float(winner_row["server_learning_rate"])
    edge_parameters: dict[str, str] = {}
    if len(lr_values) > 1 and winner_lr in (lr_values[0], lr_values[-1]):
        edge_parameters["learning_rate"] = (
            "low" if winner_lr == lr_values[0] else "high"
        )
    if len(slr_values) > 1 and winner_slr in (slr_values[0], slr_values[-1]):
        edge_parameters["server_learning_rate"] = (
            "low" if winner_slr == slr_values[0] else "high"
        )
    return edge_parameters, lr_values, slr_values


def grid_edge_warnings(selected: dict[str, Any]) -> list[str]:
    """Human-readable warning per group whose winner sits at a grid edge.

    Used by ``main`` to print loudly to stderr -- the campaign should not
    silently proceed on a grid-limited selection.
    """
    warnings = []
    for key in sorted(selected):
        choice = selected[key]
        if not choice.get("at_grid_edge"):
            continue
        params = choice.get("at_grid_edge_parameters", {})
        detail = ", ".join(f"{name}={side}" for name, side in sorted(params.items()))
        warnings.append(
            f"{key}: winning candidate is at the grid edge on {detail} -- it "
            "may only be winning because the grid did not extend further in "
            "that direction."
        )
    return warnings


def selection_values(metrics: dict[str, Any]) -> tuple[float, float]:
    validation_mse = metrics.get("equal_client_best_validation_structural_mse")
    if validation_mse is None:
        validation_mse = metrics["best_validation_mse"]
    moment = metrics[
        "equal_client_validation_moment_violation_at_best_validation"
    ]
    return float(validation_mse), float(moment)


def select(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["g0"], row["method"]), []).append(row)

    selected: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for (g0, method), candidates in sorted(groups.items()):
        key = f"{g0}:{method}"
        if len(candidates) != EXPECTED_CANDIDATES_PER_GROUP:
            raise ValueError(
                f"{key} has {len(candidates)} candidates; expected "
                f"{EXPECTED_CANDIDATES_PER_GROUP}"
            )
        evaluated = []
        missing = []
        for row in candidates:
            metrics_path = run_dir(row) / "metrics.json"
            if not metrics_path.is_file():
                missing.append(str(metrics_path))
                continue
            with metrics_path.open() as handle:
                metrics = json.load(handle)
            if bool(metrics.get("diverged", False)):
                evaluated.append(
                    {"row": row, "status": "diverged", "metrics": metrics}
                )
                continue
            values = selection_values(metrics)
            evaluated.append(
                {
                    "row": row,
                    "status": "eligible",
                    "metrics": metrics,
                    "selection_values": values,
                }
            )
        if missing:
            raise FileNotFoundError(
                f"{key} is incomplete; missing metrics: {missing}"
            )
        eligible = [
            candidate
            for candidate in evaluated
            if candidate["status"] == "eligible"
        ]
        if not eligible:
            raise ValueError(f"{key} has no non-diverged tuning candidate")
        winner = min(eligible, key=lambda item: item["selection_values"])
        val_mse, val_moment = winner["selection_values"]
        edge_parameters, lr_grid, slr_grid = grid_edge_flags(candidates, winner["row"])
        selected[key] = {
            "learning_rate": float(winner["row"]["learning_rate"]),
            "server_learning_rate": float(
                winner["row"]["server_learning_rate"]
            ),
            "run_id": winner["row"]["run_id"],
            "equal_client_validation_mse": val_mse,
            "equal_client_validation_moment_violation": val_moment,
            "selection_rule": [
                "equal_client_validation_mse",
                "equal_client_validation_moment_violation_at_best_validation",
            ],
            "test_fields_read": [],
            # Grid-edge guard: is the winner at the boundary of the explored
            # range for a tuned parameter? See grid_edge_flags docstring --
            # with only 2 server_learning_rate points, that parameter is
            # always flagged, which is intentional (see docstring).
            "at_grid_edge": bool(edge_parameters),
            "at_grid_edge_parameters": edge_parameters,
            "grid_learning_rate_candidates": lr_grid,
            "grid_server_learning_rate_candidates": slr_grid,
            # Horizon mismatch guard: a pick validated at TUNING_ROUNDS is not
            # verified to still win at FINAL_ROUNDS. Empty when the protocol
            # tunes and reports at the same budget, which is the whole point of
            # Amendment 1. See build_horizon_confirmation_manifest.
            "tuning_rounds": TUNING_ROUNDS,
            "final_rounds": FINAL_ROUNDS,
            "horizon_mismatch_warning": horizon_mismatch_warning(),
        }
        report[key] = {
            "n_candidates": len(candidates),
            "n_diverged": sum(
                item["status"] == "diverged" for item in evaluated
            ),
            "selected": selected[key],
        }
    expected_groups = {
        f"{g0}:{method}" for g0 in G0_VARIANTS for method in FEDERATED_METHODS
    }
    if set(selected) != expected_groups:
        raise ValueError(
            f"tuning groups do not match protocol: {sorted(selected)}"
        )
    return selected, report


# ---------------------------------------------------------------------------
# Horizon-confirmation manifest: generate-only, never launched from here.
# ---------------------------------------------------------------------------


def load_scenario_metadata(scenario_dir: str | Path, g0: str, scenario_seed: int) -> dict[str, Any]:
    """Load a certified Study A v2 scenario's metadata sidecar.

    Read-only: this never touches the frozen scenario artifacts, it only
    reads the metadata JSON that ``prepare_eicu_study_a_v2_scenarios.py``
    already wrote alongside them.
    """
    path = Path(scenario_dir) / f"{g0}_scenario_seed{int(scenario_seed)}_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing certified scenario metadata: {path}")
    with path.open() as handle:
        return json.load(handle)


def build_horizon_confirmation_manifest(
    selected: dict[str, Any],
    scenario_metadata_by_g0: dict[str, dict[str, Any]],
    *,
    dataset: str,
    output_root: str,
    data_cache_dir: str,
    scenario_seed: int = TUNING_SCENARIO_SEED,
    optimizer_seed: int = TUNING_OPTIMIZER_SEED,
    seed_pair_id: str = TUNING_SEED_PAIR_ID,
    rounds: int = FINAL_ROUNDS,
) -> list[dict[str, Any]]:
    """Build (never launch) the 6-row horizon-confirmation manifest.

    Re-runs each tuning-selected (g0, method) config -- identical
    learning_rate and server_learning_rate, nothing re-tuned -- at ``rounds``
    (``FINAL_ROUNDS`` by default) instead of the ``TUNING_ROUNDS`` they were
    selected at. This answers "does the pick made at the tuning horizon still
    win at the final horizon?", not "what is the best config at the final
    horizon?" -- those are different questions.

    Under Amendment 1 the two horizons are equal, which makes this a no-op
    confirmation and is the intended state: the campaign now re-tunes at the
    horizon it reports at rather than confirming a pick made at a shorter one.
    The machinery remains for any future amendment that separates them again.

    Every row is validation-only (``test_mse_used_for_selection: "false"``,
    ``selection_source: "validation_only"``), matching the discipline the
    tuning selection itself enforces.
    """
    rows = []
    for g0 in G0_VARIANTS:
        metadata = scenario_metadata_by_g0.get(g0)
        if metadata is None:
            raise KeyError(f"scenario metadata missing for g0={g0!r}")
        for method in FEDERATED_METHODS:
            key = f"{g0}:{method}"
            if key not in selected:
                raise KeyError(f"selected hyperparameters missing group {key!r}")
            choice = selected[key]
            run_id = f"horizon_confirmation_{g0}_{method}_{seed_pair_id}_r{int(rounds)}"
            rows.append(
                {
                    "run_id": run_id,
                    "protocol_version": "eicu_study_a_v2_offhours",
                    "run_group": HORIZON_CONFIRMATION_ROLE,
                    "role": HORIZON_CONFIRMATION_ROLE,
                    "training_scope": "federated",
                    "method": method,
                    "method_label": "FedOGDA" if method == "fedogda_s" else "FedGDA",
                    "g0": g0,
                    "dataset": dataset,
                    "scenario_name": f"{g0}_scenario_seed{int(scenario_seed)}",
                    "scenario_seed": int(scenario_seed),
                    "optimizer_seed": int(optimizer_seed),
                    "seed": int(optimizer_seed),
                    "seed_pair_id": seed_pair_id,
                    "comm_round": int(rounds),
                    "epochs": 1,
                    "batch_size": 4,
                    "require_multibatch_stochastic": "true",
                    "client_optimizer": "ogda" if method == "fedogda_s" else "sgd",
                    "federated_optimizer": "FedAvg",
                    "model": "lr",
                    "partition_method": "natural",
                    "partition_alpha": 0.0,
                    "learning_rate": float(choice["learning_rate"]),
                    "server_learning_rate": float(choice["server_learning_rate"]),
                    "learning_rate_status": (
                        f"frozen_from_{TUNING_ROUNDS}_round_tuning_"
                        f"pending_{int(rounds)}_round_confirmation"
                    ),
                    "selected_tuning_run_id": choice.get("run_id", ""),
                    "weight_decay": float(metadata.get("weight_decay", 0.01)),
                    "critic_multiplier": float(metadata.get("critic_multiplier", 10.0)),
                    "gradient_clip_norm": 1.0,
                    "client_num_in_total": int(metadata["n_clients"]),
                    "client_num_per_round": int(metadata["n_clients"]),
                    "input_dim_g": int(metadata["input_dim"]),
                    "input_dim_f": int(metadata["instrument_dim"]),
                    "hidden_widths": "32,32",
                    "model_activation": "relu",
                    "objective_mode": "paper_aligned",
                    "aggregation_weighting": "uniform_clients",
                    "output_root": str(output_root),
                    "data_cache_dir": str(data_cache_dir),
                    "scenario_checksum": metadata.get("scenario_checksum_sha256", ""),
                    "scenario_scope": metadata.get("scenario_scope", ""),
                    "using_gpu": "false",
                    "gpu_id": 0,
                    "skip_model_selection": "true",
                    "simple_model_selection_epochs": 100,
                    "f_history_model_selection_epochs": 60,
                    "model_selection_batch_size": 200,
                    "test_mse_used_for_selection": "false",
                    "selection_source": "validation_only",
                    "primary_selection_metric": "equal_client_validation_mse",
                    "alignment_label": "horizon_confirmation_validation_only",
                    "study_claim": "semi_synthetic_benchmark_no_clinical_claim",
                    "run_status": "not_started",
                    "implementation_status": "generated_not_launched",
                    "notes": (
                        f"validation-only re-run of the tuning-selected config "
                        f"at rounds={int(rounds)} (tuning was at "
                        f"rounds={TUNING_ROUNDS}); generated by "
                        "select_eicu_study_a_v2_tuning.py, NOT launched"
                    ),
                }
            )
    if len(rows) != 6:
        raise RuntimeError(
            f"internal error: horizon confirmation manifest has {len(rows)} "
            f"rows, expected 6"
        )
    return rows


def write_horizon_confirmation_manifest(
    rows: list[dict[str, Any]], out_path: str | Path
) -> tuple[Path, Path]:
    """Write the horizon-confirmation manifest as JSON + CSV. Never runs anything."""
    json_path = Path(out_path).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fieldnames = sorted({field for row in rows for field in row})
    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--emit-horizon-confirmation-manifest",
        default=None,
        help=(
            "If set, also GENERATE (never launch) a 6-row validation-only "
            "manifest re-running each selected config at "
            "--horizon-confirmation-rounds. Requires --scenario-dir."
        ),
    )
    parser.add_argument(
        "--scenario-dir",
        default=None,
        help="Certified scenario directory; required with --emit-horizon-confirmation-manifest.",
    )
    parser.add_argument("--dataset", default=None, help="Defaults to --scenario-dir's basename.")
    parser.add_argument("--output-root", default=None, help="Defaults to results/eicu_study_a_v2_offhours_demo.")
    parser.add_argument("--data-cache-dir", default=None, help="Defaults to --scenario-dir's parent.")
    parser.add_argument("--horizon-confirmation-rounds", type=int, default=FINAL_ROUNDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with Path(args.manifest).open() as handle:
        rows = json.load(handle)
    selected, report = select(rows)

    edge_warnings = grid_edge_warnings(selected)
    horizon_note = (
        f"tuning_rounds={TUNING_ROUNDS} vs final_rounds={FINAL_ROUNDS}: the "
        "selected (learning_rate, server_learning_rate) pairs have not been "
        "verified to remain optimal at the final horizon. Use "
        "--emit-horizon-confirmation-manifest to generate (never launch) a "
        f"validation-only re-run at rounds={FINAL_ROUNDS} before trusting "
        "these picks for the final campaign."
    )
    report["_campaign_warnings"] = {
        "grid_edge": edge_warnings,
        "horizon_mismatch": horizon_note,
    }

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(selected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report_path = output.with_name(output.stem + "_report.json")
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output}")
    print(f"wrote {report_path}")

    # Loud, not silent: grid-edge selections and the horizon mismatch print
    # to stderr regardless of whether a horizon-confirmation manifest is
    # requested this run.
    print(f"WARNING: {horizon_note}", file=sys.stderr)
    if edge_warnings:
        print("WARNING: grid-edge tuning selections detected:", file=sys.stderr)
        for line in edge_warnings:
            print(f"  - {line}", file=sys.stderr)
        print(
            "WARNING: the campaign should not silently proceed on a "
            "grid-edge selection; review before materializing the final "
            "manifest.",
            file=sys.stderr,
        )

    if args.emit_horizon_confirmation_manifest:
        if not args.scenario_dir:
            raise ValueError(
                "--scenario-dir is required with --emit-horizon-confirmation-manifest"
            )
        scenario_dir = Path(args.scenario_dir).resolve()
        dataset = args.dataset or scenario_dir.name
        output_root = args.output_root or str(
            Path(__file__).resolve().parents[1]
            / "results"
            / "eicu_study_a_v2_offhours_demo"
        )
        data_cache_dir = args.data_cache_dir or str(scenario_dir.parent)
        metadata_by_g0 = {
            g0: load_scenario_metadata(scenario_dir, g0, TUNING_SCENARIO_SEED)
            for g0 in G0_VARIANTS
        }
        horizon_rows = build_horizon_confirmation_manifest(
            selected,
            metadata_by_g0,
            dataset=dataset,
            output_root=output_root,
            data_cache_dir=data_cache_dir,
            rounds=args.horizon_confirmation_rounds,
        )
        horizon_json, horizon_csv = write_horizon_confirmation_manifest(
            horizon_rows, args.emit_horizon_confirmation_manifest
        )
        print(f"wrote {horizon_json} (GENERATED ONLY -- not launched)")
        print(f"wrote {horizon_csv} (GENERATED ONLY -- not launched)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
