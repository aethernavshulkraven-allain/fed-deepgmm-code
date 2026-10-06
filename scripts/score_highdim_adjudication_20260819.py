#!/usr/bin/env python3
"""Mechanical scorer for the 500-round, 3-seed Psi-vs-MSE adjudication and
confirmation stages, implementing
PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md's decision tree (section 0)
exactly:

1. Scoring does not begin until every planned candidate has exactly three
   config-matched, ordered 500-round artifacts. Missing or malformed work
   makes the cell incomplete; it is not a scientific exclusion.
2. A candidate is eligible only if all 3 seeds have finite required metrics
   and no sticky divergence/nonfinite evidence anywhere in the full curve.
3. 0 eligible candidates -> cell requires retuning.
4. 1 eligible candidate -> promoted directly, no ranking needed.
5. >=2 eligible candidates -> rank by M_c = median across seeds of
   S_{c,s} = mean(Psi_451:500); compare the top-ranked candidate against
   every other eligible candidate with the frozen pairwise practical-tie
   rule; form the tie set (top plus everything that ties with top); if the
   tie set has more than one member, promote whichever has the lowest
   median last-50-round mean validation MSE within that set. An exact MSE tie
   remains unresolved rather than being broken by manifest order.

Critic collapse is diagnostic-only and never enters this scorer's
decision -- by design, this module has no notion of critic output at all.

This module is imported by tests/test_adjudication_scorer_20260819.py,
which exercises it (as a pure library, synthetic Candidate/SeedResult
objects, no disk I/O) against the six required synthetic scenarios.

It is ALSO an end-to-end CLI (see `main()` / `load_seed_result` /
`build_cell_candidates` below) that reads the real completed/reused runs
for a given adjudication manifest (results/highdim_psi_adjudication_20260819_v2/
for new runs, results/highdim_deterministic_finals_20260813/ for
reused-from-finals candidates) and writes final per-cell adjudication
outcomes -- this wiring is frozen now, before any real adjudication
results exist, per the same before-seeing-the-data discipline as every
other rule in this campaign.

Usage:
  python scripts/score_highdim_adjudication_20260819.py --cells x
  python scripts/score_highdim_adjudication_20260819.py --cells signal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import METHOD_TO_OPTIMIZER, validate_artifacts  # noqa: E402

CAMPAIGN_V2_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
FINALS_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
NEW_RUN_RESULTS_ROOT = REPO_ROOT / "results/highdim_psi_adjudication_20260819_v2"
NEW_RUN_ID_PREFIX = "det_adjudicate"
REQUIRED_ARTIFACTS = (
    "effective_config.json", "metrics.json", "mse_by_round.csv",
    "predictions.npz", "checkpoints/best_validation.pt", "checkpoints/final.pt",
)
LAST_N_ROUNDS = 50
REQUIRED_COMM_ROUNDS = 500
COMPLETE_STATUS = "complete"
TERMINAL_STATUS = "terminal_ineligible"
INCOMPLETE_STATUSES = {"incomplete", "invalid"}


@dataclass(frozen=True)
class SeedResult:
    seed: int
    psi_last50_mean: float | None
    mse_last50_mean: float | None
    diverged: bool
    artifacts_complete: bool
    finite: bool
    best_round_psi: float | None = None  # diagnostic only, never used for scoring
    run_dir: str = ""
    note: str = ""
    status: str = COMPLETE_STATUS

    @property
    def ok(self) -> bool:
        return (
            self.status == COMPLETE_STATUS
            and self.artifacts_complete
            and self.finite
            and not self.diverged
            and self.psi_last50_mean is not None
            and self.mse_last50_mean is not None
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    label: str
    seeds: tuple[SeedResult, ...]

    @property
    def eligible(self) -> bool:
        return len(self.seeds) == 3 and all(s.ok for s in self.seeds)

    @property
    def incomplete(self) -> bool:
        return len(self.seeds) != 3 or any(s.status in INCOMPLETE_STATUSES for s in self.seeds)

    @property
    def psi_scores(self) -> list[float]:
        return [s.psi_last50_mean for s in self.seeds]

    @property
    def mse_scores(self) -> list[float]:
        return [s.mse_last50_mean for s in self.seeds]

    @property
    def median_psi(self) -> float:
        return median(self.psi_scores)

    @property
    def median_mse(self) -> float:
        return median(self.mse_scores)

    @property
    def psi_range(self) -> float:
        return max(self.psi_scores) - min(self.psi_scores)


def practical_tie(a: Candidate, b: Candidate) -> bool:
    """Frozen pairwise Psi tie rule from PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md."""
    diffs = [sa - sb for sa, sb in zip(a.psi_scores, b.psi_scores)]
    signs = {(1 if d > 0 else (-1 if d < 0 else 0)) for d in diffs}
    signs_differ = len(signs) > 1
    gap_small = abs(a.median_psi - b.median_psi) <= max(a.psi_range, b.psi_range)
    return signs_differ or gap_small


@dataclass
class CellOutcome:
    outcome: str
    winner: Candidate | None
    eligible: list[Candidate]
    excluded: list[Candidate]
    tie_set: list[Candidate] = field(default_factory=list)
    detail: str = ""


def score_cell(candidates: list[Candidate]) -> CellOutcome:
    eligible = [c for c in candidates if c.eligible]
    excluded = [c for c in candidates if not c.eligible]

    incomplete = [c for c in candidates if c.incomplete]
    if incomplete:
        return CellOutcome(
            outcome="incomplete", winner=None, eligible=eligible, excluded=excluded,
            detail=(
                "planned candidate data are missing or invalid: "
                f"{[candidate.candidate_id for candidate in incomplete]}"
            ),
        )

    if not eligible:
        return CellOutcome(
            outcome="retune_required", winner=None, eligible=eligible, excluded=excluded,
            detail="no eligible candidate: every candidate failed the three-seed stability rule",
        )

    if len(eligible) == 1:
        return CellOutcome(
            outcome="promoted", winner=eligible[0], eligible=eligible, excluded=excluded,
            tie_set=list(eligible), detail="only one eligible candidate, no ranking needed",
        )

    ranked = sorted(eligible, key=lambda c: -c.median_psi)
    top = ranked[0]
    tie_set = [top] + [c for c in ranked[1:] if practical_tie(top, c)]

    if len(tie_set) == 1:
        return CellOutcome(
            outcome="promoted", winner=top, eligible=eligible, excluded=excluded,
            tie_set=tie_set, detail="top Psi candidate, no practical tie with any other eligible candidate",
        )

    minimum_mse = min(c.median_mse for c in tie_set)
    mse_winners = [c for c in tie_set if c.median_mse == minimum_mse]
    if len(mse_winners) != 1:
        return CellOutcome(
            outcome="mse_tie_unresolved", winner=None, eligible=eligible, excluded=excluded,
            tie_set=tie_set,
            detail=(
                "practical Psi tie and exact median-MSE tie among "
                f"{[c.candidate_id for c in mse_winners]}; no order-based promotion"
            ),
        )
    winner = mse_winners[0]
    return CellOutcome(
        outcome="tie_resolved_by_mse", winner=winner, eligible=eligible, excluded=excluded,
        tie_set=tie_set,
        detail=f"practical tie among {[c.candidate_id for c in tie_set]}, resolved by lowest median MSE",
    )


# --------------------------------------------------------------------------
# Real-data wiring: reads actual completed/reused runs off disk and scores
# them. Everything above this line is a pure library with no disk I/O.
# --------------------------------------------------------------------------

def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _json_numbers_are_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _load_json_object(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_seed_config(
    config: dict,
    run_dir: Path,
    seed: int,
    expected_config: dict | None,
) -> None:
    if int(config.get("random_seed")) != int(seed):
        raise ValueError(
            f"random_seed mismatch: {config.get('random_seed')!r} != {seed}"
        )
    if str(config.get("run_id")) != run_dir.name:
        raise ValueError(f"run_id mismatch: {config.get('run_id')!r} != {run_dir.name!r}")
    if int(config.get("comm_round")) != REQUIRED_COMM_ROUNDS:
        raise ValueError(
            f"comm_round must be {REQUIRED_COMM_ROUNDS}, got {config.get('comm_round')!r}"
        )
    if str(config.get("server_buffer_policy")) != "direct_client_aggregate":
        raise ValueError(
            "server_buffer_policy must be 'direct_client_aggregate', got "
            f"{config.get('server_buffer_policy')!r}"
        )
    if expected_config is None:
        return
    exact_fields = ("dataset", "variant", "client_optimizer")
    for field_name in exact_fields:
        expected = expected_config.get(field_name)
        if expected is None:
            continue
        if str(config.get(field_name)) != str(expected):
            raise ValueError(
                f"{field_name} mismatch: {config.get(field_name)!r} != {expected!r}"
            )
    for field_name in ("learning_rate", "critic_multiplier"):
        expected = expected_config.get(field_name)
        if expected is None:
            continue
        if not math.isclose(
            float(config.get(field_name)), float(expected), rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"{field_name} mismatch: {config.get(field_name)!r} != {expected!r}"
            )


def load_seed_result(
    run_dir: Path,
    seed: int,
    expected_config: dict | None = None,
    manifest_row: dict[str, str] | None = None,
) -> SeedResult:
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        return SeedResult(
            seed=seed, psi_last50_mean=None, mse_last50_mean=None,
            diverged=False, artifacts_complete=False, finite=False,
            run_dir=str(run_dir), note=f"missing artifacts: {missing}", status="incomplete",
        )
    try:
        config = _load_json_object(run_dir / "effective_config.json")
        metrics = _load_json_object(run_dir / "metrics.json")
        _validate_seed_config(config, run_dir, seed, expected_config)
        validation_row = dict(manifest_row) if manifest_row is not None else {
            "run_id": run_dir.name,
            "dataset": str(config.get("dataset", "")),
            "method": str(config.get("variant", "")),
            "seed": str(seed),
            "client_optimizer": str(config.get("client_optimizer", "")),
            "server_buffer_policy": "direct_client_aggregate",
        }
        artifact_validation = validate_artifacts(
            run_dir,
            validation_row,
            expected_config={
                **(expected_config or {}),
                "run_id": run_dir.name,
                "random_seed": seed,
                "comm_round": REQUIRED_COMM_ROUNDS,
                "server_buffer_policy": "direct_client_aggregate",
            },
        )
        if str(metrics.get("run_status")) != "completed":
            raise ValueError("metrics.run_status must be 'completed'")
        if int(metrics.get("rounds_completed")) != REQUIRED_COMM_ROUNDS:
            raise ValueError(
                "metrics.rounds_completed must be "
                f"{REQUIRED_COMM_ROUNDS}, got {metrics.get('rounds_completed')!r}"
            )
        diverged = bool(metrics.get("diverged", False))
        with (run_dir / "mse_by_round.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != REQUIRED_COMM_ROUNDS:
            raise ValueError(
                f"mse_by_round.csv must have {REQUIRED_COMM_ROUNDS} rows, got {len(rows)}"
            )
        curve_diverged = False
        all_values_finite = True
        all_psi_vals = []
        all_mse_vals = []
        for index, row in enumerate(rows):
            if int(row.get("round", "")) != index:
                raise ValueError(
                    f"mse_by_round.csv row {index} has round={row.get('round')!r}"
                )
            finite_text = str(row.get("finite", ""))
            diverged_text = str(row.get("diverged", ""))
            if finite_text not in {"True", "False"}:
                raise ValueError(f"row {index} finite flag is not boolean")
            if diverged_text not in {"True", "False"}:
                raise ValueError(f"row {index} diverged flag is not boolean")
            psi = float(row["gmm_eval"])
            mse = float(row["val_mse"])
            all_psi_vals.append(psi)
            all_mse_vals.append(mse)
            row_values_finite = math.isfinite(psi) and math.isfinite(mse)
            all_values_finite = all_values_finite and row_values_finite
            curve_diverged = curve_diverged or finite_text == "False" or diverged_text == "True"

        tail = rows[-LAST_N_ROUNDS:]
        psi_vals = [float(row["gmm_eval"]) for row in tail]
        mse_vals = [float(row["val_mse"]) for row in tail]
        finite = all_values_finite and _json_numbers_are_finite(metrics)
        terminal = (
            diverged
            or curve_diverged
            or not finite
            or bool(artifact_validation["terminal_ineligible"])
        )
        return SeedResult(
            seed=seed,
            psi_last50_mean=(sum(psi_vals) / len(psi_vals)) if not terminal else None,
            mse_last50_mean=(sum(mse_vals) / len(mse_vals)) if not terminal else None,
            diverged=terminal, artifacts_complete=True, finite=finite,
            best_round_psi=max(
                (value for value in all_psi_vals if math.isfinite(value)), default=None
            ),
            run_dir=str(run_dir),
            note="" if not terminal else "sticky divergence/nonfinite evidence in full curve or metrics",
            status=TERMINAL_STATUS if terminal else COMPLETE_STATUS,
        )
    except Exception as exc:  # noqa: BLE001
        return SeedResult(
            seed=seed, psi_last50_mean=None, mse_last50_mean=None,
            diverged=False, artifacts_complete=True, finite=False,
            run_dir=str(run_dir), note=f"{type(exc).__name__}: {exc}", status="invalid",
        )


def resolve_new_run_dir(dataset: str, method: str, seed: int, lr: float, cm: float) -> Path:
    run_id = f"{NEW_RUN_ID_PREFIX}_{dataset}_{method}_seed{seed}_alpha0p5_lr{token(lr)}_cm{token(cm)}"
    return NEW_RUN_RESULTS_ROOT / dataset / method / f"seed_{seed}" / run_id


def resolve_reused_run_dir(finals_index: dict, dataset: str, method: str, seed: int, lr: float, cm: float) -> Path | None:
    key = (dataset, method, round(lr, 6), round(cm, 6), seed, 0.5)
    row = finals_index.get(key)
    if row is None:
        return None
    d = Path(row["final_result_dir"])
    return d if d.is_absolute() else REPO_ROOT / d


def load_finals_index() -> dict:
    with (FINALS_DIR / "finals_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (r["dataset"], r["method"], round(float(r["learning_rate"]), 6),
         round(float(r["critic_multiplier"]), 6), int(r["seed"]), round(float(r["alpha"]), 6)): r
        for r in rows
    }


def _expected_seed_config(
    dataset: str,
    method: str,
    lr: float,
    cm: float,
) -> dict:
    return {
        "dataset": dataset,
        "variant": method,
        "client_optimizer": METHOD_TO_OPTIMIZER[method],
        "learning_rate": lr,
        "critic_multiplier": cm,
        "server_buffer_policy": "direct_client_aggregate",
    }


def _load_new_run_manifest(summary: dict, cells: str) -> dict[str, dict[str, str]]:
    manifest_path = CAMPAIGN_V2_DIR / f"adjudication_{cells}_manifest.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_run_id: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            raise ValueError(f"{manifest_path.name} contains a blank run_id")
        if run_id in by_run_id:
            raise ValueError(f"{manifest_path.name} contains duplicate run_id {run_id!r}")
        scientific_status = str(row.get("scientific_status", "eligible") or "eligible")
        if scientific_status != "eligible":
            raise ValueError(
                f"{manifest_path.name} run {run_id!r} is not score-eligible: "
                f"scientific_status={scientific_status!r}"
            )
        by_run_id[run_id] = row

    expected: dict[str, tuple[str, str, int, float, float]] = {}
    for cell_plan in summary["plan"]:
        dataset, method = cell_plan["dataset"], cell_plan["method"]
        for candidate in cell_plan["candidates"]:
            if candidate["reused_from_finals"]:
                continue
            lr, cm = float(candidate["lr"]), float(candidate["cm"])
            for seed in (0, 1, 2):
                run_id = resolve_new_run_dir(dataset, method, seed, lr, cm).name
                expected[run_id] = (dataset, method, seed, lr, cm)

    if int(summary.get("new_runs", -1)) != len(expected):
        raise ValueError(
            f"summary new_runs={summary.get('new_runs')!r} but plan requires {len(expected)}"
        )
    if set(by_run_id) != set(expected):
        missing = sorted(set(expected) - set(by_run_id))
        unexpected = sorted(set(by_run_id) - set(expected))
        raise ValueError(
            f"summary/manifest run mismatch; missing={missing}, unexpected={unexpected}"
        )
    for run_id, (dataset, method, seed, lr, cm) in expected.items():
        row = by_run_id[run_id]
        checks = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "client_optimizer": METHOD_TO_OPTIMIZER[method],
            "comm_round": REQUIRED_COMM_ROUNDS,
            "server_buffer_policy": "direct_client_aggregate",
        }
        for field_name, expected_value in checks.items():
            if str(row.get(field_name)) != str(expected_value):
                raise ValueError(
                    f"manifest {run_id} {field_name}={row.get(field_name)!r}; "
                    f"expected {expected_value!r}"
                )
        for field_name, expected_value in (("learning_rate", lr), ("critic_multiplier", cm)):
            if not math.isclose(
                float(row.get(field_name)), float(expected_value), rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError(
                    f"manifest {run_id} {field_name}={row.get(field_name)!r}; "
                    f"expected {expected_value!r}"
                )
    return by_run_id


def build_cell_candidates(
    cell_plan: dict,
    finals_index: dict,
    seeds=(0, 1, 2),
    new_run_manifest: dict[str, dict[str, str]] | None = None,
) -> list[Candidate]:
    dataset, method = cell_plan["dataset"], cell_plan["method"]
    candidates = []
    for cand in cell_plan["candidates"]:
        lr, cm = cand["lr"], cand["cm"]
        candidate_id = f"{dataset}/{method}/lr{lr:g}_cm{cm:g}"
        label = ",".join(cand["labels"])
        seed_results = []
        for seed in seeds:
            manifest_row = None
            if cand["reused_from_finals"]:
                run_dir = resolve_reused_run_dir(finals_index, dataset, method, seed, lr, cm)
                if run_dir is None:
                    seed_results.append(SeedResult(
                        seed=seed, psi_last50_mean=None, mse_last50_mean=None,
                        diverged=False, artifacts_complete=False, finite=False,
                        note="reused_from_finals=True but no matching finals_manifest.csv row found",
                        status="incomplete",
                    ))
                    continue
            else:
                run_dir = resolve_new_run_dir(dataset, method, seed, lr, cm)
                if new_run_manifest is not None and run_dir.name not in new_run_manifest:
                    seed_results.append(SeedResult(
                        seed=seed, psi_last50_mean=None, mse_last50_mean=None,
                        diverged=False, artifacts_complete=False, finite=False,
                        run_dir=str(run_dir), note="planned run is absent from adjudication manifest",
                        status="incomplete",
                    ))
                    continue
                if new_run_manifest is not None:
                    manifest_row = new_run_manifest[run_dir.name]
            seed_results.append(load_seed_result(
                run_dir,
                seed,
                expected_config=_expected_seed_config(dataset, method, lr, cm),
                manifest_row=manifest_row,
            ))
        candidates.append(Candidate(candidate_id=candidate_id, label=label, seeds=tuple(seed_results)))
    return candidates


def seed_result_to_dict(s: SeedResult) -> dict:
    return asdict(s)


def candidate_to_dict(c: Candidate) -> dict:
    return {
        "candidate_id": c.candidate_id, "label": c.label, "eligible": c.eligible,
        "incomplete": c.incomplete,
        "median_psi": c.median_psi if c.eligible else None,
        "median_mse": c.median_mse if c.eligible else None,
        "seeds": [seed_result_to_dict(s) for s in c.seeds],
    }


def score_manifest(cells: str) -> dict:
    with (CAMPAIGN_V2_DIR / f"adjudication_{cells}_summary.json").open() as handle:
        summary = json.load(handle)
    reuses_finals = any(
        candidate["reused_from_finals"]
        for cell_plan in summary["plan"]
        for candidate in cell_plan["candidates"]
    )
    finals_index = load_finals_index() if reuses_finals else {}
    new_run_manifest = _load_new_run_manifest(summary, cells)

    cell_results = {}
    for cell_plan in summary["plan"]:
        dataset, method = cell_plan["dataset"], cell_plan["method"]
        candidates = build_cell_candidates(
            cell_plan, finals_index, new_run_manifest=new_run_manifest
        )
        outcome = score_cell(candidates)
        cell_results[f"{dataset}/{method}"] = {
            "dataset": dataset, "method": method,
            "outcome": outcome.outcome,
            "winner": candidate_to_dict(outcome.winner) if outcome.winner else None,
            "eligible_candidates": [c.candidate_id for c in outcome.eligible],
            "excluded_candidates": [c.candidate_id for c in outcome.excluded],
            "tie_set": [c.candidate_id for c in outcome.tie_set],
            "detail": outcome.detail,
            "all_candidates": [candidate_to_dict(c) for c in candidates],
        }
    return cell_results


def main() -> int:
    global CAMPAIGN_V2_DIR, FINALS_DIR, NEW_RUN_ID_PREFIX, NEW_RUN_RESULTS_ROOT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", choices=["x", "signal"], required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_V2_DIR)
    parser.add_argument("--results-root", type=Path, default=NEW_RUN_RESULTS_ROOT)
    parser.add_argument("--finals-dir", type=Path, default=FINALS_DIR)
    parser.add_argument("--run-id-prefix", default=NEW_RUN_ID_PREFIX)
    args = parser.parse_args()

    CAMPAIGN_V2_DIR = args.campaign_dir
    NEW_RUN_RESULTS_ROOT = args.results_root
    FINALS_DIR = args.finals_dir
    NEW_RUN_ID_PREFIX = args.run_id_prefix

    results = score_manifest(args.cells)
    out_path = Path(args.out) if args.out else CAMPAIGN_V2_DIR / f"adjudication_{args.cells}_results.json"
    incomplete_cells = [
        cell for cell, result in results.items() if result["outcome"] == "incomplete"
    ]
    if incomplete_cells:
        for cell in incomplete_cells:
            print(f"INCOMPLETE {cell}: {results[cell]['detail']}", file=sys.stderr)
        print("No adjudication results were written.", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for cell, r in sorted(results.items()):
        winner = r["winner"]["candidate_id"] if r["winner"] else None
        print(f"{cell:28s} {r['outcome']:22s} winner={winner}  {r['detail']}")
    print(f"\nWritten: {out_path}")
    unresolved_cells = [
        cell
        for cell, result in results.items()
        if result["outcome"] in {"retune_required", "mse_tie_unresolved"}
    ]
    if unresolved_cells:
        print(
            "Stage has no frozen promotion for: " + ", ".join(unresolved_cells),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
