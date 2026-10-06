#!/usr/bin/env python3
"""Analyze the deterministic runtime-profiling diagnostic.

Produces three things:

1. auxiliary-regression cost per scenario (aux-on vs aux-off wall clock);
2. per-round phase breakdown at the protocol-v2 configuration (aux off);
3. a refreshed protocol-v2 GPU-hour projection from measured per-round cost.

The earlier ~1780-1820 GPU-hour projection descends from the nine original
deterministic candidates, which ran with auxiliary regression on *and* the
accidental nested 3x3=9 auxiliary loop. Neither applies to protocol v2, so the
projection has to be rebuilt from measured aux-off cost rather than scaled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/highdim_deterministic_runtime_profile_20260805"
DEFAULT_PROFILE_ROOT = (
    REPO_ROOT / "results/_profiling/highdim_deterministic_runtime_profile_20260805"
)

MEASURED = ("femnist_z", "femnist_x", "cifar10_xz")
METHOD = "fedgda_d"

# Protocol-v2 campaign shape (planning notes S16).
ALPHAS = 3
TUNING_CANDIDATES_PER_CELL = 4      # 2 methods x 2 candidates, seed 0
TUNING_ROUNDS = 150
FINAL_RUNS_PER_CELL = 10            # 2 methods x 5 seeds
FINAL_ROUNDS = 500
ALL_SCENARIOS = ("femnist_z", "femnist_x", "femnist_xz",
                 "cifar10_z", "cifar10_x", "cifar10_xz")

# Cost model: a scenario's per-round cost tracks how many CNNs it carries.
# femnist_z (MLP g + CNN f) and femnist_x (CNN g + MLP f) both measured ~equal,
# confirming the driver is "one CNN + one MLP" vs "two CNNs".
COST_CLASS = {
    "femnist_z": "femnist_one_cnn",
    "femnist_x": "femnist_one_cnn",
    "femnist_xz": "femnist_two_cnn",
    "cifar10_z": "cifar_one_cnn",
    "cifar10_x": "cifar_one_cnn",
    "cifar10_xz": "cifar_two_cnn",
}


def run_dir(root: Path, scenario: str, arm: str) -> Path:
    run_id = f"det_profile_{scenario}_{METHOD}_seed0_alpha0p5_{arm}_r6"
    return root / scenario / METHOD / "seed_0" / run_id


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def setup_and_per_round(profile_root: Path, scenario: str):
    """Return (setup_seconds, per_round_seconds, phase_totals, rounds)."""
    summary = load_json(run_dir(profile_root, scenario, "auxoff") / "profile_summary.json")
    if not summary:
        return None, None, None, None
    totals = summary.get("phase_totals_seconds", {})
    counts = summary.get("phase_counts", {})
    # runner_init already nests model_selection and setup_clients -- do not sum
    # those separately or setup is double counted.
    setup = float(totals.get("data_load", 0.0)) + float(totals.get("runner_init", 0.0))
    rounds = int(counts.get("round_total") or 0)
    per_round = float(totals.get("round_total", 0.0)) / rounds if rounds else None
    return setup, per_round, totals, rounds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--profile-root", default=str(DEFAULT_PROFILE_ROOT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root) / "unprofiled"
    profile_root = Path(args.profile_root)

    report: dict = {}
    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(s)
        print(s)

    # ---------- 1. auxiliary regression cost ----------
    emit("=" * 76)
    emit("1. AUXILIARY REGRESSION COST (unprofiled wall clock, 6 rounds)")
    emit("=" * 76)
    emit(f"  {'scenario':<13}{'aux ON':>10}{'aux OFF':>10}{'saved':>9}"
         f"{'% total':>9}{'% train':>9}")
    emit(f"  {'-'*13}{'-'*10:>10}{'-'*10:>10}{'-'*9:>9}{'-'*9:>9}{'-'*9:>9}")

    aux_rows = {}
    setups = {}
    for scenario in MEASURED:
        on = load_json(run_dir(result_root, scenario, "auxon") / "metrics.json")
        off = load_json(run_dir(result_root, scenario, "auxoff") / "metrics.json")
        if not (on and off):
            emit(f"  {scenario:<13}  (missing)")
            continue
        t_on = float(on["runtime_seconds"])
        t_off = float(off["runtime_seconds"])
        setup, _, _, _ = setup_and_per_round(profile_root, scenario)
        setups[scenario] = setup
        saved = t_on - t_off
        pct_total = 100.0 * saved / t_on
        # Training-only share strips the one-time setup from both arms.
        pct_train = ""
        if setup is not None:
            tr_on, tr_off = t_on - setup, t_off - setup
            if tr_on > 0:
                pct_train = f"{100.0*(tr_on-tr_off)/tr_on:8.1f}%"
        emit(f"  {scenario:<13}{t_on:9.1f}s{t_off:9.1f}s{saved:8.1f}s"
             f"{pct_total:8.1f}%{pct_train:>9}")
        aux_rows[scenario] = {
            "aux_on_seconds": t_on, "aux_off_seconds": t_off,
            "saved_seconds": saved, "pct_of_total": pct_total,
        }
    report["auxiliary_cost"] = aux_rows

    # ---------- 2. per-round breakdown ----------
    emit()
    emit("=" * 76)
    emit("2. PER-ROUND BREAKDOWN AT PROTOCOL-V2 CONFIG (aux off)")
    emit("=" * 76)

    per_round_measured = {}
    breakdown = {}
    for scenario in MEASURED:
        setup, per_round, totals, rounds = setup_and_per_round(profile_root, scenario)
        if per_round is None:
            emit(f"  {scenario:<13} (no profile)")
            continue
        per_round_measured[scenario] = per_round
        rt = float(totals.get("round_total", 0.0))
        client = float(totals.get("client_train_gmm", 0.0))
        local = float(totals.get("trainer_gmm_local_training", 0.0))
        agg = float(totals.get("aggregate_gmm", 0.0))
        ev = float(totals.get("eval_global_model", 0.0))
        emit(f"  {scenario:<13} setup {setup:6.1f}s   round {per_round:6.2f}s")
        emit(f"      client loop      {100*client/rt:5.1f}%   "
             f"of which gradient math {100*local/rt:5.1f}%")
        emit(f"      aggregation      {100*agg/rt:5.1f}%   "
             f"evaluation {100*ev/rt:5.1f}%   everything else "
             f"{100*(rt-client-agg-ev)/rt:5.1f}%")
        breakdown[scenario] = {
            "setup_seconds": setup, "per_round_seconds": per_round,
            "client_loop_pct": 100*client/rt, "gradient_math_pct": 100*local/rt,
            "aggregation_pct": 100*agg/rt, "evaluation_pct": 100*ev/rt,
        }
    report["breakdown"] = breakdown

    # ---------- 3. refreshed projection ----------
    emit()
    emit("=" * 76)
    emit("3. REFRESHED PROTOCOL-V2 PROJECTION")
    emit("=" * 76)

    if not per_round_measured:
        emit("  no measured per-round costs; cannot project")
        return 0

    one_cnn = per_round_measured.get("femnist_x") or per_round_measured.get("femnist_z")
    two_cnn = per_round_measured.get("cifar10_xz")
    fem_setup = setups.get("femnist_x") or 60.0
    cif_setup = setups.get("cifar10_xz") or 170.0

    # Interpolate the three unmeasured scenarios between the measured extremes.
    cost = {
        "femnist_one_cnn": (one_cnn, fem_setup, "measured"),
        "femnist_two_cnn": ((one_cnn + two_cnn) / 2, fem_setup, "interpolated"),
        "cifar_one_cnn": ((one_cnn + two_cnn) / 2, cif_setup, "interpolated"),
        "cifar_two_cnn": (two_cnn, cif_setup, "measured"),
    }

    tuning_h = finals_h = 0.0
    per_scenario = {}
    emit(f"  {'scenario':<13}{'s/round':>9}{'basis':>14}"
         f"{'tuning':>10}{'finals':>10}")
    for scenario in ALL_SCENARIOS:
        pr, setup, basis = cost[COST_CLASS[scenario]]
        t = ALPHAS * TUNING_CANDIDATES_PER_CELL * (setup + pr * TUNING_ROUNDS) / 3600.0
        f = ALPHAS * FINAL_RUNS_PER_CELL * (setup + pr * FINAL_ROUNDS) / 3600.0
        tuning_h += t
        finals_h += f
        per_scenario[scenario] = {
            "per_round_seconds": pr, "basis": basis,
            "tuning_gpu_hours": t, "finals_gpu_hours": f,
        }
        emit(f"  {scenario:<13}{pr:9.2f}{basis:>14}{t:9.1f}h{f:9.1f}h")

    combined = tuning_h + finals_h
    # Bracket using the measured extremes applied to every scenario.
    lo = (ALPHAS * TUNING_CANDIDATES_PER_CELL * len(ALL_SCENARIOS)
          * (fem_setup + one_cnn * TUNING_ROUNDS) / 3600.0
          + ALPHAS * FINAL_RUNS_PER_CELL * len(ALL_SCENARIOS)
          * (fem_setup + one_cnn * FINAL_ROUNDS) / 3600.0)
    hi = (ALPHAS * TUNING_CANDIDATES_PER_CELL * len(ALL_SCENARIOS)
          * (cif_setup + two_cnn * TUNING_ROUNDS) / 3600.0
          + ALPHAS * FINAL_RUNS_PER_CELL * len(ALL_SCENARIOS)
          * (cif_setup + two_cnn * FINAL_ROUNDS) / 3600.0)

    emit()
    emit(f"  tuning  (72 runs @150 rounds) : {tuning_h:8.1f} GPU-h")
    emit(f"  finals (180 runs @500 rounds) : {finals_h:8.1f} GPU-h")
    emit(f"  COMBINED                      : {combined:8.1f} GPU-h"
         f"   ({combined/48:.1f} quota weeks)")
    emit(f"  bracket (all-cheap..all-dear) : {lo:8.1f} .. {hi:.1f} GPU-h"
         f"   ({lo/48:.1f} .. {hi/48:.1f} weeks)")
    emit()
    emit(f"  earlier S17 estimate          : ~1780-1820 GPU-h (~37-38 weeks)")
    emit(f"  ratio                         : {1800/combined:8.1f}x lower")

    report["projection"] = {
        "per_scenario": per_scenario,
        "tuning_gpu_hours": tuning_h,
        "finals_gpu_hours": finals_h,
        "combined_gpu_hours": combined,
        "quota_weeks_at_48": combined / 48.0,
        "bracket_gpu_hours": [lo, hi],
        "earlier_estimate_gpu_hours": [1780, 1820],
    }

    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
