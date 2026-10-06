"""Initialization x data-schedule crossover on corrected FEMNIST-X.

Localizes whether a seed's failure follows its INITIAL MODEL STATE or the rest
of its execution environment (partition, client sequence, batch ordering).

Why it is needed. Changing the seed changes all three at once: fedml.init()
seeds random, numpy and torch from a single args.random_seed, which then drives
model initialization, the Dirichlet partition and client sampling together. So
"seed 33 fails" is currently unattributable. (scenario_seed and optimizer_seed
are recorded in every artifact but consumed nowhere -- a separate provenance
issue, not a fix for this.)

What the early-round analysis already established, replicated across all four
critic rates in critic_rate_ladder_20260922:

  * rounds 0-10 are indistinguishable across seeds -- gentle, near-identical
    descent;
  * at round ~10-11 s31 and s32 begin descending in earnest and reach their
    best at rounds 83-118;
  * s33 turns UP at round 11 and never returns; it never reaches even 5% below
    its own round-0 value at ANY critic rate, while s31 and s32 cross that
    threshold at rounds 10-14.

And at initialization and round 0 the three seeds are NOT distinguishable by
the critic/BN probes -- s33 in fact starts with the least degenerate critic
(|mean|/sd 0.80 vs 1.03 and 1.66). So the fork is a training-dynamics event
around round 11, not a visibly bad starting point.

The design, at a fixed cm = 30 chosen in advance:

                          | data/sampling 31 | data/sampling 33
    initial state from 31 |  diagonal control |    crossover
    initial state from 33 |     crossover     | diagonal control

init_state_override loads the saved post-model-selection initial g/f state,
including BatchNorm buffers, at exactly the boundary where training begins --
after model selection, since the models are reinitialized there and overriding
their first construction would miss the state training actually starts from.
Nothing else is touched: no reseeding, no RNG consumption, no change to
partition, client sequence or batch ordering.

CRITICAL GATE: both diagonals must reproduce their corresponding existing
cm30 ladder runs. Until they do, the off-diagonal arms mean nothing.

Scope. This asks whether failure follows the initial model state or the rest of
the environment. It does NOT separate partitioning from participation order --
that is the next split if the environment side wins. One dataset, one rate,
150 rounds. No promotion.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stochastic_eg_campaign as base                      # noqa: E402
import run_stochastic_eg_stability as stability                # noqa: E402

sys.path.insert(0, str(base.ROOT / "fedgmm/sp_decentralized_mnist_lr_example"))
from experiment_utils import resolve_server_learning_rates      # noqa: E402

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/init_data_crossover_20260922"
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "crossover"
CM = 30
CELLS = (("init31_data31", 31, 31), ("init31_data33", 31, 33),
         ("init33_data31", 33, 31), ("init33_data33", 33, 33))
LADDER = base.ROOT / ("experiments/highdim_coauthor_protocol_v1/critic_rate_ladder_20260922/results/ladder/femnist_x/fed_eg_s")
CRITIC_SERVER_LR = 1.5
ALPHA = 0.5
ROUNDS = 150
G_LR = 0.003
SERVER_LR = 1.5




def corrected_archive():
    audit = base.read_json(PAIRED / "data/class_consistent/generation_audit.json")
    record = audit["datasets"][DATASET]
    archive = Path(record["generated"])
    expected = record["generated_sha256"]
    if base.file_sha256(archive) != expected:
        raise RuntimeError(f"Corrected archive no longer matches its audit: {archive}")
    return archive, expected


def initial_state_path(seed):
    path = LADDER / f"seed_{seed}" / f"critladder_cm{CM}_s{seed}" / "checkpoints" / "initial_valid.pt"
    if not path.is_file():
        raise SystemExit(f"missing initial state for seed {seed}: {path}")
    return path


def make_rows(campaign, source, archive, archive_sha):
    rows = []
    for index, (cell, init_seed, data_seed) in enumerate(CELLS):
        init_path = initial_state_path(init_seed)
        row = base.make_row(campaign, source, STAGE, DATASET, ALPHA, "fed_eg_s", data_seed,
                            {"candidate": index, "learning_rate": G_LR,
                             "server_learning_rate": SERVER_LR}, ROUNDS)
        run_id = f"cross_{cell}"
        row.update(stability.POLICY)
        row.update({
            "run_id": run_id, "arm": cell,
            "init_seed": init_seed, "data_seed": data_seed,
            "is_diagonal": init_seed == data_seed,
            "protocol_version": "init_data_crossover_20260922_v1",
            "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
            "aggregation_weighting": "uniform_clients",
            "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
            "critic_multiplier": CM,
            "eg_predictor_critic_server_lr": CRITIC_SERVER_LR,
            "eg_corrector_critic_server_lr": CRITIC_SERVER_LR,
            "eg_predictor_server_lr": SERVER_LR,
            "eg_corrector_server_lr": SERVER_LR,
            "comm_round": ROUNDS,
            # random_seed IS the data/sampling trajectory: it drives the
            # partition and the client sequence. The initial model state it
            # would also have produced is then replaced by the override.
            "random_seed": data_seed,
            "scenario_seed": data_seed, "optimizer_seed": init_seed,
            "init_state_override": str(init_path),
            # capture the round ~11 fork; the per-round CSV is dense for free,
            # these are for the critic/BN probes
            "periodic_checkpoint_interval": 5,
            "data_cache_dir": str(archive.parent.parent),
            "scenario_name": str(archive.with_suffix("")),
            "scenario_checksum": archive_sha,
            "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                    "fed_eg_s" / f"seed_{data_seed}" / run_id),
            "notes": ("Initialization x data-schedule crossover at cm 30. "
                      "Diagonals must reproduce the cm30 ladder runs."),
        })
        rows.append(row)
    return rows


def assert_arms_are_distinguishable(rows):
    """Resolve every cell rather than trusting key names, and require the full
    2x2 with both diagonals present -- the diagonals are the gate."""
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        if lrs["predictor_f"] != CRITIC_SERVER_LR or lrs["corrector_f"] != CRITIC_SERVER_LR:
            raise SystemExit(f"{row['run_id']}: critic server rate moved")
        if lrs["predictor_g"] != SERVER_LR or lrs["corrector_g"] != SERVER_LR:
            raise SystemExit(f"{row['run_id']}: structural server rate moved")
        if float(row["critic_multiplier"]) != CM:
            raise SystemExit(f"{row['run_id']}: critic_multiplier is not {CM}")
        override = Path(row["init_state_override"])
        if not override.is_file():
            raise SystemExit(f"{row['run_id']}: init_state_override missing: {override}")
        if f"_s{row['init_seed']}" not in override.name and \
                f"s{row['init_seed']}" not in str(override):
            raise SystemExit(
                f"{row['run_id']}: override {override} does not come from seed "
                f"{row['init_seed']}")
        if int(row["random_seed"]) != int(row["data_seed"]):
            raise SystemExit(f"{row['run_id']}: random_seed must be the data seed")
        key = (int(row["init_seed"]), int(row["data_seed"]))
        if key in seen:
            raise SystemExit(f"{row['run_id']} duplicates {seen[key]}")
        seen[key] = row["run_id"]
    expected = {(i, d) for _, i, d in CELLS}
    if seen.keys() != expected:
        raise SystemExit(f"grid mismatch: missing {sorted(expected - seen.keys())}")
    if sum(1 for i, d in seen if i == d) != 2:
        raise SystemExit("both diagonal controls are required as the gate")


def prepare(campaign):
    if (campaign / "freeze.json").exists():
        base.verify_freeze(campaign, full_data=True)
        return
    if any((campaign / name).exists() for name in ("runtime", "configs", "protocol.json")):
        raise FileExistsError(f"Partial preparation preserved: {campaign}")
    archive, archive_sha = corrected_archive()
    sources = base.launcher._load_rows(
        base.ROOT / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_stochastic.csv")
    source = next(row for row in sources if row["dataset"] == DATASET)
    rows = make_rows(campaign, source, archive, archive_sha)
    assert_arms_are_distinguishable(rows)

    protocol = {
        "dataset": DATASET, "alpha": ALPHA, "rounds": ROUNDS, "critic_multiplier": CM,
        "cells": [{"cell": c, "init_seed": i, "data_seed": d} for c, i, d in CELLS],
        "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
        "critic_server_lr": CRITIC_SERVER_LR, "run_count": len(rows),
        "archive": str(archive), "archive_sha256": archive_sha,
        "predecessor": "critic_rate_ladder_20260922",
        "early_divergence": "rounds 0-10 indistinguishable; fork at round ~11; s33 never reaches 5% below its round-0 value at any critic rate",
        "question": ("Does the failure follow the initial model state or the rest of the "
                     "execution environment (partition, client sequence, batch order)?"),
        "held_fixed": ["structural local and server rates", "initialization",
                       "warm-up", "partitions", "participation", "batches",
                       "reference schedule", "round budget"],
        "motivation": {
            "source": "critic_bn_diagnostic_20260922/diagnostics.json",
            "critic_grad_over_structural_grad": "0.10-0.52 at every selected checkpoint",
            "independent_basis_worst_bin_t": "2.35-5.34 in every checkpoint",
        },
        "decision_rule": (
            "GATE: both diagonals must reproduce their cm30 ladder counterparts; "
            "until then the off-diagonals mean nothing. Then: failure follows the "
            "initial state -> split g from f initialization next. Failure follows "
            "the environment -> split partitioning from participation order. Neither "
            "cleanly -> the fork is an interaction and needs the round-11 probes."
        ),
        "scope": "One dataset, one alpha, three seeds, 150 rounds. Screen, not a finals matrix.",
        "automatic_promotion": False,
        "selection_metric_source": "validation", "test_mse_used_for_selection": False,
        "client_execution_mode": "sp",
        "watchdog": {"startup_seconds": 1800, "stall_seconds": 900, "total_seconds": 14400},
        "deterioration_policy": stability.POLICY,
        "source": source,
    }
    base.save_json(campaign / "protocol.json", protocol, immutable=True)
    base.write_manifest(campaign / f"{STAGE}_manifest.csv", rows)
    base.snapshot(campaign)
    runtime = campaign / "runtime"
    for name in ("run_init_data_crossover_20260922.py", "run_stochastic_eg_stability.py"):
        shutil.copy2(base.ROOT / "scripts" / name, runtime / "scripts" / name)
    hashes = [{"path": str(path.relative_to(runtime)), "sha256": base.file_sha256(path)}
              for path in runtime.rglob("*.py") if "__pycache__" not in path.parts]
    base.save_json(runtime / "training_hashes.json", hashes, immutable=True)
    configs = []
    for job in base.make_jobs(campaign, rows):
        base.launcher.write_config(job.config_path, job.config)
        configs.append(job.config_path)
    files = [p for p in runtime.rglob("*") if p.is_file() and not p.is_symlink()
             and "__pycache__" not in p.parts]
    files += configs + [campaign / "protocol.json", campaign / f"{STAGE}_manifest.csv"]
    base.save_json(campaign / "freeze.json", {
        "files": {str(p.relative_to(campaign)): base.file_sha256(p) for p in files},
        "data": {str(archive): {"sha256": archive_sha, "size": archive.stat().st_size,
                                "mtime_ns": archive.stat().st_mtime_ns}},
    }, immutable=True)
    print(f"Prepared {len(rows)} arms on corrected {DATASET}: {campaign}", flush=True)


def run(campaign):
    if base.ROOT != campaign / "runtime":
        raise RuntimeError("Launch the frozen runtime script, not the live source")
    allocated = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not 1 <= len(allocated) <= 2 or len(set(allocated)) != len(allocated):
        raise RuntimeError(f"Require one or two distinct broker GPUs; got {allocated!r}")
    base.verify_freeze(campaign, full_data=True)
    protocol = base.read_json(campaign / "protocol.json")
    rows = base.launcher._load_rows(campaign / f"{STAGE}_manifest.csv")
    assert_arms_are_distinguishable(rows)
    records = base.run_stage(campaign, STAGE, rows, allocated, protocol["watchdog"],
                             resolver=stability.resolve, summarizer=stability.summarize)
    base.save_json(campaign / "status.json", {
        "status": f"{STAGE}_complete_review_required", "stage": STAGE,
        "resolved": len(records), "total": len(rows), "gpus_used": len(allocated),
        "finals_launched": False, "winner": None,
        "reason": "Crossover; diagonals must reproduce the cm30 ladder before reading off-diagonals.",
        "time": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "prepare":
        prepare(campaign)
    elif args.action == "verify":
        base.verify_freeze(campaign, full_data=True)
        print("Frozen campaign verified")
    else:
        with (campaign / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(campaign)
            except Exception as error:
                base.save_json(campaign / "status.json", {
                    "status": "paused", "stage": STAGE,
                    "reason": str(error), "time": time.time()})
                raise
