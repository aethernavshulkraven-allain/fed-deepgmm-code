"""g x f initialization crossover on corrected FEMNIST-X.

The init x data crossover established that the collapse follows the INITIAL
MODEL STATE, not the data schedule: both init-31 cells forked at rounds 11-12
and reached 98-100% prediction spread under either data schedule, while both
init-33 cells never forked and stayed at 3-18%, under either. Both diagonals
reproduced their cm30 ladder counterparts bit-for-bit (max |diff| 0.0 over 150
rounds), so that machinery is trusted.

This splits "the initial state" into its two halves, at fixed cm 30 and a
fixed environment (data/sampling seed 31):

              |  f from 31        |  f from 33
    g from 31 |  control (=init31)|  mixed
    g from 33 |  mixed            |  control (=init33)

Each network's COMPLETE state dict transfers, buffers included.
init_g_state_override and init_f_state_override load them independently at the
same post-model-selection boundary; RNG, partition, client sequence, batch
ordering and optimizer-state handling are untouched.

GATE: the two unmixed controls must reproduce cross_init31_data31 and
cross_init33_data31 from the previous campaign before any mixed cell is read.

Measurement is the point, not just another outcome table. checkpoint_rounds
captures EVERY round across 8-15 -- the observed fork -- so the critic/BN
probes can run through it: feature versus head gradients, within-batch critic
variation, clipping, and independent residual moments. An outcome table alone
would say where the sensitivity lives without saying why.

Reading it. Sensitivity follows g -> the structural model's starting point
decides, and the next step holds f and the environment fixed and asks whether
a controlled BN intervention moves both the gradient measurements and the
outcome. Sensitivity follows f -> the critic's starting point decides, which
would connect to the critic-rate effect. Only a particular PAIRING works ->
investigate compatibility rather than blaming one network.

This does not by itself prove BN failure, a bad basin, critic weakness or an
optimizer defect. One environment, one rate, n=1 per cell; the decisive
comparison is repeated under the other environment next, and only then
replicated across further initializations.
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

def default_campaign():
    suffix = "" if data_seed_value() == DEFAULT_DATA_SEED else f"_env{data_seed_value()}"
    return base.ROOT / ("experiments/highdim_coauthor_protocol_v1/"
                        f"gf_init_crossover{suffix}_20260922")
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "gfcross"
CM = 30
DEFAULT_DATA_SEED = 31              # environment; override with --data-seed
CHECKPOINT_ROUNDS = "8 9 10 11 12 13 14 15"
_DATA_SEED = {"value": DEFAULT_DATA_SEED}


def data_seed_value():
    return _DATA_SEED["value"]


CELLS = (("g31_f31", 31, 31), ("g31_f33", 31, 33),
         ("g33_f31", 33, 31), ("g33_f33", 33, 33))
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
    for index, (cell, g_seed, f_seed) in enumerate(CELLS):
        row = base.make_row(campaign, source, STAGE, DATASET, ALPHA, "fed_eg_s", data_seed_value(),
                            {"candidate": index, "learning_rate": G_LR,
                             "server_learning_rate": SERVER_LR}, ROUNDS)
        run_id = f"gfcross_{cell}"
        row.update(stability.POLICY)
        row.update({
            "run_id": run_id, "arm": cell,
            "g_init_seed": g_seed, "f_init_seed": f_seed, "data_seed": data_seed_value(),
            "is_control": g_seed == f_seed,
            "protocol_version": "gf_init_crossover_20260922_v1",
            "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
            "aggregation_weighting": "uniform_clients",
            "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
            "critic_multiplier": CM,
            "eg_predictor_critic_server_lr": CRITIC_SERVER_LR,
            "eg_corrector_critic_server_lr": CRITIC_SERVER_LR,
            "eg_predictor_server_lr": SERVER_LR,
            "eg_corrector_server_lr": SERVER_LR,
            "comm_round": ROUNDS,
            "random_seed": data_seed_value(),
            "scenario_seed": data_seed_value(), "optimizer_seed": data_seed_value(),
            "init_g_state_override": str(initial_state_path(g_seed)),
            "init_f_state_override": str(initial_state_path(f_seed)),
            # every round across the observed fork, for the probes
            "checkpoint_rounds": CHECKPOINT_ROUNDS,
            "periodic_checkpoint_interval": 25,
            "data_cache_dir": str(archive.parent.parent),
            "scenario_name": str(archive.with_suffix("")),
            "scenario_checksum": archive_sha,
            "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                    "fed_eg_s" / f"seed_{data_seed_value()}" / run_id),
            "notes": ("g x f initialization crossover at cm 30, environment 31. "
                      "Unmixed controls must reproduce the init x data crossover."),
        })
        rows.append(row)
    return rows


def assert_arms_are_distinguishable(rows):
    """Resolve every cell rather than trusting key names; require the full 2x2
    with both unmixed controls, and require g and f to come from the sources
    each cell names."""
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        if lrs["predictor_f"] != CRITIC_SERVER_LR or lrs["corrector_f"] != CRITIC_SERVER_LR:
            raise SystemExit(f"{row['run_id']}: critic server rate moved")
        if lrs["predictor_g"] != SERVER_LR or lrs["corrector_g"] != SERVER_LR:
            raise SystemExit(f"{row['run_id']}: structural server rate moved")
        if float(row["critic_multiplier"]) != CM:
            raise SystemExit(f"{row['run_id']}: critic_multiplier is not {CM}")
        if int(row["random_seed"]) != data_seed_value():
            raise SystemExit(f"{row['run_id']}: environment must be seed {data_seed_value()}")
        for key, seed in (("init_g_state_override", row["g_init_seed"]),
                          ("init_f_state_override", row["f_init_seed"])):
            path = Path(row[key])
            if not path.is_file():
                raise SystemExit(f"{row['run_id']}: {key} missing: {path}")
            if f"seed_{seed}" not in str(path):
                raise SystemExit(
                    f"{row['run_id']}: {key} is {path}, not from seed {seed}")
        if not row["checkpoint_rounds"]:
            raise SystemExit(f"{row['run_id']}: dense fork capture is required")
        key = (int(row["g_init_seed"]), int(row["f_init_seed"]))
        if key in seen:
            raise SystemExit(f"{row['run_id']} duplicates {seen[key]}")
        seen[key] = row["run_id"]
    expected = {(g, f) for _, g, f in CELLS}
    if seen.keys() != expected:
        raise SystemExit(f"grid mismatch: missing {sorted(expected - seen.keys())}")
    if sum(1 for g, f in seen if g == f) != 2:
        raise SystemExit("both unmixed controls are required as the gate")


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
        "cells": [{"cell": c, "g_init_seed": g, "f_init_seed": f} for c, g, f in CELLS],
        "data_seed": data_seed_value(), "checkpoint_rounds": CHECKPOINT_ROUNDS,
        "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
        "critic_server_lr": CRITIC_SERVER_LR, "run_count": len(rows),
        "archive": str(archive), "archive_sha256": archive_sha,
        "predecessor": "init_data_crossover_20260922",
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
            "GATE: both unmixed controls must reproduce the init x data crossover "
            "cells init31_data31 and init33_data31. Then: follows g -> hold f and "
            "the environment fixed and test whether a BN intervention moves the "
            "gradient measurements AND the outcome. Follows f -> connects to the "
            "critic-rate effect. Only one pairing works -> compatibility, not blame."
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
    for name in ("run_gf_init_crossover_20260922.py", "run_stochastic_eg_stability.py"):
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
    if int(protocol["data_seed"]) != data_seed_value():
        raise RuntimeError(
            f"--data-seed {data_seed_value()} does not match the frozen campaign's "
            f"{protocol['data_seed']}; refusing to run a different environment than "
            "the one that was prepared")
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
    parser.add_argument("--campaign", type=Path, default=None)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED,
                        help="environment: partition and client-sampling seed")
    args = parser.parse_args()
    _DATA_SEED["value"] = args.data_seed
    campaign = (args.campaign or default_campaign()).resolve()
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
