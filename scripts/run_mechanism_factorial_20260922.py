"""Gated 2x2x2 factorial across four established cases.

Replaces another round of observational checkpoint summaries. The crossovers
identified WHERE the sensitivity lives; more seed tables will not say WHY. This
tests the three standing hypotheses TOGETHER, so an interaction that separate
sweeps would miss is visible.

Factors (8 combinations):

    local BatchNorm       batch (current) | frozen (stored statistics)
    local critic rate     cm 30 | cm 100
    critic-only server    1.5 | 0.3        (structural server rates unchanged)

Cases (4), all with fixed initial states and a fixed client schedule:

    g31/f31 env31   better-learning reference   (R2 +0.102, forks at 11)
    g33/f31 env31   change structural init      (R2 +0.015, never forks)
    g31/f33 env31   works here                  (R2 +0.100, forks at 14)
    g31/f33 env33   SAME networks, fails        (R2 +0.017, never forks)

The last pair is the point: identical starting networks, opposite outcomes
under different environments. Any mechanism claim has to explain that.

Predeclared readings, fixed before the runs:

  * BN frozen improves the poor cases AND restores the missing learning
    signal -> evidence for a normalization-mediated mechanism.
  * Smaller critic server step helps where raising cm does not -> evidence
    pointing at server update dynamics.
  * Raising cm helps at BOTH server rates -> insufficient local critic
    adaptation.
  * Only a combination helps -> an interaction separate sweeps would miss.
  * None helps -> these three do not explain the failure; go to faithful
    upstream reproduction and identification checks rather than widening the
    same sweep.

A failed frozen-BN arm does NOT eliminate every BatchNorm mechanism: freezing
changes the forward map and its Jacobian, and LeakySoftmaxCNN already bypasses
BN for single-sample batches. One improved score does not establish a root
cause either. The stopping rule is all three of: the intervention reproducibly
changes the failure, the corresponding internal measurement changes as
predicted, and it survives a fresh initialization/environment pair.
"""

import argparse
import fcntl
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

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/mechanism_factorial_20260922"
LADDER = base.ROOT / ("experiments/highdim_coauthor_protocol_v1/"
                      "critic_rate_ladder_20260922/results/ladder/femnist_x/fed_eg_s")
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "factorial"
ALPHA = 0.5
ROUNDS = 150
G_LR = 0.003
SERVER_LR = 1.5
CHECKPOINT_ROUNDS = "8 10 12 15"

CASES = (("g31f31_e31", 31, 31, 31), ("g33f31_e31", 33, 31, 31),
         ("g31f33_e31", 31, 33, 31), ("g31f33_e33", 31, 33, 33))
BN_MODES = ("batch", "frozen")
CRITIC_MULTIPLIERS = (30, 100)
CRITIC_SERVER_LRS = (1.5, 0.3)


def initial_state_path(seed):
    path = LADDER / f"seed_{seed}" / f"critladder_cm30_s{seed}" / "checkpoints" / "initial_valid.pt"
    if not path.is_file():
        raise SystemExit(f"missing initial state for seed {seed}: {path}")
    return path


def corrected_archive():
    audit = base.read_json(PAIRED / "data/class_consistent/generation_audit.json")
    record = audit["datasets"][DATASET]
    archive = Path(record["generated"])
    expected = record["generated_sha256"]
    if base.file_sha256(archive) != expected:
        raise RuntimeError(f"Corrected archive no longer matches its audit: {archive}")
    return archive, expected


def make_rows(campaign, source, archive, archive_sha):
    rows, index = [], 0
    for case, g_seed, f_seed, env in CASES:
        for bn in BN_MODES:
            for cm in CRITIC_MULTIPLIERS:
                for csrv in CRITIC_SERVER_LRS:
                    row = base.make_row(campaign, source, STAGE, DATASET, ALPHA,
                                        "fed_eg_s", env,
                                        {"candidate": index, "learning_rate": G_LR,
                                         "server_learning_rate": SERVER_LR}, ROUNDS)
                    index += 1
                    srv_tag = str(csrv).replace(".", "p")
                    run_id = f"fact_{case}_bn{bn}_cm{cm}_srv{srv_tag}"
                    row.update(stability.POLICY)
                    row.update({
                        "run_id": run_id, "arm": case,
                        "case": case, "g_init_seed": g_seed, "f_init_seed": f_seed,
                        "env_seed": env, "bn_mode": bn,
                        "protocol_version": "mechanism_factorial_20260922_v1",
                        "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
                        "aggregation_weighting": "uniform_clients",
                        "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
                        "critic_multiplier": cm,
                        "eg_predictor_critic_server_lr": csrv,
                        "eg_corrector_critic_server_lr": csrv,
                        "eg_predictor_server_lr": SERVER_LR,
                        "eg_corrector_server_lr": SERVER_LR,
                        "local_bn_mode": bn,
                        "comm_round": ROUNDS,
                        "random_seed": env,
                        "scenario_seed": env, "optimizer_seed": env,
                        "init_g_state_override": str(initial_state_path(g_seed)),
                        "init_f_state_override": str(initial_state_path(f_seed)),
                        "checkpoint_rounds": CHECKPOINT_ROUNDS,
                        "periodic_checkpoint_interval": 50,
                        "data_cache_dir": str(archive.parent.parent),
                        "scenario_name": str(archive.with_suffix("")),
                        "scenario_checksum": archive_sha,
                        "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                                "fed_eg_s" / f"seed_{env}" / run_id),
                        "notes": ("2x2x2 mechanism factorial. Diagnostic; frozen BN "
                                  "changes the forward map and is not a proposed fix."),
                    })
                    rows.append(row)
    return rows


def assert_grid_is_complete(rows):
    """Resolve every cell rather than trusting key names, and require the full
    4 x 2 x 2 x 2 with each baseline present for its case."""
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        csrv = float(row["eg_predictor_critic_server_lr"])
        if lrs["predictor_f"] != csrv or lrs["corrector_f"] != csrv:
            raise SystemExit(f"{row['run_id']}: critic server rate did not take effect")
        if lrs["predictor_g"] != SERVER_LR or lrs["corrector_g"] != SERVER_LR:
            raise SystemExit(f"{row['run_id']}: structural server rate moved")
        if row["local_bn_mode"] not in BN_MODES:
            raise SystemExit(f"{row['run_id']}: bad local_bn_mode {row['local_bn_mode']!r}")
        if float(row["critic_multiplier"]) not in CRITIC_MULTIPLIERS:
            raise SystemExit(f"{row['run_id']}: critic_multiplier not on the grid "
                             "-- make_row may have clobbered it")
        if int(row["random_seed"]) != int(row["env_seed"]):
            raise SystemExit(f"{row['run_id']}: environment mismatch")
        for key, seed in (("init_g_state_override", row["g_init_seed"]),
                          ("init_f_state_override", row["f_init_seed"])):
            if f"seed_{seed}" not in str(row[key]):
                raise SystemExit(f"{row['run_id']}: {key} is not from seed {seed}")
        if not row["checkpoint_rounds"]:
            raise SystemExit(f"{row['run_id']}: fork capture is required")
        key = (row["case"], row["local_bn_mode"], int(float(row["critic_multiplier"])), csrv)
        if key in seen:
            raise SystemExit(f"{row['run_id']} duplicates {seen[key]}")
        seen[key] = row["run_id"]
    expected = {(c, b, m, s) for c, _, _, _ in CASES for b in BN_MODES
                for m in CRITIC_MULTIPLIERS for s in CRITIC_SERVER_LRS}
    if seen.keys() != expected:
        raise SystemExit(f"grid mismatch: missing {sorted(expected - seen.keys())}")


def prepare(campaign):
    if (campaign / "freeze.json").exists():
        base.verify_freeze(campaign, full_data=True)
        return
    if any((campaign / n).exists() for n in ("runtime", "configs", "protocol.json")):
        raise FileExistsError(f"Partial preparation preserved: {campaign}")
    archive, archive_sha = corrected_archive()
    sources = base.launcher._load_rows(
        base.ROOT / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_stochastic.csv")
    source = next(r for r in sources if r["dataset"] == DATASET)
    rows = make_rows(campaign, source, archive, archive_sha)
    assert_grid_is_complete(rows)
    protocol = {
        "dataset": DATASET, "alpha": ALPHA, "rounds": ROUNDS, "run_count": len(rows),
        "factors": {"local_bn_mode": list(BN_MODES),
                    "critic_multiplier": list(CRITIC_MULTIPLIERS),
                    "critic_server_lr": list(CRITIC_SERVER_LRS)},
        "cases": [{"case": c, "g_init_seed": g, "f_init_seed": f, "env_seed": e}
                  for c, g, f, e in CASES],
        "archive": str(archive), "archive_sha256": archive_sha,
        "checkpoint_rounds": CHECKPOINT_ROUNDS,
        "question": ("Do local BatchNorm statistics, local critic rate, or the "
                     "critic-only server step -- singly or in combination -- change "
                     "the failure?"),
        "predeclared_readings": {
            "bn_frozen_helps_poor_cases": "normalization-mediated mechanism",
            "smaller_server_step_helps_where_cm_does_not": "server update dynamics",
            "cm_helps_at_both_server_rates": "insufficient local critic adaptation",
            "only_a_combination_helps": "interaction separate sweeps would miss",
            "none_helps": ("these three do not explain it; go to faithful upstream "
                           "reproduction and identification checks"),
        },
        "caveats": ("A failed frozen-BN arm does not eliminate every BatchNorm "
                    "mechanism: freezing changes the forward map and its Jacobian, and "
                    "LeakySoftmaxCNN already bypasses BN for single-sample batches. One "
                    "improved score does not establish a root cause."),
        "stopping_rule": ("Mechanism supported only when all three hold: the "
                          "intervention reproducibly changes the failure, the internal "
                          "measurement changes as predicted, and it survives a fresh "
                          "initialization/environment pair."),
        "baseline_reference": {"g31f31_e31": 0.102, "g33f31_e31": 0.015,
                               "g31f33_e31": 0.100, "g31f33_e33": 0.017},
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
    for name in ("run_mechanism_factorial_20260922.py", "run_stochastic_eg_stability.py"):
        shutil.copy2(base.ROOT / "scripts" / name, runtime / "scripts" / name)
    hashes = [{"path": str(p.relative_to(runtime)), "sha256": base.file_sha256(p)}
              for p in runtime.rglob("*.py") if "__pycache__" not in p.parts]
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
    print(f"Prepared {len(rows)} runs ({len(CASES)} cases x 2x2x2): {campaign}", flush=True)


def run(campaign):
    if base.ROOT != campaign / "runtime":
        raise RuntimeError("Launch the frozen runtime script, not the live source")
    allocated = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not 1 <= len(allocated) <= 2 or len(set(allocated)) != len(allocated):
        raise RuntimeError(f"Require one or two distinct broker GPUs; got {allocated!r}")
    base.verify_freeze(campaign, full_data=True)
    protocol = base.read_json(campaign / "protocol.json")
    rows = base.launcher._load_rows(campaign / f"{STAGE}_manifest.csv")
    assert_grid_is_complete(rows)
    records = base.run_stage(campaign, STAGE, rows, allocated, protocol["watchdog"],
                             resolver=stability.resolve, summarizer=stability.summarize)
    base.save_json(campaign / "status.json", {
        "status": f"{STAGE}_complete_review_required", "stage": STAGE,
        "resolved": len(records), "total": len(rows), "gpus_used": len(allocated),
        "finals_launched": False, "winner": None,
        "reason": "Mechanism factorial; read against protocol.predeclared_readings.",
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
