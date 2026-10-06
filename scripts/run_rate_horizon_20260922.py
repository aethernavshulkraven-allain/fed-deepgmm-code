"""Horizon extension for the discovery shortlist: 500 rounds, same seeds.

The discovery grid ran 150 rounds and the predeclared horizon guard fired:
three of the four shortlisted configurations took their best validation round
at 106, 118 and 133 of 150, i.e. they were still improving when the budget
ran out. Their reported values understate them and the ranking among them is
not trustworthy.

Shortlist from the balanced 72-run factorial (within 1 SE = 0.039 of the best
mean +0.191; within-config seed SD 0.055):

    eta_g  eta_f  g_srv  f_srv   mean R2   seeds            max peak
    0.01   0.3    1.5    1.5     +0.191    +0.166 / +0.215        14
    0.003  1.0    1.5    0.3     +0.186    +0.177 / +0.195       133
    0.003  0.3    1.5    0.3     +0.169    +0.112 / +0.226       106
    0.003  0.3    1.5    1.5     +0.166    +0.140 / +0.192       118

ONLY the horizon changes. Same development seeds 31 and 32, same corrected
archive, same paper-aligned objective, same rates. Introducing fresh seeds here
would confound horizon truncation with seed variation; fresh seeds belong to
confirmation, which is a separate stage.

Read it as: a config whose value rises materially at 500 rounds was truncated
at 150 and its discovery number was an underestimate. A config flat between 150
and 500 was already converged. The all-important case is the one that peaked at
round 14 -- if it stays flat while the late peakers rise, the discovery ranking
inverts and the shortlist was ordered by truncation rather than by quality.

Still development, not confirmation. No promotion, no selection on test.
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

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/rate_horizon_20260922"
LADDER = base.ROOT / ("experiments/highdim_coauthor_protocol_v1/"
                      "critic_rate_ladder_20260922/results/ladder/femnist_x/fed_eg_s")
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "horizon"
ALPHA = 0.5
ROUNDS = 500
G_LR = 0.003
SERVER_LR = 1.5
CHECKPOINT_ROUNDS = "10 12 14 20"

SEEDS = (31, 32)
# (eta_g, eta_f, structural server, critic server) -- the discovery shortlist
SHORTLIST = ((0.01,  0.3, 1.5, 1.5),
             (0.003, 1.0, 1.5, 0.3),
             (0.003, 0.3, 1.5, 0.3),
             (0.003, 0.3, 1.5, 1.5))
DISCOVERY_MEANS = {(0.01, 0.3, 1.5, 1.5): 0.191, (0.003, 1.0, 1.5, 0.3): 0.186,
                   (0.003, 0.3, 1.5, 0.3): 0.169, (0.003, 0.3, 1.5, 1.5): 0.166}
DISCOVERY_MAX_PEAK = {(0.01, 0.3, 1.5, 1.5): 14, (0.003, 1.0, 1.5, 0.3): 133,
                      (0.003, 0.3, 1.5, 0.3): 106, (0.003, 0.3, 1.5, 1.5): 118}


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
    for eta_g, eta_f, gsrv, fsrv in SHORTLIST:
        cm = eta_f / eta_g
        for seed in SEEDS:
            row = base.make_row(campaign, source, STAGE, DATASET, ALPHA, "fed_eg_s", seed,
                                {"candidate": index, "learning_rate": eta_g,
                                 "server_learning_rate": SERVER_LR}, ROUNDS)
            index += 1
            tag = (f"g{str(eta_g).replace('.','p')}_f{str(eta_f).replace('.','p')}"
                   f"_gs{str(gsrv).replace('.','p')}_fs{str(fsrv).replace('.','p')}")
            run_id = f"hz_{tag}_s{seed}"
            row.update(stability.POLICY)
            row.update({
                "run_id": run_id, "arm": tag,
                "eta_g": eta_g, "eta_f": eta_f,
                "structural_server_lr": gsrv, "critic_server_lr": fsrv,
                "discovery_mean_r2_150": DISCOVERY_MEANS[(eta_g, eta_f, gsrv, fsrv)],
                "discovery_max_peak_150": DISCOVERY_MAX_PEAK[(eta_g, eta_f, gsrv, fsrv)],
                "protocol_version": "rate_horizon_20260922_v1",
                "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
                "aggregation_weighting": "uniform_clients",
                "learning_rate": eta_g, "critic_multiplier": cm,
                "server_learning_rate": SERVER_LR,
                "eg_predictor_server_lr": gsrv, "eg_corrector_server_lr": gsrv,
                "eg_predictor_critic_server_lr": fsrv,
                "eg_corrector_critic_server_lr": fsrv,
                "local_bn_mode": "batch",
                "comm_round": ROUNDS,
                "random_seed": seed, "scenario_seed": seed, "optimizer_seed": seed,
                "checkpoint_rounds": CHECKPOINT_ROUNDS,
                "periodic_checkpoint_interval": 100,
                "data_cache_dir": str(archive.parent.parent),
                "scenario_name": str(archive.with_suffix("")),
                "scenario_checksum": archive_sha,
                "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                        "fed_eg_s" / f"seed_{seed}" / run_id),
                "notes": ("Horizon extension of the discovery shortlist; ONLY the "
                          "round budget differs from discovery. No promotion."),
            })
            rows.append(row)
    return rows


def assert_grid_is_complete(rows):
    """Only the horizon may differ from discovery. Verify the rates land where
    intended -- cm is derived and make_row has clobbered it before -- and that
    the seeds are the SAME development seeds, not fresh ones."""
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        gsrv, fsrv = float(row["structural_server_lr"]), float(row["critic_server_lr"])
        if lrs["predictor_g"] != gsrv or lrs["corrector_g"] != gsrv:
            raise SystemExit(f"{row['run_id']}: structural server rate did not take effect")
        if lrs["predictor_f"] != fsrv or lrs["corrector_f"] != fsrv:
            raise SystemExit(f"{row['run_id']}: critic server rate did not take effect")
        eta_g, eta_f = float(row["eta_g"]), float(row["eta_f"])
        applied = float(row["critic_multiplier"]) * float(row["learning_rate"])
        if abs(applied - eta_f) > 1e-9:
            raise SystemExit(
                f"{row['run_id']}: cm x eta_g = {applied} but eta_f should be {eta_f}")
        if int(row["comm_round"]) != ROUNDS:
            raise SystemExit(f"{row['run_id']}: horizon must be {ROUNDS} rounds")
        if int(row["seed"]) not in SEEDS:
            raise SystemExit(
                f"{row['run_id']}: seed {row['seed']} is not a development seed; "
                "fresh seeds belong to confirmation and would confound horizon "
                "with seed variation")
        if row["local_bn_mode"] != "batch":
            raise SystemExit(f"{row['run_id']}: ordinary BN only")
        key = (eta_g, eta_f, gsrv, fsrv, int(row["seed"]))
        if key in seen:
            raise SystemExit(f"{row['run_id']} duplicates {seen[key]}")
        seen[key] = row["run_id"]
    expected = {(g, f, gs, fs, s) for g, f, gs, fs in SHORTLIST for s in SEEDS}
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
        "shortlist": [{"eta_g": g, "eta_f": f, "structural_server_lr": gs,
                       "critic_server_lr": fs,
                       "discovery_mean_r2_150": DISCOVERY_MEANS[(g, f, gs, fs)],
                       "discovery_max_peak_150": DISCOVERY_MAX_PEAK[(g, f, gs, fs)]}
                      for g, f, gs, fs in SHORTLIST],
        "seeds": list(SEEDS), "local_bn_mode": "batch",
        "seeds": list(SEEDS), "local_bn_mode": "batch (calibrated BN nested later)",
        
        "archive": str(archive), "archive_sha256": archive_sha,
        "checkpoint_rounds": CHECKPOINT_ROUNDS,
        "question": ("Were the shortlisted configurations truncated at 150 rounds, and "
                     "does the ranking survive a 500-round budget?"),
        "predeclared_readings": {
            "late peakers rise, the round-14 config stays flat":
                "the discovery ranking was ordered by truncation, not quality",
            "all four rise together": "150 rounds understated every config equally",
            "all four flat": "150 rounds was adequate; the discovery ranking stands",
            "a config still peaking near 500": "range STILL unresolved, not an optimum",
        },
        "selection_rule": ("Development stage. Re-rank the shortlist; do NOT select a "
                           "winner here. Confirmation runs fresh seeds."),
        "selection_rule": ("Factorial main effects and interactions, NOT argmax. "
                           "Output is a shortlist: every config within 1 SE of the "
                           "best mean. Measured selection optimism for ~36 configs at "
                           "SE 0.054 is on the order of the effect itself."),
        "horizon_guard": ("Good cells peak at mean round 82, max 118 of 150. Any "
                          "shortlisted config peaking after round 120 is re-run at 500 "
                          "rounds before confirmation."),
        "caveats": ("Same development seeds as discovery, by design: fresh seeds here "
                    "would confound horizon with seed variation. Two seeds only, so "
                    "the re-ranking is indicative and confirmation still adjudicates. "
                    "Every shortlisted config sits at structural server 1.5, the "
                    "boundary of the tested range, which remains unresolved."),
        "stopping_rule": ("Mechanism supported only when all three hold: the "
                          "intervention reproducibly changes the failure, the internal "
                          "measurement changes as predicted, and it survives a fresh "
                          "initialization/environment pair."),
        "baseline_reference": {
            "discovery_at_150_rounds": {
                f"eta_g{g}_eta_f{f}_gs{gs}_fs{fs}": v
                for (g, f, gs, fs), v in DISCOVERY_MEANS.items()}},
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
    for name in ("run_rate_horizon_20260922.py", "run_stochastic_eg_stability.py"):
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
    print(f"Prepared {len(rows)} runs ({len(SHORTLIST)} configs x {len(SEEDS)} seeds at {ROUNDS} rounds): {campaign}", flush=True)


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
        "reason": "Horizon extension; re-rank the shortlist, do not select a winner.",
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
