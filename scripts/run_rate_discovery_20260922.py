"""Discovery grid: client and server rates crossed independently, per player.

Final experiment, discovery stage. Analysed as a FACTORIAL, not a leaderboard.

Why not argmax. Measured from the critic ladder: within-config seed SD is 0.077,
the SE of a 3-seed mean is 0.044, and the entire cm effect (best minus worst
mean) is 0.087. The expected maximum of ~96 noisy means is about +0.12 --
larger than the effect being measured, so an argmax winner is mostly noise. The
same runs are well powered for FACTOR effects: a main effect averages 36 runs,
SE about 0.013. So selection is by main effects and interactions, and the output
is a SHORTLIST (every config within 1 SE of the best mean) for the confirmation
stage to adjudicate on fresh seeds.

Why absolute rates rather than another cm grid. The implementation sets
eta_f = critic_multiplier * eta_g, so changing eta_g at fixed cm moves BOTH
players. Here eta_g and eta_f are chosen independently and cm is emitted as
eta_f / eta_g, which isolates each player:

    eta_g in {0.003, 0.01, 0.03}     eta_f in {0.1, 0.3, 1.0}

eta_f = 0.03 (the current cm 10 at eta_g 0.003) is dropped: already measured as
suboptimal. Both server coefficients are crossed independently -- the structural
server rate has never been varied in this investigation, held at 1.5 in all 32
factorial runs and every crossover cell.

    structural server in {0.3, 1.5}   critic server in {0.3, 1.5}

36 configurations x 2 development seeds = 72 runs at 150 rounds.

Not included here, deliberately. Calibrated-statistics BatchNorm is nested at
the discovered best rate cells in a following stage, not crossed with all 36 --
BN has now failed the evaluation-mode test, the small-batch-necessity test (the
pooled control fails with 256 real samples) and the epsilon intervention
(regime moved 180x and 1700x, game unchanged against a predeclared bar). Nesting
is proportionate; it does mean BN x rate interactions are not measured here, and
that limit should be stated rather than glossed.

Seeds 31 and 32 are DEVELOPMENT blocks. Confirmation runs fresh seeds, includes
the g33/f31 case that failed in 8 of 8 cells as a locked stress test, and
reports selection optimism so the expected discovery-to-confirmation regression
is not misread as a failed campaign.
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

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/rate_discovery_20260922"
LADDER = base.ROOT / ("experiments/highdim_coauthor_protocol_v1/"
                      "critic_rate_ladder_20260922/results/ladder/femnist_x/fed_eg_s")
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "discovery"
ALPHA = 0.5
ROUNDS = 150
G_LR = 0.003
SERVER_LR = 1.5
CHECKPOINT_ROUNDS = "10 12 14 20"

SEEDS = (31, 32)
STRUCTURAL_LRS = (0.003, 0.01, 0.03)
CRITIC_LRS = (0.1, 0.3, 1.0)
STRUCTURAL_SERVER_LRS = (1.5, 0.3)
CRITIC_SERVER_LRS = (1.5, 0.3)


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
    for eta_g in STRUCTURAL_LRS:
        for eta_f in CRITIC_LRS:
            cm = eta_f / eta_g
            for gsrv in STRUCTURAL_SERVER_LRS:
                for fsrv in CRITIC_SERVER_LRS:
                    for seed in SEEDS:
                        row = base.make_row(campaign, source, STAGE, DATASET, ALPHA,
                                            "fed_eg_s", seed,
                                            {"candidate": index, "learning_rate": eta_g,
                                             "server_learning_rate": SERVER_LR}, ROUNDS)
                        index += 1
                        tag = (f"g{str(eta_g).replace('.','p')}"
                               f"_f{str(eta_f).replace('.','p')}"
                               f"_gs{str(gsrv).replace('.','p')}"
                               f"_fs{str(fsrv).replace('.','p')}")
                        run_id = f"disc_{tag}_s{seed}"
                        row.update(stability.POLICY)
                        row.update({
                            "run_id": run_id, "arm": tag,
                            "eta_g": eta_g, "eta_f": eta_f,
                            "structural_server_lr": gsrv, "critic_server_lr": fsrv,
                            "protocol_version": "rate_discovery_20260922_v1",
                            "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
                            "aggregation_weighting": "uniform_clients",
                            "learning_rate": eta_g,
                            # cm is DERIVED so the two players are set independently
                            "critic_multiplier": cm,
                            "server_learning_rate": SERVER_LR,
                            "eg_predictor_server_lr": gsrv,
                            "eg_corrector_server_lr": gsrv,
                            "eg_predictor_critic_server_lr": fsrv,
                            "eg_corrector_critic_server_lr": fsrv,
                            "local_bn_mode": "batch",
                            "comm_round": ROUNDS,
                            "random_seed": seed,
                            "scenario_seed": seed, "optimizer_seed": seed,
                            "checkpoint_rounds": CHECKPOINT_ROUNDS,
                            "periodic_checkpoint_interval": 50,
                            "data_cache_dir": str(archive.parent.parent),
                            "scenario_name": str(archive.with_suffix("")),
                            "scenario_checksum": archive_sha,
                            "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                                    "fed_eg_s" / f"seed_{seed}" / run_id),
                            "notes": ("Rate discovery; eta_g and eta_f independent, cm "
                                      "derived. Factorial analysis, shortlist output, "
                                      "no promotion."),
                        })
                        rows.append(row)
    return rows


def assert_grid_is_complete(rows):
    """Resolve every cell rather than trusting key names. The critical check is
    that eta_f actually lands where intended: cm is derived, and make_row has
    clobbered critic_multiplier before."""
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        gsrv, fsrv = float(row["structural_server_lr"]), float(row["critic_server_lr"])
        if lrs["predictor_g"] != gsrv or lrs["corrector_g"] != gsrv:
            raise SystemExit(f"{row['run_id']}: structural server rate did not take effect")
        if lrs["predictor_f"] != fsrv or lrs["corrector_f"] != fsrv:
            raise SystemExit(f"{row['run_id']}: critic server rate did not take effect")
        eta_g, eta_f = float(row["eta_g"]), float(row["eta_f"])
        if abs(float(row["learning_rate"]) - eta_g) > 1e-12:
            raise SystemExit(f"{row['run_id']}: structural LR is not {eta_g}")
        applied = float(row["critic_multiplier"]) * float(row["learning_rate"])
        if abs(applied - eta_f) > 1e-9:
            raise SystemExit(
                f"{row['run_id']}: cm x eta_g = {applied} but eta_f should be {eta_f} "
                "-- critic_multiplier may have been clobbered")
        if row["local_bn_mode"] != "batch":
            raise SystemExit(f"{row['run_id']}: discovery is ordinary BN only")
        if not row["checkpoint_rounds"]:
            raise SystemExit(f"{row['run_id']}: fork capture is required")
        key = (eta_g, eta_f, gsrv, fsrv, int(row["seed"]))
        if key in seen:
            raise SystemExit(f"{row['run_id']} duplicates {seen[key]}")
        seen[key] = row["run_id"]
    expected = {(g, f, gs, fs, s) for g in STRUCTURAL_LRS for f in CRITIC_LRS
                for gs in STRUCTURAL_SERVER_LRS for fs in CRITIC_SERVER_LRS for s in SEEDS}
    if seen.keys() != expected:
        raise SystemExit(f"grid mismatch: missing {sorted(expected - seen.keys())[:4]}")


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
        "factors": {"eta_g": list(STRUCTURAL_LRS), "eta_f": list(CRITIC_LRS),
                    "structural_server_lr": list(STRUCTURAL_SERVER_LRS),
                    "critic_server_lr": list(CRITIC_SERVER_LRS)},
        "seeds": list(SEEDS), "local_bn_mode": "batch (calibrated BN nested later)",
        
        "archive": str(archive), "archive_sha256": archive_sha,
        "checkpoint_rounds": CHECKPOINT_ROUNDS,
        "question": ("Do the structural and critic rates, at client and server, singly "
                     "or in interaction, change the failure?"),
        "predeclared_readings": {
            "eta_g helps at fixed eta_f": "structural local-step effect",
            "eta_f helps at fixed eta_g": "critic local-step effect",
            "local gains depend on server coefficients": "client/server interaction",
            "improvement only at a grid boundary": "range unresolved, NOT a located optimum",
            "no factor effect exceeds 2 SE": "rates do not explain the failure",
        },
        "selection_rule": ("Factorial main effects and interactions, NOT argmax. "
                           "Output is a shortlist: every config within 1 SE of the "
                           "best mean. Measured selection optimism for ~36 configs at "
                           "SE 0.054 is on the order of the effect itself."),
        "horizon_guard": ("Good cells peak at mean round 82, max 118 of 150. Any "
                          "shortlisted config peaking after round 120 is re-run at 500 "
                          "rounds before confirmation."),
        "caveats": ("BN x rate interactions are NOT measured here; calibrated BN is "
                    "nested at the discovered best cells in a following stage. Seeds 31 "
                    "and 32 are development blocks and both are responsive seeds, so "
                    "this grid does not test whether a configuration rescues the hard "
                    "g33 case -- that is a locked confirmation stress test."),
        "stopping_rule": ("Mechanism supported only when all three hold: the "
                          "intervention reproducibly changes the failure, the internal "
                          "measurement changes as predicted, and it survives a fresh "
                          "initialization/environment pair."),
        "baseline_reference": {"eta_g 0.003, eta_f 0.3, both servers 1.5": 0.120},
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
    for name in ("run_rate_discovery_20260922.py", "run_stochastic_eg_stability.py"):
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
    print(f"Prepared {len(rows)} runs (36 configs x {len(SEEDS)} seeds): {campaign}", flush=True)


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
        "reason": "Rate discovery; analyse as a factorial and emit a shortlist.",
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
