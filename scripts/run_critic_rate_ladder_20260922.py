"""Critic-rate ladder and seed replication on corrected FEMNIST-X.

Follows the four-arm screen (critic_rate_screen_20260922), which found that
raising the LOCAL critic rate moved mechanism and outcome together while the
critic-only SERVER step did not:

  arm              cm  critic srv   R2     corr  spread  peak    const/actual  worst t
  a_control        10     1.5     +0.058  0.241    22%   13/150      26%        3.40
  b_local_critic   30     1.5     +0.102  0.547    98%   83/150      13%        1.98
  c_server_critic  10     0.3     +0.072  0.275    31%   14/150      56%        3.26
  d_both           30     0.3     +0.064  0.295    44%   17/150      16%        3.56

Arm C made the critic MORE degenerate, so the server dimension is dropped here
and the critic server coefficient is held at 1.5 in every cell.

Two questions, one grid, so the ladder and the replication share a freeze:

  Ladder      does the trend continue as cm rises, or turn over? cm in
              {10, 30, 100, 300}. A turnover would bound the mechanism; a
              monotone trend toward the ceiling would strengthen it.
  Replication is arm B's gain reproducible? seeds {31, 32, 33} at every cm,
              so control and treatment are replicated together rather than
              comparing a replicated treatment against a single-seed control.

Everything else is held fixed and identical to the screen: g LR 0.003, server
rates 1.5, paper-aligned objective, uniform-client weighting, corrected
class-consistent archive pinned by checksum, 150 rounds, alpha 0.5.

Still a screen. R2 +0.102 against a ceiling of 1.0 is a poor estimate, and a
root-cause claim needs a reversal or ablation beyond this. No promotion.
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

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/critic_rate_ladder_20260922"
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "ladder"
SEEDS = (31, 32, 33)
CRITIC_SERVER_LR = 1.5
ALPHA = 0.5
ROUNDS = 150
G_LR = 0.003
SERVER_LR = 1.5

CRITIC_MULTIPLIERS = (10, 30, 100, 300)


def corrected_archive():
    audit = base.read_json(PAIRED / "data/class_consistent/generation_audit.json")
    record = audit["datasets"][DATASET]
    archive = Path(record["generated"])
    expected = record["generated_sha256"]
    if base.file_sha256(archive) != expected:
        raise RuntimeError(f"Corrected archive no longer matches its audit: {archive}")
    return archive, expected


def make_rows(campaign, source, archive, archive_sha):
    rows = []
    for index, cm in enumerate(CRITIC_MULTIPLIERS):
        for seed in SEEDS:
            row = base.make_row(campaign, source, STAGE, DATASET, ALPHA, "fed_eg_s", seed,
                                {"candidate": index, "learning_rate": G_LR,
                                 "server_learning_rate": SERVER_LR}, ROUNDS)
            arm = f"cm{cm}"
            run_id = f"critladder_{arm}_s{seed}"
            row.update(stability.POLICY)
            row.update({
                "run_id": run_id, "arm": arm,
                "protocol_version": "critic_rate_ladder_20260922_v1",
                "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
                "aggregation_weighting": "uniform_clients",
                "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
                "critic_multiplier": cm,
                "eg_predictor_critic_server_lr": CRITIC_SERVER_LR,
                "eg_corrector_critic_server_lr": CRITIC_SERVER_LR,
                "eg_predictor_server_lr": SERVER_LR,
                "eg_corrector_server_lr": SERVER_LR,
                "comm_round": ROUNDS,
                "scenario_seed": seed, "optimizer_seed": seed,
                "data_cache_dir": str(archive.parent.parent),
                "scenario_name": str(archive.with_suffix("")),
                "scenario_checksum": archive_sha,
                "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                        "fed_eg_s" / f"seed_{seed}" / run_id),
                "notes": ("Critic-rate ladder and seed replication; only "
                          "critic_multiplier varies. Diagnostic, no promotion."),
            })
            rows.append(row)
    return rows


def assert_arms_are_distinguishable(rows):
    """A mistyped critic key is accepted silently by fedml/arguments.py and
    never read, producing a clean-looking run that tests nothing. Resolve each
    cell rather than trusting key names."""
    expected = {(cm, seed) for cm in CRITIC_MULTIPLIERS for seed in SEEDS}
    seen = set()
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        cm = float(row["critic_multiplier"])
        if lrs["predictor_f"] != CRITIC_SERVER_LR or lrs["corrector_f"] != CRITIC_SERVER_LR:
            raise SystemExit(f"{row['run_id']}: critic server rate is not {CRITIC_SERVER_LR}")
        if lrs["predictor_g"] != SERVER_LR or lrs["corrector_g"] != SERVER_LR:
            raise SystemExit(f"{row['run_id']}: structural server rate moved")
        if cm not in CRITIC_MULTIPLIERS:
            raise SystemExit(f"{row['run_id']}: critic_multiplier {cm} is not on the ladder "
                             "-- make_row may have clobbered it")
        key = (int(cm), int(row["seed"]))
        if key in seen:
            raise SystemExit(f"{row['run_id']}: duplicate cell {key}")
        seen.add(key)
    if seen != expected:
        raise SystemExit(f"grid mismatch: missing {sorted(expected - seen)}")


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
        "dataset": DATASET, "seeds": list(SEEDS), "alpha": ALPHA, "rounds": ROUNDS,
        "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
        "critic_multipliers": list(CRITIC_MULTIPLIERS),
        "critic_server_lr": CRITIC_SERVER_LR, "run_count": len(rows),
        "archive": str(archive), "archive_sha256": archive_sha,
        "predecessor": "critic_rate_screen_20260922",
        "question": ("Does the local-critic gain continue as cm rises or turn over, "
                     "and does it reproduce across seeds?"),
        "held_fixed": ["structural local and server rates", "initialization",
                       "warm-up", "partitions", "participation", "batches",
                       "reference schedule", "round budget"],
        "motivation": {
            "source": "critic_bn_diagnostic_20260922/diagnostics.json",
            "critic_grad_over_structural_grad": "0.10-0.52 at every selected checkpoint",
            "independent_basis_worst_bin_t": "2.35-5.34 in every checkpoint",
        },
        "decision_rule": (
            "Monotone improvement to cm 300 -> the mechanism is rate-limited and the "
            "bound is not yet found. Turnover -> the optimum is interior and locates "
            "the mechanism. Gain present at seed 31 only -> not reproducible, treat "
            "the screen result as noise. Judge mechanism measurements together with "
            "validation improvement; reduced volatility alone does not count."
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
    for name in ("run_critic_rate_ladder_20260922.py", "run_stochastic_eg_stability.py"):
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
        "reason": "Critic-rate ladder and replication; read against protocol.decision_rule.",
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
