"""Four-arm critic-rate screen on corrected FEMNIST-X.

Tests the hypothesis that the critic is under-powered relative to the
structural model: let clients fit the critic harder while making the server's
critic extrapolation more conservative. It is a hypothesis, not a requirement
that a larger critic rate must help.

What motivates it, from the checkpoint diagnostics
(critic_bn_diagnostic_20260922): the critic's pre-clip gradient is only
0.10-0.52x the structural model's at every selected checkpoint, and a fixed
8-quantile instrument basis detects conditional moment violations at t =
2.35-5.34 in every checkpoint including the federated one -- violations the
learned critic is not eliminating.

Arms, varying ONLY the two critic knobs. Structural rates, initialization,
warm-up, partitions, participation, batches, reference schedule and budget are
identical across arms:

    A control          local f LR 0.03 (cm 10),  critic server 1.5
    B local critic     local f LR 0.09 (cm 30),  critic server 1.5
    C server critic    local f LR 0.03 (cm 10),  critic server 0.3
    D both             local f LR 0.09 (cm 30),  critic server 0.3

The critic server coefficient is the one added in Phase 3.1; before it, g and f
shared one coefficient per phase and arms C and D were not expressible.

Reading it. If C or D helps, split predictor-only versus corrector-only next.
If B fails, distinguish clipped/unstable updates from an uninformative critic
before raising cm further. If all four fail the grid is inconclusive, NOT proof
that rates are irrelevant. Judge on the mechanism measurements together with
validation improvement -- reduced volatility alone does not count, and neither
does a better final/best ratio.

Note that cm and the server coefficient are coupled through the trajectory:
raising cm does not change any server coefficient, but it changes the deltas
the server receives. The arms are only interpretable alongside the per-client
delta measurements.
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

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/critic_rate_screen_20260922"
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
DATASET = "femnist_x"
STAGE = "screen"
SEED = 31
ALPHA = 0.5
ROUNDS = 150
G_LR = 0.003
SERVER_LR = 1.5

ARMS = {
    "a_control":      {"critic_multiplier": 10, "critic_server_lr": 1.5},
    "b_local_critic": {"critic_multiplier": 30, "critic_server_lr": 1.5},
    "c_server_critic":{"critic_multiplier": 10, "critic_server_lr": 0.3},
    "d_both":         {"critic_multiplier": 30, "critic_server_lr": 0.3},
}


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
    for index, (arm, spec) in enumerate(ARMS.items()):
        row = base.make_row(campaign, source, STAGE, DATASET, ALPHA, "fed_eg_s", SEED,
                            {"candidate": index, "learning_rate": G_LR,
                             "server_learning_rate": SERVER_LR}, ROUNDS)
        run_id = f"critrate_{arm}_s{SEED}"
        row.update(stability.POLICY)
        row.update({
            "run_id": run_id, "arm": arm,
            "protocol_version": "critic_rate_screen_20260922_v1",
            "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
            "aggregation_weighting": "uniform_clients",
            "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
            # the two knobs under test
            "critic_multiplier": spec["critic_multiplier"],
            "eg_predictor_critic_server_lr": spec["critic_server_lr"],
            "eg_corrector_critic_server_lr": spec["critic_server_lr"],
            # structural server rates held fixed in every arm
            "eg_predictor_server_lr": SERVER_LR,
            "eg_corrector_server_lr": SERVER_LR,
            "comm_round": ROUNDS,
            "scenario_seed": SEED, "optimizer_seed": SEED,
            "data_cache_dir": str(archive.parent.parent),
            # The loader prefers its cwd's data/.../main.npz, so name the
            # reviewed archive absolutely and pin its hash.
            "scenario_name": str(archive.with_suffix("")),
            "scenario_checksum": archive_sha,
            "final_result_dir": str(campaign / "results" / STAGE / DATASET /
                                    "fed_eg_s" / f"seed_{SEED}" / run_id),
            "notes": ("Critic-rate screen; only critic_multiplier and the critic "
                      "server coefficient vary. Diagnostic, no promotion."),
        })
        rows.append(row)
    return rows


def assert_arms_are_distinguishable(rows):
    """A mistyped critic key is accepted silently by fedml/arguments.py and
    simply never read, producing a clean-looking run that tests nothing. Rather
    than trust the key names, resolve each arm and require the four arms to
    differ in exactly the intended way."""
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        spec = ARMS[row["arm"]]
        if lrs["predictor_f"] != spec["critic_server_lr"]:
            raise SystemExit(
                f"{row['run_id']}: resolved predictor_f={lrs['predictor_f']} but the arm "
                f"intends {spec['critic_server_lr']}. The critic key did not take effect.")
        if lrs["corrector_f"] != spec["critic_server_lr"]:
            raise SystemExit(f"{row['run_id']}: resolved corrector_f mismatch")
        if lrs["predictor_g"] != SERVER_LR or lrs["corrector_g"] != SERVER_LR:
            raise SystemExit(
                f"{row['run_id']}: structural server rate moved; arms must differ "
                "only in the critic coefficients")
        if float(row["critic_multiplier"]) != spec["critic_multiplier"]:
            raise SystemExit(
                f"{row['run_id']}: critic_multiplier is {row['critic_multiplier']}, "
                f"expected {spec['critic_multiplier']} -- make_row may have clobbered it")
        key = (spec["critic_multiplier"], spec["critic_server_lr"])
        if key in seen:
            raise SystemExit(f"{row['run_id']} duplicates {seen[key]}: arms are not distinct")
        seen[key] = row["run_id"]
    if len(seen) != 4:
        raise SystemExit(f"expected 4 distinct arms, got {len(seen)}")


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
        "dataset": DATASET, "seed": SEED, "alpha": ALPHA, "rounds": ROUNDS,
        "learning_rate": G_LR, "server_learning_rate": SERVER_LR,
        "arms": ARMS, "run_count": len(rows),
        "archive": str(archive), "archive_sha256": archive_sha,
        "question": ("Does a stronger local critic, a more conservative critic-only "
                     "server step, or their combination recover structural signal on "
                     "corrected FEMNIST-X?"),
        "held_fixed": ["structural local and server rates", "initialization",
                       "warm-up", "partitions", "participation", "batches",
                       "reference schedule", "round budget"],
        "motivation": {
            "source": "critic_bn_diagnostic_20260922/diagnostics.json",
            "critic_grad_over_structural_grad": "0.10-0.52 at every selected checkpoint",
            "independent_basis_worst_bin_t": "2.35-5.34 in every checkpoint",
        },
        "decision_rule": (
            "C or D helps -> split predictor-only vs corrector-only next. B fails -> "
            "distinguish clipped/unstable updates from an uninformative critic before "
            "raising cm further. All four fail -> inconclusive, NOT proof that rates "
            "are irrelevant. Judge mechanism measurements together with validation "
            "improvement; reduced volatility alone does not count."
        ),
        "scope": "One dataset, one seed, one alpha, 150 rounds. Screen, not a finals matrix.",
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
    for name in ("run_critic_rate_screen_20260922.py", "run_stochastic_eg_stability.py"):
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
        "reason": "Critic-rate screen; read arms against protocol.decision_rule.",
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
