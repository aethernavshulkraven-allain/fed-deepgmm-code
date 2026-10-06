"""FedEG-S validation stage: reduced two-candidate bank, full 1500-round horizon.

Implements the locked design in
experiments/highdim_coauthor_protocol_v1/fedeg_s_final_design_20260922/, with
the recorded amendment reducing the candidate bank from four per family to two.
That halves tuning cost (216 -> 108 validation runs) while leaving the 90-run
finals matrix and every run's 1500-round budget untouched. The tradeoff is a
narrower search, NOT evidence that the excluded settings are worse.

Candidates, two per scenario family, structural server held at 1.5 throughout:

    X    g 0.003, f 0.3     critic server 0.3 vs 1.5
    XZ   g 0.003, f 0.03 vs 0.3     critic server 1.5
    Z    g 0.01,  f 0.1  vs 0.3     critic server 1.5

Validation seeds 31, 32 and 33 are all retained. Seed 33 is the poorly
responding case that failed in 8 of 8 crossover and factorial cells; dropping it
would make selection look better without making it better.

Full horizons deliberately. The investigation established that good cells peak
late -- mean round 82, max 133 of 150 -- and that early rankings mislead, so
runs are neither shortened to 150 rounds nor pruned on early performance.
Validation-deterioration stopping is DISABLED so finite runs actually complete
1500 rounds; numerical-failure stopping stays enabled.

SCOPE OF THIS LAUNCH. Four of six scenario families are archive-ready:

    femnist_x    corrected class-consistent archive
    femnist_xz   corrected class-consistent archive
    femnist_z    original archive -- correct, the image-treatment inconsistency
    cifar10_z    original archive    is exactly zero when X stays scalar

cifar10_x and cifar10_xz are marked `status: blocked` in the design's own
datasets.yaml: their corrected archives do not exist and the protocol forbids
substituting the originals. They are excluded here and added once generated and
audited, which needs a CIFAR paired scenario -- the existing generator is
hardcoded to the two FEMNIST datasets.

4 scenarios x 3 alphas x 2 candidates x 3 seeds = 72 runs, about 20 GPU-hours
at the measured 0.278 GPU-h per 1500-round run. Resumable: re-running the same
command skips completed runs, so a stop at quota exhaustion costs only the runs
in flight.
"""

import argparse
import fcntl
import math
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

DEFAULT = base.ROOT / "experiments/highdim_coauthor_protocol_v1/fedeg_s_validation_v3_20260922"
PAIRED = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_20260919"
PAIRED_CIFAR = base.ROOT / "experiments/highdim_coauthor_protocol_v1/paired_dgp_cifar10_20260922"
ORIGINAL = base.ROOT / "fedgmm/sp_decentralized_mnist_lr_example/data"
STAGE = "validation"
SEEDS = (31, 32, 33)
ALPHAS = (0.1, 0.5, 1.0)
ROUNDS = 1500
STRUCTURAL_SERVER_LR = 1.5
CHECKPOINT_ROUNDS = "0 1 2 5 8 9 10 11 12 13 14 15 20 25 50"

# dataset -> (family, archive variant, audit root for corrected archives)
#
# Z-image scenarios legitimately use the ORIGINAL archives: the class-consistency
# defect is that `g` is recomputed on the treatment image's class while `y` keeps
# the continuous-treatment signal, so it is exactly zero wherever X stays scalar.
REGISTRY = {
    "femnist_x":  ("x",  "corrected", PAIRED),
    "femnist_xz": ("xz", "corrected", PAIRED),
    "femnist_z":  ("z",  "original",  None),
    "cifar10_x":  ("x",  "corrected", PAIRED_CIFAR),
    "cifar10_xz": ("xz", "corrected", PAIRED_CIFAR),
    "cifar10_z":  ("z",  "original",  None),
}
READY = ("femnist_x", "femnist_xz", "femnist_z", "cifar10_z")

# family -> {candidate: (eta_g, eta_f, critic_server_lr)}
CANDIDATES = {
    "x":  {"x_ref":      (0.003, 0.3,  1.5), "x_damped":   (0.003, 0.3,  0.3)},
    "xz": {"xz_control": (0.003, 0.03, 1.5), "xz_ref":     (0.003, 0.3,  1.5)},
    "z":  {"z_reference":(0.01,  0.1,  1.5), "z_stronger": (0.01,  0.3,  1.5)},
}
# Finite runs must complete the full horizon; only numerical failure stops them.
# Zero is the documented disable value -- POLICY_DEFAULTS calls it "preserves
# existing training" -- and ValidationDeteriorationGuard returns immediately on it.
NO_DETERIORATION_STOP = {**stability.POLICY, "validation_deterioration_window": 0}

# The rolling window used to REPORT stability, which is a different quantity from
# the window used to STOP on deterioration even though one config key feeds both
# in run_stochastic_eg_stability.summarize. Stopping is disabled here, so reusing
# the stopping window would ask for the median of an empty slice.
STABILITY_REPORT_WINDOW = 25


def summarize(job):
    """Summarize stability with a pinned reporting window.

    `stability.summarize` passes `validation_deterioration_window` straight into
    `stability_summary`, conflating the deterioration-stopping window with the
    reporting window. With stopping disabled that window is 0 and the report dies
    on `median(values[i:i + 0])`. The two windows are independent, so this pins
    the reporting one and leaves stopping off.
    """
    result = base.validation_record(job)
    values = stability.read_values(job)
    window = min(STABILITY_REPORT_WINDOW, len(values))
    result.update(stability.stability_summary(
        values, window=window, tail=min(50, len(values))))
    result["stability_report_window"] = window
    # The design's eligibility rule is full-horizon completion, and it is decided
    # per CANDIDATE across all three seeds, not per run: "A candidate is eligible
    # only if all three runs complete 1,500 finite rounds", ranked by the mean of
    # each run's minimum validation MSE. So this records the per-run fact that
    # rule needs and leaves the decision to selection.
    result["completed_full_horizon"] = (
        int(result["rounds"]) == int(job.config["comm_round"])
        and math.isfinite(result["best"]))
    # Descriptive only. These thresholds come from the stochastic-EG stability
    # study and are NOT part of this stage's eligibility rule; naming them
    # "selection_eligible" would quietly impose a second, stricter criterion.
    result["tail_stability_within_stability_study_thresholds"] = (
        result["tail_median_over_best_window"] <= 2
        and result["tail_p90_over_best_window"] <= 5)
    return result


def archive_for(dataset):
    family, variant, audit_root = REGISTRY[dataset]
    if variant == "corrected":
        audit_path = audit_root / "data/class_consistent/generation_audit.json"
        if not audit_path.is_file():
            raise SystemExit(
                f"{dataset}: no corrected archive has been generated and audited "
                f"({audit_path}); the design forbids substituting the original")
        record = base.read_json(audit_path)["datasets"][dataset]
        path, expected = Path(record["generated"]), record["generated_sha256"]
    else:
        path = ORIGINAL / dataset / "main.npz"
        expected = base.file_sha256(path)
    if not path.is_file():
        raise SystemExit(f"{dataset}: archive missing at {path}")
    if base.file_sha256(path) != expected:
        raise SystemExit(f"{dataset}: archive does not match its recorded hash")
    return path, expected


def make_rows(campaign, sources, datasets, alphas, seeds, rounds):
    rows, index = [], 0
    for dataset in datasets:
        family, variant, _ = REGISTRY[dataset]
        archive, archive_sha = archive_for(dataset)
        for alpha in alphas:
            for candidate, (eta_g, eta_f, fsrv) in CANDIDATES[family].items():
                for seed in seeds:
                    row = base.make_row(campaign, sources[dataset], STAGE, dataset,
                                        alpha, "fed_eg_s", seed,
                                        {"candidate": index, "learning_rate": eta_g,
                                         "server_learning_rate": STRUCTURAL_SERVER_LR},
                                        rounds)
                    index += 1
                    tag = f"{alpha:g}".replace(".", "p")
                    run_id = f"val_{dataset}_a{tag}_{candidate}_s{seed}"
                    row.update(NO_DETERIORATION_STOP)
                    row.update({
                        "run_id": run_id, "arm": candidate,
                        "family": family, "candidate_name": candidate,
                        "archive_variant": variant,
                        "eta_g": eta_g, "eta_f": eta_f,
                        "structural_server_lr": STRUCTURAL_SERVER_LR,
                        "critic_server_lr": fsrv,
                        "protocol_version": "fedeg_s_validation_20260922_v1",
                        "objective_mode": "paper_aligned", "objective_lambda_1": 0.25,
                        "aggregation_weighting": "uniform_clients",
                        "server_buffer_policy": "direct_client_aggregate",
                        "local_bn_mode": "batch",
                        "learning_rate": eta_g,
                        "critic_multiplier": eta_f / eta_g,
                        "server_learning_rate": STRUCTURAL_SERVER_LR,
                        "eg_predictor_server_lr": STRUCTURAL_SERVER_LR,
                        "eg_corrector_server_lr": STRUCTURAL_SERVER_LR,
                        "eg_predictor_critic_server_lr": fsrv,
                        "eg_corrector_critic_server_lr": fsrv,
                        "alpha": alpha, "partition_alpha": alpha,
                        "comm_round": rounds,
                        "client_num_in_total": 1000, "client_num_per_round": 10,
                        "epochs": 3, "batch_size": 256,
                        "stop_on_numerical_failure": True,
                        "random_seed": seed, "scenario_seed": seed, "optimizer_seed": seed,
                        "checkpoint_rounds": CHECKPOINT_ROUNDS,
                        "periodic_checkpoint_interval": 100,
                        "compact_predictions_only": True,
                        "data_cache_dir": str(archive.parent.parent),
                        "scenario_name": str(archive.with_suffix("")),
                        "scenario_checksum": archive_sha,
                        "final_result_dir": str(campaign / "results" / STAGE / dataset /
                                                "fed_eg_s" / f"seed_{seed}" / run_id),
                        "notes": ("FedEG-S validation, reduced two-candidate bank, full "
                                  "1500-round horizon. Selection on validation only."),
                    })
                    rows.append(row)
    return rows


def archive_records(datasets):
    """{dataset: {path, sha256, variant}}, the shape protocol.json stores."""
    records = {}
    for dataset in datasets:
        path, sha = archive_for(dataset)
        records[dataset] = {"path": str(path), "sha256": sha,
                            "variant": REGISTRY[dataset][1]}
    return records


def assert_grid_is_complete(rows, datasets, archives, alphas, seeds, rounds):
    """Resolve every cell rather than trusting key names.

    `archives` is passed in rather than recomputed, because the module constants
    resolve against `base.ROOT`, which is the repository at prepare time and the
    frozen runtime copy at run time. Re-deriving an archive path here would look
    correct while preparing and point into the runtime tree while running. The
    campaign's own recorded archives are the stable reference, and
    `verify_freeze(full_data=True)` has already checked their bytes.
    """
    seen = {}
    for row in rows:
        lrs = resolve_server_learning_rates(row)
        fsrv = float(row["critic_server_lr"])
        if lrs["predictor_g"] != STRUCTURAL_SERVER_LR or lrs["corrector_g"] != STRUCTURAL_SERVER_LR:
            raise SystemExit(f"{row['run_id']}: structural server rate moved")
        if lrs["predictor_f"] != fsrv or lrs["corrector_f"] != fsrv:
            raise SystemExit(f"{row['run_id']}: critic server rate did not take effect")
        eta_g, eta_f = float(row["eta_g"]), float(row["eta_f"])
        applied = float(row["critic_multiplier"]) * float(row["learning_rate"])
        if abs(applied - eta_f) > 1e-9:
            raise SystemExit(
                f"{row['run_id']}: cm x eta_g = {applied}, eta_f should be {eta_f}")
        if float(row["alpha"]) != float(row["partition_alpha"]):
            raise SystemExit(f"{row['run_id']}: alpha and partition_alpha disagree")
        if int(row["comm_round"]) != rounds:
            raise SystemExit(f"{row['run_id']}: horizon must be {rounds}")
        if int(row["validation_deterioration_window"]) != 0:
            raise SystemExit(
                f"{row['run_id']}: deterioration stopping must be OFF so finite runs "
                "complete the full horizon")
        if not row["stop_on_numerical_failure"]:
            raise SystemExit(f"{row['run_id']}: numerical-failure stopping must stay on")
        if row["dataset"] not in datasets:
            raise SystemExit(f"{row['run_id']}: dataset outside this campaign's scope")
        record = archives.get(row["dataset"])
        if record is None:
            raise SystemExit(f"{row['run_id']}: no archive recorded for this dataset")
        if row["archive_variant"] != record["variant"]:
            raise SystemExit(
                f"{row['run_id']}: expected the {record['variant']} archive, row says "
                f"{row['archive_variant']}")
        if row["scenario_name"] + ".npz" != record["path"]:
            raise SystemExit(f"{row['run_id']}: archive path is not the recorded one")
        if row["scenario_checksum"] != record["sha256"]:
            raise SystemExit(f"{row['run_id']}: archive checksum is not the recorded one")
        key = row["run_id"]
        if key in seen:
            raise SystemExit(f"duplicate run_id {key}")
        seen[key] = True
    expected = len(datasets) * len(alphas) * 2 * len(seeds)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, built {len(rows)}")


def prepare(campaign, datasets, alphas=ALPHAS, seeds=SEEDS, rounds=ROUNDS):
    if (campaign / "freeze.json").exists():
        base.verify_freeze(campaign, full_data=True)
        return
    if any((campaign / n).exists() for n in ("runtime", "configs", "protocol.json")):
        raise FileExistsError(f"Partial preparation preserved: {campaign}")
    source_rows = base.launcher._load_rows(
        base.ROOT / "experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_stochastic.csv")
    sources = {d: next(r for r in source_rows if r["dataset"] == d) for d in datasets}
    archives = archive_records(datasets)
    rows = make_rows(campaign, sources, datasets, alphas, seeds, rounds)
    assert_grid_is_complete(rows, datasets, archives, alphas, seeds, rounds)
    protocol = {
        "stage": STAGE, "datasets": list(datasets), "alphas": list(alphas),
        "seeds": list(seeds), "rounds": rounds, "run_count": len(rows),
        "is_smoke": (list(alphas), list(seeds), rounds) != (list(ALPHAS), list(SEEDS), ROUNDS),
        "candidates": {f: {k: {"eta_g": v[0], "eta_f": v[1], "critic_server_lr": v[2]}
                           for k, v in c.items()} for f, c in CANDIDATES.items()},
        "structural_server_lr": STRUCTURAL_SERVER_LR,
        "archives": archives,
        "excluded_datasets": [d for d in REGISTRY if d not in datasets],
        "amendment": ("Candidate bank reduced from four per family to two, halving "
                      "validation from 216 to 108 runs. The 90-run finals matrix and "
                      "the 1500-round budget are unchanged. Narrower search, NOT "
                      "evidence that excluded settings are worse."),
        "deterioration_stopping": "disabled so finite runs complete the full horizon",
        "numerical_failure_stopping": "enabled",
        "selection_metric_source": "validation", "test_mse_used_for_selection": False,
        "automatic_promotion": False,
        "eligibility_rule": ("A candidate is eligible only if all three seeds complete "
                             "the full finite horizon; rank eligible candidates by the "
                             "arithmetic mean of each run's minimum validation MSE. "
                             "Tail-stability fields are descriptive, never a filter."),
        "resumable": "re-run the same command; completed runs are skipped",
        "client_execution_mode": "sp",
        "watchdog": {"startup_seconds": 1800, "stall_seconds": 1800, "total_seconds": 43200},
        "deterioration_policy": NO_DETERIORATION_STOP,
        "sources": sources,
    }
    base.save_json(campaign / "protocol.json", protocol, immutable=True)
    base.write_manifest(campaign / f"{STAGE}_manifest.csv", rows)
    base.snapshot(campaign)
    runtime = campaign / "runtime"
    for name in ("run_fedeg_s_validation_20260922.py", "run_stochastic_eg_stability.py"):
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
        "data": {r["path"]: {"sha256": r["sha256"],
                             "size": Path(r["path"]).stat().st_size,
                             "mtime_ns": Path(r["path"]).stat().st_mtime_ns}
                 for r in archives.values()},
    }, immutable=True)
    print(f"Prepared {len(rows)} validation runs "
          f"({len(datasets)} datasets x {len(alphas)} alphas x 2 candidates x "
          f"{len(seeds)} seeds, {rounds} rounds): {campaign}", flush=True)


def run(campaign):
    if base.ROOT != campaign / "runtime":
        raise RuntimeError("Launch the frozen runtime script, not the live source")
    allocated = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not 1 <= len(allocated) <= 4 or len(set(allocated)) != len(allocated):
        raise RuntimeError(f"Require one to four distinct broker GPUs; got {allocated!r}")
    base.verify_freeze(campaign, full_data=True)
    protocol = base.read_json(campaign / "protocol.json")
    rows = base.launcher._load_rows(campaign / f"{STAGE}_manifest.csv")
    assert_grid_is_complete(rows, set(protocol["datasets"]), protocol["archives"],
                            protocol["alphas"], protocol["seeds"], protocol["rounds"])
    records = base.run_stage(campaign, STAGE, rows, allocated, protocol["watchdog"],
                             resolver=stability.resolve, summarizer=summarize)
    base.save_json(campaign / "status.json", {
        "status": f"{STAGE}_complete_review_required", "stage": STAGE,
        "resolved": len(records), "total": len(rows), "gpus_used": len(allocated),
        "finals_launched": False, "winner": None,
        "reason": "Validation stage; select per dataset/alpha on validation MSE only.",
        "time": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    parser.add_argument("--datasets", nargs="+", default=list(READY),
                        choices=sorted(REGISTRY))
    # Smoke-test overrides. Defaults are the real design; anything else marks the
    # campaign is_smoke in protocol.json so it can never be mistaken for a result.
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "prepare":
        prepare(campaign, list(dict.fromkeys(args.datasets)),
                list(dict.fromkeys(args.alphas)), list(dict.fromkeys(args.seeds)),
                args.rounds)
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
