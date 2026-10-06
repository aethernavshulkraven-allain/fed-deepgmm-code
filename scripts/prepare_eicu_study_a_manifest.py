"""Manifest generator for the Study A federated campaign.

Produces CSV manifests in the same schema ``run_manifest.py`` already
consumes (extended with ``scenario_name``, ``input_dim_g/f``,
``objective_mode``, ``aggregation_weighting`` -- all wired into
``run_manifest.py``'s ``build_config``/``write_config``), so launching,
resuming, and validating reuses that existing, tested machinery rather than
duplicating it.

Three stages, matching the frozen Study A protocol:

* ``tuning``       -- selected tuning seed pairs, 3 g0 x 2 methods x 6
                      LR/server-LR candidates per pair.
* ``confirmatory`` -- 3 g0 x 5 seeds x 2 methods = 30 rows, frozen
                      hyperparameters, 500 rounds.
* ``ablation``     -- all 3 g0 variants x 5 seeds x 2 methods x sample_size
                      weighting = 30 *additional* rows (the uniform_clients
                      arm is already covered by the confirmatory stage).

Every row uses: client_optimizer sgd/ogda (batch_size>0 -> the *_s variants,
matching how every other manifest in this repo derives fedgda_s/fedogda_s),
partition_method natural, objective_mode paper_aligned. Every row explicitly
sets skip_model_selection=true (Study A always uses exactly one g/f/learning
setup; the alternative model-selection path does not thread theta~ through and
is out of scope for this protocol -- see PaperAlignedMomentObjective's usage
notes).

Usage:
    python scripts/prepare_eicu_study_a_manifest.py --stage tuning \\
        --scenario-dir fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth \\
        --out experiments/eicu_study_a/tuning_manifest.csv
"""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

G0_VARIANTS = ("linear", "interaction", "mlp")  # "mlp" is the implementation
# label for protocol_v1.md's frozen_random_mlp variant (S4.1: "displayed
# label frozen_random_mlp maps to implementation label mlp only if metadata
# records that mapping" -- prepare_eicu_semisynth.py's make_g0("mlp", ...)
# records g0.kind="mlp" and every manifest row below records g0="mlp" too;
# G0_DISPLAY_LABEL is the recorded mapping.
G0_DISPLAY_LABEL = {"linear": "linear", "interaction": "interaction", "mlp": "frozen_random_mlp"}
METHODS = ("fedgda", "fedogda")  # -> fedgda_s / fedogda_s (client_optimizer sgd/ogda)

# protocol_v1.md S7.1-S7.3: scenario_seed (DGP/split) and optimizer_seed
# (model init/minibatch/optimizer) are separate randomness domains bound by a
# seed_pair_id; reusing one scalar for both is explicitly noncompliant even
# when the two numbers happen to match. These are the frozen/proposed pairs.
TUNING_SEED_PAIRS = (
    ("tuning_01", 11, 1011),
    ("tuning_02", 22, 1022),
    ("tuning_03", 33, 1033),
)
CONFIRMATORY_SEED_PAIRS = (
    ("confirmatory_01", 101, 1101),
    ("confirmatory_02", 102, 1102),
    ("confirmatory_03", 103, 1103),
    ("confirmatory_04", 104, 1104),
    ("confirmatory_05", 105, 1105),
)
PROTOCOL_VERSION = "eicu_study_a_v1"

# protocol_v1.md S6.2: centralized baseline methods, already canonical labels
# (gda_d/sgda_s/oadam_s) -- these manifest rows are descriptive/tracking only
# (run_eicu_centralized_baselines.py launches them directly, not through
# run_manifest.py, matching this repo's existing convention that
# training_scope != "federated" rows are listed but not launched by it).
CENTRALIZED_METHODS = ("gda_d", "sgda_s", "oadam_s")
CENTRALIZED_BATCH_SIZE = {"gda_d": 0, "sgda_s": 256, "oadam_s": 256}
CENTRALIZED_CLIENT_OPTIMIZER = {"gda_d": "sgd", "sgda_s": "sgd", "oadam_s": "oadam"}
# Centralized baselines do not yet have their own frozen validation-only
# tuning process (protocol_v1.md S8.1: "Centralized baselines also require
# comparable validation-only tuning" remains open) -- these are the same
# defaults run_eicu_centralized_baselines.py already uses, marked explicitly
# as not-yet-tuned rather than silently implying they were.
CENTRALIZED_DEFAULT_G_LR = 0.001
CENTRALIZED_DEFAULT_F_LR = 0.01
CENTRALIZED_DEFAULT_WEIGHT_DECAY = 0.01

# LR multiplier applied to the verified smoke config's learning_rate; server LR
# candidates. 3 x 2 = 6 candidates per (g0, method), matching the frozen protocol.
TUNING_LR_MULTIPLIERS = (0.5, 1.0, 2.0)
TUNING_SERVER_LRS = (1.0, 1.5)

# From the verified smoke config (experiments/eicu_v1_demo/smoke_config.yaml).
BASE_LEARNING_RATE = 0.001
BASE_WEIGHT_DECAY = 0.01
BASE_CRITIC_MULTIPLIER = 10.0
BASE_GRADIENT_CLIP_NORM = 1.0

TUNING_COMM_ROUND = 150
CONFIRMATORY_COMM_ROUND = 500

# protocol_v1.md: every row shares these facts about the study/selection
# policy; kept as constants rather than per-role variation since they are
# frozen campaign-wide, not per-row decisions.
ALIGNMENT_LABEL = {
    "tuning": "validation_only_tuning",
    "confirmatory": "primary_extension",
    "aggregation_ablation": "non_paper_aligned_aggregation_ablation",
    "centralized_baseline": "centralized_reference",
}
STUDY_CLAIM = "extension_no_published_target"
PRIMARY_SELECTION_METRIC = "equal_client_validation_mse"
SELECTION_SOURCE = "validation_only"

CSV_FIELDNAMES = [
    "run_id", "protocol_version", "run_group", "training_scope", "method", "method_label",
    "dataset", "seed", "alpha", "output_root", "final_result_dir", "implementation_status",
    "run_status", "preflight_required", "preflight_status", "model", "federated_optimizer",
    "client_optimizer", "client_num_in_total", "client_num_per_round", "comm_round", "epochs",
    "batch_size", "partition_method", "partition_alpha", "data_cache_dir", "learning_rate",
    "learning_rate_status", "weight_decay", "critic_multiplier", "server_learning_rate",
    "gradient_clip_norm", "simple_model_selection_epochs", "f_history_model_selection_epochs",
    "model_selection_batch_size", "using_gpu", "gpu_id", "notes",
    "scenario_name", "objective_mode", "aggregation_weighting", "input_dim_g", "input_dim_f",
    "g0", "skip_model_selection",
    "scenario_seed", "optimizer_seed", "seed_pair_id", "campaign_role", "scenario_checksum",
    "role", "g0_display_label",
    "alignment_label", "primary_selection_metric", "test_mse_used_for_selection",
    "selection_source", "study_claim", "scenario_scope", "scenario_metadata_path",
    "config_path", "result_path",
]


def load_scenario_metadata(scenario_dir, g0, scenario_seed):
    meta_path = os.path.join(scenario_dir, f"{g0}_scenario_seed{scenario_seed}_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"missing {meta_path} -- run scripts/prepare_eicu_semisynth.py "
            f"--g0 {g0} --scenario-seed {scenario_seed} first"
        )
    with open(meta_path) as handle:
        return json.load(handle)


def selection_key(g0, method):
    """Must match select_eicu_study_a_tuning.py's grouping key exactly: it
    groups by the manifest row's full ``method`` column (e.g. ``fedgda_s``),
    not the bare method name used to loop over METHODS here.
    """
    return f"{g0}:{method}_s"


def base_row(
    run_id, run_group, g0, scenario_seed, optimizer_seed, seed_pair_id, method, metadata,
    comm_round, output_root, role, campaign_role="", scenario_dir="",
):
    variant_suffix = "s"  # every row is stochastic-minibatch (batch_size > 0)
    client_optimizer = "ogda" if method == "fedogda" else "sgd"
    n_clients = int(metadata["n_clients"])
    manifest_method = f"{method}_{variant_suffix}"
    # Matches run_manifest.py's _run_dir exactly: output_root/dataset/method/seed_{seed}/run_id.
    result_path = os.path.join("eicu_semisynth", manifest_method, f"seed_{optimizer_seed}", run_id)
    return {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "run_group": run_group,
        "training_scope": "federated",
        "method": manifest_method,
        "method_label": f"{method.upper()}-S",
        "dataset": "eicu_semisynth",
        # run_manifest.py treats row["seed"] as the optimizer/training seed
        # (random_seed, run-dir path, seed-mismatch check) -- scenario_seed
        # is carried separately below and never conflated with this value.
        "seed": optimizer_seed,
        "alpha": "",  # not applicable -- eicu uses natural partitioning, not Dirichlet
        "output_root": output_root,
        "final_result_dir": "",
        "implementation_status": "implemented",
        "run_status": "not_started",
        "preflight_required": "false",
        "preflight_status": "",
        "model": "lr",
        "federated_optimizer": "FedAvg",
        "client_optimizer": client_optimizer,
        "client_num_in_total": n_clients,
        "client_num_per_round": n_clients,  # full participation, per protocol
        "comm_round": comm_round,
        "epochs": 1,
        "batch_size": max(min(n_clients, 32), 1),
        "partition_method": "natural",
        "partition_alpha": 0.0,
        "data_cache_dir": "data",
        "critic_multiplier": BASE_CRITIC_MULTIPLIER,
        "gradient_clip_norm": BASE_GRADIENT_CLIP_NORM,
        "simple_model_selection_epochs": 100,
        "f_history_model_selection_epochs": 60,
        "model_selection_batch_size": 200,
        "using_gpu": "false",  # these runs complete in well under a second on CPU
        "gpu_id": 0,
        "notes": "",
        "scenario_name": f"{g0}_scenario_seed{scenario_seed}",
        "objective_mode": "paper_aligned",
        "aggregation_weighting": "uniform_clients",
        "input_dim_g": int(metadata["n_features_x"]),
        "input_dim_f": int(metadata["n_features_z"]),
        "g0": g0,
        "g0_display_label": G0_DISPLAY_LABEL[g0],
        "skip_model_selection": "true",
        "scenario_seed": scenario_seed,
        "optimizer_seed": optimizer_seed,
        "seed_pair_id": seed_pair_id,
        "campaign_role": campaign_role,
        "scenario_checksum": metadata.get("scenario_checksum_sha256", ""),
        "role": role,
        "alignment_label": ALIGNMENT_LABEL[role],
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "test_mse_used_for_selection": "false",
        "selection_source": SELECTION_SOURCE,
        "study_claim": STUDY_CLAIM,
        "scenario_scope": metadata.get("scenario_scope", ""),
        "scenario_metadata_path": os.path.join(
            scenario_dir, f"{g0}_scenario_seed{scenario_seed}_metadata.json"
        ) if scenario_dir else "",
        # Populated once run_manifest.py generates configs for these rows;
        # left blank here since prelaunch validation does not require it
        # (config cross-checks are opt-in via the validator's --config-dir).
        "config_path": "",
        "result_path": result_path,
    }


def generate_tuning(scenario_dir, output_root, seed_pairs=TUNING_SEED_PAIRS[:1]):
    """Screening grid for each requested tuning seed pair.

    protocol_v1.md S7.2 (proposed): run the full LR/server-LR grid on
    ``tuning_01`` first; a candidate is only eligible for final selection once
    it also has results on ``tuning_02``/``tuning_03``. The shortlist/racing
    policy for deciding which candidates earn that second and third pair is
    explicitly unresolved pending full-eICU runtime preflight, so this
    defaults to screening on ``tuning_01`` alone (36 rows, unchanged) and lets
    a caller re-invoke with additional seed pairs once a shortlist exists.
    """
    rows = []
    for g0 in G0_VARIANTS:
        for method in METHODS:
            for lr_mult in TUNING_LR_MULTIPLIERS:
                for server_lr in TUNING_SERVER_LRS:
                    for seed_pair_id, scenario_seed, optimizer_seed in seed_pairs:
                        metadata = load_scenario_metadata(scenario_dir, g0, scenario_seed)
                        run_id = (
                            f"tuning_{g0}_{method}_{seed_pair_id}_lr{lr_mult}_slr{server_lr}"
                            .replace(".", "p")
                        )
                        row = base_row(
                            run_id, "tuning", g0, scenario_seed, optimizer_seed, seed_pair_id,
                            method, metadata, TUNING_COMM_ROUND, output_root, role="tuning",
                            scenario_dir=scenario_dir,
                        )
                        row["learning_rate"] = BASE_LEARNING_RATE * lr_mult
                        row["learning_rate_status"] = "tuning_candidate"
                        row["weight_decay"] = BASE_WEIGHT_DECAY
                        row["server_learning_rate"] = server_lr
                        row["notes"] = f"lr_multiplier={lr_mult};server_lr_candidate={server_lr}"
                        rows.append(row)
    return rows


def generate_confirmatory(scenario_dir, output_root, selected_hyperparameters):
    """``selected_hyperparameters``: {(g0, method): {"learning_rate":, "server_learning_rate":}}
    from scripts/select_eicu_study_a_tuning.py's frozen selection.
    """
    rows = []
    for g0 in G0_VARIANTS:
        for method in METHODS:
            key = selection_key(g0, method)
            if key not in selected_hyperparameters:
                raise KeyError(f"no frozen hyperparameters for {key}; run tuning selection first")
            selected = selected_hyperparameters[key]
            for seed_pair_id, scenario_seed, optimizer_seed in CONFIRMATORY_SEED_PAIRS:
                metadata = load_scenario_metadata(scenario_dir, g0, scenario_seed)
                run_id = f"confirmatory_{g0}_{method}_{seed_pair_id}"
                row = base_row(
                    run_id, "confirmatory", g0, scenario_seed, optimizer_seed, seed_pair_id,
                    method, metadata, CONFIRMATORY_COMM_ROUND, output_root, role="confirmatory",
                    scenario_dir=scenario_dir,
                )
                row["learning_rate"] = selected["learning_rate"]
                row["learning_rate_status"] = selected.get(
                    "learning_rate_status", "frozen_from_tuning"
                )
                row["weight_decay"] = BASE_WEIGHT_DECAY
                row["server_learning_rate"] = selected["server_learning_rate"]
                if selected.get("selection_note"):
                    row["notes"] = str(selected["selection_note"])
                rows.append(row)
    return rows


def generate_ablation(scenario_dir, output_root, selected_hyperparameters):
    """Sample-size aggregation ablation across all three g0 variants.

    protocol_v1.md S6.4/decision_register.md D16: the required ablation role
    is 3 g0 x 5 confirmatory seed pairs x 2 methods = 30 rows -- the
    uniform_clients arm already exists as the confirmatory rows, so only the
    sample_size arm is generated here, but across every g0 (not linear only).
    ``campaign_role: aggregation_ablation`` is what authorizes
    ``aggregation_weighting: sample_size`` past FedAvgAPI's eICU guard (see
    ``check_eicu_aggregation_weighting`` in experiment_utils.py).
    """
    rows = []
    for g0 in G0_VARIANTS:
        for method in METHODS:
            key = selection_key(g0, method)
            if key not in selected_hyperparameters:
                raise KeyError(f"no frozen hyperparameters for {key}; run tuning selection first")
            selected = selected_hyperparameters[key]
            for seed_pair_id, scenario_seed, optimizer_seed in CONFIRMATORY_SEED_PAIRS:
                metadata = load_scenario_metadata(scenario_dir, g0, scenario_seed)
                run_id = f"ablation_{g0}_{method}_sample_size_{seed_pair_id}"
                row = base_row(
                    run_id, "ablation", g0, scenario_seed, optimizer_seed, seed_pair_id,
                    method, metadata, CONFIRMATORY_COMM_ROUND, output_root,
                    role="aggregation_ablation", campaign_role="aggregation_ablation",
                    scenario_dir=scenario_dir,
                )
                row["learning_rate"] = selected["learning_rate"]
                row["learning_rate_status"] = selected.get(
                    "learning_rate_status", "frozen_from_tuning"
                )
                row["weight_decay"] = BASE_WEIGHT_DECAY
                row["server_learning_rate"] = selected["server_learning_rate"]
                row["aggregation_weighting"] = "sample_size"
                note = "aggregation ablation: sample_size arm"
                if selected.get("selection_note"):
                    note += f"; {selected['selection_note']}"
                row["notes"] = note
                rows.append(row)
    return rows


def centralized_base_row(
    run_id, g0, scenario_seed, optimizer_seed, seed_pair_id, method, metadata, scenario_dir=""
):
    """Descriptive/tracking row for a centralized baseline -- launched by
    run_eicu_centralized_baselines.py directly, not by run_manifest.py (whose
    select_rows only launches training_scope == "federated" rows, matching
    every other manifest already in this repo).
    """
    n_clients = int(metadata["n_clients"])
    # Matches run_eicu_centralized_baselines.py's run_dir_for exactly.
    result_path = os.path.join(f"{g0}_{seed_pair_id}", method, f"seed_{optimizer_seed}")
    return {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "run_group": "centralized_baseline",
        "training_scope": "centralized",
        "method": method,
        "method_label": method.upper(),
        "dataset": "eicu_semisynth",
        "seed": optimizer_seed,
        "alpha": "",
        "output_root": "",
        "final_result_dir": "",
        "implementation_status": "implemented",
        "run_status": "not_started",
        "preflight_required": "false",
        "preflight_status": "",
        "model": "lr",
        "federated_optimizer": "",
        "client_optimizer": CENTRALIZED_CLIENT_OPTIMIZER[method],
        "client_num_in_total": n_clients,
        "client_num_per_round": n_clients,
        "comm_round": CONFIRMATORY_COMM_ROUND,
        "epochs": 1,
        "batch_size": CENTRALIZED_BATCH_SIZE[method],
        "partition_method": "not_applicable_centralized",
        "partition_alpha": 0.0,
        "data_cache_dir": "data",
        "learning_rate": CENTRALIZED_DEFAULT_G_LR,
        "learning_rate_status": "default_not_tuned",
        "weight_decay": CENTRALIZED_DEFAULT_WEIGHT_DECAY,
        "critic_multiplier": CENTRALIZED_DEFAULT_F_LR / CENTRALIZED_DEFAULT_G_LR,
        "server_learning_rate": "",
        "gradient_clip_norm": BASE_GRADIENT_CLIP_NORM,
        "simple_model_selection_epochs": "",
        "f_history_model_selection_epochs": "",
        "model_selection_batch_size": "",
        "using_gpu": "false",
        "gpu_id": 0,
        "notes": "",
        "scenario_name": f"{g0}_scenario_seed{scenario_seed}",
        "objective_mode": "paper_aligned",
        # "none": centralized rows have no client aggregation step to weight
        # (the campaign validator hard-checks this literal value for any
        # role declared federated: false -- see validate_eicu_study_a_campaign.py).
        "aggregation_weighting": "none",
        "input_dim_g": int(metadata["n_features_x"]),
        "input_dim_f": int(metadata["n_features_z"]),
        "g0": g0,
        "g0_display_label": G0_DISPLAY_LABEL[g0],
        "skip_model_selection": "true",
        "scenario_seed": scenario_seed,
        "optimizer_seed": optimizer_seed,
        "seed_pair_id": seed_pair_id,
        "campaign_role": "",
        "scenario_checksum": metadata.get("scenario_checksum_sha256", ""),
        "role": "centralized_baseline",
        "alignment_label": ALIGNMENT_LABEL["centralized_baseline"],
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "test_mse_used_for_selection": "false",
        "selection_source": SELECTION_SOURCE,
        "study_claim": STUDY_CLAIM,
        "scenario_scope": metadata.get("scenario_scope", ""),
        "scenario_metadata_path": os.path.join(
            scenario_dir, f"{g0}_scenario_seed{scenario_seed}_metadata.json"
        ) if scenario_dir else "",
        "config_path": "",
        "result_path": result_path,
    }


def generate_centralized_baseline(scenario_dir, output_root):
    """3 g0 x 5 confirmatory seed pairs x 3 methods = 45 rows (protocol_v1.md
    S6.2), using the identical scenario artifacts as the federated
    confirmatory/ablation rows for the same (g0, seed_pair_id).
    """
    rows = []
    for g0 in G0_VARIANTS:
        for seed_pair_id, scenario_seed, optimizer_seed in CONFIRMATORY_SEED_PAIRS:
            metadata = load_scenario_metadata(scenario_dir, g0, scenario_seed)
            for method in CENTRALIZED_METHODS:
                run_id = f"centralized_{g0}_{method}_{seed_pair_id}"
                row = centralized_base_row(
                    run_id, g0, scenario_seed, optimizer_seed, seed_pair_id, method, metadata,
                    scenario_dir=scenario_dir,
                )
                row["output_root"] = output_root
                rows.append(row)
    return rows


def write_manifest(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})

    json_path = os.path.splitext(out_csv)[0] + ".json"
    with open(json_path, "w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_csv, json_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", required=True,
        choices=("tuning", "confirmatory", "ablation", "centralized_baseline", "all"),
    )
    parser.add_argument(
        "--scenario-dir",
        default=os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example", "data", "eicu_semisynth"),
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "results", "eicu_study_a"),
        help="passed through as each row's output_root; still pass --output-root explicitly to run_manifest.py",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--selected-hyperparameters",
        default=None,
        help="JSON from select_eicu_study_a_tuning.py; required for confirmatory/ablation",
    )
    parser.add_argument(
        "--tuning-seed-pairs",
        default="tuning_01",
        help=(
            "comma-separated seed_pair_id values from TUNING_SEED_PAIRS to generate "
            "tuning rows for (default: tuning_01 screening only). Re-invoke with the "
            "remaining pairs for shortlisted candidates once a shortlist policy is set."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    selected = {}
    if args.selected_hyperparameters:
        with open(args.selected_hyperparameters) as handle:
            selected = json.load(handle)

    if args.stage == "tuning":
        requested_pair_ids = {p.strip() for p in args.tuning_seed_pairs.split(",") if p.strip()}
        seed_pairs = tuple(p for p in TUNING_SEED_PAIRS if p[0] in requested_pair_ids)
        if not seed_pairs:
            raise ValueError(f"--tuning-seed-pairs matched no known pair: {args.tuning_seed_pairs!r}")
        tuning_rows = generate_tuning(args.scenario_dir, args.output_root, seed_pairs=seed_pairs)
    # "all" deliberately excludes tuning: protocol_v1.json's required_matrix
    # is confirmatory + centralized_baseline + aggregation_ablation only (105
    # rows); tuning is a separate, non-required, non-frozen stage.
    if args.stage in ("confirmatory", "ablation", "centralized_baseline", "all") and not selected:
        raise ValueError(f"--selected-hyperparameters is required for --stage {args.stage}")
    if args.stage in ("confirmatory", "all"):
        confirmatory_rows = generate_confirmatory(args.scenario_dir, args.output_root, selected)
    if args.stage in ("ablation", "all"):
        ablation_rows = generate_ablation(args.scenario_dir, args.output_root, selected)
    if args.stage in ("centralized_baseline", "all"):
        centralized_rows = generate_centralized_baseline(args.scenario_dir, args.output_root)

    if args.stage == "tuning":
        rows = tuning_rows
    elif args.stage == "confirmatory":
        rows = confirmatory_rows
    elif args.stage == "ablation":
        rows = ablation_rows
    elif args.stage == "centralized_baseline":
        rows = centralized_rows
    else:
        rows = confirmatory_rows + centralized_rows + ablation_rows

    csv_path, json_path = write_manifest(rows, args.out)
    print(f"stage={args.stage} rows={len(rows)}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
