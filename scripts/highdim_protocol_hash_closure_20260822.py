"""Shared, canonical list of decision-critical and execution-critical source
files for the post-BatchNorm-fix high-dimensional protocol's hash freezes.

Every stage (diagnostic, corrected screen, V4 signal/X) hashes this same
core set plus whatever scorer is specific to that stage -- kept in one
place so the closures can't silently drift apart between prep scripts.

Deliberately does not include the full vendored fedml/ framework: most of
it (hierarchical_fl, mime, turboaggregate, fedprox, feddyn, scaffold,
fednova, async_fedavg, fedgan, fedgkt, fedopt, fednas, fedseg,
classical_vertical_fl, llm training, the MLOps/API/scheduler modules, ...)
implements algorithms and features this campaign never invokes. This list
is every file actually on the execution path for a federated_optimizer=
FedAvg, model=lr/DeepGMM run through main.py: model construction, the
model trainer FedAvg actually uses, the scenario/dataset loaders for the
datasets this campaign trains on, and the server-update math itself.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_SOURCES: tuple[str, ...] = (
    "fedgmm/sp_decentralized_mnist_lr_example/main.py",
    "fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py",
    "fedgmm/sp_decentralized_mnist_lr_example/model_selection_class.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/model/model_hub.py",
    "fedgmm/sp_decentralized_mnist_lr_example/models/cnn_models.py",
    "fedgmm/sp_decentralized_mnist_lr_example/models/mlp_model.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/trainer_creator.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/my_model_trainer_classification.py",
    "fedgmm/sp_decentralized_mnist_lr_example/scenarios/abstract_scenario.py",
    "fedgmm/sp_decentralized_mnist_lr_example/scenarios/cifar10_scenario.py",
    "fedgmm/sp_decentralized_mnist_lr_example/scenarios/mnist_scenarios.py",
    "fedgmm/sp_decentralized_mnist_lr_example/scenarios/toy_scenarios.py",
    # The four scenarios/*.py entries above are retained for over-coverage
    # (harmless if unused) but are NOT what femnist_x/z/xz or cifar10_x/z/xz
    # actually load through at training time -- traced 2026-08-26 (closeout
    # plan SS4.5): fedml.data.load() dispatches through the two loaders below,
    # which build an AbstractScenario directly from the dataset NPZ, bypassing
    # the scenarios/*_scenario.py wrapper classes entirely for this campaign.
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/data/data_loader.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/data/MNIST/data_loader.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/data/cifar10/efficient_loader.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/data/cifar10/without_reload.py",
    # Psi / model-selection modules -- omitted before 2026-08-26; the legacy
    # Psi definition frozen in PROTOCOL_DECISION_ADDENDUM_20260826.md lives
    # here, not just in model_selection_class.py above.
    "fedgmm/sp_decentralized_mnist_lr_example/game_objectives/approximate_psi_objective.py",
    "fedgmm/sp_decentralized_mnist_lr_example/game_objectives/abstract_objective.py",
    "fedgmm/sp_decentralized_mnist_lr_example/game_objectives/simple_moment_objective.py",
    "fedgmm/sp_decentralized_mnist_lr_example/model_selection/learning_eval_nostop.py",
    "fedgmm/sp_decentralized_mnist_lr_example/model_selection/abstract_learning_eval.py",
    "fedgmm/sp_decentralized_mnist_lr_example/model_selection/simple_model_eval.py",
    # Optimizer implementations -- also omitted before 2026-08-26.
    "fedgmm/sp_decentralized_mnist_lr_example/optimizers/optimizer_factory.py",
    "fedgmm/sp_decentralized_mnist_lr_example/optimizers/Customsgd.py",
    "fedgmm/sp_decentralized_mnist_lr_example/optimizers/ogda.py",
    "fedgmm/sp_decentralized_mnist_lr_example/optimizers/oadam.py",
    "fedgmm/sp_decentralized_mnist_lr_example/optimizers/extragradient.py",
    # Directly imported by fedavg_api.py (`.client`, `.multiprocess_client`)
    # and model_selection_class.py (`from model_selection.learning_eval
    # import f_history_g_eval` -- the base learning_eval.py, not just its
    # NoStop subclass above) -- omitted before 2026-08-27, found via a
    # review plus confirmation against fedavg_api.py's actual import list.
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/client.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/multiprocess_client.py",
    "fedgmm/sp_decentralized_mnist_lr_example/model_selection/learning_eval.py",
    "fedgmm/sp_decentralized_mnist_lr_example/plotting.py",
    # main.py's FedMLRunner(...) call resolves through these two files --
    # confirmed directly from a real traceback captured 2026-08-27 (a
    # genuine ModelSelectionFailure during the screen-failure certification
    # reruns), not just static analysis:
    #   fedml/runner.py:53 in __init__
    #   fedml/simulation/simulator.py:41 in __init__
    #   fedml/simulation/sp/fedavg/fedavg_api.py:562 in __init__
    # Their own top-level imports (fedml/constants.py, fedml/core/__init__.py)
    # are included too, since Python executes both files' module bodies on
    # import regardless of which names actually get used. fedml/core/__init__.py
    # itself re-exports classes from the same excluded vendored subsystems
    # named in this file's module docstring (contribution assessment,
    # differential privacy, security attacker/defender, distributed comm) --
    # those deeper files are not included, on the same "never invoked by
    # this campaign" basis, since nothing in this campaign ever instantiates
    # or calls any of them; only fedml/core/__init__.py's own shallow
    # re-export list is execution-critical here.
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/runner.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/simulator.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/constants.py",
    "fedgmm/sp_decentralized_mnist_lr_example/fedml/core/__init__.py",
    "scripts/run_manifest.py",
    "scripts/verify_protocol_hashes.py",
    "scripts/check_manifest_stage_complete.py",
)

# The six source dataset NPZ files this campaign trains on (closeout plan
# SS4.5) -- hashed separately from CORE_SOURCES because they are data, not
# code, but are equally execution-critical: a silently-swapped NPZ would
# invalidate every result exactly like a silently-changed source file would.
CORE_DATASET_FILES: tuple[str, ...] = (
    "fedgmm/sp_decentralized_mnist_lr_example/data/femnist_x/main.npz",
    "fedgmm/sp_decentralized_mnist_lr_example/data/femnist_z/main.npz",
    "fedgmm/sp_decentralized_mnist_lr_example/data/femnist_xz/main.npz",
    "fedgmm/sp_decentralized_mnist_lr_example/data/cifar10_x/main.npz",
    "fedgmm/sp_decentralized_mnist_lr_example/data/cifar10_z/main.npz",
    "fedgmm/sp_decentralized_mnist_lr_example/data/cifar10_xz/main.npz",
)

# The frozen protocol rules every scorer/preparer in this campaign actually
# implements -- omitted from every stage's hash closure before 2026-08-27,
# including this document's own explicit promise in SS7 that it and every
# script depending on it would be covered. A silent edit to any of these
# would change scoring/boundary outcomes exactly like a silently-changed .py
# file would, so they belong in the same closure.
CORE_PROTOCOL_DOCS: tuple[str, ...] = (
    "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813/PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md",
    "experiments/highdim_coauthor_protocol_v1/deterministic_screen_20260813/BOUNDARY_RULE_AMENDMENT_20260818.md",
    "experiments/highdim_coauthor_protocol_v1/deterministic_screen_post_bn_20260822/PROTOCOL_DECISION_ADDENDUM_20260826.md",
)


def git_provenance(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Git revision plus a checksum of any intentional dirty diff, for the
    prelaunch hash closure (closeout plan SS4.5). dirty_diff_sha256 is None
    only when the tree is exactly clean at git_revision; any dirty diff --
    intentional staged work included in this launch -- is still recorded via
    its checksum, not silently dropped, so a launch from a dirty tree remains
    auditable rather than merely tolerated. Python/PyTorch/CUDA/GPU
    environment metadata is captured separately, per run, by
    RuntimeProfiler.record_environment() into environment.json -- not
    duplicated here, since it legitimately varies run to run."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    diff_text = subprocess.run(
        ["git", "diff", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    dirty_diff_sha256 = (
        hashlib.sha256(diff_text.encode("utf-8")).hexdigest() if diff_text else None
    )
    return {"git_revision": revision, "dirty_diff_sha256": dirty_diff_sha256}
