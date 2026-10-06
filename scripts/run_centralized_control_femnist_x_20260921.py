"""SUPERSEDED AND INVALID -- DO NOT RUN. Kept only as provenance.

This script produced experiments/highdim_coauthor_protocol_v1/
centralized_control_20260921/, whose GMM arms (R^2 -0.44764 and +0.05932) are
void. Its training loop placed two unrestricted backward() calls before both
optimizer steps:

    g_obj.backward(retain_graph=True)
    f_obj.backward()
    opt_g.step(); opt_f.step()

backward() accumulates into every parameter in the graph, so each network
received the gradient of L_g + L_f = M + (-M + P) = P. The moment terms
cancelled exactly -- verified at 9.2e-16 against a penalty-only gradient -- and
both networks trained on the penalty alone. The conclusions drawn from it
("federation ruled out", "BatchNorm ruled out") were withdrawn; no hypothesis
was eliminated.

Replacement: scripts/run_pooled_control_femnist_x_20260922.py, which takes
isolated per-player gradients and refuses to start unless a non-degenerate
gradient gate passes.

The original docstring follows.
"""

"""Centralized control for the femnist_x collapse: is the failure federated or fundamental?

Every diagnostic so far has been run inside the federated loop. Four candidate
causes have been eliminated there -- server BatchNorm buffer handling, objective
signs and penalty, outer weighting, and the DGP target/moment mismatch -- and
none moved femnist_x off R^2 ~ 0. The audit noted no centralized experiment was
ever run, so the federated layer itself has never been controlled for.

Three facts motivate this control:

  1. The evaluation target is a deterministic function of the image class:
     within-class variance of train_g is 3.6e-33 and the ceiling R^2 is exactly
     1.0. The task reduces to 10-way image classification with 6 distinct output
     values, so it is not information-limited.

  2. Federated data throughput is severe. The median client holds 11 samples
     (82% hold fewer than 32), and the local loop fills a 256-row batch by
     itertools.cycle over those 11, so a "batch of 256" carries 11 samples of
     information. Each round touches ~200 of 20,000 train samples: 1.5 effective
     epochs at 150 rounds, 15 at 1500.

  3. A BatchNorm inference check came back negative. Re-running the best
     femnist_x checkpoint under stored running statistics versus batch
     statistics at n = 11/32/256/2000 leaves R^2 at 0.00-0.04 in every mode, so
     normalization is not corrupting the output at evaluation time.

This script removes federation and nothing else it can avoid: same DefaultCNN g,
same width-20 LeakyReLU MLP critic f, same OptimalMomentObjective(lambda_1=0.1)
that the 412 historical runs used, same femnist_x archive, same structural-MSE
evaluation against true g with validation-only checkpoint selection.

Three arms:

  supervised   g alone, MSE against the true target. Not a causal method and not
               a baseline to report -- it is a ceiling check that answers
               whether this CNN can represent and learn this mapping from pooled
               data at all. If this fails, no federated fix matters.

  gmm_matched  full GMM game at the federated rates (g 0.003, critic x10).
               Pooled data and real batches are the ONLY difference from the
               failing runs, so this isolates federation.

  gmm_upstream full GMM game at CausalML/DeepGMM's X rates (g 5e-6, critic
               ratio 1000, Adam betas (0.5, 0.9) approximating their OAdam).
               Tests the rate imbalance independently of federation.

Read it as: supervised fails -> architecture or data pipeline, and every
federated experiment so far debugged the wrong layer. supervised works but
gmm_matched fails -> the adversarial objective or its rates, not federation.
Both work -> federated fragmentation is the cause and the next move is
participation rate and client count.

Validation-only selection; test metrics are computed after selection and never
used to select. No dataset is written or modified.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "fedgmm/sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(EXAMPLE))

from models.cnn_models import DefaultCNN                      # noqa: E402
from models.mlp_model import MLPModel                         # noqa: E402
from game_objectives.simple_moment_objective import (         # noqa: E402
    OptimalMomentObjective,
)

DATASET = "femnist_x"
LAMBDA_1 = 0.1          # matches objective_lambda_1 in all 412 historical runs
BATCH_SIZE = 256        # real samples, not 23 repeats of 11
EVAL_EVERY = 200        # steps


def load_data(device):
    archive = np.load(EXAMPLE / "data" / DATASET / "main.npz")
    out = {}
    for split in ("train", "dev", "test"):
        out[split] = {
            "x": torch.tensor(archive[f"{split}_x"]).double().to(device),
            "z": torch.tensor(archive[f"{split}_z"]).double().to(device),
            "y": torch.tensor(archive[f"{split}_y"]).double().to(device),
            "g": torch.tensor(archive[f"{split}_g"]).double().to(device),
        }
    return out


def build_models(device):
    g = DefaultCNN(cuda=(device.type == "cuda")).double().to(device)
    f = MLPModel(input_dim=1, layer_widths=[20],
                 activation=nn.LeakyReLU).double().to(device)
    return g, f


@torch.no_grad()
def structural_mse(g, split, batch=2000):
    """Mean squared error of g against the true structural function."""
    g.eval()
    preds = []
    for i in range(0, split["x"].shape[0], batch):
        preds.append(torch.squeeze(g(split["x"][i:i + batch])))
    pred = torch.cat(preds)
    target = torch.squeeze(split["g"])
    mse = float(((pred - target) ** 2).mean())
    r2 = 1.0 - mse / float(target.var(unbiased=False))
    spread = float(pred.std()) / float(target.std())
    return mse, r2, spread


def run_arm(arm, data, device, steps, seed, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    g, f = build_models(device)

    if arm == "supervised":
        opt_g = torch.optim.Adam(g.parameters(), lr=1e-3)
        opt_f = None
        objective = None
    elif arm == "gmm_matched":
        opt_g = torch.optim.SGD(g.parameters(), lr=0.003, momentum=0.0)
        opt_f = torch.optim.SGD(f.parameters(), lr=0.03, momentum=0.0)
        objective = OptimalMomentObjective(lambda_1=LAMBDA_1)
    elif arm == "gmm_upstream":
        opt_g = torch.optim.Adam(g.parameters(), lr=5e-6, betas=(0.5, 0.9))
        opt_f = torch.optim.Adam(f.parameters(), lr=5e-3, betas=(0.5, 0.9))
        objective = OptimalMomentObjective(lambda_1=LAMBDA_1)
    else:
        raise ValueError(arm)

    train = data["train"]
    n = train["x"].shape[0]
    best = {"dev_mse": float("inf"), "step": -1}
    rng = np.random.default_rng(seed)

    for step in range(1, steps + 1):
        idx = torch.tensor(rng.choice(n, BATCH_SIZE, replace=False), device=device)
        xb, zb, yb = train["x"][idx], train["z"][idx], train["y"][idx]

        g.train()
        if arm == "supervised":
            # ceiling check only: uses the true target directly
            opt_g.zero_grad()
            loss = ((torch.squeeze(g(xb)) - torch.squeeze(train["g"][idx])) ** 2).mean()
            loss.backward()
            opt_g.step()
        else:
            f.train()
            g_obj, f_obj = objective.calc_objective(g, f, xb, zb, yb)
            # Both gradients are taken from the SAME forward graph before either
            # player steps. The federated inner loop steps g before calling
            # f_obj.backward(), which leaves g's parameters modified in place
            # underneath f's graph; that ordering raises a version-counter error
            # here. Simultaneous updates are the standard reading of the game
            # and avoid depending on that ordering.
            opt_g.zero_grad()
            opt_f.zero_grad()
            g_obj.backward(retain_graph=True)
            f_obj.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
            opt_g.step()
            opt_f.step()

        if step % EVAL_EVERY == 0 or step == steps:
            dev_mse, dev_r2, _ = structural_mse(g, data["dev"])
            if not np.isfinite(dev_mse):
                log.append({"arm": arm, "step": step, "event": "nonfinite"})
                break
            if dev_mse < best["dev_mse"]:
                test_mse, test_r2, spread = structural_mse(g, data["test"])
                best = {"dev_mse": dev_mse, "dev_r2": dev_r2, "step": step,
                        "test_mse": test_mse, "test_r2": test_r2,
                        "test_spread_kept": spread}
            log.append({"arm": arm, "step": step, "dev_mse": dev_mse,
                        "dev_r2": dev_r2})
            print(f"  [{arm}] step {step:6d}  dev_mse={dev_mse:.5f}  dev_R2={dev_r2:+.3f}",
                  flush=True)

    best["arm"] = arm
    best["steps_run"] = steps
    best["effective_epochs"] = steps * BATCH_SIZE / n
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments/highdim_coauthor_protocol_v1"
                                       "/centralized_control_20260921")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  steps={args.steps}  batch={BATCH_SIZE}  seed={args.seed}",
          flush=True)

    data = load_data(device)
    n = data["train"]["x"].shape[0]
    print(f"pooled train samples={n} "
          f"({args.steps * BATCH_SIZE / n:.1f} effective epochs)\n", flush=True)

    # federated reference, for the report
    target = torch.squeeze(data["test"]["g"])
    const_mse = float(target.var(unbiased=False))
    print(f"constant-predictor (learn-nothing) test MSE = {const_mse:.5f}\n", flush=True)

    results, log = [], []
    started = time.time()
    for arm in ("supervised", "gmm_matched", "gmm_upstream"):
        print(f"=== {arm} ===", flush=True)
        t0 = time.time()
        best = run_arm(arm, data, device, args.steps, args.seed, log)
        best["wall_seconds"] = time.time() - t0
        results.append(best)
        print(f"  -> best dev at step {best['step']}: "
              f"test_mse={best.get('test_mse', float('nan')):.5f} "
              f"test_R2={best.get('test_r2', float('nan')):+.3f}\n", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET,
        "question": ("Is the femnist_x collapse federated or fundamental? "
                     "Pooled data, same models/objective/evaluation as the "
                     "federated runs."),
        "batch_size": BATCH_SIZE, "steps": args.steps, "seed": args.seed,
        "lambda_1": LAMBDA_1,
        "constant_predictor_test_mse": const_mse,
        "ceiling_r2": 1.0,
        "federated_reference": {
            "source": "stochastic_eg_confirm_20260910 / paired_dgp_20260919",
            "femnist_x_test_mse_at_best_validation": 0.1741,
            "femnist_x_r2": 0.061,
            "note": "500-round federated FedEG, alpha 0.5, 3 seeds",
        },
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "supervised_arm_caveat": ("Ceiling check only. Trains g directly on the "
                                  "true target, is not a causal estimator, and "
                                  "must not be reported as a result."),
        "results": results,
        "total_wall_seconds": time.time() - started,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    (args.out / "eval_log.json").write_text(json.dumps(log, indent=2))

    print("=== SUMMARY ===")
    print(f"{'arm':14s} {'test MSE':>9s} {'test R2':>8s} {'spread':>7s} {'best step':>10s}")
    for r in results:
        print(f"{r['arm']:14s} {r.get('test_mse', float('nan')):9.5f} "
              f"{r.get('test_r2', float('nan')):+8.3f} "
              f"{r.get('test_spread_kept', float('nan')):7.0%} {r['step']:10d}")
    print(f"\nfederated femnist_x for comparison: test_mse=0.1741  R2=+0.061")
    print(f"constant predictor:                 test_mse={const_mse:.5f}  R2= 0.000")
    print(f"\nwrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
