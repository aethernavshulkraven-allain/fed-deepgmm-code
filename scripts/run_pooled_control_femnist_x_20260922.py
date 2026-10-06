"""Pooled control for the image-X collapse, with its own gradient gate.

Replaces scripts/run_centralized_control_femnist_x_20260921.py, whose GMM arms
are void: it placed two unrestricted ``backward()`` calls before both optimizer
steps, so each network accumulated the gradient of ``L_g + L_f = M + (-M + P)
= P``. The moment terms cancelled exactly (9.2e-16 against a penalty-only
gradient) and both networks trained on the penalty alone. Its conclusions --
"federation ruled out", "BatchNorm ruled out" -- were withdrawn.

The lesson that shapes this file: the federated trainer's gradient tests say
nothing about a freshly written pooled loop, which is exactly where that defect
happened. So this harness refuses to run until ``assert_gradient_gate`` passes,
and the gate exercises the SAME function the training loop uses. A gate that
tests a different code path certifies nothing.

What this control can and cannot establish. Pooled success localizes a damaging
transition between the pooled and federated settings; it does not isolate which
component caused it. Pooled failure does not exclude an additional federated
mechanism. It also changes two things at once relative to FedEG -- federation
AND extragradient -- so ``gmm_matched`` here is pooled simultaneous GDA, not
pooled FedEG, and is labelled that way in the output.

Arms:

  supervised      g alone, MSE against the true target. Not a causal method and
                  not a reportable baseline -- a ceiling check for whether this
                  CNN can represent and learn this mapping from pooled data.
  gmm_matched     the game at the federated rates (g 0.003, critic x10), plain
                  SGD, pooled batches.
  gmm_upstream    the game at CausalML/DeepGMM's MNIST-X rates (g 5e-6, critic
                  ratio 1000) using the repo's real OAdam, which implements the
                  optimistic correction theta <- theta - 2u_t + u_{t-1}
                  (optimizers/oadam.py:110-111). The earlier "upstream" arm used
                  plain Adam with matched betas and omitted that correction
                  entirely; its R^2 +0.059 must not be cited as an upstream
                  result. OAdam is used HERE ONLY -- wiring it into FedEG's
                  local factory would add another algorithm change to the system
                  under investigation.

Validation-only selection; test metrics are computed after selection and never
used to select. No dataset is written or modified.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "fedgmm/sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(EXAMPLE))

from game_objectives.simple_moment_objective import OptimalMomentObjective  # noqa: E402
from models.cnn_models import DefaultCNN                                    # noqa: E402
from models.mlp_model import MLPModel                                       # noqa: E402
from optimizers.oadam import OAdam                                          # noqa: E402

DATASET = "femnist_x"
LAMBDA_1 = 0.1
BATCH_SIZE = 256
EVAL_EVERY = 200
CLIP = 1.0
ARCHIVE_PATH = EXAMPLE / "data" / DATASET / "main.npz"


def game_gradients(objective, g, f, xb, zb, yb):
    """Per-player gradients, isolated by construction.

    ``torch.autograd.grad`` against an explicit parameter list cannot deposit
    anything in the other player, regardless of call order -- unlike
    ``backward()``, which accumulates into every parameter in the graph. This is
    the single function the gate and the training loop share.
    """
    g_params = [p for p in g.parameters() if p.requires_grad]
    f_params = [p for p in f.parameters() if p.requires_grad]
    g_obj, f_obj = objective.calc_objective(g, f, xb, zb, yb)
    g_grads = torch.autograd.grad(g_obj, g_params, retain_graph=True, allow_unused=True)
    f_grads = torch.autograd.grad(f_obj, f_params, allow_unused=True)
    return list(g_grads), list(f_grads), g_obj, f_obj


def reference_gradients_via_backward(objective, g, f, xb, zb, yb):
    """The intended gradients, computed by a DIFFERENT mechanism.

    Uses backward() with the interleaved zero_grad the federated trainer and
    upstream both rely on: g's gradient is read before f's backward runs, and
    f's parameters are cleared of g's contamination before f's backward. Two
    independent implementations agreeing is far stronger evidence than either
    one merely differing from a known-bad value.
    """
    g_params = [p for p in g.parameters() if p.requires_grad]
    f_params = [p for p in f.parameters() if p.requires_grad]
    for param in g_params + f_params:
        param.grad = None
    g_obj, f_obj = objective.calc_objective(g, f, xb, zb, yb)
    g_obj.backward(retain_graph=True)
    g_out = [None if p.grad is None else p.grad.detach().clone() for p in g_params]
    for param in f_params:                 # drop what g's backward deposited
        param.grad = None
    f_obj.backward()
    f_out = [None if p.grad is None else p.grad.detach().clone() for p in f_params]
    for param in g_params + f_params:
        param.grad = None
    return g_out, f_out


def assert_gradient_gate(device):
    """Refuse to train unless the harness computes the intended game.

    Requiring only "differs from the cancelled gradient" is necessary but not
    sufficient: returning zeros, or returning the correct gradients with their
    signs reversed, both differ from penalty-only and both would have passed an
    earlier version of this gate. So the primary check is EQUALITY to an
    independent reference, with shape, participation and finiteness checks, and
    the K1 rejection retained as an additional guard.
    """
    torch.manual_seed(101)
    g = DefaultCNN(cuda=False).double().to(device)
    f = MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double().to(device)
    gen = torch.Generator().manual_seed(7)
    xb = torch.rand((6, 1, 28, 28), generator=gen, dtype=torch.float64).to(device)
    zb = torch.randn((6, 1), generator=gen, dtype=torch.float64).to(device)
    yb = torch.randn((6, 1), generator=gen, dtype=torch.float64).to(device)
    objective = OptimalMomentObjective(lambda_1=LAMBDA_1)
    g.train()
    f.train()

    g_params = [p for p in g.parameters() if p.requires_grad]
    f_params = [p for p in f.parameters() if p.requires_grad]

    actual_g, actual_f, _, _ = game_gradients(objective, g, f, xb, zb, yb)
    expected_g, expected_f = reference_gradients_via_backward(
        objective, g, f, xb, zb, yb)

    # 1. Arity, shape, participation, finiteness.
    for label, params, actual, expected in (
        ("g", g_params, actual_g, expected_g),
        ("f", f_params, actual_f, expected_f),
    ):
        if len(actual) != len(params):
            raise RuntimeError(
                f"pooled gradient gate FAILED: {label} returned {len(actual)} "
                f"gradients for {len(params)} parameters")
        for i, (got, want, param) in enumerate(zip(actual, expected, params)):
            if want is not None and got is None:
                raise RuntimeError(
                    f"pooled gradient gate FAILED: {label}[{i}] is None but the "
                    "reference says this parameter participates")
            if got is None:
                continue
            if got.shape != param.shape:
                raise RuntimeError(
                    f"pooled gradient gate FAILED: {label}[{i}] shape {tuple(got.shape)} "
                    f"does not match parameter {tuple(param.shape)}")
            if not torch.isfinite(got).all():
                raise RuntimeError(
                    f"pooled gradient gate FAILED: {label}[{i}] contains non-finite values")

    # 2. Equality to the independent reference. Catches zeros, sign reversal,
    #    scaling, and any other wrong-but-not-cancelled gradient.
    def max_abs_diff(a, b):
        return max((float((x - y).abs().max())
                    for x, y in zip(a, b) if x is not None and y is not None),
                   default=0.0)

    g_err = max_abs_diff(actual_g, expected_g)
    f_err = max_abs_diff(actual_f, expected_f)
    if g_err > 1e-10 or f_err > 1e-10:
        raise RuntimeError(
            "pooled gradient gate FAILED: returned gradients do not match the "
            f"independent backward-ordering reference (g={g_err:.3e}, f={f_err:.3e}). "
            "The harness is not computing the intended game.")

    # 3. The gradients must not be trivially zero, or equality above is vacuous.
    g_scale = max((float(x.abs().max()) for x in actual_g if x is not None), default=0.0)
    f_scale = max((float(x.abs().max()) for x in actual_f if x is not None), default=0.0)
    if g_scale < 1e-12 or f_scale < 1e-12:
        raise RuntimeError(
            f"pooled gradient gate FAILED: gradients are identically zero "
            f"(g={g_scale:.3e}, f={f_scale:.3e})")

    # 4. K1 rejection, retained: the returned gradients must not be the
    #    penalty-only gradient that two unrestricted backward() calls produce.
    g_obj2, f_obj2 = objective.calc_objective(g, f, xb, zb, yb)
    combined = g_obj2 + f_obj2
    cancelled_g = list(torch.autograd.grad(combined, g_params, retain_graph=True,
                                           allow_unused=True))
    cancelled_f = list(torch.autograd.grad(combined, f_params, allow_unused=True))
    g_gap = max_abs_diff(actual_g, cancelled_g)
    f_gap = max_abs_diff(actual_f, cancelled_f)
    if g_gap < 1e-6 or f_gap < 1e-6:
        raise RuntimeError(
            "pooled gradient gate FAILED: the harness is computing the cancelled "
            f"(penalty-only) gradient. max|intended - cancelled| g={g_gap:.3e} "
            f"f={f_gap:.3e}. This is the defect that voided "
            "centralized_control_20260921; refusing to produce more of it.")

    # 5. The loop must APPLY them: one step through clipping and the optimizer
    #    must move parameters against the gradient.
    before = [p.detach().clone() for p in g_params]
    opt = torch.optim.SGD(g_params, lr=0.1, momentum=0.0)
    for param, grad in zip(g_params, actual_g):
        param.grad = None if grad is None else grad.detach().clone()
    torch.nn.utils.clip_grad_norm_(g_params, CLIP)
    applied = [None if p.grad is None else p.grad.detach().clone() for p in g_params]
    opt.step()
    for i, (param, prior, grad) in enumerate(zip(g_params, before, applied)):
        if grad is None:
            continue
        expected_param = prior - 0.1 * grad
        if not torch.allclose(param.detach(), expected_param, atol=1e-12):
            raise RuntimeError(
                f"pooled gradient gate FAILED: parameter {i} did not move by "
                "-lr * clipped gradient; the loop does not apply what it computes")

    return {"g_max_abs_diff_vs_reference": g_err,
            "f_max_abs_diff_vs_reference": f_err,
            "g_gap_vs_cancelled": g_gap, "f_gap_vs_cancelled": f_gap,
            "g_grad_scale": g_scale, "f_grad_scale": f_scale}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(archive_path):
    """Everything an independent reader needs to reproduce or audit this run.

    Metric summaries alone are not auditable: they do not say which code, which
    archive, or which optimizer settings produced them.
    """
    sources = {}
    for rel in ("scripts/run_pooled_control_femnist_x_20260922.py",
                "fedgmm/sp_decentralized_mnist_lr_example/models/cnn_models.py",
                "fedgmm/sp_decentralized_mnist_lr_example/models/mlp_model.py",
                "fedgmm/sp_decentralized_mnist_lr_example/optimizers/oadam.py",
                "fedgmm/sp_decentralized_mnist_lr_example/game_objectives/"
                "simple_moment_objective.py"):
        path = ROOT / rel
        if path.exists():
            sources[rel] = sha256_file(path)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    text=True, capture_output=True,
                                    check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {
        "git_commit": commit, "git_worktree_dirty": dirty,
        "source_sha256": sources,
        "data_archive": str(archive_path),
        "data_sha256": sha256_file(archive_path),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def optimizer_settings(arm, opt_g, opt_f):
    def describe(opt):
        if opt is None:
            return None
        group = opt.param_groups[0]
        return {"class": type(opt).__name__,
                **{k: v for k, v in group.items() if k != "params"}}
    return {"arm": arm, "g": describe(opt_g), "f": describe(opt_f)}


def load_data(device):
    archive = np.load(ARCHIVE_PATH)
    return {
        split: {key: torch.tensor(archive[f"{split}_{key}"]).double().to(device)
                for key in ("x", "z", "y", "g")}
        for split in ("train", "dev", "test")
    }


@torch.no_grad()
def structural_mse(g, split, batch=2000):
    g.eval()
    pred = torch.cat([torch.squeeze(g(split["x"][i:i + batch]))
                      for i in range(0, split["x"].shape[0], batch)])
    target = torch.squeeze(split["g"])
    mse = float(((pred - target) ** 2).mean())
    return (mse, 1.0 - mse / float(target.var(unbiased=False)),
            float(pred.std()) / float(target.std()))


@torch.no_grad()
def predict(g, split, batch=2000):
    """Compact predictions against the true target, for later critic/BN analysis."""
    g.eval()
    pred = torch.cat([torch.squeeze(g(split["x"][i:i + batch]))
                      for i in range(0, split["x"].shape[0], batch)])
    return {"prediction": pred.detach().cpu().numpy().astype("float32"),
            "true_g": torch.squeeze(split["g"]).detach().cpu().numpy().astype("float32")}


def build_optimizers(arm, g, f):
    if arm == "supervised":
        return torch.optim.Adam(g.parameters(), lr=1e-3), None
    if arm == "gmm_matched":
        return (torch.optim.SGD(g.parameters(), lr=0.003, momentum=0.0),
                torch.optim.SGD(f.parameters(), lr=0.03, momentum=0.0))
    if arm == "gmm_upstream":
        return (OAdam(g.parameters(), lr=5e-6, betas=(0.5, 0.9)),
                OAdam(f.parameters(), lr=5e-3, betas=(0.5, 0.9)))
    raise ValueError(arm)


def run_arm(arm, data, device, steps, seed, log):
    torch.manual_seed(seed)
    g = DefaultCNN(cuda=(device.type == "cuda")).double().to(device)
    f = MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double().to(device)
    opt_g, opt_f = build_optimizers(arm, g, f)
    objective = None if arm == "supervised" else OptimalMomentObjective(lambda_1=LAMBDA_1)

    train = data["train"]
    n = train["x"].shape[0]
    best = {"dev_mse": float("inf"), "step": -1}
    selected = {}
    rng = np.random.default_rng(seed)

    for step in range(1, steps + 1):
        idx = torch.tensor(rng.choice(n, BATCH_SIZE, replace=False), device=device)
        xb, zb, yb = train["x"][idx], train["z"][idx], train["y"][idx]
        g.train()

        if arm == "supervised":
            opt_g.zero_grad()
            (((torch.squeeze(g(xb)) - torch.squeeze(train["g"][idx])) ** 2).mean()).backward()
            opt_g.step()
        else:
            f.train()
            g_grads, f_grads, _, _ = game_gradients(objective, g, f, xb, zb, yb)
            g_params = [p for p in g.parameters() if p.requires_grad]
            f_params = [p for p in f.parameters() if p.requires_grad]
            for param, grad in zip(g_params, g_grads):
                param.grad = None if grad is None else grad.detach()
            for param, grad in zip(f_params, f_grads):
                param.grad = None if grad is None else grad.detach()
            torch.nn.utils.clip_grad_norm_(g_params, CLIP)
            torch.nn.utils.clip_grad_norm_(f_params, CLIP)
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
                selected["g_state"] = {k: v.detach().cpu().clone()
                                       for k, v in g.state_dict().items()}
                selected["f_state"] = {k: v.detach().cpu().clone()
                                       for k, v in f.state_dict().items()}
                selected["predictions"] = predict(g, data["test"])
            log.append({"arm": arm, "step": step, "dev_mse": dev_mse, "dev_r2": dev_r2})
            print(f"  [{arm}] step {step:6d}  dev_mse={dev_mse:.5f}  dev_R2={dev_r2:+.3f}",
                  flush=True)

    best.update({"arm": arm, "steps_run": steps,
                 "effective_epochs": steps * BATCH_SIZE / n,
                 "optimizer_settings": optimizer_settings(arm, opt_g, opt_f)})
    artifacts = {
        "selected_g_state": selected.get("g_state"),
        "selected_f_state": selected.get("f_state"),
        "selected_predictions": selected.get("predictions"),
        "final_g_state": {k: v.detach().cpu().clone() for k, v in g.state_dict().items()},
        "final_f_state": {k: v.detach().cpu().clone() for k, v in f.state_dict().items()},
        "final_predictions": predict(g, data["test"]),
    }
    return best, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--gate-only", action="store_true",
                        help="run the gradient gate and exit, without training")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments/highdim_coauthor_protocol_v1"
                                       "/pooled_control_20260922")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== pooled gradient gate ===", flush=True)
    gate = assert_gradient_gate(device)
    print(f"  PASS: intended vs cancelled gradient differs by "
          f"g={gate['g_gap_vs_cancelled']:.3e} f={gate['f_gap_vs_cancelled']:.3e}\n",
          flush=True)
    if args.gate_only:
        return

    data = load_data(device)
    args.out.mkdir(parents=True, exist_ok=True)
    results, log = [], []
    started = time.time()
    for arm in ("supervised", "gmm_matched", "gmm_upstream"):
        print(f"=== {arm} ===", flush=True)
        t0 = time.time()
        best, artifacts = run_arm(arm, data, device, args.steps, args.seed, log)
        best["wall_seconds"] = time.time() - t0
        results.append(best)

        arm_dir = args.out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"step": best["step"],
                    "g_state_dict": artifacts["selected_g_state"],
                    "f_state_dict": artifacts["selected_f_state"]},
                   arm_dir / "selected_checkpoint.pt")
        torch.save({"step": best["steps_run"],
                    "g_state_dict": artifacts["final_g_state"],
                    "f_state_dict": artifacts["final_f_state"]},
                   arm_dir / "final_checkpoint.pt")
        for name, payload in (("selected_predictions", artifacts["selected_predictions"]),
                              ("final_predictions", artifacts["final_predictions"])):
            if payload is not None:
                np.savez_compressed(arm_dir / f"{name}.npz", arm=arm, **payload)
        print(f"  -> best dev at step {best['step']}: "
              f"test_R2={best.get('test_r2', float('nan')):+.3f}"
              f"  (artifacts in {arm_dir.name}/)\n", flush=True)

    (args.out / "results.json").write_text(json.dumps({
        "dataset": DATASET, "steps": args.steps, "seed": args.seed,
        "batch_size": BATCH_SIZE, "lambda_1": LAMBDA_1,
        "gradient_clip_norm": CLIP,
        "gradient_gate": gate,
        "provenance": provenance(ARCHIVE_PATH),
        "artifacts_per_arm": ["selected_checkpoint.pt", "final_checkpoint.pt",
                              "selected_predictions.npz", "final_predictions.npz"],
        "supersedes": "centralized_control_20260921 (void: penalty-only gradients)",
        "arm_semantics": {
            "gmm_matched": "pooled simultaneous GDA, NOT pooled FedEG -- removes "
                           "federation and extragradient together",
            "gmm_upstream": "real OAdam with the optimistic correction, upstream "
                            "MNIST-X rates",
            "supervised": "ceiling check only; not a causal estimator",
        },
        "selection_metric_source": "validation",
        "test_mse_used_for_selection": False,
        "results": results, "total_wall_seconds": time.time() - started,
    }, indent=2))
    (args.out / "eval_log.json").write_text(json.dumps(log, indent=2))
    print(f"wrote {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
