"""Critic and BatchNorm gradient diagnostics on saved checkpoints.

Runs on artifacts that already exist -- no training. Every probe restores BN
running statistics, per-module training flags and RNG state, so probes cannot
perturb one another or the checkpoint.

Four measurements, in the order a review specified:

  1. Within-batch critic variation, and feature-block versus output-bias
     gradients. A critic that varies across the evaluation set can be nearly
     constant inside one training batch; if it is, it supplies almost no
     moment signal to g's features while still moving g's output bias.
  2. Actual versus constant versus centered critic signals, with state
     restored between probes, to separate "the critic is uninformative" from
     "the critic has a large constant component".
  3. Pre-clipping gradient norms and the update size that would actually be
     applied, for both players.
  4. Residual moments against an INDEPENDENT instrument basis rather than the
     learned critic. A small learned-critic loss can reflect a weak critic, not
     satisfied conditional moments.

None of these establishes a cause on its own; they are the measurements that
choose the next intervention.
"""

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "fedgmm/sp_decentralized_mnist_lr_example"
sys.path.insert(0, str(EXAMPLE))

from game_objectives.simple_moment_objective import OptimalMomentObjective  # noqa: E402
from models.cnn_models import DefaultCNN                                    # noqa: E402
from models.mlp_model import MLPModel                                       # noqa: E402

LAMBDA_1 = 0.1
CLIP = 1.0


@contextlib.contextmanager
def frozen_observation(models):
    """Observe without changing anything the next probe would see."""
    saved_bn, saved_mode = [], []
    for model in models:
        for module in model.modules():
            saved_mode.append((module, module.training))
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                saved_bn.append((
                    module,
                    None if module.running_mean is None else module.running_mean.clone(),
                    None if module.running_var is None else module.running_var.clone(),
                    None if module.num_batches_tracked is None
                    else module.num_batches_tracked.clone(),
                ))
    rng = torch.get_rng_state()
    try:
        yield
    finally:
        for module, mean, var, count in saved_bn:
            if mean is not None:
                module.running_mean.copy_(mean)
            if var is not None:
                module.running_var.copy_(var)
            if count is not None:
                module.num_batches_tracked.copy_(count)
        for module, was_training in saved_mode:
            module.training = was_training
        torch.set_rng_state(rng)


def block_of(name):
    for prefix in ("cnn.", "linear_1.", "linear_2."):
        if name.startswith(prefix):
            return prefix.rstrip(".")
    if name.startswith("bn."):
        return "bn"
    if name == "linear_3.bias":
        return "linear_3_bias"
    if name.startswith("linear_3."):
        return "linear_3_weight"
    return "other"


def block_grad_norms(model, grads):
    out = {}
    for (name, _), grad in zip(model.named_parameters(), grads):
        if grad is None:
            continue
        key = block_of(name)
        out[key] = out.get(key, 0.0) + float((grad ** 2).sum())
    return {k: float(np.sqrt(v)) for k, v in out.items()}


def critic_signal_probe(g, f, objective, xb, zb, yb, mode):
    """g's gradient under the actual / constant / centered critic."""
    class Wrapped(nn.Module):
        def __init__(self, inner, kind):
            super().__init__()
            self.inner = inner
            self.kind = kind
            with torch.no_grad():
                self.mean = inner(zb).mean()

        def forward(self, z):
            out = self.inner(z)
            if self.kind == "constant":
                return torch.full_like(out, float(self.mean))
            if self.kind == "centered":
                return out - self.mean
            return out

    critic = Wrapped(f, mode)
    g_params = [p for p in g.parameters() if p.requires_grad]
    g_obj, _ = objective.calc_objective(g, critic, xb, zb, yb)
    grads = torch.autograd.grad(g_obj, g_params, allow_unused=True)
    return list(grads)


def independent_moment_test(g, xb, zb, yb, bins=8):
    """E[residual | Z] against a fixed instrument basis, not the learned critic."""
    with torch.no_grad():
        residual = (torch.squeeze(yb) - torch.squeeze(g(xb))).cpu().numpy()
    z = torch.squeeze(zb).cpu().numpy()
    edges = np.quantile(z, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    stats = []
    for i in range(bins):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        if mask.sum() < 2:
            continue
        chunk = residual[mask]
        stats.append({"bin": i, "n": int(mask.sum()),
                      "mean_residual": float(chunk.mean()),
                      "se": float(chunk.std(ddof=1) / np.sqrt(mask.sum()))})
    worst = max(stats, key=lambda s: abs(s["mean_residual"]) / max(s["se"], 1e-12))
    return {"bins": stats,
            "max_abs_mean_residual": max(abs(s["mean_residual"]) for s in stats),
            "worst_bin_t_stat": abs(worst["mean_residual"]) / max(worst["se"], 1e-12),
            "residual_sd": float(residual.std())}


def diagnose(label, g_state, f_state, data, device, batch_size, seed):
    g = DefaultCNN(cuda=(device.type == "cuda")).double().to(device)
    f = MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double().to(device)
    g.load_state_dict(g_state)
    f.load_state_dict(f_state)
    objective = OptimalMomentObjective(lambda_1=LAMBDA_1)

    rng = np.random.default_rng(seed)
    idx = torch.tensor(rng.choice(data["train"]["x"].shape[0], batch_size, replace=False),
                       device=device)
    xb = data["train"]["x"][idx]
    zb = data["train"]["z"][idx]
    yb = data["train"]["y"][idx]

    report = {"label": label, "batch_size": batch_size}
    g_params = [p for p in g.parameters() if p.requires_grad]
    f_params = [p for p in f.parameters() if p.requires_grad]

    # (1) within-batch critic variation
    with frozen_observation([g, f]):
        g.train()
        f.train()
        with torch.no_grad():
            fz = torch.squeeze(f(zb))
            gx = torch.squeeze(g(xb))
            eps = gx - torch.squeeze(yb)
        report["critic_within_batch"] = {
            "mean": float(fz.mean()), "sd": float(fz.std()),
            "abs_mean_over_sd": float(fz.mean().abs() / (fz.std() + 1e-12)),
            "min": float(fz.min()), "max": float(fz.max()),
            "corr_with_residual": float(np.corrcoef(fz.cpu(), eps.cpu())[0, 1]),
        }

    # (1b, 3) block gradients, pre-clip norms and applied update size
    with frozen_observation([g, f]):
        g.train()
        f.train()
        g_obj, f_obj = objective.calc_objective(g, f, xb, zb, yb)
        g_grads = list(torch.autograd.grad(g_obj, g_params, retain_graph=True,
                                           allow_unused=True))
        f_grads = list(torch.autograd.grad(f_obj, f_params, allow_unused=True))
        g_total = float(torch.norm(torch.stack(
            [torch.norm(t, 2) for t in g_grads if t is not None]), 2))
        f_total = float(torch.norm(torch.stack(
            [torch.norm(t, 2) for t in f_grads if t is not None]), 2))
        report["block_grad_norms_g"] = block_grad_norms(g, g_grads)
        report["gradients"] = {
            "g_total_preclip": g_total, "f_total_preclip": f_total,
            "g_clip_factor": min(1.0, CLIP / (g_total + 1e-6)),
            "f_clip_factor": min(1.0, CLIP / (f_total + 1e-6)),
            "g_clip_binds": g_total > CLIP, "f_clip_binds": f_total > CLIP,
            "g_param_norm": float(torch.norm(torch.stack(
                [torch.norm(p, 2) for p in g_params]), 2)),
        }

    # (2) actual vs constant vs centered critic
    signals = {}
    for mode in ("actual", "constant", "centered"):
        with frozen_observation([g, f]):
            g.train()
            f.train()
            grads = critic_signal_probe(g, f, objective, xb, zb, yb, mode)
            signals[mode] = {
                "blocks": block_grad_norms(g, grads),
                "total": float(torch.norm(torch.stack(
                    [torch.norm(t, 2) for t in grads if t is not None]), 2)),
            }
    report["critic_signal"] = signals

    # (4) independent instrument basis
    with frozen_observation([g, f]):
        g.eval()
        report["independent_moments"] = independent_moment_test(g, xb, zb, yb)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments/highdim_coauthor_protocol_v1"
                                       "/critic_bn_diagnostic_20260922")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    archive = np.load(EXAMPLE / "data" / "femnist_x" / "main.npz")
    data = {"train": {k: torch.tensor(archive[f"train_{k}"]).double().to(device)
                      for k in ("x", "z", "y", "g")}}

    pooled = ROOT / "experiments/highdim_coauthor_protocol_v1/pooled_control_20260922"
    targets = []
    for arm in ("gmm_matched", "gmm_upstream"):
        for which in ("selected", "final"):
            path = pooled / arm / f"{which}_checkpoint.pt"
            if path.exists():
                ck = torch.load(path, map_location=device, weights_only=False)
                targets.append((f"pooled/{arm}/{which}", ck["g_state_dict"],
                                ck["f_state_dict"]))

    fed = ROOT / ("experiments/highdim_coauthor_protocol_v1/stochastic_eg_confirm_20260910"
                  "/results/seed31/femnist_x/fed_eg_s/seed_31")
    for run in sorted(fed.glob("segcfg*")):
        ck_path = run / "checkpoints" / "best_validation.pt"
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            targets.append(("federated/fed_eg_s/best_validation",
                            ck["g_state_dict"], ck["f_state_dict"]))

    reports = []
    for label, g_state, f_state in targets:
        print(f"=== {label} ===", flush=True)
        rep = diagnose(label, g_state, f_state, data, device, args.batch_size, args.seed)
        c = rep["critic_within_batch"]
        sig = rep["critic_signal"]
        gr = rep["gradients"]
        im = rep["independent_moments"]
        print(f"  critic within batch: mean={c['mean']:+.4f} sd={c['sd']:.4f} "
              f"|mean|/sd={c['abs_mean_over_sd']:.2f} corr_resid={c['corr_with_residual']:+.3f}")
        print(f"  g grad blocks: " + " ".join(
            f"{k}={v:.3e}" for k, v in sorted(rep["block_grad_norms_g"].items())))
        print(f"  g |grad| preclip={gr['g_total_preclip']:.3e} (clip binds={gr['g_clip_binds']}) "
              f"f |grad| preclip={gr['f_total_preclip']:.3e}")
        print(f"  critic signal total: actual={sig['actual']['total']:.3e} "
              f"constant={sig['constant']['total']:.3e} centered={sig['centered']['total']:.3e}")
        print(f"  independent moments: max|E[resid|Z]|={im['max_abs_mean_residual']:.4f} "
              f"worst-bin t={im['worst_bin_t_stat']:.2f} resid_sd={im['residual_sd']:.4f}")
        print(flush=True)
        reports.append(rep)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnostics.json").write_text(json.dumps({
        "batch_size": args.batch_size, "seed": args.seed,
        "note": "Measurements only. None of these establishes a cause.",
        "reports": reports}, indent=2))
    print(f"wrote {args.out / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
