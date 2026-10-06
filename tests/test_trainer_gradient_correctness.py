"""Phase 1 gate: the real trainer applies the intended game's gradients.

Existing coverage was narrower than it looked.
``tests/test_deterministic_update_ordering.py`` reproduces the ``train_gmm``
sequence BY HAND on scalar ``Linear1D`` models -- it imports the objective and
the optimizers but never imports or calls ``ModelTrainerCLS``. So it shows the
pattern is sound; it does not show the trainer implements that pattern, and it
never exercises the CNN, paper-aligned mode, or real clipping.

These tests call the real trainer methods with the real femnist_x model pair
(DefaultCNN structural model, width-20 LeakyReLU MLP critic), under both
objective modes, with clipping enabled, and compare what the optimizer
actually received against an independently derived reference.

Why this matters concretely: a centralized control written during this
investigation placed two unrestricted ``backward()`` calls before both
optimizer steps. ``backward()`` accumulates into every parameter in the graph,
so each network received the gradient of ``L_g + L_f = M + (-M + P) = P`` --
the moment terms cancelled exactly and both networks trained on the penalty
alone. It scored R^2 -0.45 and was read as evidence about federation for two
days. ``PenaltyOnlyCancellationTest`` is the check that would have caught it in
one second.

The two paths reach isolation differently and both are tested:
  train_gmm     -- ``backward()`` + interleaved ``zero_grad()``; isolation is a
                   property of the ORDERING, so contamination is possible.
  train_gmm_eg  -- ``torch.autograd.grad(loss, explicit_params)``; isolation is
                   structural, but the helper passes ``allow_unused=True`` and
                   writes ``None`` straight into ``.grad``.
"""

import copy
import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, "fedgmm", "sp_decentralized_mnist_lr_example")
sys.path.insert(0, EXAMPLE)

from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS  # noqa: E402
from game_objectives.simple_moment_objective import (  # noqa: E402
    OptimalMomentObjective,
    PaperAlignedMomentObjective,
)
from models.cnn_models import DefaultCNN  # noqa: E402
from models.mlp_model import MLPModel  # noqa: E402
from optimizers.Customsgd import CustomSGD  # noqa: E402
from optimizers.extragradient import ExtraGradient  # noqa: E402

CLIP = 1.0
G_LR = 0.003
F_LR = 0.03


def make_batch(n=6, seed=17):
    """A femnist_x-shaped batch: image treatment, scalar instrument."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 1, 28, 28), generator=generator, dtype=torch.float64)
    z = torch.randn((n, 1), generator=generator, dtype=torch.float64)
    y = torch.randn((n, 1), generator=generator, dtype=torch.float64)
    return (None, None, x, y, z)


class RecordingOptimizer:
    """Captures ``.grad`` at the instant the optimizer consumes it.

    The trainer steps immediately after clipping, so the applied gradient is
    not observable afterwards. Recording at step/extrapolation time captures
    exactly what was applied, including the effect of clipping.
    """

    def __init__(self, base):
        self._base = base
        self.recorded = []

    def _snapshot(self):
        self.recorded.append([
            None if p.grad is None else p.grad.detach().clone()
            for group in self._base.param_groups for p in group["params"]
        ])

    def step(self, *args, **kwargs):
        self._snapshot()
        return self._base.step(*args, **kwargs)

    def extrapolation(self, *args, **kwargs):
        self._snapshot()
        return self._base.extrapolation(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def make_trainer(objective_mode, algorithm="fed_eg", seed=11):
    torch.manual_seed(seed)
    trainer = object.__new__(ModelTrainerCLS)
    trainer.g = DefaultCNN(cuda=False).double()
    trainer.f = MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()
    optimizer_cls = ExtraGradient if algorithm == "fed_eg_double" else CustomSGD
    kwargs = {} if algorithm == "fed_eg_double" else {"momentum": 0.0}
    trainer.g_optimizer = RecordingOptimizer(
        optimizer_cls(trainer.g.parameters(), lr=G_LR, **kwargs))
    trainer.f_optimizer = RecordingOptimizer(
        optimizer_cls(trainer.f.parameters(), lr=F_LR, **kwargs))
    trainer.game_objective = (
        PaperAlignedMomentObjective() if objective_mode == "paper_aligned"
        else OptimalMomentObjective(lambda_1=0.1)
    )
    trainer.args = SimpleNamespace(
        epochs=1, client_optimizer=algorithm, gradient_clip_norm=CLIP,
        stop_on_numerical_failure=True, zo_mu=1e-3, zo_num_directions=2,
        max_local_steps_per_round=0, dataloader_pin_memory=False,
    )
    trainer.id = 0
    return trainer


def clipped(grads, max_norm=CLIP):
    """Apply torch's clip_grad_norm_ semantics to a detached gradient list."""
    grads = [None if gr is None else gr.clone() for gr in grads]
    present = [gr for gr in grads if gr is not None]
    total = torch.norm(torch.stack([torch.norm(gr, 2) for gr in present]), 2)
    factor = max_norm / (total + 1e-6)
    if factor < 1:
        grads = [None if gr is None else gr * factor for gr in grads]
    return grads


def reference_gradients(trainer, batch, mode):
    """Independently derived per-player gradients at the trainer's start state.

    Uses ``torch.autograd.grad`` against an explicit parameter list, so no
    gradient can leak between players regardless of call order -- the property
    the ``train_gmm`` ordering is supposed to achieve by other means.
    """
    g = copy.deepcopy(trainer.g)
    f = copy.deepcopy(trainer.f)
    objective = copy.deepcopy(trainer.game_objective)
    if hasattr(objective, "set_theta_tilde"):
        objective.set_theta_tilde(g)
    _, _, x, y, z = batch
    g.train()
    f.train()
    g_obj, f_obj = objective.calc_objective(g, f, x, z, y)
    g_grads = torch.autograd.grad(g_obj, list(g.parameters()), retain_graph=True,
                                  allow_unused=True)
    f_grads = torch.autograd.grad(f_obj, list(f.parameters()), allow_unused=True)
    if mode == "raw":
        return list(g_grads), list(f_grads)
    return clipped(list(g_grads)), clipped(list(f_grads))


def assert_grads_close(actual, expected, label, places=10):
    assert len(actual) == len(expected), f"{label}: arity {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e is None:
            assert a is None or torch.count_nonzero(a) == 0, f"{label}[{i}] expected no gradient"
            continue
        assert a is not None, f"{label}[{i}] missing gradient"
        assert torch.allclose(a, e, atol=10 ** -places), (
            f"{label}[{i}] max|diff|={float((a - e).abs().max()):.3e}")


class TrainGmmAppliesIntendedGradientsTest(unittest.TestCase):
    """train_gmm's isolation is a property of its zero_grad ordering."""

    def _run(self, objective_mode, profile_batches):
        trainer = make_trainer(objective_mode)
        batch = make_batch()
        expected_g, expected_f = reference_gradients(trainer, batch, "clipped")
        if profile_batches:
            # Exercise the profiled arm, which is byte-identical logic guarded
            # by a span. Patching only one arm would instrument half the runs.
            trainer.args._fedgmm_runtime_profiler = SimpleNamespace(
                profile_batches=True,
                span=lambda *a, **k: __import__("contextlib").nullcontext(),
            )
        trainer.train_gmm([batch], torch.device("cpu"), trainer.args)
        assert_grads_close(trainer.g_optimizer.recorded[0], expected_g,
                           f"g/{objective_mode}/profiled={profile_batches}")
        assert_grads_close(trainer.f_optimizer.recorded[0], expected_f,
                           f"f/{objective_mode}/profiled={profile_batches}")

    def test_legacy_unprofiled(self):
        self._run("legacy", profile_batches=False)

    def test_legacy_profiled_arm(self):
        self._run("legacy", profile_batches=True)

    def test_paper_aligned_unprofiled(self):
        self._run("paper_aligned", profile_batches=False)

    def test_paper_aligned_profiled_arm(self):
        self._run("paper_aligned", profile_batches=True)


def eg_reference(trainer, batch):
    """Independently replay both EG phases from the trainer's start state.

    Gradients are taken with torch.autograd.grad against explicit parameter
    lists, so no gradient can leak between players; the real ExtraGradient
    optimizer is reused as a known-good component. The frozen reference is set
    ONCE from the original g, as the trainer does -- the corrector is evaluated
    at the look-ahead point but must still score against the round base.

    Returns (predictor_g, predictor_f, corrector_g, corrector_f, final_g_state).
    """
    g = copy.deepcopy(trainer.g)
    f = copy.deepcopy(trainer.f)
    objective = copy.deepcopy(trainer.game_objective)
    if hasattr(objective, "set_theta_tilde"):
        objective.set_theta_tilde(g)
    g_params = [p for p in g.parameters() if p.requires_grad]
    f_params = [p for p in f.parameters() if p.requires_grad]
    g_opt = ExtraGradient(g_params, lr=G_LR)
    f_opt = ExtraGradient(f_params, lr=F_LR)
    _, _, x, y, z = batch
    g.train()
    f.train()

    def phase():
        g_obj, f_obj = objective.calc_objective(g, f, x, z, y)
        gg = list(torch.autograd.grad(g_obj, g_params, retain_graph=True, allow_unused=True))
        fg = list(torch.autograd.grad(f_obj, f_params, allow_unused=True))
        for param, grad in zip(g_params, gg):
            param.grad = None if grad is None else grad.clone()
        for param, grad in zip(f_params, fg):
            param.grad = None if grad is None else grad.clone()
        torch.nn.utils.clip_grad_norm_(g_params, CLIP)
        torch.nn.utils.clip_grad_norm_(f_params, CLIP)
        return ([None if p.grad is None else p.grad.detach().clone() for p in g_params],
                [None if p.grad is None else p.grad.detach().clone() for p in f_params])

    predictor_g, predictor_f = phase()
    g_opt.extrapolation()
    f_opt.extrapolation()
    corrector_g, corrector_f = phase()
    g_opt.step()
    f_opt.step()
    return predictor_g, predictor_f, corrector_g, corrector_f, copy.deepcopy(g.state_dict())


class TrainGmmEgAppliesIntendedGradientsTest(unittest.TestCase):
    """The EG path isolates structurally via torch.autograd.grad.

    BOTH phases are checked. Counting two optimizer calls does not validate the
    second update: a review forced the corrector gradients to zero and an
    earlier version of this test still passed, because it compared only
    recorded[0].
    """

    def _run(self, objective_mode):
        trainer = make_trainer(objective_mode, algorithm="fed_eg_double")
        batch = make_batch()
        exp_pg, exp_pf, exp_cg, exp_cf, exp_final = eg_reference(trainer, batch)
        trainer.train_gmm_eg([batch], torch.device("cpu"), trainer.args)

        self.assertEqual(len(trainer.g_optimizer.recorded), 2,
                         "expected a predictor and a corrector step")
        assert_grads_close(trainer.g_optimizer.recorded[0], exp_pg,
                           f"eg-predictor-g/{objective_mode}")
        assert_grads_close(trainer.f_optimizer.recorded[0], exp_pf,
                           f"eg-predictor-f/{objective_mode}")
        assert_grads_close(trainer.g_optimizer.recorded[1], exp_cg,
                           f"eg-corrector-g/{objective_mode}")
        assert_grads_close(trainer.f_optimizer.recorded[1], exp_cf,
                           f"eg-corrector-f/{objective_mode}")

        actual_final = trainer.g.state_dict()
        for key, expected in exp_final.items():
            if not expected.dtype.is_floating_point:
                continue
            self.assertTrue(
                torch.allclose(actual_final[key], expected, atol=1e-10),
                f"{objective_mode}: final g[{key}] diverges from the replayed reference")

    def test_legacy(self):
        self._run("legacy")

    def test_paper_aligned(self):
        self._run("paper_aligned")

    def test_corrector_gradients_are_not_trivially_zero(self):
        """Guards the fixture: if the corrector gradients were zero anyway, the
        comparison above could not distinguish a zeroed corrector from a
        correct one."""
        trainer = make_trainer("legacy", algorithm="fed_eg_double")
        _, _, corrector_g, corrector_f, _ = eg_reference(trainer, make_batch())
        self.assertGreater(
            max(float(gr.abs().max()) for gr in corrector_g if gr is not None), 1e-9)
        self.assertGreater(
            max(float(gr.abs().max()) for gr in corrector_f if gr is not None), 1e-9)


class PenaltyOnlyCancellationTest(unittest.TestCase):
    """The K1 check, in a fixture where the two candidates provably differ.

    Deliberately NOT asserted on every training step: real gradients can
    legitimately coincide or vanish, and a blanket per-step check would produce
    spurious failures and train people to ignore it.
    """

    def _penalty_only_gradients(self, trainer, batch):
        """What both players receive if L_g and L_f are summed into both."""
        g = copy.deepcopy(trainer.g)
        f = copy.deepcopy(trainer.f)
        objective = copy.deepcopy(trainer.game_objective)
        if hasattr(objective, "set_theta_tilde"):
            objective.set_theta_tilde(g)
        _, _, x, y, z = batch
        g.train()
        f.train()
        g_obj, f_obj = objective.calc_objective(g, f, x, z, y)
        combined = g_obj + f_obj
        g_grads = torch.autograd.grad(combined, list(g.parameters()),
                                      retain_graph=True, allow_unused=True)
        f_grads = torch.autograd.grad(combined, list(f.parameters()), allow_unused=True)
        return list(g_grads), list(f_grads)

    def test_fixture_is_non_degenerate(self):
        """The intended and penalty-only gradients must actually differ here,
        or the test below cannot distinguish correct from cancelled."""
        trainer = make_trainer("legacy")
        batch = make_batch()
        intended_g, intended_f = reference_gradients(trainer, batch, "raw")
        penalty_g, penalty_f = self._penalty_only_gradients(trainer, batch)
        g_gap = max(float((a - b).abs().max())
                    for a, b in zip(intended_g, penalty_g) if a is not None)
        f_gap = max(float((a - b).abs().max())
                    for a, b in zip(intended_f, penalty_f) if a is not None)
        self.assertGreater(g_gap, 1e-6, "fixture degenerate: g candidates coincide")
        self.assertGreater(f_gap, 1e-6, "fixture degenerate: f candidates coincide")

    def test_trainer_does_not_apply_the_cancelled_gradient(self):
        trainer = make_trainer("legacy")
        batch = make_batch()
        penalty_g, penalty_f = self._penalty_only_gradients(trainer, batch)
        trainer.train_gmm([batch], torch.device("cpu"), trainer.args)

        applied_g = trainer.g_optimizer.recorded[0]
        expected_if_cancelled = clipped(penalty_g)
        differs = any(
            a is not None and e is not None and not torch.allclose(a, e, atol=1e-10)
            for a, e in zip(applied_g, expected_if_cancelled)
        )
        self.assertTrue(differs, "g received the penalty-only gradient: losses cancelled")


class CancellationDetectorIsNotVacuousTest(unittest.TestCase):
    """A gate that cannot fail is worthless, so reproduce K1 and prove it fires.

    Scope, stated precisely: this is a STANDALONE reproduction of the defective
    sequence -- two unrestricted backward() calls before both optimizer steps --
    asserting that the detector's comparison would report it. It is NOT an
    injected mutation of the production trainer followed by the normal gate
    failing. It shows the comparison is sensitive to the mechanism; it does not
    by itself show the trainer would be caught if it regressed that way.
    Measured here: the applied gradient matches the penalty-only gradient to
    ~1.8e-15, i.e. exactly.
    """

    def test_injected_cancellation_is_detected(self):
        trainer = make_trainer("legacy")
        batch = make_batch()
        _, _, x, y, z = batch

        g, f = trainer.g, trainer.f
        g.train()
        f.train()
        g_obj, f_obj = trainer.game_objective.calc_objective(g, f, x, z, y)
        trainer.g_optimizer.zero_grad()
        trainer.f_optimizer.zero_grad()
        g_obj.backward(retain_graph=True)   # the defect: both losses land
        f_obj.backward()                    # in both parameter sets
        torch.nn.utils.clip_grad_norm_(g.parameters(), CLIP)
        applied_g = [None if p.grad is None else p.grad.detach().clone()
                     for p in g.parameters()]

        penalty_g, _ = PenaltyOnlyCancellationTest()._penalty_only_gradients(
            make_trainer("legacy"), batch)
        expected_if_cancelled = clipped(penalty_g)

        differs = any(
            a is not None and e is not None and not torch.allclose(a, e, atol=1e-10)
            for a, e in zip(applied_g, expected_if_cancelled)
        )
        self.assertFalse(
            differs,
            "injected K1 did not reproduce the cancellation, so the detector "
            "in PenaltyOnlyCancellationTest is not exercised by this fixture")


class ExpectedParticipationTest(unittest.TestCase):
    """_set_objective_gradients passes allow_unused=True and writes None into
    .grad. A blanket prohibition would be wrong -- some parameters legitimately
    do not participate -- so declare which ones must, and check only those."""

    def test_declared_participants_receive_gradients(self):
        trainer = make_trainer("paper_aligned", algorithm="fed_eg_double")
        batch = make_batch()
        g_names = [n for n, _ in trainer.g.named_parameters()]
        f_names = [n for n, _ in trainer.f.named_parameters()]
        trainer.train_gmm_eg([batch], torch.device("cpu"), trainer.args)

        for label, names, recorded in (
            ("g", g_names, trainer.g_optimizer.recorded[0]),
            ("f", f_names, trainer.f_optimizer.recorded[0]),
        ):
            missing = [n for n, gr in zip(names, recorded) if gr is None]
            self.assertEqual(missing, [], f"{label}: expected participants got None")


def zo_reference(trainer, batch, seed, mu, num_directions):
    """Independently reproduce the SPSA estimate and update for BOTH players.

    Rademacher directions come from torch.empty_like(...).bernoulli_(0.5), which
    draws from the global RNG, so seeding identically and consuming in the same
    order (g then f, per direction) reproduces them exactly. The probe forwards
    run in train mode and therefore move BatchNorm buffers; the reference
    performs the same forwards in the same order so the two track.

    Returns (final_g_params, final_f_params, g_estimate, f_estimate).
    """
    g = copy.deepcopy(trainer.g)
    f = copy.deepcopy(trainer.f)
    objective = copy.deepcopy(trainer.game_objective)
    if hasattr(objective, "set_theta_tilde"):
        objective.set_theta_tilde(g)
    g_params = [p for p in g.parameters() if p.requires_grad]
    f_params = [p for p in f.parameters() if p.requires_grad]
    g_est = [torch.zeros_like(p) for p in g_params]
    f_est = [torch.zeros_like(p) for p in f_params]
    _, _, x, y, z = batch
    g.train()
    f.train()

    def shift(params, directions, scale):
        with torch.no_grad():
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=scale)

    torch.manual_seed(seed)
    for _ in range(num_directions):
        gd = [torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0) for p in g_params]
        fd = [torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0) for p in f_params]
        shift(g_params, gd, mu)
        shift(f_params, fd, mu)
        with torch.no_grad():
            g_plus, f_plus = objective.calc_objective(g, f, x, z, y)
        shift(g_params, gd, -2.0 * mu)
        shift(f_params, fd, -2.0 * mu)
        with torch.no_grad():
            g_minus, f_minus = objective.calc_objective(g, f, x, z, y)
        shift(g_params, gd, mu)
        shift(f_params, fd, mu)
        g_coeff = (g_plus.item() - g_minus.item()) / (2.0 * mu)
        f_coeff = (f_plus.item() - f_minus.item()) / (2.0 * mu)
        for est, direction in zip(g_est, gd):
            est.add_(direction, alpha=g_coeff / num_directions)
        for est, direction in zip(f_est, fd):
            est.add_(direction, alpha=f_coeff / num_directions)

    def clip_and_apply(params, estimates, lr):
        total = sum(e.pow(2).sum() for e in estimates).sqrt()
        scale = min(1.0, CLIP / (total.item() + 1e-12))
        with torch.no_grad():
            for param, est in zip(params, estimates):
                param.add_(est, alpha=-lr * scale)

    clip_and_apply(g_params, g_est, G_LR)
    clip_and_apply(f_params, f_est, F_LR)
    return ([p.detach().clone() for p in g_params],
            [p.detach().clone() for p in f_params], g_est, f_est)


class SetObjectiveGradientsContractTest(unittest.TestCase):
    """Phase 1.2: pin the contract of _set_objective_gradients.

    It is four lines, but every one of them carries a property the EG path
    depends on and none of them was covered:

        gradients = torch.autograd.grad(loss, parameters,
                                        retain_graph=retain_graph,
                                        allow_unused=True)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.detach()

    It ASSIGNS rather than accumulates, it detaches, it silently writes None
    for an unused parameter, and it takes gradients against an explicit
    parameter list -- which is what makes the EG path immune by construction to
    the cross-loss accumulation that produced K1 on the backward() path.
    """

    def _setup(self):
        torch.manual_seed(3)
        g = nn.Linear(2, 1).double()
        f = nn.Linear(2, 1).double()
        x = torch.randn(5, 2, dtype=torch.float64)
        return g, f, x

    def test_assigns_rather_than_accumulates(self):
        """A pre-existing .grad must be REPLACED by exactly the new gradient.

        Checking only that the value changed is not enough: 99 + new_gradient
        also differs from 99, so an accumulating implementation passes that
        weaker check. Compare against the independently computed gradient.
        """
        g, _, x = self._setup()
        params = list(g.parameters())
        stale = 99.0
        expected = [gr.detach().clone() for gr in
                    torch.autograd.grad(g(x).sum(), params, retain_graph=False)]
        for param in params:
            param.grad = torch.full_like(param, stale)

        loss = g(x).sum()
        ModelTrainerCLS._set_objective_gradients(loss, params)

        for i, (param, want) in enumerate(zip(params, expected)):
            self.assertTrue(
                torch.allclose(param.grad, want, atol=1e-12),
                f"param {i}: .grad is not exactly the new gradient "
                f"(max|diff|={float((param.grad - want).abs().max()):.3e}); "
                "an accumulating implementation would land at stale + new")
            accumulated = want + stale
            self.assertFalse(
                torch.allclose(param.grad, accumulated, atol=1e-12),
                f"param {i}: .grad equals stale + new -- assignment became accumulation")

    def test_written_gradient_is_detached(self):
        g, _, x = self._setup()
        loss = g(x).sum()
        ModelTrainerCLS._set_objective_gradients(loss, list(g.parameters()))
        for param in g.parameters():
            self.assertIsNone(param.grad.grad_fn, "gradient still carries a graph")
            self.assertFalse(param.grad.requires_grad)

    def test_cross_player_isolation_is_structural(self):
        """The property the backward() path achieves only through ordering.

        Taking g's gradient from a loss that also depends on f must leave f's
        .grad untouched -- this is why the EG path could not have produced K1.
        """
        g, f, x = self._setup()
        for param in f.parameters():
            param.grad = torch.zeros_like(param)
        joint = (g(x) * f(x)).sum()
        ModelTrainerCLS._set_objective_gradients(joint, list(g.parameters()))
        for param in f.parameters():
            self.assertTrue(torch.all(param.grad == 0),
                            "f received gradient from a call scoped to g")

    def test_unused_parameter_receives_none_not_zero(self):
        """allow_unused=True writes None, not a zero tensor. Downstream code
        (clipping, the optimizers) must tolerate that, so pin the behaviour
        rather than assume it is impossible."""
        g, _, x = self._setup()
        spare = nn.Parameter(torch.ones(3, dtype=torch.float64))
        # Give it a stale gradient first: None must be WRITTEN, not merely
        # left unset, or a stale value would survive into the optimizer.
        spare.grad = torch.full_like(spare, 42.0)
        loss = g(x).sum()
        ModelTrainerCLS._set_objective_gradients(loss, list(g.parameters()) + [spare])
        self.assertIsNone(spare.grad, "stale gradient survived on an unused parameter")
        for param in g.parameters():
            self.assertIsNotNone(param.grad)

    def test_retain_graph_contract_supports_the_two_call_sequence(self):
        """The EG path calls it twice on one graph: g with retain_graph=True,
        then f. Without the retain the second call raises, so the ordering is
        load-bearing and worth pinning."""
        g, f, x = self._setup()
        joint = (g(x) * f(x)).sum()
        ModelTrainerCLS._set_objective_gradients(joint, list(g.parameters()),
                                                 retain_graph=True)
        ModelTrainerCLS._set_objective_gradients(joint, list(f.parameters()))
        for param in list(g.parameters()) + list(f.parameters()):
            self.assertIsNotNone(param.grad)

        g2, f2, x2 = self._setup()
        joint2 = (g2(x2) * f2(x2)).sum()
        ModelTrainerCLS._set_objective_gradients(joint2, list(g2.parameters()),
                                                 retain_graph=False)
        with self.assertRaises(RuntimeError):
            ModelTrainerCLS._set_objective_gradients(joint2, list(f2.parameters()))


class ZeroOrderMatchesFiniteDifferenceTest(unittest.TestCase):
    """ZO estimates a gradient by finite differences, so an autograd-equality
    assertion would be the wrong reference entirely. Reproduce the seeded SPSA
    estimate independently instead, for BOTH players.

    An earlier version only checked that some g parameter moved by less than a
    loose bound. A review replaced the entire ZO method with a one-line nudge
    to a single parameter and that version still passed, so it certified
    nothing. These assertions compare the actual post-update parameters of g
    and f against an independently computed SPSA reference.

    Scoped to the legacy objective: paper-aligned adds a third network forward
    per probe, and fed_zo_eg is not the algorithm under investigation.
    """

    SEED = 5
    MU = 1e-3
    DIRECTIONS = 2

    def _run_both(self):
        trainer = make_trainer("legacy")
        trainer.args.zo_mu = self.MU
        trainer.args.zo_num_directions = self.DIRECTIONS
        batch = make_batch(n=4)
        exp_g, exp_f, g_est, f_est = zo_reference(
            trainer, batch, self.SEED, self.MU, self.DIRECTIONS)
        torch.manual_seed(self.SEED)
        trainer.train_gmm_zo([batch], torch.device("cpu"), trainer.args)
        actual_g = [p.detach().clone() for p in trainer.g.parameters() if p.requires_grad]
        actual_f = [p.detach().clone() for p in trainer.f.parameters() if p.requires_grad]
        return actual_g, actual_f, exp_g, exp_f, g_est, f_est

    def test_structural_parameters_match_the_spsa_reference(self):
        actual_g, _, exp_g, _, _, _ = self._run_both()
        for i, (a, e) in enumerate(zip(actual_g, exp_g)):
            self.assertTrue(torch.allclose(a, e, atol=1e-12),
                            f"g[{i}] max|diff|={float((a - e).abs().max()):.3e}")

    def test_critic_parameters_match_the_spsa_reference(self):
        """The critic was entirely unchecked before."""
        _, actual_f, _, exp_f, _, _ = self._run_both()
        for i, (a, e) in enumerate(zip(actual_f, exp_f)):
            self.assertTrue(torch.allclose(a, e, atol=1e-12),
                            f"f[{i}] max|diff|={float((a - e).abs().max()):.3e}")

    def test_reference_is_non_degenerate(self):
        """A zero estimate or a no-op update would make the comparisons above
        pass against a broken implementation."""
        trainer = make_trainer("legacy")
        trainer.args.zo_mu = self.MU
        trainer.args.zo_num_directions = self.DIRECTIONS
        batch = make_batch(n=4)
        before = [p.detach().clone() for p in trainer.g.parameters() if p.requires_grad]
        exp_g, _, g_est, f_est = zo_reference(
            trainer, batch, self.SEED, self.MU, self.DIRECTIONS)
        self.assertGreater(max(float(e.abs().max()) for e in g_est), 1e-9,
                           "SPSA g estimate is identically zero")
        self.assertGreater(max(float(e.abs().max()) for e in f_est), 1e-9,
                           "SPSA f estimate is identically zero")
        moved = max(float((a - b).abs().max()) for a, b in zip(exp_g, before))
        self.assertGreater(moved, 1e-12, "reference update is a no-op")


if __name__ == "__main__":
    unittest.main()
