# import torch
# from torch import nn

# from ...core.alg_frame.client_trainer import ClientTrainer
# from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
# import logging
# import copy
# import logging
# import random
# import math
# from optimizers import fedoptimizer
# import itertools
# # from functorch import grad_and_value, make_functional, vmap


# class ModelTrainerCLS(ClientTrainer):
#     def get_g_model_params(self):
#         return self.g.state_dict()
    
#     def get_f_model_params(self):
#         return self.f.state_dict()
    
#     def get_model_params(self):
#         return self.reg_model.cpu().state_dict()

#     def set_model_params(self, model_parameters):
#         self.reg_model.load_state_dict(model_parameters)
#         self.reg_model = self.reg_model.train()  

#     def set_g_model_params(self, model_parameters):
#         new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
#         self.g.load_state_dict(new_state_dict)
#         self.g = self.g.train()
        
#     def set_f_model_params(self, model_parameters):
#         new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
#         self.f.load_state_dict(new_state_dict)
#         self.f = self.f.train()
        
#     def train(self, client_data, device, args):
#         model = self.reg_model
#         # model = model.load_state_dict(self.get_model_params())
#         model.to(device)
#         model.train()

#         # train and update
#         criterion = nn.MSELoss().to(device)  # pylint: disable=E1102
#         if args.client_optimizer == "sgd":
#             optimizer = torch.optim.SGD(
#                 filter(lambda p: p.requires_grad, self.reg_model.parameters()),
#                 lr=args.learning_rate,
#             )
#         else:
#             optimizer = torch.optim.Adam(
#                 filter(lambda p: p.requires_grad, self.model.parameters()),
#                 lr=args.learning_rate,
#                 weight_decay=args.weight_decay,
#                 amsgrad=True,
#             )

#         epoch_loss = []
#         for epoch in range(args.epochs):
#             batch_loss = []

#             for epoch in range(args.epochs):
#                 for batch in client_data:
#                     x_batch = batch[2]
#                     y_batch = batch[3]
#                     model.zero_grad()
#                     preds = torch.squeeze(model(x_batch))
#                     truee = torch.squeeze(y_batch)
#                     loss = criterion(preds, truee)  # pylint: disable=E1102
#                     loss.backward()
#                     optimizer.step()

#                 # Uncommet this following line to avoid nan loss
#                 # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 # logging.info(
#                 #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 #         epoch,
#                 #         (batch_idx + 1) * args.batch_size,
#                 #         len(train_data) * args.batch_size,
#                 #         100.0 * (batch_idx + 1) / len(train_data),
#                 #         loss.item(),
#                 #     )
#                 # )

#                 batch_loss.append(loss.item())
#             if len(batch_loss) == 0:
#                 epoch_loss.append(0.0)
#             else:
#                 epoch_loss.append(sum(batch_loss) / len(batch_loss))
#             # logging.info(
#             #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
#             #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
#             #     )
#             # )
    

#     # Function to get a snapshot of the model parameters
#     def get_params_snapshot(self, model):
#         return {name: param.clone() for name, param in model.named_parameters()}

#     def compare_params(self, initial_params, model, model_name):
#         changed = False
#         for name, initial_param in initial_params.items():
#             current_param = model.state_dict()[name]
#             if not torch.equal(current_param, initial_param):
#                 changed = True
#                 print(f"Parameter {name} of model {model_name} has changed.")
#         if not changed:
#             print(f"No parameters of model {model_name} have changed.")
    
        
#     def train_gmm(self, client_data, device, args):
#         g = self.g
#         f = self.f
        
#         g.to(device)
#         f.to(device)
#         g.train()
#         f.train()
        
#     # Snapshot of parameters before training
#         # initial_g_params = self.get_params_snapshot(g)
#         # initial_f_params = self.get_params_snapshot(f)
        
#     # loop through training data
#         for epoch in range(args.epochs):
#             for batch in client_data:
#                 x_batch = batch[2]
#                 y_batch = batch[3]
#                 z_batch = batch[4]
#                 g_obj, f_obj = self.game_objective.calc_objective(
#                    g,f, x_batch, z_batch, y_batch)
#                 # do single step optimization on f and g
#                 # final_g.zero_grad()
#                 self.g_optimizer.zero_grad()
#                 # optimizer_g.zero_grad()
#                 g_obj.backward(retain_graph=True)
#                 # final_g.step()
#                 self.g_optimizer.step()
#                 # optimizer_g.step()

#                 self.f_optimizer.zero_grad()
#                 # optimizer_f.zero_grad()
#                 # final_f.zero_grad()
#                 f_obj.backward()
#                 # final_f.step()
#                 self.f_optimizer.step()
#                 # optimizer_f.step()
#             # scheduler_g.step()
#             # scheduler_f.step()
                
#         # self.compare_params(initial_g_params, g, "g")
#         # self.compare_params(initial_f_params, f, "f")
        
#         self.set_g_model_params(g.state_dict())
#         self.set_f_model_params(f.state_dict())
        
#     def train_iterations(self, train_data, device, args):
#         model = self.model

#         model.to(device)
#         model.train()

#         # train and update
#         criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
#         if args.client_optimizer == "sgd":
#             optimizer = torch.optim.SGD(
#                 filter(lambda p: p.requires_grad, self.model.parameters()),
#                 lr=args.learning_rate,
#             )
#         else:
#             optimizer = torch.optim.Adam(
#                 filter(lambda p: p.requires_grad, self.model.parameters()),
#                 lr=args.learning_rate,
#                 weight_decay=args.weight_decay,
#                 amsgrad=True,
#             )

#         epoch_loss = []

#         current_steps = 0
#         current_epoch = 0
#         while current_steps < args.local_iterations:
#             batch_loss = []
#             for batch_idx, (x, labels) in enumerate(train_data):
#                 x, labels = x.to(device), labels.to(device)
#                 model.zero_grad()
#                 log_probs = model(x)
#                 labels = labels.long()
#                 loss = criterion(log_probs, labels)  # pylint: disable=E1102
#                 loss.backward()

#                 # Uncommet this following line to avoid nan loss
#                 # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 optimizer.step()
#                 # logging.info(
#                 #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 #         epoch,
#                 #         (batch_idx + 1) * args.batch_size,
#                 #         len(train_data) * args.batch_size,
#                 #         100.0 * (batch_idx + 1) / len(train_data),
#                 #         loss.item(),
#                 #     )
#                 # )
#                 batch_loss.append(loss.item())
#                 current_steps += 1
#                 if current_steps == args.local_iterations:
#                     break
#             current_epoch += 1
#             epoch_loss.append(sum(batch_loss) / len(batch_loss))
#             logging.info(
#                 "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
#                     self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
#                 )
#             )

#     def test(self, test_data, device, args):
#         model = self.model

#         model.to(device)
#         model.eval()

#         metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

#         criterion = nn.CrossEntropyLoss().to(device)

#         with torch.no_grad():
#             for batch_idx, (x, target) in enumerate(test_data):
#                 x = x.to(device)
#                 target = target.to(device)
#                 pred = model(x)
#                 target = target.long()
#                 loss = criterion(pred, target)  # pylint: disable=E1102

#                 _, predicted = torch.max(pred, -1)
#                 correct = predicted.eq(target).sum()

#                 metrics["test_correct"] += correct.item()
#                 metrics["test_loss"] += loss.item() * target.size(0)
#                 metrics["test_total"] += target.size(0)
#         return metrics
import torch
from numerical_failure import NumericalFailure, clip_gradients, ensure_finite
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging
import random
import math
from contextlib import nullcontext
from optimizers import fedoptimizer
import itertools
# from functorch import grad_and_value, make_functional, vmap


class ModelTrainerCLS(ClientTrainer):
    def _profiler(self):
        return getattr(getattr(self, "args", None), "_fedgmm_runtime_profiler", None)

    def _profile_span(self, phase, round_idx=None, client_id=None, detail=""):
        profiler = self._profiler()
        if profiler is None:
            return nullcontext()
        return profiler.span(phase, round_idx=round_idx, client_id=client_id, detail=detail)

    def get_g_model_params(self):
        with self._profile_span("trainer_get_g_params"):
            return self.g.state_dict()
    
    def get_f_model_params(self):
        with self._profile_span("trainer_get_f_params"):
            return self.f.state_dict()
    
    def get_model_params(self):
        with self._profile_span("trainer_get_reg_params"):
            if self.reg_model is None:
                return None
            args = getattr(self, "args", None)
            state_device = str(getattr(args, "auxiliary_regression_state_device", "device")).lower()
            if state_device == "cpu":
                return self.reg_model.cpu().state_dict()
            return self.reg_model.state_dict()

    def set_model_params(self, model_parameters):
        with self._profile_span("trainer_set_reg_params"):
            if self.reg_model is None:
                if model_parameters is not None:
                    raise ValueError("Cannot load regression state when auxiliary regression is disabled")
                return
            self.reg_model.load_state_dict(model_parameters)
            self.reg_model = self.reg_model.train()

    def set_g_model_params(self, model_parameters):
        with self._profile_span("trainer_set_g_params"):
            new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
            self.g.load_state_dict(new_state_dict)
            self.g = self.g.train()
        
    def set_f_model_params(self, model_parameters):
        with self._profile_span("trainer_set_f_params"):
            new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
            self.f.load_state_dict(new_state_dict)
            self.f = self.f.train()
        
    def train(self, client_data, device, args):
        model = self.reg_model
        if model is None:
            raise RuntimeError("Auxiliary regression training is disabled")
        # model = model.load_state_dict(self.get_model_params())
        profiler = self._profiler()
        profile_batches = bool(getattr(profiler, "profile_batches", False))
        with self._profile_span("trainer_reg_model_to_device", client_id=getattr(self, "id", None)):
            model.to(device)
            model.train()

        # train and update
        with self._profile_span("trainer_reg_optimizer_init", client_id=getattr(self, "id", None)):
            criterion = nn.MSELoss().to(device)  # pylint: disable=E1102
            if args.client_optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.learning_rate,
                )
            else:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    amsgrad=True,
                )

        non_blocking = bool(getattr(args, "dataloader_pin_memory", False))
        auxiliary_epochs = int(getattr(args, "auxiliary_regression_epochs", args.epochs))
        if profiler is not None:
            profiler.record_once(
                "trainer_reg_epoch_nesting",
                "trainer_reg_epoch_loop_shape",
                detail=(
                    f"configured_epochs={int(args.epochs)}; "
                    f"auxiliary_regression_epochs={auxiliary_epochs}; "
                    f"effective_passes={auxiliary_epochs}"
                ),
            )
        with self._profile_span("trainer_reg_local_training", client_id=getattr(self, "id", None)):
            for epoch in range(auxiliary_epochs):
                for batch in client_data:
                    if profile_batches:
                        with self._profile_span("trainer_reg_batch_to_device", client_id=getattr(self, "id", None)):
                            x_batch = batch[2].to(device, non_blocking=non_blocking)
                            y_batch = batch[3].to(device, non_blocking=non_blocking)
                    else:
                        x_batch = batch[2].to(device, non_blocking=non_blocking)
                        y_batch = batch[3].to(device, non_blocking=non_blocking)
                    if profile_batches:
                        with self._profile_span("trainer_reg_batch_compute", client_id=getattr(self, "id", None)):
                            model.zero_grad()
                            preds = torch.squeeze(model(x_batch))
                            truee = torch.squeeze(y_batch)
                            loss = criterion(preds, truee)  # pylint: disable=E1102
                            loss.backward()
                            optimizer.step()
                    else:
                        model.zero_grad()
                        preds = torch.squeeze(model(x_batch))
                        truee = torch.squeeze(y_batch)
                        loss = criterion(preds, truee)  # pylint: disable=E1102
                        loss.backward()
                        optimizer.step()

                # Uncommet this following line to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Avoid per-batch loss.item() here; train_reg does not return
                # the loss and item() forces a CUDA sync in every local step.
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )
    

    # Function to get a snapshot of the model parameters
    def get_params_snapshot(self, model):
        return {name: param.clone() for name, param in model.named_parameters()}

    def compare_params(self, initial_params, model, model_name):
        changed = False
        for name, initial_param in initial_params.items():
            current_param = model.state_dict()[name]
            if not torch.equal(current_param, initial_param):
                changed = True
                print(f"Parameter {name} of model {model_name} has changed.")
        if not changed:
            print(f"No parameters of model {model_name} have changed.")
    
        
    def train_gmm(self, client_data, device, args, g_reference=None):
        fail_fast = bool(getattr(args, "stop_on_numerical_failure", False))
        g = self.g
        f = self.f
        gradient_clip_norm = float(getattr(args, "gradient_clip_norm", 1.0))
        profiler = self._profiler()
        profile_batches = bool(getattr(profiler, "profile_batches", False))
        non_blocking = bool(getattr(args, "dataloader_pin_memory", False))
        
        # Clear optimizer states because the model weights were just updated with global weights!
        # This is critical for stateful optimizers like OGDA, otherwise prev_grad is stale.
        with self._profile_span("trainer_gmm_optimizer_state_clear", client_id=getattr(self, "id", None)):
            self.g_optimizer.state.clear()
            self.f_optimizer.state.clear()
        with self._profile_span("trainer_gmm_model_to_device", client_id=getattr(self, "id", None)):
            g.to(device)
            f.to(device)
            g.train()
            f.train()
            _bn_mode = str(getattr(args, "local_bn_mode", "batch"))
            if _bn_mode == "calibrated":
                # Refresh at THIS phase's starting weights, then hold fixed
                # for the phase. Affine parameters stay trainable.
                self._bn_calibrated = self.calibrate_bn_statistics(
                    g, getattr(args, "_fedgmm_calibration_panel", None))
            self._local_bn_switched = self.apply_local_bn_mode((g, f), _bn_mode)

        # theta~ (paper-aligned objective only): snapshot g's parameters as they
        # are right now -- the global iterate received at the start of this
        # round, before any local steps below mutate them -- and freeze it for
        # the rest of this call. EG correction receives the original round's
        # state explicitly: the live g is already at the server look-ahead.
        # Legacy objectives do not define this method and are unaffected.
        with self._profile_span("trainer_gmm_set_theta_tilde", client_id=getattr(self, "id", None)):
            if hasattr(self.game_objective, "set_theta_tilde"):
                self.game_objective.set_theta_tilde(g, state_dict=g_reference)

    # loop through training data
        # max_local_steps_per_round: 0 (or unset/negative) means unlimited --
        # exactly the pre-existing behavior, byte-for-byte, so the frozen
        # demo campaign (which never sets this) is unaffected. When > 0, the
        # cap counts actual optimizer steps taken (one g-step + one f-step
        # per batch, always together) *per round*, i.e. summed across all
        # args.epochs local epochs, not reset at each epoch boundary --
        # local_epochs>1 with a cap is not part of this campaign's protocol
        # but this keeps the semantics well-defined if it is ever used.
        #
        # Rotation, not a fixed prefix: client_data here is a plain, fixed-
        # order Python list materialized ONCE at data-load time for eICU/zoo
        # datasets (fedml/data/data_loader.py: "Convert DataLoaders to lists
        # of batches"), not a reshuffling DataLoader -- it is the SAME list,
        # in the SAME order, every round this trainer is called. Naively
        # always taking the first max_local_steps batches of that fixed list
        # would train on only those batches for the entire run, permanently
        # excluding the rest of a large client's data. Instead we rotate the
        # starting offset every call, tracked as state on this trainer
        # instance -- one Client (and its deep-copied model_trainer) lives
        # for the whole run, one per real client, see fedavg_api.py's
        # _setup_clients (called once in __init__, not per round) -- so
        # every batch is eventually used, within ceil(n_batches /
        # max_local_steps) calls. Deterministic for a fixed run: this is
        # pure call-count rotation, no RNG involved.
        max_local_steps = int(getattr(args, "max_local_steps_per_round", 0) or 0)
        local_step_count = 0
        training_batches = client_data
        rotation_offset = 0
        rotation_n_batches = 0
        if max_local_steps > 0:
            materialized = client_data if isinstance(client_data, list) else list(client_data)
            rotation_n_batches = len(materialized)
            if rotation_n_batches > 0:
                rotation_offset = int(getattr(self, "_max_local_steps_rotation_offset", 0)) % rotation_n_batches
                training_batches = materialized[rotation_offset:] + materialized[:rotation_offset]
            else:
                training_batches = materialized
        with self._profile_span("trainer_gmm_local_training", client_id=getattr(self, "id", None)):
            for epoch in range(args.epochs):
                if max_local_steps > 0 and local_step_count >= max_local_steps:
                    break
                for batch in training_batches:
                    if max_local_steps > 0 and local_step_count >= max_local_steps:
                        break
                    if profile_batches:
                        with self._profile_span("trainer_gmm_batch_to_device", client_id=getattr(self, "id", None)):
                            x_batch = batch[2].to(device, non_blocking=non_blocking)
                            y_batch = batch[3].to(device, non_blocking=non_blocking)
                            z_batch = batch[4].to(device, non_blocking=non_blocking)
                    else:
                        x_batch = batch[2].to(device, non_blocking=non_blocking)
                        y_batch = batch[3].to(device, non_blocking=non_blocking)
                        z_batch = batch[4].to(device, non_blocking=non_blocking)
                    if profile_batches:
                        with self._profile_span("trainer_gmm_batch_compute", client_id=getattr(self, "id", None)):
                            g_obj, f_obj = self.game_objective.calc_objective(
                               g,f, x_batch, z_batch, y_batch)
                            ensure_finite((g_obj, f_obj), fail_fast, "local gradient objective")
                            # do single step optimization on f and g
                            # final_g.zero_grad()
                            self.g_optimizer.zero_grad()
                            # optimizer_g.zero_grad()
                            g_obj.backward(retain_graph=True)
                            clip_gradients(g.parameters(), gradient_clip_norm, fail_fast)
                            # final_g.step()
                            self.g_optimizer.step()
                            # optimizer_g.step()

                            self.f_optimizer.zero_grad()
                            # optimizer_f.zero_grad()
                            # final_f.zero_grad()
                            f_obj.backward()
                            clip_gradients(f.parameters(), gradient_clip_norm, fail_fast)
                            # final_f.step()
                            self.f_optimizer.step()
                            # optimizer_f.step()
                    else:
                        g_obj, f_obj = self.game_objective.calc_objective(
                           g,f, x_batch, z_batch, y_batch)
                        ensure_finite((g_obj, f_obj), fail_fast, "local gradient objective")
                        # do single step optimization on f and g
                        # final_g.zero_grad()
                        self.g_optimizer.zero_grad()
                        # optimizer_g.zero_grad()
                        g_obj.backward(retain_graph=True)
                        clip_gradients(g.parameters(), gradient_clip_norm, fail_fast)
                        # final_g.step()
                        self.g_optimizer.step()
                        # optimizer_g.step()

                        self.f_optimizer.zero_grad()
                        # optimizer_f.zero_grad()
                        # final_f.zero_grad()
                        f_obj.backward()
                        clip_gradients(f.parameters(), gradient_clip_norm, fail_fast)
                        # final_f.step()
                        self.f_optimizer.step()
                        # optimizer_f.step()
                    local_step_count += 1
            # scheduler_g.step()
            # scheduler_f.step()

        if max_local_steps > 0 and rotation_n_batches > 0:
            # Advance the rotation by exactly how many batches this round
            # actually consumed (== local_step_count here, since one batch
            # is always exactly one step in this loop), so next round picks
            # up where this one left off instead of restarting at batch 0.
            self._max_local_steps_rotation_offset = (
                rotation_offset + local_step_count
            ) % rotation_n_batches

        # self.compare_params(initial_g_params, g, "g")
        # self.compare_params(initial_f_params, f, "f")
        
        self.set_g_model_params(g.state_dict())
        self.set_f_model_params(f.state_dict())

    @staticmethod
    def calibrate_bn_statistics(model, panel):
        """Set BatchNorm buffers to the ACTUAL pre-BN statistics at these weights.

        Not an EMA seeded at variance 1. In ordinary training the stored
        variance is 0.9^n to six decimals for the first ~20 rounds -- pure
        decaying initialization carrying no feature information -- so freezing
        it pins an uninformed prior. This measures the real statistics of the
        tensor entering BatchNorm, at the weights this phase is about to start
        from, and writes them into the buffers.

        The panel is a fixed, reproducible set of TRAINING IMAGES only: no
        validation or test images, no outcomes, no true structural targets.

        This is a DIAGNOSTIC policy, and a shared one: every client calibrates
        on the same panel, so a global statistic crosses clients. That extra
        training-input access is recorded in the run artifacts. A win here is
        evidence about statistic quality, NOT a deployable federated method.

        Returns the number of BatchNorm modules calibrated.
        """
        modules = [m for m in model.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not modules or panel is None:
            return 0
        captured = {}

        def make_hook(mod):
            def hook(_module, inputs):
                captured[mod] = inputs[0].detach()
            return hook

        handles = [m.register_forward_pre_hook(make_hook(m)) for m in modules]
        modes = [(m, m.training) for m in model.modules()]
        try:
            model.eval()          # eval so this pass cannot move the buffers
            with torch.no_grad():
                model(panel.to(next(model.parameters()).device))
        finally:
            for handle in handles:
                handle.remove()
            for module, flag in modes:
                module.training = flag
        calibrated = 0
        with torch.no_grad():
            for module in modules:
                seen = captured.get(module)
                if seen is None:
                    continue
                flat = seen.transpose(0, 1).reshape(seen.shape[1], -1)
                if module.running_mean is not None:
                    module.running_mean.copy_(flat.mean(dim=1))
                if module.running_var is not None:
                    module.running_var.copy_(flat.var(dim=1, unbiased=False))
                calibrated += 1
        return calibrated

    @staticmethod
    def apply_local_bn_mode(models, mode):
        """Optionally run BatchNorm on stored statistics during local updates.

        mode "batch" (default) is ordinary training: BN uses this batch's
        statistics and updates its running estimates. mode "frozen" puts only
        the BatchNorm modules into eval, so they use the stored running
        statistics and do not update them, while every weight -- including the
        BN affine parameters -- stays trainable.

        This is a DIAGNOSTIC, not a proposed fix. It changes the forward map
        and its Jacobian, not merely which statistics are reported, so an
        improvement under it is evidence about a normalization-mediated
        mechanism rather than proof of one. Motivation: the median client holds
        11 samples, so batch statistics are estimated from very few points --
        though note LeakySoftmaxCNN already bypasses BN entirely for
        single-sample batches, so some clients are frozen-statistics already.

        Returns the number of BatchNorm modules switched, so a run can record
        that the intervention actually had something to act on.
        """
        if mode == "batch":
            return 0
        if mode not in ("frozen", "calibrated"):
            raise ValueError(
                "local_bn_mode must be 'batch', 'frozen' or 'calibrated', "
                f"got {mode!r}")
        switched = 0
        for model in models:
            if model is None:
                continue
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    switched += 1
        return switched

    @staticmethod
    def _set_objective_gradients(loss, parameters, retain_graph=False):
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=retain_graph, allow_unused=True
        )
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.detach()

    def train_gmm_eg(self, client_data, device, args, g_reference=None):
        """Apply local ExtraGradient for every FedEG_double client batch."""
        g = self.g.to(device)
        f = self.f.to(device)
        g.train()
        f.train()
        _bn_mode = str(getattr(args, "local_bn_mode", "batch"))
        if _bn_mode == "calibrated":
            # Refresh at THIS phase's starting weights, then hold fixed
            # for the phase. Affine parameters stay trainable.
            self._bn_calibrated = self.calibrate_bn_statistics(
                g, getattr(args, "_fedgmm_calibration_panel", None))
        self._local_bn_switched = self.apply_local_bn_mode((g, f), _bn_mode)
        if not hasattr(self.g_optimizer, "extrapolation") or not hasattr(
            self.f_optimizer, "extrapolation"
        ):
            raise TypeError("FedEG_double requires ExtraGradient optimizers")

        self.g_optimizer.state.clear()
        self.f_optimizer.state.clear()
        g_parameters = [
            parameter for parameter in g.parameters() if parameter.requires_grad
        ]
        f_parameters = [
            parameter for parameter in f.parameters() if parameter.requires_grad
        ]
        gradient_clip_norm = float(getattr(args, "gradient_clip_norm", 1.0))
        non_blocking = bool(getattr(args, "dataloader_pin_memory", False))

        if hasattr(self.game_objective, "set_theta_tilde"):
            self.game_objective.set_theta_tilde(g, state_dict=g_reference)

        for _ in range(args.epochs):
            for batch in client_data:
                x_batch = batch[2].to(device, non_blocking=non_blocking)
                y_batch = batch[3].to(device, non_blocking=non_blocking)
                z_batch = batch[4].to(device, non_blocking=non_blocking)

                self.g_optimizer.zero_grad(set_to_none=True)
                self.f_optimizer.zero_grad(set_to_none=True)
                predictor_g_obj, predictor_f_obj = self.game_objective.calc_objective(
                    g, f, x_batch, z_batch, y_batch
                )
                self._set_objective_gradients(
                    predictor_g_obj, g_parameters, retain_graph=True
                )
                self._set_objective_gradients(predictor_f_obj, f_parameters)
                torch.nn.utils.clip_grad_norm_(g_parameters, gradient_clip_norm)
                torch.nn.utils.clip_grad_norm_(f_parameters, gradient_clip_norm)
                self.g_optimizer.extrapolation()
                self.f_optimizer.extrapolation()

                self.g_optimizer.zero_grad(set_to_none=True)
                self.f_optimizer.zero_grad(set_to_none=True)
                corrector_g_obj, corrector_f_obj = self.game_objective.calc_objective(
                    g, f, x_batch, z_batch, y_batch
                )
                self._set_objective_gradients(
                    corrector_g_obj, g_parameters, retain_graph=True
                )
                self._set_objective_gradients(corrector_f_obj, f_parameters)
                torch.nn.utils.clip_grad_norm_(g_parameters, gradient_clip_norm)
                torch.nn.utils.clip_grad_norm_(f_parameters, gradient_clip_norm)
                self.g_optimizer.step()
                self.f_optimizer.step()

        self.set_g_model_params(g.state_dict())
        self.set_f_model_params(f.state_dict())
        

    @staticmethod
    def _rademacher_directions(parameters):
        """Generate independent SPSA directions with entries in {-1, +1}."""
        return [
            torch.empty_like(parameter).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            for parameter in parameters
        ]


    @staticmethod
    def _apply_perturbation(parameters, directions, scale):
        with torch.no_grad():
            for parameter, direction in zip(parameters, directions):
                parameter.add_(direction, alpha=scale)


    @staticmethod
    def _clip_and_apply_zo_update(parameters, estimates, learning_rate, max_norm=1.0, fail_fast=False):
        if not estimates:
            return
        total_norm = sum(estimate.pow(2).sum() for estimate in estimates).sqrt()
        ensure_finite((total_norm,), fail_fast, "ZO estimated gradient norm")
        clip_scale = min(1.0, max_norm / (total_norm.item() + 1e-12))
        with torch.no_grad():
            for parameter, estimate in zip(parameters, estimates):
                parameter.add_(estimate, alpha=-learning_rate * clip_scale)

    def train_gmm_zo(self, client_data, device, args, g_reference=None):
        """Apply forward-only SPSA updates for the FedZO-EG correction phase."""
        fail_fast = bool(getattr(args, "stop_on_numerical_failure", False))
        g = self.g.to(device)
        f = self.f.to(device)
        g.train()
        f.train()
        _bn_mode = str(getattr(args, "local_bn_mode", "batch"))
        if _bn_mode == "calibrated":
            # Refresh at THIS phase's starting weights, then hold fixed
            # for the phase. Affine parameters stay trainable.
            self._bn_calibrated = self.calibrate_bn_statistics(
                g, getattr(args, "_fedgmm_calibration_panel", None))
        self._local_bn_switched = self.apply_local_bn_mode((g, f), _bn_mode)

        mu = float(getattr(args, "zo_mu", 1e-3))
        num_directions = int(getattr(args, "zo_num_directions", 1))
        if mu <= 0.0:
            raise ValueError("zo_mu must be positive")
        if num_directions < 1:
            raise ValueError("zo_num_directions must be at least 1")

        gradient_clip_norm = float(getattr(args, "gradient_clip_norm", 1.0))
        non_blocking = bool(getattr(args, "dataloader_pin_memory", False))
        g_lr = float(self.g_optimizer.param_groups[0]["lr"])
        f_lr = float(self.f_optimizer.param_groups[0]["lr"])
        g_parameters = [parameter for parameter in g.parameters() if parameter.requires_grad]
        f_parameters = [parameter for parameter in f.parameters() if parameter.requires_grad]

        if hasattr(self.game_objective, "set_theta_tilde"):
            self.game_objective.set_theta_tilde(g, state_dict=g_reference)

        for _ in range(args.epochs):
            for batch in client_data:
                x_batch = batch[2].to(device, non_blocking=non_blocking)
                y_batch = batch[3].to(device, non_blocking=non_blocking)
                z_batch = batch[4].to(device, non_blocking=non_blocking)
                g_estimates = [torch.zeros_like(parameter) for parameter in g_parameters]
                f_estimates = [torch.zeros_like(parameter) for parameter in f_parameters]

                for _ in range(num_directions):
                    g_directions = self._rademacher_directions(g_parameters)
                    f_directions = self._rademacher_directions(f_parameters)

                    self._apply_perturbation(g_parameters, g_directions, mu)
                    self._apply_perturbation(f_parameters, f_directions, mu)
                    with torch.no_grad():
                        g_plus, f_plus = self.game_objective.calc_objective(
                            g, f, x_batch, z_batch, y_batch
                        )
                    ensure_finite((g_plus, f_plus), fail_fast, "ZO positive probe")

                    self._apply_perturbation(g_parameters, g_directions, -2.0 * mu)
                    self._apply_perturbation(f_parameters, f_directions, -2.0 * mu)
                    with torch.no_grad():
                        g_minus, f_minus = self.game_objective.calc_objective(
                            g, f, x_batch, z_batch, y_batch
                        )
                    ensure_finite((g_minus, f_minus), fail_fast, "ZO negative probe")

                    self._apply_perturbation(g_parameters, g_directions, mu)
                    self._apply_perturbation(f_parameters, f_directions, mu)

                    g_coefficient = (g_plus.item() - g_minus.item()) / (2.0 * mu)
                    f_coefficient = (f_plus.item() - f_minus.item()) / (2.0 * mu)
                    for estimate, direction in zip(g_estimates, g_directions):
                        estimate.add_(direction, alpha=g_coefficient / num_directions)
                    for estimate, direction in zip(f_estimates, f_directions):
                        estimate.add_(direction, alpha=f_coefficient / num_directions)

                self._clip_and_apply_zo_update(
                    g_parameters, g_estimates, g_lr, gradient_clip_norm, fail_fast
                )
                self._clip_and_apply_zo_update(
                    f_parameters, f_estimates, f_lr, gradient_clip_norm, fail_fast
                )

        self.set_g_model_params(g.state_dict())
        self.set_f_model_params(f.state_dict())

    def train_iterations(self, train_data, device, args):
        model = self.reg_model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.reg_model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
