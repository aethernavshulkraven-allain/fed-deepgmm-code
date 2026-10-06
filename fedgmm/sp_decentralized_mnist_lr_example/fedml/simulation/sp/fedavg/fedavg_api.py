import copy
import json
import logging
import math
import csv
import os
import time
import numpy 
import torch
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from .multiprocess_client import (MultiprocessClientExecutor, SingleGPUMultiprocessClientExecutor, _to_cpu, _to_device)
# from fedml.core.dp.mechanisms import 
# from opacus.layers import 
from model_selection.simple_model_eval import GradientDecentSimpleModelEval
from model_selection_class import FHistoryModelSelectionV3
from game_objectives.simple_moment_objective import (
    OptimalMomentObjective,
    PaperAlignedMomentObjective,
)
from optimizers.oadam import OAdam
from optimizers.Customsgd import CustomSGD
from optimizers.extragradient import ExtraGradient
from optimizers.ogda import OGDA
from optimizers.optimizer_factory import OptimizerFactory
from numerical_failure import NumericalFailure, ensure_finite
from validation_stability import ValidationDeteriorationGuard, write_validation_stop
# from optimizers.optimizer_factory import DPOAdam
from torch.optim import Adam,sgd
from model_selection.simple_model_eval import SGDSimpleModelEval
from model_selection.learning_eval_nostop import \
    FHistoryLearningEvalSGDNoStop
from game_objectives.approximate_psi_objective import approx_psi_eval
# from fedgmm.sp_decentralized_mnist_lr_example.plotting import PlotElement
from plotting import PlotElement
import matplotlib.pyplot as plt
from experiment_utils import (
    AGGREGATION_WEIGHTING_CHOICES,
    DEFAULT_SERVER_BUFFER_POLICY,
    OBJECTIVE_MODE_CHOICES,
    PAPER_ALIGNED_LAMBDA,
    SERVER_BUFFER_POLICY_CHOICES,
    apply_parameter_server_update,
    resolve_server_learning_rates,
    optional_path,
    append_aggregation_weights_by_round,
    append_mse_by_round,
    append_test_mse_by_round,
    batchnorm_running_var_min,
    check_eicu_aggregation_weighting,
    check_eicu_objective_mode,
    check_max_local_steps_participation,
    client_indices_for_round,
    compute_client_weights,
    config_checksum,
    default_critic_multiplier,
    env_truthy,
    ensure_run_dirs,
    equal_client_moment_violation,
    equal_client_structural_mse,
    get_effective_config,
    moment_violation,
    prepare_run_dir,
    predictions_for_split,
    run_dir_from_config,
    RuntimeProfiler,
    save_predictions_npz,
    save_predictions_npz_compact,
    metric_values_are_finite,
    state_is_finite,
    structural_mse,
    structural_mse_from_predictions,
    now_seconds,
    weighted_average_state_dicts,
    write_effective_config,
    write_aggregation_weights_by_round,
    write_batching_summary,
    write_metrics,
    write_mse_by_round,
    write_per_client_metrics,
    write_test_mse_by_round,
)

def log_results_to_csv(file_path, round_number, mse):
       if os.path.dirname(file_path):
           os.makedirs(os.path.dirname(file_path), exist_ok=True)
       file_exists = os.path.isfile(file_path)
       with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['Round', 'MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Round': round_number, 'MSE': mse}) 
class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        # Defensive initialization of research-specific arguments
        research_args = {
            'video_plotter': False,
            'print_freq': 1,
            'eval_freq': 1,
            'burn_in': 0,
            'max_no_progress': 10000,
            'verbose': True,
            'print_freq_mul': 1
        }
        for arg_name, default_value in research_args.items():
            if not hasattr(self.args, arg_name):
                setattr(self.args, arg_name, default_value)
        if not hasattr(self.args, "critic_multiplier"):
            setattr(self.args, "critic_multiplier", default_critic_multiplier(args.dataset))
        if not hasattr(self.args, "server_learning_rate"):
            setattr(self.args, "server_learning_rate", 1.5)
        if not hasattr(self.args, "server_buffer_policy"):
            setattr(
                self.args,
                "server_buffer_policy",
                DEFAULT_SERVER_BUFFER_POLICY,
            )
        if not hasattr(self.args, "eg_predictor_server_lr"):
            setattr(self.args, "eg_predictor_server_lr", None)
        if not hasattr(self.args, "eg_corrector_server_lr"):
            setattr(self.args, "eg_corrector_server_lr", None)
        if not hasattr(self.args, "eg_predictor_critic_server_lr"):
            setattr(self.args, "eg_predictor_critic_server_lr", None)
        if not hasattr(self.args, "eg_corrector_critic_server_lr"):
            setattr(self.args, "eg_corrector_critic_server_lr", None)
        if not hasattr(self.args, "init_state_override"):
            setattr(self.args, "init_state_override", None)
        for _name in ("init_g_state_override", "init_f_state_override"):
            if not hasattr(self.args, _name):
                setattr(self.args, _name, None)
        if not hasattr(self.args, "checkpoint_rounds"):
            setattr(self.args, "checkpoint_rounds", None)
        if not hasattr(self.args, "zo_mu"):
            setattr(self.args, "zo_mu", 1e-3)
        if not hasattr(self.args, "zo_num_directions"):
            setattr(self.args, "zo_num_directions", 1)
        if not hasattr(self.args, "enable_multiprocessing"):
            setattr(self.args, "enable_multiprocessing", False)
        if not hasattr(self.args, "multiprocessing_num_workers"):
            setattr(self.args, "multiprocessing_num_workers", 0)
        if not hasattr(self.args, "multiprocessing_gpu_ids"):
            setattr(self.args, "multiprocessing_gpu_ids", None)
        if not hasattr(self.args, "client_execution_mode"):
            setattr(self.args, "client_execution_mode", None)
        if not hasattr(self.args, "multiprocessingsinglegpu_num_workers"):
            setattr(self.args, "multiprocessingsinglegpu_num_workers", 2)
        if not hasattr(self.args, "multiprocessingsinglegpu_gpu_id"):
            setattr(self.args, "multiprocessingsinglegpu_gpu_id", None)
        if not hasattr(self.args, "objective_lambda_1"):
            setattr(self.args, "objective_lambda_1", 0.1)
        if not hasattr(self.args, "aggregation_weighting"):
            setattr(self.args, "aggregation_weighting", "sample_size")
        if not hasattr(self.args, "objective_mode"):
            setattr(self.args, "objective_mode", "legacy")
        if not hasattr(self.args, "campaign_role"):
            setattr(self.args, "campaign_role", "")
        if not hasattr(self.args, "output_dir"):
            setattr(self.args, "output_dir", "results")
        if not hasattr(self.args, "run_id"):
            setattr(self.args, "run_id", "0")
        if not hasattr(self.args, "random_seed"):
            setattr(self.args, "random_seed", 0)
        if not hasattr(self.args, "gradient_clip_norm"):
            setattr(self.args, "gradient_clip_norm", 1.0)
        if not hasattr(self.args, "simple_model_selection_epochs"):
            setattr(self.args, "simple_model_selection_epochs", 100)
        if not hasattr(self.args, "f_history_model_selection_epochs"):
            setattr(self.args, "f_history_model_selection_epochs", 60)
        if not hasattr(self.args, "model_selection_batch_size"):
            setattr(self.args, "model_selection_batch_size", 200)
        if not hasattr(self.args, "model_selection_max_samples"):
            setattr(self.args, "model_selection_max_samples", 0)
        if not hasattr(self.args, "skip_model_selection"):
            setattr(self.args, "skip_model_selection", False)
        if not hasattr(self.args, "skip_gmm_eval"):
            setattr(self.args, "skip_gmm_eval", bool(getattr(self.args, "skip_model_selection", False)))
        if not hasattr(self.args, "gmm_eval_proxy"):
            setattr(self.args, "gmm_eval_proxy", "negative_val_mse" if bool(getattr(self.args, "skip_gmm_eval", False)) else "approx_psi")
        if bool(getattr(self.args, "skip_model_selection", False)):
            setattr(self.args, "skip_gmm_eval", True)
            setattr(self.args, "gmm_eval_proxy", "negative_val_mse")
        if not hasattr(self.args, "auxiliary_regression"):
            setattr(self.args, "auxiliary_regression", False)
        if not hasattr(self.args, "auxiliary_regression_epochs"):
            setattr(self.args, "auxiliary_regression_epochs", int(getattr(self.args, "epochs", 0)) if bool(self.args.auxiliary_regression) else 0)
        if not hasattr(self.args, "auxiliary_regression_state_device"):
            setattr(self.args, "auxiliary_regression_state_device", "device")
        if not hasattr(self.args, "append_round_csv"):
            setattr(self.args, "append_round_csv", True)
        if not hasattr(self.args, "periodic_checkpoint_interval"):
            setattr(self.args, "periodic_checkpoint_interval", 200)
        if not hasattr(self.args, "enable_legacy_outputs"):
            setattr(self.args, "enable_legacy_outputs", True)
        if not hasattr(self.args, "overwrite"):
            setattr(self.args, "overwrite", False)
        if not hasattr(self.args, "require_multibatch_stochastic"):
            setattr(self.args, "require_multibatch_stochastic", False)
        if not hasattr(self.args, "max_local_steps_per_round"):
            setattr(self.args, "max_local_steps_per_round", 0)
        if not hasattr(self.args, "log_test_mse_by_round"):
            setattr(self.args, "log_test_mse_by_round", False)
        if not hasattr(self.args, "test_mse_used_for_selection"):
            setattr(self.args, "test_mse_used_for_selection", False)
        if not hasattr(self.args, "selection_metric_source"):
            setattr(self.args, "selection_metric_source", "validation")
        if bool(getattr(self.args, "test_mse_used_for_selection", False)):
            raise ValueError("test_mse_used_for_selection must remain false")
        if str(getattr(self.args, "selection_metric_source", "validation")).lower() != "validation":
            raise ValueError("selection_metric_source must be validation")
        _aggregation_weighting = str(getattr(self.args, "aggregation_weighting", "sample_size"))
        if _aggregation_weighting not in AGGREGATION_WEIGHTING_CHOICES:
            raise ValueError(
                f"aggregation_weighting must be one of {AGGREGATION_WEIGHTING_CHOICES}, "
                f"got {_aggregation_weighting!r}"
            )
        _objective_mode = str(getattr(self.args, "objective_mode", "legacy"))
        if _objective_mode not in OBJECTIVE_MODE_CHOICES:
            raise ValueError(
                f"objective_mode must be one of {OBJECTIVE_MODE_CHOICES}, "
                f"got {_objective_mode!r}"
            )
        _server_buffer_policy = str(
            getattr(self.args, "server_buffer_policy", DEFAULT_SERVER_BUFFER_POLICY)
        )
        if _server_buffer_policy not in SERVER_BUFFER_POLICY_CHOICES:
            raise ValueError(
                f"server_buffer_policy must be one of {SERVER_BUFFER_POLICY_CHOICES}, "
                f"got {_server_buffer_policy!r}"
            )
        _campaign_role = str(getattr(self.args, "campaign_role", ""))
        check_eicu_aggregation_weighting(
            getattr(self.args, "dataset", ""), _aggregation_weighting, _campaign_role
        )
        check_eicu_objective_mode(getattr(self.args, "dataset", ""), _objective_mode)
        if bool(getattr(self.args, "require_multibatch_stochastic", False)) and int(getattr(self.args, "batch_size", 0)) <= 0:
            raise ValueError("require_multibatch_stochastic requires a positive batch_size")
        check_max_local_steps_participation(
            getattr(self.args, "client_execution_mode", ""),
            getattr(self.args, "enable_multiprocessing", False),
            getattr(self.args, "max_local_steps_per_round", 0),
            getattr(self.args, "client_num_per_round", 0),
            getattr(self.args, "client_num_in_total", 0),
        )
        self.effective_config = get_effective_config(self.args)
        self.run_dir = run_dir_from_config(self.effective_config)
        self.profiler = getattr(self.args, "_fedgmm_runtime_profiler", None)
        if self.profiler is None:
            self.profiler = RuntimeProfiler.from_config(self.effective_config)
            setattr(self.args, "_fedgmm_runtime_profiler", self.profiler)
            self.profiler.start_telemetry()
        self.profiler.configure(self.effective_config, self.run_dir)
        self.profiler.record_environment(self.effective_config, self.run_dir, self.device)
        self.profile_skip_aux_reg = (
            bool(getattr(self.profiler, "enabled", False))
            and env_truthy("FEDGMM_PROFILE_SKIP_AUX_REG", default=False)
        )
        self.auxiliary_regression_enabled = bool(self.effective_config.get("auxiliary_regression", False))
        self.skip_auxiliary_regression = self.profile_skip_aux_reg or not self.auxiliary_regression_enabled
        self.append_round_csv = bool(self.effective_config.get("append_round_csv", True))
        self.periodic_checkpoint_interval = int(self.effective_config.get("periodic_checkpoint_interval", 200))
        # An explicit round list, for densely capturing a window the interval
        # cannot express -- e.g. every round across the observed fork at 8-15.
        _rounds = optional_path(self.effective_config.get("checkpoint_rounds")) or ""
        self.checkpoint_rounds = frozenset(
            int(part) for part in str(_rounds).replace(",", " ").split() if part != "")
        self.skip_model_selection = bool(self.effective_config.get("skip_model_selection", False))
        self.skip_gmm_eval = bool(self.effective_config.get("skip_gmm_eval", self.skip_model_selection))
        self.gmm_eval_proxy = str(self.effective_config.get("gmm_eval_proxy", "negative_val_mse" if self.skip_gmm_eval else "approx_psi"))
        if self.skip_auxiliary_regression:
            self.profiler.record_once(
                "skip_auxiliary_regression",
                "skip_auxiliary_regression",
                detail=(
                    "Auxiliary regression train/aggregate/set disabled; "
                    f"profile_env={self.profile_skip_aux_reg}; config_enabled={self.auxiliary_regression_enabled}"
                ),
            )
        if self.skip_model_selection:
            self.profiler.record_once(
                "skip_model_selection",
                "skip_model_selection",
                detail="Fast final mode: model-selection critic history is bypassed and gmm_eval uses proxy diagnostics.",
            )
        with self.profiler.span("prepare_run_dir"):
            prepare_run_dir(self.run_dir, overwrite=bool(getattr(self.args, "overwrite", False)))
            ensure_run_dirs(self.run_dir)
            write_effective_config(self.run_dir, self.effective_config)
        self.server_learning_rate = float(self.effective_config["server_learning_rate"])
        # Four coefficients: g and f, predictor and corrector. Resolved from the
        # effective config so what the artifact records is what is applied.
        self.server_lrs = resolve_server_learning_rates(self.effective_config)
        # Calibration panel for local_bn_mode="calibrated": a fixed, reproducible
        # set of TRAINING IMAGES only -- no validation or test images, no
        # outcomes, no true structural targets. Built once, from the training
        # partitions the run already holds, with a dedicated generator so the
        # global RNG stream is untouched. Shared across clients by construction,
        # which is the diagnostic's known cost and is recorded below.
        self.calibration_panel = None
        if str(self.effective_config.get("local_bn_mode", "batch")) == "calibrated":
            size = int(self.effective_config.get("calibration_panel_size", 512) or 512)
            pool = []
            for client_idx in sorted(self.train_data_local_dict):
                for batch in self.train_data_local_dict[client_idx]:
                    pool.append(batch[2])
                    if sum(t.shape[0] for t in pool) >= size * 4:
                        break
                if sum(t.shape[0] for t in pool) >= size * 4:
                    break
            if pool:
                stacked = torch.cat(pool, dim=0)
                generator = torch.Generator().manual_seed(
                    int(self.effective_config.get("random_seed", 0)))
                take = min(size, stacked.shape[0])
                idx = torch.randperm(stacked.shape[0], generator=generator)[:take]
                self.calibration_panel = stacked[idx].detach().clone()
                logging.info(
                    f"BN calibration panel: {take} training images "
                    f"(shared across clients; diagnostic only)")
        setattr(self.args, "_fedgmm_calibration_panel", self.calibration_panel)
        self.server_buffer_policy = str(self.effective_config["server_buffer_policy"])
        self.critic_multiplier = float(self.effective_config["critic_multiplier"])
        self.objective_lambda_1 = float(self.effective_config["objective_lambda_1"])
        self.aggregation_weighting = str(self.effective_config["aggregation_weighting"])
        self.objective_mode = str(self.effective_config["objective_mode"])
        self.mse_rows = []
        self.test_mse_rows = []
        self.aggregation_weight_rows = []
        self.log_test_mse_by_round = bool(self.effective_config.get("log_test_mse_by_round", False))
        # closeout plan Phase 1 SS4.4: future image runs (V4/stability/finals)
        # opt into the compact prediction schema instead of the ~10 GiB-
        # across-the-campaign full test-tensor one. Defaults False so every
        # other campaign's predictions.npz schema is completely unaffected.
        self.compact_predictions_only = bool(
            self.effective_config.get("compact_predictions_only", False)
        )
        self.best_validation_mse = float("inf")
        self.best_validation_round = None
        self.best_g_state = None
        self.best_f_state = None
        self.best_reg_state = None
        # Second, independent checkpoint: lowest held-out moment violation.
        # This is the g0-free selection criterion (works even without ground
        # truth, unlike best_validation_mse above), tracked separately so both
        # can be compared -- the two need not select the same round.
        self.best_moment_violation = float("inf")
        self.best_moment_violation_round = None
        self.best_moment_state = None
        self.final_g_state = None
        self.final_f_state = None
        self.final_reg_state = None
        self.best_state = None
        self.final_state = None
        self.delta_g_prev = None
        self.delta_f_prev = None
        self.diverged = False
        self.nonfinite_first_round = None
        self.nonfinite_diagnostics = []
        self.batching_summary_rows = []
        self.start_time = now_seconds()
        [
        train_data_num,
        test_data_num,
        val_data_num,
        train_data_global,
        test_data_global,
        val_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        class_num, 
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = val_data_global
        for split_name, split in [("train", self.train_global), ("val", self.val_global), ("test", self.test_global)]:
            if not hasattr(split, "g") or split.g is None:
                raise ValueError(f"{split_name} split does not expose ground-truth structural g")
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.val_data_num_in_total = val_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_dict = val_data_local_dict


        logging.info("model = {}".format(model))
        self.model = model
        self.device = device

        # Move models to device
        with self.profiler.span("model_to_device"):
            if isinstance(self.model, list):
                for model_list in self.model:
                    if isinstance(model_list, list):
                        for m in model_list:
                            if isinstance(m, torch.nn.Module):
                                m.to(self.device).double()
                    elif isinstance(model_list, torch.nn.Module):
                        model_list.to(self.device).double()
            elif isinstance(self.model, torch.nn.Module):
                self.model.to(self.device).double()

        # FedML's resolved device is authoritative. Keep every global
        # split on the same device as the models before model selection.
        with self.profiler.span("global_tensor_to_device"):
            for split in (self.train_global, self.val_global, self.test_global):
                for tensor_name in ("x", "z", "y", "g", "w"):
                    tensor = getattr(split, tensor_name, None)
                    if torch.is_tensor(tensor):
                        setattr(
                            split,
                            tensor_name,
                            tensor.to(self.device),
                        )
        with self.profiler.span("eval_target_cache"):
            self._train_y_cpu = self.train_global.y.detach().cpu()
            self._train_g_cpu = self.train_global.g.detach().cpu()
            self._val_y_cpu = self.val_global.y.detach().cpu()
            self._val_g_cpu = self.val_global.g.detach().cpu()
            self._test_g_cpu = self.test_global.g.detach().cpu()
            self._test_y_cpu = self.test_global.y.detach().cpu()

        # g_learning_rates = [0.010, 0.050, 0.020]
        ##g_learning_rates =[0.01, 0.001,0.0001,0.0005]
        # g_learning_rates = [0.0005]
        # g_learning_rates=[0.1,0.2,0.5]
        # g_learning_rates =[0.00010, 0.000050, 0.000020]
        # Use the learning rate from the YAML configuration
        g_learning_rates = [self.args.learning_rate]
        if self.objective_mode == "paper_aligned":
            # lambda is fixed at 1/4 for this mode -- not objective_lambda_1,
            # which only tunes the legacy objective. See PaperAlignedMomentObjective.
            game_objectives = [
                PaperAlignedMomentObjective(lambda_1=PAPER_ALIGNED_LAMBDA),
            ]
        else:
            game_objectives = [
                OptimalMomentObjective(lambda_1=self.objective_lambda_1),
            ]
        learning_setups = []
        critic_multiplier = self.critic_multiplier
       
        for g_lr in g_learning_rates:
            for game_objective in game_objectives:
                # learning_setup = {
                #     "g_optimizer_factory": OptimizerFactory(
                #         CustomSGD, lr=float(g_lr), betas=(0.5,0.9)),
                #     "f_optimizer_factory": OptimizerFactory(
                #         CustomSGD, lr=1000*float(g_lr),betas =(0.5,0.9)),
                #     "game_objective": game_objective
                # }
                # learning_setups.append(learning_setup)
                if args.client_optimizer == "ogda":
                    learning_setup = {
                        "g_optimizer_factory": OptimizerFactory(
                            OGDA, lr=float(g_lr)),
                        "f_optimizer_factory": OptimizerFactory(
                            OGDA, lr=critic_multiplier*float(g_lr)),
                        "game_objective": game_objective
                    }
                elif args.client_optimizer == "fed_eg_double":
                    learning_setup = {
                        "g_optimizer_factory": OptimizerFactory(
                            ExtraGradient, lr=float(g_lr)),
                        "f_optimizer_factory": OptimizerFactory(
                            ExtraGradient, lr=critic_multiplier*float(g_lr)),
                        "game_objective": game_objective
                    }
                else:
                    learning_setup = {
                        "g_optimizer_factory": OptimizerFactory(
                            CustomSGD, lr=float(g_lr), momentum=0.0),
                        "f_optimizer_factory": OptimizerFactory(
                            CustomSGD, lr=critic_multiplier*float(g_lr), momentum=0.0),
                        "game_objective": game_objective
                    }
                learning_setups.append(learning_setup)
        
#              learning_setup = {
#                       "g_optimizer_factory": OptimizerFactory(
#                        CustomSGD, lr=float(g_lr), momentum=0.9),  # Using SGD with momentum
#                       "f_optimizer_factory": OptimizerFactory(
#                        CustomSGD, lr=5.0*float(g_lr), momentum=0.9),  # Note the increased learning rate for f_optimizer
#                       "game_objective": game_objective
# }
        if args.client_optimizer == "ogda":
            default_g_opt_factory = OptimizerFactory(OGDA, lr=0.01)
            # default_f_opt_factory = OptimizerFactory(OGDA, lr=0.01)
            default_f_opt_factory = OptimizerFactory(OGDA, lr=critic_multiplier*args.learning_rate)
        elif args.client_optimizer == "fed_eg_double":
            default_g_opt_factory = OptimizerFactory(ExtraGradient, lr=0.01)
            default_f_opt_factory = OptimizerFactory(
                ExtraGradient, lr=critic_multiplier*args.learning_rate
            )
        else:
            default_g_opt_factory = OptimizerFactory(CustomSGD, lr=0.01, momentum=0.0)
            # default_f_opt_factory = OptimizerFactory(CustomSGD, lr=0.01, momentum=0.0)
            default_f_opt_factory = OptimizerFactory(CustomSGD, lr=critic_multiplier*args.learning_rate, momentum=0.0)
            
        # default_g_opt_factory = OptimizerFactory(
        #     sgd, lr=0.0001, betas=(0.5, 0.9))
        # default_f_opt_factory = OptimizerFactory(
        #     sgd, lr=0.001, betas=(0.5, 0.9))
        # default_g_opt_factory = OptimizerFactory(
        #     CustomSGD, lr=0.01, momentum=0.9)
        # default_f_opt_factory = OptimizerFactory(
        #     CustomSGD, lr=0.01, momentum=0.9)
        # g_simple_model_eval = GradientDecentSimpleModelEval(
        #     max_num_iter=100, max_no_progress=10, eval_freq=1)      
        g_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=int(self.args.simple_model_selection_epochs),
            max_no_progress=10,
            batch_size=int(self.args.model_selection_batch_size),
            eval_freq=1)
        f_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=int(self.args.simple_model_selection_epochs),
            max_no_progress=10,
            batch_size=int(self.args.model_selection_batch_size),
            eval_freq=1)
        learning_eval = FHistoryLearningEvalSGDNoStop(
            num_epochs=int(self.args.f_history_model_selection_epochs),
            eval_freq=1,
            batch_size=int(self.args.model_selection_batch_size))
        psi_eval_burn_in = min(
            30,
            max(0, int(self.args.f_history_model_selection_epochs) - 1),
        )
        self.model_selection = FHistoryModelSelectionV3(
            g_model_list=model[0],
            f_model_list=model[1],
            learning_args_list=learning_setups,
            default_g_optimizer_factory=default_g_opt_factory,
            default_f_optimizer_factory=default_f_opt_factory,
            g_simple_model_eval=g_simple_model_eval,
            f_simple_model_eval=f_simple_model_eval,
            learning_eval=learning_eval,
            psi_eval_max_no_progress=10, psi_eval_burn_in=psi_eval_burn_in,
            failure_context={
                "dataset": self.args.dataset,
                "random_seed": self.args.random_seed,
                "simple_model_selection_epochs": self.args.simple_model_selection_epochs,
                "f_history_model_selection_epochs": self.args.f_history_model_selection_epochs,
                "model_selection_batch_size": self.args.model_selection_batch_size,
                "learning_rate": self.args.learning_rate,
                "critic_multiplier": self.critic_multiplier,
                "f_learning_rate": self.critic_multiplier * float(self.args.learning_rate),
                "client_optimizer": self.args.client_optimizer,
            },
        )
        self.default_g_opt_factory = default_g_opt_factory
        # g_simple_model_eval = SGDSimpleModelEval()
        # f_simple_model_eval = SGDSimpleModelEval()
        # learning_eval = FHistoryLearningEvalSGDNoStop(num_epochs=args.epochs_model_selection, eval_freq=args.eval_freq, print_freq=args.print_freq, batch_size=args.batch_size)
        self.reg_model = model[2][0]
        # self.model_selection = FHistoryModelSelectionV3(
        #     g_model_list=model[0],
        #     f_model_list=model[1],
        #     learning_args_list=learning_setups,
        #     default_g_optimizer_factory=default_g_opt_factory,
        #     default_f_optimizer_factory=default_f_opt_factory,
        #     g_simple_model_eval=g_simple_model_eval,
        #     f_simple_model_eval=f_simple_model_eval,
        #     learning_eval=learning_eval,
        #     psi_eval_max_no_progress=self.args.psi_eval_max_no_progress, psi_eval_burn_in=self.args.psi_eval_burn_in)
        # model_linear_sgd_fedavg = torch.load('./fedgmm/sp_decentralized_mnist_lr_example/model_linear_fedsgd')
        # fedavg_sgd = model_linear_sgd_fedavg(self.test_global.x)
        # sgd_plain = model_linear_sgd_plain(self.test_global.x)
        # mse = float(((fedavg_sgd - self.test_global.g) ** 2).mean())
        model_selection_max_samples = int(self.args.model_selection_max_samples)
        if model_selection_max_samples > 0:
            train_limit = min(model_selection_max_samples, int(train_data_global.y.shape[0]))
            dev_limit = min(model_selection_max_samples, int(val_data_global.y.shape[0]))
            # A capped model-selection critic history has ``dev_limit`` rows.
            # Keep the smoke-only validation view aligned with that history so
            # downstream approximate-psi evaluation compares like-sized data.
            for tensor_name in ("x", "z", "y", "g", "w"):
                tensor = getattr(val_data_global, tensor_name, None)
                if torch.is_tensor(tensor):
                    setattr(val_data_global, tensor_name, tensor[:dev_limit])
            self.val_global = val_data_global
            self.val_data_num_in_total = dev_limit
            x_train_selection = train_data_global.x[:train_limit]
            z_train_selection = train_data_global.z[:train_limit]
            y_train_selection = train_data_global.y[:train_limit]
            x_dev_selection = val_data_global.x
            z_dev_selection = val_data_global.z
            y_dev_selection = val_data_global.y
        else:
            x_train_selection = train_data_global.x
            z_train_selection = train_data_global.z
            y_train_selection = train_data_global.y
            x_dev_selection = val_data_global.x
            z_dev_selection = val_data_global.z
            y_dev_selection = val_data_global.y
        if self.skip_model_selection:
            if len(model[0]) != 1 or len(model[1]) != 1 or len(learning_setups) != 1:
                raise ValueError(
                    "skip_model_selection is only valid when the generated config has exactly "
                    "one g model, one f model, and one learning setup"
                )
            with self.profiler.span("model_selection_fast_init"):
                g_global = model[0][0]
                f_global = model[1][0]
                g_global.initialize()
                f_global.initialize()
                learning_args = learning_setups[0]
                dev_f_collection = []
                e_dev_tilde = torch.zeros_like(y_dev_selection.detach().reshape(-1))
            self.skip_gmm_eval = True
            self.gmm_eval_proxy = "negative_val_mse"
        else:
            with self.profiler.span("model_selection"):
                g_global, f_global, learning_args, dev_f_collection, e_dev_tilde = \
                    self.model_selection.do_model_selection(
                        x_train=x_train_selection, z_train=z_train_selection, y_train=y_train_selection,
                        x_dev=x_dev_selection, z_dev=z_dev_selection, y_dev=y_dev_selection, verbose=True)
        
        self.eval_history = []
        self.g_state_history = []
        self.epsilon_dev_history = []
        self.epsilon_train_history = []

        self.g_of_x_train_list = []
        self.g_of_x_dev_list = []

        self.mse_list = []
        self.eval_list = []
        self.dev_f_collection = dev_f_collection
        self.e_dev_tilde = e_dev_tilde
        
        with self.profiler.span("create_model_trainer"):
            self.model_trainer = create_model_trainer([g_global, f_global, model[2][0]], learning_args, args)
        # named_parameters() alone would include a parameter someone froze
        # (requires_grad=False) -- the server optimizer math (learning-rate
        # interpolation, OGDA optimistic extrapolation) must only ever touch
        # keys the client actually trains, for the same reason it must not
        # touch BatchNorm buffers: a frozen parameter should be aggregated
        # like a buffer (direct_client_aggregate), never extrapolated.
        self.g_parameter_keys = frozenset(
            name for name, param in self.model_trainer.g.named_parameters()
            if param.requires_grad
        )
        self.f_parameter_keys = frozenset(
            name for name, param in self.model_trainer.f.named_parameters()
            if param.requires_grad
        )
        batchnorm_running_var_min(self.model_trainer.g, "g")
        batchnorm_running_var_min(self.model_trainer.f, "f")

        with self.profiler.span("setup_clients"):
            self.client_executor = self._create_client_executor()
            if self.client_executor is None:
                self._setup_clients(
                    train_data_local_num_dict, train_data_local_dict,
                    test_data_local_dict, self.model_trainer,
                )
            else:
                self._prime_client_loaders_for_sp_equivalence()

    def _create_client_executor(self):
        execution_mode = self.args.client_execution_mode
        if execution_mode is None or not str(execution_mode).strip():
            execution_mode = "multi_gpu_processes" if self.args.enable_multiprocessing else "sp"
        execution_mode = str(execution_mode).strip().lower()
        if execution_mode not in {"sp", "multi_gpu_processes", "multiprocessingsinglegpu"}:
            raise ValueError(f"Unknown client_execution_mode {execution_mode!r}")
        self.client_execution_mode = execution_mode
        if execution_mode == "sp":
            return None
        if not torch.cuda.is_available():
            logging.warning("Multi-GPU multiprocessing requested without CUDA; using SP execution")
            self.client_execution_mode = "sp"
            return None
        if execution_mode == "multiprocessingsinglegpu":
            gpu_id = self.args.multiprocessingsinglegpu_gpu_id
            if gpu_id is None:
                gpu_id = self.device.index
                if gpu_id is None:
                    gpu_id = torch.cuda.current_device()
            gpu_id = int(gpu_id)
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid multiprocessingsinglegpu_gpu_id {gpu_id}; "
                    f"PyTorch sees {torch.cuda.device_count()} GPUs"
                )
            coordinator_id = self.device.index
            if coordinator_id is None:
                coordinator_id = torch.cuda.current_device()
            if gpu_id != coordinator_id:
                raise ValueError(
                    "multiprocessingsinglegpu_gpu_id must match device_args.gpu_id"
                )
            worker_count = min(
                int(self.args.multiprocessingsinglegpu_num_workers),
                int(self.args.client_num_per_round),
            )
            if worker_count < 2:
                logging.warning(
                    "Single-GPU multiprocessing requires at least two workers; "
                    "using SP execution"
                )
                self.client_execution_mode = "sp"
                return None
            return SingleGPUMultiprocessClientExecutor(
                self.model_trainer, self.args, gpu_id, worker_count
            )
        gpu_ids = self.args.multiprocessing_gpu_ids
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, str):
            gpu_ids = [int(value.strip()) for value in gpu_ids.split(",") if value.strip()]
        else:
            gpu_ids = [int(value) for value in gpu_ids]
        invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= torch.cuda.device_count()]
        if invalid_ids:
            raise ValueError(f"Invalid multiprocessing_gpu_ids {invalid_ids}; PyTorch sees {torch.cuda.device_count()} GPUs")
        worker_limit = int(self.args.multiprocessing_num_workers)
        if worker_limit > 0:
            gpu_ids = gpu_ids[:worker_limit]
        gpu_ids = gpu_ids[:int(self.args.client_num_per_round)]
        if len(gpu_ids) < 2:
            logging.warning("Multi-GPU multiprocessing requires at least two workers; using SP execution")
            self.client_execution_mode = "sp"
            return None
        return MultiprocessClientExecutor(self.model_trainer, self.args, gpu_ids)

    @staticmethod
    def _materialize_client_data(data_loader, num_epochs):
        return [[_to_cpu(batch) for batch in data_loader] for _ in range(int(num_epochs))]

    def _run_primary_client_updates(self, client_indexes, g_global, f_global, reg_global):
        if getattr(self, "client_executor", None) is None:
            gmm_locals = []
            reg_locals = []
            for client_idx, client in zip(client_indexes, self.client_list):
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx], self.test_data_local_dict[client_idx], self.train_data_local_num_dict[client_idx])
                sample_number = client.get_sample_number()
                gmm_locals.append((sample_number, client.train(g_global, f_global)))
                if reg_global is not None:
                    reg_locals.append((sample_number, client.train_reg(reg_global)))
            return gmm_locals, reg_locals
        tasks = []
        for client_idx in client_indexes:
            tasks.append({
                "phase": "primary", "client_idx": int(client_idx),
                "gmm_train_epochs": self._materialize_client_data(self.train_data_local_dict[client_idx], self.args.epochs),
                "reg_train_epochs": self._materialize_client_data(self.train_data_local_dict[client_idx], self.args.auxiliary_regression_epochs) if reg_global is not None else None,
                "sample_number": self.train_data_local_num_dict[client_idx],
                "g_global": _to_cpu(g_global), "f_global": _to_cpu(f_global),
                "reg_global": _to_cpu(reg_global),
            })
        results = self.client_executor.run(tasks)
        gmm_locals = []
        reg_locals = []
        for client_idx, result in zip(client_indexes, results):
            sample_number = self.train_data_local_num_dict[client_idx]
            gmm_locals.append((sample_number, _to_device(result["gmm"], self.device)))
            if reg_global is not None:
                reg_locals.append((sample_number, _to_device(result["regression"], self.device)))
        return gmm_locals, reg_locals


    def _prime_client_loaders_for_sp_equivalence(self):
        """Mirror the loader passes consumed while constructing SP clients."""
        for client_idx in range(int(self.args.client_num_per_round)):
            list(self.train_data_local_dict[client_idx])
            list(self.test_data_local_dict[client_idx])
    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        # numpy.random.seed(0)
        # staggler_ids = random.sample(range(10), 5)
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            # additional_epochs = 0
            # if client_idx in staggler_ids:
            #     additional_epochs=10
            c = Client(
                client_idx,
                list(train_data_local_dict[client_idx])[0],
                list(test_data_local_dict[client_idx])[0],
                # train_data_local_dict[client_idx][0],
                # test_data_local_dict[client_idx][0],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(model_trainer),
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _run_correction_client_updates(
        self, client_indexes, g_global, f_global, g_reference=None
    ):
        """Run phase two on the same sampled clients used by the predictor."""
        if (getattr(self, "objective_mode", "legacy") == "paper_aligned"
                and g_reference is None):
            raise ValueError(
                "Paper-aligned EG correction requires the round-base g reference"
            )
        reference_kwargs = {} if g_reference is None else {"g_reference": g_reference}
        use_zeroth_order = self.args.client_optimizer == "fed_zo_eg"
        if getattr(self, "client_executor", None) is None:
            correction_locals = []
            for client_idx, client in zip(client_indexes, self.client_list):
                if client.client_idx != client_idx:
                    raise RuntimeError("FedEG correction client order differs from predictor phase")
                train_method = client.train_zo if use_zeroth_order else client.train
                correction = train_method(
                    copy.deepcopy(g_global), copy.deepcopy(f_global), **reference_kwargs
                )
                correction_locals.append((client.get_sample_number(), copy.deepcopy(correction)))
            return correction_locals
        tasks = []
        for client_idx in client_indexes:
            tasks.append({
                "phase": "correction", "use_zeroth_order": use_zeroth_order,
                "client_idx": int(client_idx),
                "gmm_train_epochs": self._materialize_client_data(self.train_data_local_dict[client_idx], self.args.epochs),
                "sample_number": self.train_data_local_num_dict[client_idx],
                "g_global": _to_cpu(g_global), "f_global": _to_cpu(f_global),
                **({"g_reference": _to_cpu(g_reference)} if g_reference is not None else {}),
            })
        results = self.client_executor.run(tasks)
        return [
            (self.train_data_local_num_dict[client_idx], _to_device(result["gmm"], self.device))
            for client_idx, result in zip(client_indexes, results)
        ]
    def train(self):
        validation_guard = ValidationDeteriorationGuard(self.effective_config)
        validation_stop = None
        # logging.info("self.model_trainer = {}".format(self.model_trainer))
        # print("Round"+" "+"mse")
        g_global = self.model_trainer.get_g_model_params()
        f_global = self.model_trainer.get_f_model_params()
        reg_global = None if self.skip_auxiliary_regression else self.model_trainer.get_model_params()
        # init_state_override: load another run's post-model-selection initial
        # g/f state, INCLUDING BatchNorm buffers, at exactly this boundary.
        #
        # This point matters. The models are reinitialized after model
        # selection, so overriding their first construction would miss the
        # state training actually starts from -- this is the state written to
        # checkpoints/initial_valid.pt below.
        #
        # Nothing else is touched: no reseeding, no RNG consumption, no change
        # to the partition, the client sequence, or batch ordering. That is the
        # point -- it crosses initial model state against an otherwise
        # untouched execution environment.
        # init_state_override sets both players from one checkpoint;
        # init_g_state_override / init_f_state_override set them independently,
        # which is what lets g and f initialization be crossed. Each transfers
        # that network's COMPLETE state dict, buffers included.
        init_both = optional_path(getattr(self.args, "init_state_override", None))
        init_g = optional_path(getattr(self.args, "init_g_state_override", None)) or init_both
        init_f = optional_path(getattr(self.args, "init_f_state_override", None)) or init_both
        if init_g or init_f:
            def _load(path, key):
                state = torch.load(path, map_location=self.device, weights_only=False)
                if key not in state:
                    raise ValueError(f"init override {path} has no {key}")
                return state[key]
            if init_g:
                self.model_trainer.set_g_model_params(_load(init_g, "g_state_dict"))
            if init_f:
                self.model_trainer.set_f_model_params(_load(init_f, "f_state_dict"))
            g_global = self.model_trainer.get_g_model_params()
            f_global = self.model_trainer.get_f_model_params()
            logging.info(f"init override applied: g<-{init_g} f<-{init_f}")
        if bool(getattr(self.args, "stop_on_numerical_failure", False)):
            ensure_finite(list(g_global.values()) + list(f_global.values()), True, "initial state")
            torch.save({"round": -1, "g_state_dict": g_global, "f_state_dict": f_global},
                       os.path.join(self.run_dir, "checkpoints", "initial_valid.pt"))
        # Written unconditionally: the client-size distribution is evidence about
        # what the run actually trained on, not a numerical-failure artifact. It
        # was previously gated on stop_on_numerical_failure, so a diagnostic run
        # launched without that flag silently lost its partition record.
        partition_audit = {
            "configured_batch_size": int(self.args.batch_size),
            "client_sample_counts": {str(key): int(value) for key, value in self.train_data_local_num_dict.items()},
            "client_batch_counts": {str(key): len(value) for key, value in self.train_data_local_dict.items()},
        }
        with open(os.path.join(self.run_dir, "partition_audit.json"), "w") as handle:
            json.dump(partition_audit, handle, sort_keys=True)
        fedAvg=[]
        # mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        # mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        # mlops.log_round_info(self.args.comm_round, -1)
        current_no_progress = 0
        
        start_round = 0
        if hasattr(self.args, 'resume') and self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                logging.info(f"=> loading checkpoint '{self.args.resume}'")
                checkpoint = torch.load(self.args.resume, map_location=self.device)
                start_round = checkpoint['round'] + 1
                self.model_trainer.set_g_model_params(checkpoint['g_state_dict'])
                self.model_trainer.set_f_model_params(checkpoint['f_state_dict'])
                
                # Initialize self.g and self.f which are normally set in eval_global_model
                self.g = self.model_trainer.g
                self.f = self.model_trainer.f
                
                logging.info(f"=> loaded checkpoint '{self.args.resume}' (round {checkpoint['round']})")
                
                # Update local globals for the first round of resumed training
                g_global = self.model_trainer.get_g_model_params()
                f_global = self.model_trainer.get_f_model_params()
            else:
                logging.error(f"=> no checkpoint found at '{self.args.resume}'")

        training_profile_wall = time.perf_counter()
        training_profile_cpu = time.process_time()
        for round_idx in range(start_round, self.args.comm_round):
            self.args._failure_round = round_idx
            self.args._failure_phase = "predictor"
            round_profile_wall = time.perf_counter()
            round_profile_cpu = time.process_time()

            # logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            w_locals_reg = []
            w_locals_prev = []
            # obj_sum=[]
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            with self.profiler.span("client_sampling", round_idx=round_idx):
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )
            # logging.info("client_indexes = " + str(client_indexes))
            with self.profiler.span("record_stochastic_batching", round_idx=round_idx):
                self._record_stochastic_batching(round_idx, client_indexes)

            with self.profiler.span("client_train", round_idx=round_idx):
                w_locals, w_locals_reg = self._run_primary_client_updates(
                    client_indexes, g_global, f_global,
                    None if self.skip_auxiliary_regression else reg_global,
                )
            if self.skip_auxiliary_regression:
                self.profiler.record(
                    "client_train_reg_skipped", round_idx=round_idx,
                    detail="auxiliary_regression=false",
                )
            with self.profiler.span("record_aggregation_weights", round_idx=round_idx):
                self._record_aggregation_weights(round_idx, client_indexes, w_locals)
            # update global weights
            with self.profiler.span("aggregate_gmm", round_idx=round_idx):
                w_agg = self._aggregate(w_locals)

            with self.profiler.span("server_update_gmm", round_idx=round_idx):
                if self.args.client_optimizer in (
                    "fed_eg", "fed_eg_double", "fed_zo_eg"
                ):
                    g_base = self.model_trainer.get_g_model_params()
                    f_base = self.model_trainer.get_f_model_params()
                    lrs = self.server_lrs
                    predictor_lr = lrs["predictor_g"]
                    corrector_lr = lrs["corrector_g"]
                    predictor_critic_lr = lrs["predictor_f"]
                    corrector_critic_lr = lrs["corrector_f"]

                    g_lookahead, _ = apply_parameter_server_update(
                        g_base,
                        w_agg[0],
                        self.g_parameter_keys,
                        predictor_lr,
                        buffer_policy=self.server_buffer_policy,
                    )
                    f_lookahead, _ = apply_parameter_server_update(
                        f_base,
                        w_agg[1],
                        self.f_parameter_keys,
                        predictor_critic_lr,
                        buffer_policy=self.server_buffer_policy,
                    )

                    with self.profiler.span(
                        "client_correction_gmm", round_idx=round_idx
                    ):
                        self.args._failure_phase = "correction"
                        reference_kwargs = (
                            {"g_reference": g_base}
                            if self.objective_mode == "paper_aligned" else {}
                        )
                        correction_locals = self._run_correction_client_updates(
                            client_indexes, g_lookahead, f_lookahead, **reference_kwargs
                        )
                    correction_agg = self._aggregate(correction_locals)
                    g_new, _ = apply_parameter_server_update(
                        g_base,
                        correction_agg[0],
                        self.g_parameter_keys,
                        corrector_lr,
                        delta_base_state=g_lookahead,
                        buffer_policy=self.server_buffer_policy,
                    )
                    f_new, _ = apply_parameter_server_update(
                        f_base,
                        correction_agg[1],
                        self.f_parameter_keys,
                        corrector_critic_lr,
                        delta_base_state=f_lookahead,
                        buffer_policy=self.server_buffer_policy,
                    )
                    w_global = [g_new, f_new]
                elif self.args.client_optimizer == "ogda":
                    server_lr = self.server_learning_rate

                    # Current weights (theta_t)
                    g_old = self.model_trainer.get_g_model_params()
                    f_old = self.model_trainer.get_f_model_params()

                    use_optimistic_update = not (
                        round_idx == start_round or self.delta_g_prev is None
                    )
                    g_new, delta_g = apply_parameter_server_update(
                        g_old,
                        w_agg[0],
                        self.g_parameter_keys,
                        server_lr,
                        previous_parameter_delta=self.delta_g_prev,
                        optimistic=use_optimistic_update,
                        buffer_policy=self.server_buffer_policy,
                    )
                    f_new, delta_f = apply_parameter_server_update(
                        f_old,
                        w_agg[1],
                        self.f_parameter_keys,
                        server_lr,
                        previous_parameter_delta=self.delta_f_prev,
                        optimistic=use_optimistic_update,
                        buffer_policy=self.server_buffer_policy,
                    )

                    # Store deltas for the next round
                    self.delta_g_prev = delta_g
                    self.delta_f_prev = delta_f
                    w_global = [g_new, f_new]
                else:
                    # w_global = w_agg
                    server_lr = self.server_learning_rate

                    # Current weights (theta_t)
                    g_old = self.model_trainer.get_g_model_params()
                    f_old = self.model_trainer.get_f_model_params()

                    g_new, _ = apply_parameter_server_update(
                        g_old,
                        w_agg[0],
                        self.g_parameter_keys,
                        server_lr,
                        buffer_policy=self.server_buffer_policy,
                    )
                    f_new, _ = apply_parameter_server_update(
                        f_old,
                        w_agg[1],
                        self.f_parameter_keys,
                        server_lr,
                        buffer_policy=self.server_buffer_policy,
                    )

                    w_global = [g_new, f_new]

            w_global_reg = None
            if self.skip_auxiliary_regression:
                self.profiler.record(
                    "aggregate_reg_skipped",
                    round_idx=round_idx,
                    detail="auxiliary_regression=false",
                )
            else:
                with self.profiler.span("aggregate_reg", round_idx=round_idx):
                    w_global_reg = self._aggregate_reg(w_locals_reg)
            with self.profiler.span("set_global_params", round_idx=round_idx):
                self.args._failure_phase = "server_update"
                ensure_finite(list(w_global[0].values()) + list(w_global[1].values()),
                              bool(getattr(self.args, "stop_on_numerical_failure", False)), "server update")
                self.model_trainer.set_g_model_params(w_global[0])
                self.model_trainer.set_f_model_params(w_global[1])
                if not self.skip_auxiliary_regression:
                    self.model_trainer.set_model_params(w_global_reg)
                g_bn_min_running_var = batchnorm_running_var_min(
                    self.model_trainer.g, "g"
                )
                f_bn_min_running_var = batchnorm_running_var_min(
                    self.model_trainer.f, "f"
                )
                g_global = self.model_trainer.get_g_model_params()
                f_global = self.model_trainer.get_f_model_params()
                if not self.skip_auxiliary_regression:
                    reg_global = self.model_trainer.get_model_params()
            # mlops.event("agg", event_started=False, event_value=str(round_idx))

            # at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            with self.profiler.span("eval_global_model", round_idx=round_idx):
                self.args._failure_phase = "evaluation"
                eval_result = self.eval_global_model()
            mse = eval_result["train_mse"]
            obj_train = eval_result["gmm_train_objective"]
            obj_dev = eval_result["gmm_val_objective"]
            curr_eval = eval_result["gmm_eval"]
            max_recent_eval = eval_result["max_recent_gmm_eval"]
            f_of_z_train = eval_result["f_of_z_train"]
            f_of_z_dev = eval_result["f_of_z_dev"]
            state_finite = (
                state_is_finite(self.model_trainer.get_g_model_params())
                and state_is_finite(self.model_trainer.get_f_model_params())
            )
            metrics_finite = metric_values_are_finite(
                eval_result[key]
                for key in (
                    "train_mse",
                    "val_mse",
                    "primary_val_mse",
                    "equal_client_val_mse",
                    "sample_weighted_val_mse",
                    "train_moment_violation",
                    "val_moment_violation",
                    "equal_client_val_moment_violation",
                    "sample_weighted_val_moment_violation",
                    "gmm_train_objective",
                    "gmm_val_objective",
                    "gmm_eval",
                )
            )
            critic_outputs_finite = metric_values_are_finite(
                (f_of_z_train, f_of_z_dev)
            )
            finite = state_finite and metrics_finite and critic_outputs_finite
            diverged = not finite
            if diverged and self.nonfinite_first_round is None:
                self.nonfinite_first_round = round_idx
            if diverged:
                diagnostic = {
                    "round": round_idx,
                    "state_finite": state_finite,
                    "metrics_finite": metrics_finite,
                    "critic_outputs_finite": critic_outputs_finite,
                    "g_bn_min_running_var": g_bn_min_running_var,
                    "f_bn_min_running_var": f_bn_min_running_var,
                }
                self.nonfinite_diagnostics.append(diagnostic)
                logging.error("Nonfinite global evaluation: %s", diagnostic)
            self.diverged = self.diverged or diverged
            mse_row = {
                "round": round_idx,
                "train_mse": eval_result["train_mse"],
                "val_mse": eval_result["val_mse"],
                "primary_val_mse": eval_result["primary_val_mse"],
                "equal_client_val_mse": eval_result["equal_client_val_mse"],
                "train_moment_violation": eval_result["train_moment_violation"],
                "val_moment_violation": eval_result["val_moment_violation"],
                "gmm_train_objective": obj_train,
                "gmm_val_objective": obj_dev,
                "gmm_eval": curr_eval,
                "g_bn_min_running_var": g_bn_min_running_var,
                "f_bn_min_running_var": f_bn_min_running_var,
                "finite": finite,
                "diverged": diverged,
            }
            self.mse_rows.append(mse_row)
            with self.profiler.span("write_mse_by_round", round_idx=round_idx, detail=f"rows={len(self.mse_rows)}"):
                if self.append_round_csv:
                    append_mse_by_round(self.run_dir, mse_row)
                else:
                    write_mse_by_round(self.run_dir, self.mse_rows)
            if self.log_test_mse_by_round:
                with self.profiler.span("test_mse_by_round_eval", round_idx=round_idx):
                    test_mse = structural_mse(self.model_trainer.g, self.test_global)
                test_finite = math.isfinite(float(test_mse))
                test_mse_row = {
                    "round": round_idx,
                    "test_mse": test_mse,
                    "finite": test_finite,
                    "diverged": diverged or not test_finite,
                }
                self.test_mse_rows.append(test_mse_row)
                with self.profiler.span("write_test_mse_by_round", round_idx=round_idx, detail=f"rows={len(self.test_mse_rows)}"):
                    if self.append_round_csv:
                        append_test_mse_by_round(self.run_dir, test_mse_row)
                    else:
                        write_test_mse_by_round(self.run_dir, self.test_mse_rows)
            if diverged and bool(getattr(self.args, "stop_on_numerical_failure", False)):
                raise NumericalFailure(f"Non-finite global evaluation at round {round_idx}: {diagnostic}")
            if getattr(self.args, "enable_legacy_outputs", True):
                with self.profiler.span("legacy_csv_write", round_idx=round_idx):
                    log_results_to_csv(f"csv/{self.args.client_optimizer}_{self.args.dataset}newtrial.csv", round_idx, mse)
            # Primary checkpoint selector: equal-client validation structural
            # MSE for eICU (protocol_v1.md S5.1), which is identical to the
            # legacy pooled val_mse for every other dataset (primary_val_mse
            # == val_mse whenever client_id is unavailable).
            if eval_result["primary_val_mse"] < self.best_validation_mse:
                with self.profiler.span("best_validation_checkpoint", round_idx=round_idx):
                    self.best_validation_mse = eval_result["primary_val_mse"]
                    self.best_validation_round = round_idx
                    self.best_state = self._capture_training_state()
                    self.best_g_state = copy.deepcopy(self.best_state["g_state_dict"])
                    self.best_f_state = copy.deepcopy(self.best_state["f_state_dict"])
                    self.best_reg_state = copy.deepcopy(self.best_state["reg_state_dict"])
                    self._save_checkpoint(
                        os.path.join(self.run_dir, "checkpoints", "best_validation.pt"),
                        round_idx,
                        self.best_state,
                        eval_result,
                        "best_validation",
                    )
            _primary_val_moment_violation = (
                eval_result["equal_client_val_moment_violation"]
                if eval_result["equal_client_val_moment_violation"] is not None
                else eval_result["val_moment_violation"]
            )
            if _primary_val_moment_violation < self.best_moment_violation:
                with self.profiler.span("best_moment_violation_checkpoint", round_idx=round_idx):
                    self.best_moment_violation = _primary_val_moment_violation
                    self.best_moment_violation_round = round_idx
                    self.best_moment_state = self._capture_training_state()
                    self._save_checkpoint(
                        os.path.join(self.run_dir, "checkpoints", "best_moment_violation.pt"),
                        round_idx,
                        self.best_moment_state,
                        eval_result,
                        "best_moment_violation",
                    )

            # Save periodic checkpoints inside the unique run directory.
            if ((self.periodic_checkpoint_interval > 0
                 and round_idx % self.periodic_checkpoint_interval == 0)
                    or round_idx in self.checkpoint_rounds):
                checkpoint_path = os.path.join(self.run_dir, "checkpoints", f"round_{round_idx}.pt")
                with self.profiler.span("periodic_checkpoint", round_idx=round_idx):
                    self._save_checkpoint(
                        checkpoint_path,
                        round_idx,
                        self._capture_training_state(),
                        eval_result,
                        "periodic",
                    )
                if getattr(self.args, "enable_legacy_outputs", True):
                    with self.profiler.span("legacy_checkpoint", round_idx=round_idx):
                        legacy_checkpoint_path = f"checkpoints/{self.args.client_optimizer}_{self.args.dataset}_round_{round_idx}.pt"
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save({
                            "round": round_idx,
                            "g_state_dict": self.g.state_dict(),
                            "f_state_dict": self.f.state_dict(),
                            "mse": mse,
                        }, legacy_checkpoint_path)
            # wandb.log({"round":round_idx,"MSE" :mse})
            # logging.info(f"{round_idx}: {mse:.4f}")
            # print(round_idx,end=" ")
            # print(mse)
            # fedAvg.append(mse)
            if round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    pass
                
                if self.args.video_plotter and round_idx % self.args.print_freq == 0:
                    frame = self.video_plotter.get_new_frame("iter = %d" % round_idx)

                    self.f = self.f.eval()
                    self.g = self.g.eval()

                    # plot f(z)
                    frame.add_plot(PlotElement(
                        self.train_global.w.cpu().numpy(), f_of_z_train.numpy(),
                        "estimated f(z)", normalize=True))

                    # plot g(x)
                    g_of_x_plot = self.epsilon_train_history[-1] + self.train_global.y.cpu()
                    frame.add_plot(PlotElement(self.train_global.w.cpu().numpy(), g_of_x_plot.numpy(),
                                            "fitted g(x)"))

                    self.f = self.f.train()
                    self.g = self.g.train()
                    
                # if round_idx % self.args.print_freq == 0 and self.args.verbose:
                #     mean_eval = numpy.mean(self.eval_history[-self.args.print_freq_mul:])
                #     print("iteration %d, dev-MSE=%f, train-loss=%f,"
                #         " dev-loss=%f, mean-recent-eval=%f"
                #         % (round_idx, mse, obj_train, obj_dev, mean_eval))
                    # wandb.log({"round": round_idx, "MSE": mse, "Train-Loss": obj_train, "Test-Loss": obj_dev, "Objective": mean_eval})

            self.profiler.record(
                "round_total",
                time.perf_counter() - round_profile_wall,
                process_cpu_seconds=time.process_time() - round_profile_cpu,
                round_idx=round_idx,
            )

            validation_stop = validation_guard.update(round_idx, eval_result["primary_val_mse"])
            if validation_stop:
                logging.warning("VALIDATION DETERIORATION STOP: %s", validation_stop)
                break

            # check stopping conditions if we are past burn-in
            if round_idx % self.args.eval_freq == 0 and round_idx >= self.args.burn_in:
                if curr_eval > max_recent_eval:
                    current_no_progress = 0
                else:
                    current_no_progress += 1

                if current_no_progress >= self.args.max_no_progress:
                    break
        self.profiler.record(
            "training_total",
            time.perf_counter() - training_profile_wall,
            process_cpu_seconds=time.process_time() - training_profile_cpu,
        )
        if self.batching_summary_rows:
            write_batching_summary(self.run_dir, self.batching_summary_rows)
        # plot relationship between MSE and eval
        # if self.args.video_plotter:
        #     plt.figure()
        #     data = pandas.DataFrame({"eval": self.eval_list, "mse": self.mse_list})
        #     data.plot.scatter(x="eval", y="mse")
        #     plt.savefig("eval_mse.png")
        finalization_profile_wall = time.perf_counter()
        finalization_profile_cpu = time.process_time()
        max_i = max(range(len(self.eval_history)), key=lambda i_: self.eval_history[i_])
        if self.args.verbose:
            # print("best iteration:", self.args.eval_freq * max_i)
            pass
            # mlops.log_round_info(self.args.comm_round, round_idx)
        best_gmm_eval_round = self.args.eval_freq * max_i
        final_round = round_idx if "round_idx" in locals() else -1
        with self.profiler.span("final_capture_state"):
            self.final_state = self._capture_training_state()
            self.final_g_state = copy.deepcopy(self.final_state["g_state_dict"])
            self.final_f_state = copy.deepcopy(self.final_state["f_state_dict"])
            self.final_reg_state = copy.deepcopy(self.final_state["reg_state_dict"])
            self._load_training_state(self.final_state)
        val_client_id = getattr(self.val_global, "client_id", None)
        test_client_id = getattr(self.test_global, "client_id", None)
        with self.profiler.span("final_validation_test_eval"):
            final_validation_mse = structural_mse(self.model_trainer.g, self.val_global)
            final_test_mse = structural_mse(self.model_trainer.g, self.test_global)
            final_prediction = predictions_for_split(self.model_trainer.g, self.test_global)
            final_train_mse = structural_mse(self.model_trainer.g, self.train_global)
            final_val_g_prediction = predictions_for_split(self.model_trainer.g, self.val_global)
            self.model_trainer.f.eval()
            with torch.no_grad():
                final_val_f_prediction = self.model_trainer.f(self.val_global.z)
            self.model_trainer.f.train()
            final_moment_violation = moment_violation(
                final_val_f_prediction, self._val_y_cpu, final_val_g_prediction
            )
            # protocol_v1.md S5.2/metric_policy.md: report equal-client and
            # sample-weighted structural MSE for the final iterate too, not
            # only the best-validation checkpoint. Gated on client_id exactly
            # like eval_global_model, so non-eICU behavior is untouched.
            equal_client_final_validation_structural_mse = None
            sample_weighted_final_validation_structural_mse = None
            primary_final_validation_mse = final_validation_mse
            if val_client_id is not None:
                (
                    equal_client_final_validation_structural_mse,
                    sample_weighted_final_validation_structural_mse,
                    _,
                ) = equal_client_structural_mse(final_val_g_prediction, self._val_g_cpu, val_client_id)
                primary_final_validation_mse = equal_client_final_validation_structural_mse
            equal_client_final_test_structural_mse = None
            sample_weighted_final_test_structural_mse = None
            primary_final_test_mse = final_test_mse
            if test_client_id is not None:
                (
                    equal_client_final_test_structural_mse,
                    sample_weighted_final_test_structural_mse,
                    _,
                ) = equal_client_structural_mse(final_prediction, self._test_g_cpu, test_client_id)
                primary_final_test_mse = equal_client_final_test_structural_mse
        with self.profiler.span("final_checkpoint"):
            self._save_checkpoint(
                os.path.join(self.run_dir, "checkpoints", "final.pt"),
                final_round,
                self.final_state,
                {
                    "train_mse": final_train_mse,
                    "val_mse": final_validation_mse,
                    "test_mse": final_test_mse,
                },
                "final",
            )
        if self.best_state is None:
            self.best_validation_round = final_round
            self.best_validation_mse = primary_final_validation_mse
            self.best_state = copy.deepcopy(self.final_state)
            self.best_g_state = copy.deepcopy(self.final_g_state)
            self.best_f_state = copy.deepcopy(self.final_f_state)
            self.best_reg_state = copy.deepcopy(self.final_reg_state)
        if self.best_moment_state is None:
            self.best_moment_violation_round = final_round
            self.best_moment_violation = final_moment_violation
            self.best_moment_state = copy.deepcopy(self.final_state)

        with self.profiler.span("best_validation_test_eval"):
            self._load_training_state(self.best_state)
            best_prediction = predictions_for_split(self.model_trainer.g, self.test_global)
            best_test_mse = structural_mse_from_predictions(best_prediction, self._test_g_cpu)
            best_val_g_prediction = predictions_for_split(self.model_trainer.g, self.val_global)
            self.model_trainer.f.eval()
            with torch.no_grad():
                best_val_f_prediction = self.model_trainer.f(self.val_global.z)
                best_test_f_prediction = self.model_trainer.f(self.test_global.z)
            self.model_trainer.f.train()

            # protocol_v1.md S5.1: for eICU the compatibility field
            # test_mse_at_best_validation must itself carry equal-client
            # semantics (not a sample-weighted/pooled value under that name)
            # -- metric_policy.md's selection firewall explicitly forbids
            # substituting one for the other.
            equal_client_test_mse_at_best_validation = None
            sample_weighted_test_mse_at_best_validation = None
            per_client_test_mse_at_best_validation = None
            primary_test_mse_at_best_validation = best_test_mse
            if test_client_id is not None:
                (
                    equal_client_test_mse_at_best_validation,
                    sample_weighted_test_mse_at_best_validation,
                    per_client_test_mse_at_best_validation,
                ) = equal_client_structural_mse(best_prediction, self._test_g_cpu, test_client_id)
                primary_test_mse_at_best_validation = equal_client_test_mse_at_best_validation

            equal_client_best_validation_structural_mse = None
            sample_weighted_validation_mse_at_best_validation = None
            equal_client_validation_moment_violation_at_best_validation = None
            sample_weighted_validation_moment_violation_at_best_validation = None
            if val_client_id is not None:
                (
                    equal_client_best_validation_structural_mse,
                    sample_weighted_validation_mse_at_best_validation,
                    _,
                ) = equal_client_structural_mse(best_val_g_prediction, self._val_g_cpu, val_client_id)
                (
                    equal_client_validation_moment_violation_at_best_validation,
                    sample_weighted_validation_moment_violation_at_best_validation,
                    _,
                ) = equal_client_moment_violation(
                    best_val_f_prediction, self._val_y_cpu, best_val_g_prediction, val_client_id
                )

            equal_client_test_moment_violation_at_best_validation = None
            sample_weighted_test_moment_violation_at_best_validation = None
            per_client_test_moment_violation_at_best_validation = None
            if test_client_id is not None:
                (
                    equal_client_test_moment_violation_at_best_validation,
                    sample_weighted_test_moment_violation_at_best_validation,
                    per_client_test_moment_violation_at_best_validation,
                ) = equal_client_moment_violation(
                    best_test_f_prediction, self._test_y_cpu, best_prediction, test_client_id
                )

            per_client_metrics_artifact = None
            if test_client_id is not None:
                per_client_metrics_artifact = write_per_client_metrics(
                    self.run_dir,
                    per_client_test_mse_at_best_validation,
                    per_client_test_moment_violation_at_best_validation,
                )
        print("MSE on test ------------------------------>>>>>>>>>>>>>>>>>>", best_test_mse)

        with self.profiler.span("best_moment_violation_test_eval"):
            self._load_training_state(self.best_moment_state)
            best_moment_prediction = predictions_for_split(self.model_trainer.g, self.test_global)
            test_mse_at_best_moment_violation = structural_mse_from_predictions(
                best_moment_prediction, self._test_g_cpu
            )
        # Restore the val-MSE-selected state as "the" checkpoint used for
        # save_predictions_npz below, matching the field names' existing meaning.
        self._load_training_state(self.best_state)

        with self.profiler.span("save_predictions_npz"):
            if self.compact_predictions_only:
                # Drops the full test-input tensor entirely -- writes the
                # compact schema *as* predictions.npz, since there is no
                # separate full artifact for this run to be additive to;
                # this run's predictions.npz never has an x array.
                save_predictions_npz_compact(
                    self.run_dir,
                    self.test_global,
                    best_prediction,
                    final_prediction,
                    self.effective_config,
                    filename="predictions.npz",
                )
            else:
                save_predictions_npz(
                    self.run_dir,
                    self.test_global,
                    best_prediction,
                    final_prediction,
                    self.effective_config,
                )

        # Stability diagnostics (metric_policy.md "Stability metrics"):
        # final-minus-best gaps and the spread of the primary (equal-client
        # when available, else pooled) validation curve over its last 50
        # recorded rounds.
        final_vs_best_validation_gap = primary_final_validation_mse - self.best_validation_mse
        final_vs_best_test_gap = primary_final_test_mse - primary_test_mse_at_best_validation
        recent_primary_val_mse = [
            row["primary_val_mse"] for row in self.mse_rows[-50:] if row.get("primary_val_mse") is not None
        ]
        if recent_primary_val_mse:
            last_50_validation_mse_std = float(numpy.std(recent_primary_val_mse))
            last_50_validation_mse_range = float(max(recent_primary_val_mse) - min(recent_primary_val_mse))
        else:
            last_50_validation_mse_std = None
            last_50_validation_mse_range = None
        g_bn_values = [
            row["g_bn_min_running_var"]
            for row in self.mse_rows
            if row.get("g_bn_min_running_var") is not None
        ]
        f_bn_values = [
            row["f_bn_min_running_var"]
            for row in self.mse_rows
            if row.get("f_bn_min_running_var") is not None
        ]

        metrics = {
            "run_id": self.effective_config.get("run_id", ""),
            "method": self.effective_config.get("variant", ""),
            "seed": self.effective_config.get("random_seed", 0),
            "scenario_seed": self.effective_config.get("scenario_seed", 0),
            "optimizer_seed": self.effective_config.get("optimizer_seed", 0),
            "best_validation_round": self.best_validation_round,
            "best_validation_mse": self.best_validation_mse,
            "test_mse_at_best_validation": primary_test_mse_at_best_validation,
            "best_moment_violation_round": self.best_moment_violation_round,
            "best_moment_violation": self.best_moment_violation,
            "test_mse_at_best_moment_violation": test_mse_at_best_moment_violation,
            "final_moment_violation": final_moment_violation,
            "final_validation_mse": primary_final_validation_mse,
            "final_test_mse": primary_final_test_mse,
            "best_gmm_eval_round": best_gmm_eval_round,
            "best_gmm_eval": self.eval_history[max_i],
            "diverged": self.diverged,
            "nonfinite_first_round": self.nonfinite_first_round,
            "nonfinite_diagnostics": self.nonfinite_diagnostics,
            "run_status": "validation_deterioration" if validation_stop else "completed",
            "failure_reason": validation_stop["reason"] if validation_stop else None,
            "rounds_completed": len(self.mse_rows),
            "runtime_seconds": now_seconds() - self.start_time,
            "test_mse_logged_by_round": self.log_test_mse_by_round,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
            "selection_metric": self.effective_config.get(
                "primary_selection_metric", "pooled_validation_mse"
            ),
            "selection_source": self.effective_config.get(
                "selection_source", "validation_only"
            ),
            "selected_round": self.best_validation_round,
            "test_mse_reported_after_selection": True,
            "is_primary": str(getattr(self.args, "campaign_role", "")) != "aggregation_ablation",
            "alignment_label": self.effective_config.get("alignment_label", ""),
            "skip_model_selection": self.skip_model_selection,
            "skip_gmm_eval": self.skip_gmm_eval,
            "gmm_eval_proxy": self.gmm_eval_proxy,
            "auxiliary_regression": not self.skip_auxiliary_regression,
            "auxiliary_regression_epochs": int(getattr(self.args, "auxiliary_regression_epochs", 0)),
            "append_round_csv": self.append_round_csv,
            "periodic_checkpoint_interval": self.periodic_checkpoint_interval,
            "aggregation_weighting": self.aggregation_weighting,
            "server_buffer_policy": self.server_buffer_policy,
            "g_bn_min_running_var": min(g_bn_values) if g_bn_values else None,
            "f_bn_min_running_var": min(f_bn_values) if f_bn_values else None,
            # Variance of the validation target g -- the MSE a constant
            # predictor (always predicting mean(g)) would achieve on this
            # run's validation set. Used only by the alpha=0.1 stability
            # check's constant-predictor escape hatch
            # (doe_review_and_revised_grid.md's "reject worse than constant
            # predictor"); never test-set-derived, so it cannot leak test MSE
            # into a selection decision.
            "val_target_variance": float(self._val_g_cpu.var(unbiased=False)),
            "objective_mode": self.objective_mode,
            "scenario_checksum": self.effective_config.get("scenario_checksum", ""),
            "config_checksum": config_checksum(self.effective_config),
            "per_client_metrics_artifact": per_client_metrics_artifact,
            "curve_artifact": "mse_by_round.csv",
            "best_checkpoint_artifact": os.path.join("checkpoints", "best_validation.pt"),
            "final_checkpoint_artifact": os.path.join("checkpoints", "final.pt"),
            "effective_config_artifact": "effective_config.json",
            "equal_client_best_validation_structural_mse": equal_client_best_validation_structural_mse,
            "sample_weighted_validation_mse_at_best_validation": sample_weighted_validation_mse_at_best_validation,
            "equal_client_final_validation_structural_mse": equal_client_final_validation_structural_mse,
            "sample_weighted_final_validation_structural_mse": sample_weighted_final_validation_structural_mse,
            "equal_client_test_mse_at_best_validation": equal_client_test_mse_at_best_validation,
            "sample_weighted_test_mse_at_best_validation": sample_weighted_test_mse_at_best_validation,
            "equal_client_final_test_structural_mse": equal_client_final_test_structural_mse,
            "sample_weighted_final_test_structural_mse": sample_weighted_final_test_structural_mse,
            "equal_client_validation_moment_violation_at_best_validation": equal_client_validation_moment_violation_at_best_validation,
            "sample_weighted_validation_moment_violation_at_best_validation": sample_weighted_validation_moment_violation_at_best_validation,
            "equal_client_test_moment_violation_at_best_validation": equal_client_test_moment_violation_at_best_validation,
            "sample_weighted_test_moment_violation_at_best_validation": sample_weighted_test_moment_violation_at_best_validation,
            # Filled in post-hoc by analyze_eicu_study_a_checkpoint.py, which
            # is the only place that knows how to build D=0/1 counterfactual
            # rows for a given Study A scenario (see metric_policy.md
            # "ATE error and individual-effect MAE" -- these are not
            # substitutable and both require the scenario's true_effect
            # array, unavailable to the dataset-agnostic trainer here).
            "equal_client_absolute_ate_error_at_best_validation": None,
            "sample_weighted_absolute_ate_error_at_best_validation": None,
            "equal_client_individual_effect_mae_at_best_validation": None,
            "sample_weighted_individual_effect_mae_at_best_validation": None,
            "final_vs_best_validation_gap": final_vs_best_validation_gap,
            "final_vs_best_test_gap": final_vs_best_test_gap,
            "last_50_validation_mse_std": last_50_validation_mse_std,
            "last_50_validation_mse_range": last_50_validation_mse_range,
        }
        with self.profiler.span("write_metrics"):
            write_metrics(self.run_dir, metrics)
        if validation_stop:
            write_validation_stop(self.run_dir, self.effective_config, validation_stop)

        if getattr(self.args, "enable_legacy_outputs", True) or getattr(self.args, "enable_legacy_plot", False):
            with self.profiler.span("legacy_plot"):
                x = self.test_global.x.detach().cpu().numpy()
                best_pred_np = best_prediction.detach().cpu().numpy()
                g_true = self.test_global.g.detach().cpu().numpy()
                indices = numpy.argsort(x, axis=0).flatten()
                x_sort = x[indices]
                pred_plot = PlotElement(x_sort, best_pred_np[indices], "FedDeepGMM-SGDA")
                true_plot = PlotElement(x_sort, g_true[indices], "Actual Causal Effect")
                plot_dir = "plots" if getattr(self.args, "enable_legacy_outputs", True) else os.path.join(self.run_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                fig, ax = plt.subplots()
                ax = true_plot.plot(ax=ax, save_path=os.path.join(plot_dir, f'aaaa_{self.args.run_name}_.png'))
                ax = pred_plot.plot(ax=ax, save_path=os.path.join(plot_dir, f'aaaa_{self.args.run_name}_.png'))
        # ax = reg_NN_plot.plot(ax=ax, save_path=f'plots/aaaacomparison_{self.args.run_name}_.png')
        self.profiler.record(
            "finalization",
            time.perf_counter() - finalization_profile_wall,
            process_cpu_seconds=time.process_time() - finalization_profile_cpu,
        )
        
        if self.client_executor is not None:
            self.client_executor.close()
            self.client_executor = None

        # mlops.log_training_finished_status()
        # mlops.log_aggregation_finished_status()
        
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        client_indexes = client_indices_for_round(
            getattr(self.args, "random_seed", 0),
            round_idx,
            client_num_in_total,
            client_num_per_round,
        )
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _record_stochastic_batching(self, round_idx, client_indexes):
        if not bool(getattr(self.args, "require_multibatch_stochastic", False)):
            return
        batch_size = int(getattr(self.args, "batch_size", 0))
        if batch_size <= 0:
            raise ValueError("require_multibatch_stochastic requires a positive batch_size")
        variant = self.effective_config.get("variant", "unknown")
        for client_idx in client_indexes:
            local_loader = self.train_data_local_dict[client_idx]
            sample_count = int(self.train_data_local_num_dict[client_idx])
            try:
                num_batches = int(len(local_loader))
            except TypeError:
                num_batches = int(math.ceil(sample_count * 1.0 / batch_size))
            first_actual_batch_size = min(sample_count, batch_size) if sample_count > 0 else 0
            row = {
                "variant": variant,
                "round": int(round_idx),
                "client_id": int(client_idx),
                "sample_count": sample_count,
                "configured_batch_size": batch_size,
                "num_batches": num_batches,
                "first_actual_batch_size": first_actual_batch_size,
            }
            self.batching_summary_rows.append(row)
            if num_batches < 2:
                write_batching_summary(self.run_dir, self.batching_summary_rows)
                raise ValueError(
                    "Stochastic smoke multibatch guard failed: "
                    f"variant={variant}, round={round_idx}, client_id={client_idx}, "
                    f"sample_count={sample_count}, configured_batch_size={batch_size}, "
                    f"num_batches={num_batches}"
                )
        # The file is written once after the round loop, not here. Rewriting the
        # whole accumulated list every round is quadratic in comm_round: the list
        # grows by client_num_per_round entries per round, so a 2000-round run
        # serialized ~358M rows to produce a 64MB artifact. The guard above still
        # flushes on failure, so a violating run keeps its forensic record.

    def _record_aggregation_weights(self, round_idx, client_indexes, w_locals):
        """Persist the resolved aggregation mode and the actual per-client
        weights used this round, so ``uniform_clients`` vs ``sample_size`` is
        independently auditable from the run directory rather than only from
        ``effective_config.json``'s static mode string. One row per round,
        matching every other ``*_by_round.csv`` artifact's granularity.
        """
        weights = self._client_weights(w_locals)
        client_weights = {
            int(client_idx): float(weight)
            for client_idx, weight in zip(client_indexes, weights)
        }
        row = {
            "round": int(round_idx),
            "aggregation_weighting": self.aggregation_weighting,
            "k_participating": len(client_weights),
            "weight_sum": float(sum(client_weights.values())),
            "client_weights_json": json.dumps(client_weights, sort_keys=True),
        }
        self.aggregation_weight_rows.append(row)
        if self.append_round_csv:
            append_aggregation_weights_by_round(self.run_dir, row)
        else:
            write_aggregation_weights_by_round(self.run_dir, self.aggregation_weight_rows)

    def _capture_training_state(self):
        return {
            "g_state_dict": copy.deepcopy(self.model_trainer.get_g_model_params()),
            "f_state_dict": copy.deepcopy(self.model_trainer.get_f_model_params()),
            "reg_state_dict": (
                None
                if self.skip_auxiliary_regression
                else copy.deepcopy(self.model_trainer.get_model_params())
            ),
        }

    def _load_training_state(self, state):
        self.model_trainer.set_g_model_params(copy.deepcopy(state["g_state_dict"]))
        self.model_trainer.set_f_model_params(copy.deepcopy(state["f_state_dict"]))
        if state.get("reg_state_dict") is not None:
            self.model_trainer.set_model_params(copy.deepcopy(state["reg_state_dict"]))
        self.g = self.model_trainer.g
        self.f = self.model_trainer.f
        self.reg_model = self.model_trainer.reg_model

    def _aggregate_t(self,obj_sum):
        total_sum=0
        for i in obj_sum:
            total_sum+=i
        return total_sum/10
    def _client_weights(self, w_locals):
        """Per-client weights for this round, per ``self.aggregation_weighting``.

        Shared by ``_aggregate`` and ``_aggregate_reg`` so FedGDA's direct
        update and FedOGDA's pseudogradient/delta (which is derived as
        ``weighted_average(w_locals) - theta_t``, see ``train()``) always use
        the identical, consistent weighting for both g and f.
        """
        sample_counts = [num for num, _ in w_locals]
        return compute_client_weights(sample_counts, self.aggregation_weighting)

    def _aggregate_reg(self, w_locals):
        weights = self._client_weights(w_locals)
        state_dicts = [params for _, params in w_locals]
        return weighted_average_state_dicts(state_dicts, weights)

    def _aggregate(self, w_locals):
        weights = self._client_weights(w_locals)
        g_dicts = [gf[0] for _, gf in w_locals]
        f_dicts = [gf[1] for _, gf in w_locals]
        g = weighted_average_state_dicts(g_dicts, weights)
        f = weighted_average_state_dicts(f_dicts, weights)
        return [g, f]

    def _save_checkpoint(self, path, round_idx, state, metrics, checkpoint_type):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "round": round_idx,
            "checkpoint_type": checkpoint_type,
            "state": {
                "g": state["g_state_dict"],
                "f": state["f_state_dict"],
                "regression": state.get("reg_state_dict"),
            },
            "g_state_dict": state["g_state_dict"],
            "f_state_dict": state["f_state_dict"],
            "reg_state_dict": state.get("reg_state_dict"),
            "metrics": metrics,
            "effective_config": self.effective_config,
        }, path)

    def _effective_batch_size(self, num_data):
        if not hasattr(self.args, "batch_size") or self.args.batch_size is None or self.args.batch_size <= 0:
            return num_data
        return self.args.batch_size

    def calc_f_g_obj(self, global_val):
        x = global_val.x
        y = global_val.y
        z = global_val.z
        num_data = x.shape[0]
        batch_size = self._effective_batch_size(num_data)
        num_batch = math.ceil(num_data * 1.0 / batch_size)
        g_of_x_batches = []
        f_of_z_batches = []
        obj_total = 0
        with torch.no_grad():
            for b in range(num_batch):
                batch_idx = slice(b * batch_size, min((b + 1) * batch_size, num_data))
                x_batch = x[batch_idx]
                z_batch = z[batch_idx]
                y_batch = y[batch_idx]
                g_obj, _ = self.model_trainer.game_objective.calc_objective(self.model_trainer.g, self.model_trainer.f, x_batch, z_batch, y_batch)
                g_of_x_batches.append(self.model_trainer.g(x_batch).detach().cpu())
                f_of_z_batches.append(self.model_trainer.f(z_batch).detach().cpu())
                obj_total += float(g_obj.detach().cpu()) * int(x_batch.shape[0]) * 1.0 / num_data
        g_of_x = torch.cat(g_of_x_batches, dim=0)
        f_of_z = torch.cat(f_of_z_batches, dim=0)
        return g_of_x, f_of_z, obj_total
    
        
    def eval_global_model(self):
        # self.model_trainer is never trained directly (each client works on
        # its own copy.deepcopy of it, see _setup_clients), so its
        # game_objective instance never gets set_theta_tilde called on it by
        # any client. calc_f_g_obj below only uses calc_objective for the
        # informational gmm_train_objective/gmm_val_objective fields (the
        # moment-violation metric is computed independently, directly from
        # g_of_x/f_of_z/y), so any theta~ is fine here -- use the current
        # (just-aggregated) global g.
        if hasattr(self.model_trainer.game_objective, "set_theta_tilde"):
            self.model_trainer.game_objective.set_theta_tilde(self.model_trainer.g)
        self.f = self.model_trainer.f.eval()
        self.g = self.model_trainer.g.eval()
        g_of_x_train, f_of_z_train, obj_train = self.calc_f_g_obj(self.train_global)
        g_of_x_dev, f_of_z_dev, obj_dev = self.calc_f_g_obj(self.val_global)
        epsilon_dev = g_of_x_dev - self._val_y_cpu
        epsilon_train = g_of_x_train - self._train_y_cpu
        train_mse = structural_mse_from_predictions(g_of_x_train, self._train_g_cpu)
        val_mse = structural_mse_from_predictions(g_of_x_dev, self._val_g_cpu)
        # Held-out moment violation: g0-free, so this is the one recovery
        # diagnostic that also works on real (Study B) data where val_mse
        # cannot be computed at all.
        train_moment_violation = moment_violation(
            f_of_z_train, self._train_y_cpu, g_of_x_train
        )
        val_moment_violation = moment_violation(f_of_z_dev, self._val_y_cpu, g_of_x_dev)

        # protocol_v1.md S5.1/metric_policy.md: Study A's primary checkpoint
        # selector is equal-client (per-hospital-averaged) validation
        # structural MSE, not this pooled/sample-weighted val_mse. client_id
        # survives on val_global untouched (Dataset.to_tensor/the eicu data
        # loader never wrap or strip it -- see abstract_scenario.py), so it is
        # available here for every eICU run and absent (None) for every other
        # dataset family, which is exactly the gate this block uses: legacy
        # datasets get val_mse unchanged, byte for byte.
        val_client_id = getattr(self.val_global, "client_id", None)
        equal_client_val_mse = None
        sample_weighted_val_mse = None
        per_client_val_mse = None
        equal_client_val_moment_violation = None
        sample_weighted_val_moment_violation = None
        per_client_val_moment_violation = None
        primary_val_mse = val_mse
        if val_client_id is not None:
            equal_client_val_mse, sample_weighted_val_mse, per_client_val_mse = (
                equal_client_structural_mse(g_of_x_dev, self._val_g_cpu, val_client_id)
            )
            (
                equal_client_val_moment_violation,
                sample_weighted_val_moment_violation,
                per_client_val_moment_violation,
            ) = equal_client_moment_violation(
                f_of_z_dev, self._val_y_cpu, g_of_x_dev, val_client_id
            )
            primary_val_mse = equal_client_val_mse
        if self.skip_gmm_eval:
            curr_eval = -float(val_mse)
        else:
            curr_eval = approx_psi_eval(epsilon_dev, self.dev_f_collection,
                                                self.e_dev_tilde)
        self.eval_list.append(curr_eval)
        self.mse_list.append(train_mse)
        if self.eval_history:
            max_recent_eval = max(self.eval_history)
        else:
            max_recent_eval = float("-inf")
        self.eval_history.append(curr_eval)
        if self.args.video_plotter:
            self.epsilon_dev_history.append(epsilon_dev)
            self.epsilon_train_history.append(epsilon_train)
            self.g_state_history.append(copy.deepcopy(self.g.state_dict()))

        self.f = self.f.train()
        self.g = self.g.train()
        return {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "primary_val_mse": primary_val_mse,
            "equal_client_val_mse": equal_client_val_mse,
            "sample_weighted_val_mse": sample_weighted_val_mse,
            "per_client_val_mse": per_client_val_mse,
            "train_moment_violation": train_moment_violation,
            "val_moment_violation": val_moment_violation,
            "equal_client_val_moment_violation": equal_client_val_moment_violation,
            "sample_weighted_val_moment_violation": sample_weighted_val_moment_violation,
            "per_client_val_moment_violation": per_client_val_moment_violation,
            "gmm_train_objective": obj_train,
            "gmm_val_objective": obj_dev,
            "gmm_eval": curr_eval,
            "max_recent_gmm_eval": max_recent_eval,
            "f_of_z_train": f_of_z_train,
            "f_of_z_dev": f_of_z_dev,
        }
