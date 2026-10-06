import copy
import logging
import random
import math
import numpy as np
import torch
import wandb
import os
import csv
import pandas
import numpy
import matplotlib.pyplot as plt
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from fedml.ml.trainer.fedprox_trainer import FedProxModelTrainer
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from model_selection_class import FHistoryModelSelectionV3
from game_objectives.simple_moment_objective import OptimalMomentObjective
from optimizers.oadam import OAdam
from optimizers.optimizer_factory import OptimizerFactory
from torch.optim import Adam
from model_selection.simple_model_eval import GradientDecentSimpleModelEval, \
    SGDSimpleModelEval
from model_selection.learning_eval_nostop import \
    FHistoryLearningEvalGradientDecentNoStop, FHistoryLearningEvalNoStop, \
    FHistoryLearningEvalSGDNoStop
from game_objectives.approximate_psi_objective import approx_psi_eval
# from fedgmm.sp_decentralized_mnist_lr_example.plotting import PlotElement
from plotting import PlotElement

from .client import Client

def log_results_to_csv(file_path, round_number, mse):
       file_exists = os.path.isfile(file_path)
       with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['Round', 'MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Round': round_number, 'MSE': mse}) 
class FedProxTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args
        research_args = {
            'video_plotter': False,
            'print_freq': 1,
            'eval_freq': 1,
            'burn_in': 0,
            'max_no_progress': 100,
            'verbose': True,
            'print_freq_mul': 1
        }
        for arg_name, default_value in research_args.items():
            if not hasattr(self.args, arg_name):
                setattr(self.args, arg_name, default_value)
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
        # self.model_trainer = create_model_trainer(model, args)
        g_learning_rates = [self.args.learning_rate]
        game_objectives = [
            OptimalMomentObjective(),
        ]
        
        learning_setups = []
        for g_lr in g_learning_rates:
            for game_objective in game_objectives:
                learning_setup = {
                    "g_optimizer_factory": OptimizerFactory(
                        OAdam, lr=g_lr, betas=(0.5, 0.9)),
                    "f_optimizer_factory": OptimizerFactory(
                        OAdam, lr=5*g_lr, betas=(0.5, 0.9)),
                    "game_objective": game_objective
                }
                learning_setups.append(learning_setup)
        default_g_opt_factory = OptimizerFactory(
            Adam, lr=0.001, betas=(0.5, 0.9))
        default_f_opt_factory = OptimizerFactory(
            Adam, lr=0.005, betas=(0.5, 0.9))
        g_simple_model_eval = SGDSimpleModelEval(max_num_epoch=1000, max_no_progress=10, batch_size=256, eval_freq=10)
        # g_simple_model_eval = GradientDecentSimpleModelEval(
        #     max_num_iter=4000, max_no_progress=10, eval_freq=100)
        f_simple_model_eval = SGDSimpleModelEval(max_num_epoch=1000, max_no_progress=10, batch_size=256, eval_freq=10)
        learning_eval = FHistoryLearningEvalSGDNoStop(num_epochs=3000, eval_freq=20, 
                                                       batch_size=1024)
        self.reg_model = model[2][0]
        self.model_selection = FHistoryModelSelectionV3(
            g_model_list=model[0],
            f_model_list=model[1],
            learning_args_list=learning_setups,
            default_g_optimizer_factory=default_g_opt_factory,
            default_f_optimizer_factory=default_f_opt_factory,
            g_simple_model_eval=g_simple_model_eval,
            f_simple_model_eval=f_simple_model_eval,
            learning_eval=learning_eval,
            psi_eval_max_no_progress=20, psi_eval_burn_in=50)
        self.default_g_opt_factory = default_g_opt_factory

        g_global, f_global, learning_args, dev_f_collection, e_dev_tilde = \
            self.model_selection.do_model_selection(
                x_train=train_data_global.x, z_train=train_data_global.z, y_train=train_data_global.y,
                x_dev=val_data_global.x, z_dev=val_data_global.z, y_dev=val_data_global.y, verbose=True)
        
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
        
        # self.model_trainer = create_model_trainer([g_global, f_global, model[2][0]], learning_args, args)
        self.model_trainer = FedProxModelTrainer([g_global, f_global, model[2][0]],learning_args, args)

        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        staggler_ids = random.sample(range(10), 5)
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            additional_epochs =0
            if client_idx in staggler_ids:
                additional_epochs=10
            c = Client(
                client_idx,
                list(train_data_local_dict[client_idx])[0],
                list(test_data_local_dict[client_idx])[0],
                # train_data_local_dict[client_idx],
                # test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(model_trainer),
                additional_epochs
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        fedProx =[]
        # logging.info("self.model_trainer = {}".format(self.model_trainer))
        g_global = self.model_trainer.get_g_model_params()
        f_global = self.model_trainer.get_f_model_params()
        reg_global = self.model_trainer.get_model_params()
        w_global = self.model_trainer.get_model_params()
        # mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        # mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        # mlops.log_round_info(self.args.comm_round, -1)
        w_locals_prev=None
        for round_idx in range(self.args.comm_round):

            # logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            w_locals_reg=[]
            
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            # logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                # mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                
                w = client.train(copy.deepcopy(g_global),copy.deepcopy(f_global),copy.deepcopy(w_global),copy.deepcopy(w_locals_prev))
                # w = client.train(copy.deepcopy(g_global),copy.deepcopy(f_global))
                w_reg = client.train_reg(copy.deepcopy(reg_global))
                w_locals_prev = copy.deepcopy(w)

                # mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                # self.logging.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                w_locals_reg.append((client.get_sample_number(), copy.deepcopy(w_reg)))

            # update global weights
            # mlops.event("agg", event_started=True, event_value=str(round_idx))
            w_global = self._aggregate(w_locals)
            w_global_reg = self._aggregate_reg(w_locals_reg)
  

            self.model_trainer.set_g_model_params(w_global[0])
            self.model_trainer.set_f_model_params(w_global[1])
            self.model_trainer.set_model_params(w_global_reg)
            mse, obj_train, obj_dev, curr_eval, max_recent_eval, f_of_z_train, f_of_z_dev = self.eval_global_model()
            # fedProx.append(mse)
            # log_results_to_csv("./fedprox_mnist_z_final.csv", round_idx, mse)
            # wandb.log({"round":round_idx,"MSE" :mse})
            # logging.info(f"{round_idx}: {mse:.4f}")
            # print(round_idx)
            # print(mse)
            # mlops.event("agg", event_started=False, event_value=str(round_idx))
            
            # test results
            # at last round
            if round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    # self._local_test_on_all_clients(round_idx)
                    mse, obj_train, obj_dev, curr_eval, max_recent_eval, f_of_z_train, f_of_z_dev = self.eval_global_model()
                
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

            # check stopping conditions if we are past burn-in
                current_no_progress = 0
                if round_idx % self.args.eval_freq == 0 and round_idx >= self.args.burn_in:
                    if curr_eval > max_recent_eval:
                        current_no_progress = 0
                    else:
                        current_no_progress += 1

                    if current_no_progress >= self.args.max_no_progress:
                        break
        # plot relationship between MSE and eval
        # if self.args.video_plotter:
        #     plt.figure()
        #     data = pandas.DataFrame({"eval": self.eval_list, "mse": self.mse_list})
        #     data.plot.scatter(x="eval", y="mse")
        #     plt.savefig("eval_mse.png")
            
        max_i = max(range(len(self.eval_history)), key=lambda i_: self.eval_history[i_])
        if self.args.verbose:
            print("best iteration:", self.args.eval_freq * max_i)
            mlops.log_round_info(self.args.comm_round, round_idx)
        self.model_trainer.set_g_model_params(self.g_state_history[max_i])
        g_final = self.g
        reg_model_final = self.reg_model
        g_final.load_state_dict(self.model_trainer.get_g_model_params())
        reg_model_final.load_state_dict(self.model_trainer.get_model_params())
        g_pred = g_final(self.test_global.x)
        reg_model_final.to(self.device)
        reg_pred = reg_model_final(self.test_global.x)
        mse = float(((g_pred - self.test_global.g) ** 2).mean())
        print("---------------")
        print("finished running methodology on scenario %s" % self.args.scenario_name)
        print("MSE on test ------------------------------>>>>>>>>>>>>>>>>>>", mse)
        print("")
        print("saving results...")
        x = self.test_global.x.detach().cpu().numpy()
        g_pred = g_pred.detach().cpu().numpy()
        g_true = self.test_global.g.detach().cpu().numpy()
        reg_pred = reg_pred.detach().cpu().numpy()
        indices = numpy.argsort(x, axis = 0).flatten() 
        x_sort = x[indices]
        g_pred_sort = g_pred[indices]
        g_true_sort = g_true[indices]
        reg_pred_sort = reg_pred[indices]
        pred_plot = PlotElement(x_sort, g_pred_sort, "Predicted Causal Effect (FedProx)")
        true_plot = PlotElement(x_sort, g_true_sort, "Actual Causal Effect")
        # for i in range(len(x_sort)):
        #     log_results_to_csv("./fedprox_linear_final.csv", x_sort[i][0], g_pred_sort[i][0])
        #     log_results_to_csv("./true_linear_final.csv", x_sort[i][0], g_true_sort[i][0])
        # reg_NN_plot = PlotElement(x_sort, reg_pred_sort, "Direct predictions from Neural Network")
        # plot_Prox = PlotElement(x_label,fedProx,"FedProx")
        
        # fig, ax = plt.subplots()
        # ax = pred_plot.plot(ax=ax)
        # ax = true_plot.plot(ax=ax, save_path=f'plots/comparison_1111{self.args.run_name}_.png')
        # ax = reg_NN_plot.plot(ax=ax, save_path=f'plots/comparison_{self.args.run_name}_.png')
        
        
        # fig, ax = plt.subplots()

# Plotting predicted plot
        # ax = pred_plot.plot(ax=ax)

# Plotting true plot
        # ax = true_plot.plot(ax=ax, save_path=f'plots/comparison_{self.args.run_name}_.png')

# Plotting reg NN plot
        # ax = reg_NN_plot.plot(ax=ax, save_path=f'plots/comparison_{self.args.run_name}_.png')

        # plt.show()
        #     if round_idx == self.args.comm_round - 1:
        #         self._local_test_on_all_clients(round_idx)
        #     # per {frequency_of_the_test} round
        #     elif round_idx % self.args.frequency_of_the_test == 0:
        #         if self.args.dataset.startswith("stackoverflow"):
        #             self._local_test_on_validation_set(round_idx)
        #         else:
        #             self._local_test_on_all_clients(round_idx)

        #     mlops.log_round_info(self.args.comm_round, round_idx)
           
        # mlops.log_training_finished_status()
        # mlops.log_aggregation_finished_status()


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    
    def _aggregate_reg(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    # def _aggregate(self, w_locals):
    #     avg_params = FedMLAggOperator.agg(self.args, w_locals)
    #     return avg_params
    def _aggregate(self, w_locals):
        training_num = sum([num for num, (_) in w_locals])

        (sample_num, (g, f)) = w_locals[0]
        for k in g.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, (local_g, _) = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    g[k] = local_g[k] * w
                else:
                    g[k] += local_g[k] * w
        
        for k in f.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, (_, local_f) = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    f[k] = local_f[k] * w
                else:
                    f[k] += local_f[k] * w
        return [g, f]
    def calc_f_g_obj(self, global_val):
        x = global_val.x
        y = global_val.y
        z = global_val.z
        num_data = x.shape[0]
        num_batch = math.ceil(num_data * 1.0 / self.args.batch_size)
        g_of_x = None
        f_of_z = None
        obj_total = 0
        for b in range(num_batch):
            if b < num_batch - 1:
                batch_idx = list(range(b*self.args.batch_size, (b+1)*self.args.batch_size))
            else:
                batch_idx = list(range(b*self.args.batch_size, num_data))
            x_batch = x[batch_idx]
            z_batch = z[batch_idx]
            y_batch = y[batch_idx]
            g_obj, _ = self.model_trainer.game_objective.calc_objective(self.model_trainer.g, self.model_trainer.f, x_batch, z_batch, y_batch)
            g_of_x_batch = self.model_trainer.g(x_batch).detach().cpu()
            f_of_z_batch = self.model_trainer.f(z_batch).detach().cpu()
            if b == 0:
                g_of_x = g_of_x_batch
                f_of_z = f_of_z_batch
            else:
                g_of_x = torch.cat([g_of_x, g_of_x_batch], dim=0)
                f_of_z = torch.cat([f_of_z, f_of_z_batch], dim=0)
            obj_total += float(g_obj.detach().cpu()) * len(batch_idx) * 1.0 / num_data
        return g_of_x, f_of_z, float(g_obj.detach().cpu())
    
    def eval_global_model(self):
        self.f = self.model_trainer.f.eval()
        self.g = self.model_trainer.g.eval()
        g_of_x_train, f_of_z_train, obj_train = self.calc_f_g_obj(self.train_global)
        g_of_x_dev, f_of_z_dev, obj_dev = self.calc_f_g_obj(self.val_global)
        epsilon_dev = g_of_x_dev - self.val_global.y.cpu()
        epsilon_train = g_of_x_train - self.train_global.y.cpu()
        curr_eval = approx_psi_eval(epsilon_dev, self.dev_f_collection,
                                            self.e_dev_tilde)
        g_error = epsilon_train + self.train_global.y.cpu() - self.train_global.g.cpu()
        mse = float((g_error ** 2).mean())
        self.eval_list.append(curr_eval)
        self.mse_list.append(mse)
        if self.eval_history:
            max_recent_eval = max(self.eval_history)
        else:
            max_recent_eval = float("-inf")
        self.eval_history.append(curr_eval)
        self.epsilon_dev_history.append(epsilon_dev)
        self.epsilon_train_history.append(epsilon_train)
        self.g_state_history.append(copy.deepcopy(self.g.state_dict()))

        self.f = self.f.train()
        self.g = self.g.train()
        self.model_trainer.set_f_model_params(self.f.state_dict())
        self.model_trainer.set_g_model_params(self.g.state_dict())
        return mse, obj_train, obj_dev, curr_eval, max_recent_eval, f_of_z_train, f_of_z_dev
    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
