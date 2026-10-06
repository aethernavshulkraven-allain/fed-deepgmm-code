
import logging
import torch
import torch.nn as nn
from fedml.model.cv.cnn import CNN_DropOut, CNN_WEB
from fedml.model.cv.darts import genotypes
from fedml.model.cv.darts.model import NetworkCIFAR
from fedml.model.cv.darts.model_search import Network
from fedml.model.cv.efficientnet import EfficientNet
from fedml.model.cv.mnist_gan import Generator, Discriminator
from fedml.model.cv.mobilenet import mobilenet
from fedml.model.cv.mobilenet_v3 import MobileNetV3
from fedml.model.cv.resnet import resnet56
from fedml.model.cv.resnet56 import resnet_client, resnet_server
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.linear.lr import LogisticRegression
from models.cnn_models import LeakySoftmaxCNN, DefaultCNN, CIFAR10CNN
from fedml.model.linear.lr_cifar10 import LogisticRegression_Cifar10
from fedml.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow, RNN_FedShakespeare
from models.mlp_model import MLPModel


def _create_auxiliary_model(enabled, factory):
    """Consume the legacy initialization RNG; retain the model only when enabled."""
    model = factory()
    return model if enabled else None


def _eicu_hidden_widths(value):
    """Parse a YAML list or a compact manifest value such as ``32,32``."""
    if value is None or value == "":
        return [64, 64]
    if isinstance(value, (list, tuple)):
        widths = [int(item) for item in value]
    else:
        text = str(value).strip().strip("[]")
        widths = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(f"eICU hidden_widths must contain positive integers, got {value!r}")
    return widths


def _eicu_activation(value):
    name = str(value or "leaky_relu").strip().lower()
    choices = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
    }
    if name not in choices:
        raise ValueError(
            f"eICU model_activation must be one of {sorted(choices)}, got {value!r}"
        )
    return choices[name]


def create(args, output_dim):
    global model
    model_name = args.model
    logging.info("create_model. model_name = %s, output_dim = %s, dataset = %s" % (model_name, output_dim, args.dataset))
    zoo_datasets = ['linear', 'abs', 'sin', 'step', 'zoo']
    is_eicu = str(args.dataset).startswith("eicu")
    auxiliary_regression = bool(getattr(args, "auxiliary_regression", False))
    if args.dataset in zoo_datasets or is_eicu:
        logging.info("MLPModel + DeepGMM for tabular dataset: %s" % args.dataset)
        if is_eicu:
            # eICU packs covariates alongside treatment and instrument
            # (x = [D, X], z = [Z, X]), so the widths depend on the cohort and
            # cannot be hardcoded. The run config carries them from the scenario.
            input_dim_g = int(getattr(args, "input_dim_g", 0))
            input_dim_f = int(getattr(args, "input_dim_f", 0))
            if input_dim_g <= 0 or input_dim_f <= 0:
                raise ValueError(
                    "eICU models need input_dim_g and input_dim_f in the run config; "
                    "read them from the scenario's *_metadata.json"
                )
            layer_widths = _eicu_hidden_widths(
                getattr(args, "hidden_widths", "64,64")
            )
            activation = _eicu_activation(
                getattr(args, "model_activation", "leaky_relu")
            )
        else:
            input_dim_g = 1
            input_dim_f = 2  # the two instruments in the zoo design
            layer_widths = [20, 20]
            activation = nn.LeakyReLU

        g_models = [
            MLPModel(input_dim=input_dim_g, layer_widths=layer_widths,
                     activation=activation).double(),
        ]
        f_models = [
            MLPModel(input_dim=input_dim_f, layer_widths=layer_widths,
                     activation=activation).double(),
        ]
        reg_models = [
            _create_auxiliary_model(auxiliary_regression, lambda: MLPModel(input_dim=input_dim_g, layer_widths=layer_widths, activation=activation).double()),
        ]
        
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    
    elif model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + DeepGMM")
        input_dim_g = 28 * 28
        input_dim_f = 1 # adversarial input is z
            
        model = [
            [LogisticRegression(input_dim_g, 1)],
            [LogisticRegression(input_dim_f, 1)],
            [_create_auxiliary_model(auxiliary_regression, lambda: LogisticRegression(input_dim_g, 1))]
        ]
    elif args.dataset in ['mnist_xz', 'femnist_xz']:
        g_models = [
            DefaultCNN(cuda=torch.cuda.is_available()),
        ]
        f_models = [
            DefaultCNN(cuda=torch.cuda.is_available()),
        ]
        reg_models = [
            _create_auxiliary_model(auxiliary_regression, lambda: DefaultCNN(cuda=torch.cuda.is_available())),
        ]
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    elif args.dataset in ["mnist_x", "femnist_x"]:
        g_models = [
            DefaultCNN(cuda=torch.cuda.is_available()),
        ]
        f_models = [
             MLPModel(input_dim=1, layer_widths=[20],
                     activation=nn.LeakyReLU).double(),
        ]
        reg_models = [
            _create_auxiliary_model(auxiliary_regression, lambda: DefaultCNN(cuda=torch.cuda.is_available())),
        ]
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    elif args.dataset in ["mnist_z", "femnist_z"]:
        g_models = [
            MLPModel(input_dim=1, layer_widths=[20],
                     activation=nn.LeakyReLU).double(),
        ]
        f_models = [
            DefaultCNN(cuda=torch.cuda.is_available()),
        ]
        reg_models = [
           _create_auxiliary_model(auxiliary_regression, lambda: MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()),
        ]
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    # elif args.dataset == "cifar_x":
    #     g_models=[
    #         DefaultCNN(cuda=True),
    #         # MLPModel(input_dim=1, layer_widths=[20],
    #         #          activation=nn.LeakyReLU).double(),
    #     ]
    #     f_models = [
    #         MLPModel(input_dim=1, layer_widths=[20],
    #                  activation=nn.LeakyReLU).double(),
    #                 # DefaultCNN(cuda=True),
    #     ]
    #     reg_models = [
    #        DefaultCNN(cuda=True),
    #     # MLPModel(input_dim=1, layer_widths=[20],
    #     #              activation=nn.LeakyReLU).double(),
    #     ]
    #     # if torch.cuda.is_available():
    #     #     for f in f_models:
    #     #         f.cuda()
    #     #     for g in g_models:
    #     #         g.cuda()
    #     if torch.cuda.is_available():
    #         for i, g in enumerate(g_models):
    #             g_models[i] = g.cuda()
    #         for i, f in enumerate(f_models):
    #             f_models[i] = f.cuda()
    #         for i, reg in enumerate(reg_models):
    #             reg_models[i] = reg.cuda()
    #     return [g_models, f_models, reg_models]
    elif args.dataset in ["cifar10_xz", "cifar_xz"]:
        g_models = [
            CIFAR10CNN(cuda=torch.cuda.is_available()),
        ]
        f_models = [
            CIFAR10CNN(cuda=torch.cuda.is_available()),
        ]
        reg_models = [
            _create_auxiliary_model(auxiliary_regression, lambda: CIFAR10CNN(cuda=torch.cuda.is_available())),
        ]
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    elif args.dataset in ["cifar10_x", "cifar_x"]:
        g_models = [
            CIFAR10CNN(cuda=torch.cuda.is_available()),
        ]
        f_models = [
             MLPModel(input_dim=1, layer_widths=[20],
                     activation=nn.LeakyReLU).double(),
        ]
        reg_models = [
            _create_auxiliary_model(auxiliary_regression, lambda: CIFAR10CNN(cuda=torch.cuda.is_available())),
        ]
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    elif args.dataset in ["cifar10_z", "cifar_z"]:
        g_models = [
            MLPModel(input_dim=1, layer_widths=[20],
                     activation=nn.LeakyReLU).double(),
        ]
        f_models = [
            CIFAR10CNN(cuda=torch.cuda.is_available()),
        ]
        reg_models = [
           _create_auxiliary_model(auxiliary_regression, lambda: MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()),
        ]
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]
    elif args.dataset == "cifar_x_old": # Kept for reference but renamed
        g_models=[
            DefaultCNN(cuda=True),
            # MLPModel(input_dim=1, layer_widths=[20],
            #          activation=nn.LeakyReLU).double(),
        ]
        f_models = [
            MLPModel(input_dim=1, layer_widths=[20],
                     activation=nn.LeakyReLU).double(),
                    # DefaultCNN(cuda=True),
        ]
        reg_models = [
           _create_auxiliary_model(auxiliary_regression, lambda: DefaultCNN(cuda=True)),
        # MLPModel(input_dim=1, layer_widths=[20],
        #              activation=nn.LeakyReLU).double(),
        ]
        # if torch.cuda.is_available():
        #     for f in f_models:
        #         f.cuda()
        #     for g in g_models:
        #         g.cuda()
        if torch.cuda.is_available():
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()
            for i, reg in enumerate(reg_models):
                if reg is not None:
                    reg_models[i] = reg.cuda()
        return [g_models, f_models, reg_models]

    elif model_name == "cnn_web" and args.dataset == "cifar10":
        logging.info("CNN_WEB + CIFAR10")
        model = CNN_WEB()
    elif model_name == "lr" and args.dataset == "cifar10":
        logging.info("LogisticRegression + CIFAR10")
        model = [
            [LogisticRegression_Cifar10(32 * 32 * 3, output_dim)],
            [LogisticRegression_Cifar10(32 * 32 * 3, output_dim)],
            [_create_auxiliary_model(auxiliary_regression, lambda: LogisticRegression_Cifar10(32 * 32 * 3, output_dim))]
        ]
    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN + MNIST")
        model = CNN_DropOut(False)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_FedShakespeare()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        if args.federated_optimizer == "FedGKT":
            client_model = resnet_client.resnet8_56(c=output_dim)
            server_model = resnet_server.resnet56_server(c=output_dim)
            model = (client_model, server_model)
        else:
            model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMpALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE")
    elif model_name == "efficientnet":
        model = EfficientNet()
    elif model_name == "darts" and args.dataset == "cifar10":
        if args.stage == "search":
            criterion = nn.CrossEntropyLoss()
            model = Network(args.init_channels, output_dim, args.layers, criterion)
        elif args.stage == "train":
            genotype = genotypes.FedNAS_V1
            model = NetworkCIFAR(args.init_channels, output_dim, args.layers, args.auxiliary, genotype)
    elif model_name == "GAN" and args.dataset == "mnist":
        gen = Generator()
        disc = Discriminator()
        model = (gen, disc)
    elif model_name == "lenet" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        from .mobile.mnn_lenet import create_mnn_lenet5_model
        
        create_mnn_lenet5_model(args.global_model_file_path)
        model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    elif model_name == "resnet20" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        from .mobile.mnn_resnet import create_mnn_resnet20_model

        create_mnn_resnet20_model(args.global_model_file_path)
        model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    else:
        raise Exception("no such model definition, please check the argument spelling or customize your own model")
    # Ensure the model is in the [g_models, f_models, reg_models] format for DeepGMM trainers
    if model is not None and (not isinstance(model, list) or len(model) < 3):
        # Fallback wrapper: wrap single model or short list into DeepGMM format
        # Note: This might share weights if we don't have access to the creation logic here,
        # but it's better than crashing. For specific models, the branches above should be used.
        model = [[model], [model], [model]]

    return model
# import logging
# import torch
# import torch.nn as nn
# from fedml.model.cv.cnn import CNN_DropOut, CNN_WEB
# from fedml.model.cv.darts import genotypes
# from fedml.model.cv.darts.model import NetworkCIFAR
# from fedml.model.cv.darts.model_search import Network
# from fedml.model.cv.efficientnet import EfficientNet
# from fedml.model.cv.mnist_gan import Generator, Discriminator
# from fedml.model.cv.mobilenet import mobilenet
# from fedml.model.cv.mobilenet_v3 import MobileNetV3
# from fedml.model.cv.resnet import resnet56
# from fedml.model.cv.resnet56 import resnet_client, resnet_server
# from fedml.model.cv.resnet_gn import resnet18
# from fedml.model.linear.lr import LogisticRegression
# from models.cnn_models import LeakySoftmaxCNN, DefaultCNN, CIFAR10CNN
# from fedml.model.linear.lr_cifar10 import LogisticRegression_Cifar10
# from fedml.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow, RNN_FedShakespeare
# from models.mlp_model import MLPModel

# def create(args, output_dim):
#     global model
#     model_name = args.model
#     logging.info("create_model. model_name = %s, output_dim = %s, dataset = %s" % (model_name, output_dim, args.dataset))
#     zoo_datasets = ['linear', 'abs', 'sin', 'step', 'zoo']
#     if args.dataset in zoo_datasets:
#         logging.info("MLPModel + DeepGMM for Zoo dataset: %s" % args.dataset)
#         input_dim_g = 1
#         input_dim_f = 2
        
#         g_models = [
#             MLPModel(input_dim=input_dim_g, layer_widths=[20, 20],
#                      activation=nn.LeakyReLU).double(),
#         ]
#         f_models = [
#             MLPModel(input_dim=input_dim_f, layer_widths=[20, 20],
#                      activation=nn.LeakyReLU).double(),
#         ]
#         reg_models = [
#             MLPModel(input_dim=input_dim_g, layer_widths=[20, 20],
#                      activation=nn.LeakyReLU).double(),
#         ]
        
#         if torch.cuda.is_available():
#             for i, g in enumerate(g_models):
#                 g_models[i] = g.cuda()
#             for i, f in enumerate(f_models):
#                 f_models[i] = f.cuda()
#             for i, reg in enumerate(reg_models):
#                 reg_models[i] = reg.cuda()
#         return [g_models, f_models, reg_models]
    
#     elif model_name == "lr" and args.dataset == "mnist":
#         logging.info("LogisticRegression + DeepGMM")
#         input_dim_g = 28 * 28
#         input_dim_f = 1 # adversarial input is z
            
#         model = [
#             [LogisticRegression(input_dim_g, 1)],
#             [LogisticRegression(input_dim_f, 1)],
#             [LogisticRegression(input_dim_g, 1)]
#         ]
#     elif args.dataset in ['mnist_xz', 'femnist_xz']:
#         g_models = [
#             DefaultCNN(cuda=True),
#         ]
#         f_models = [
#             DefaultCNN(cuda=True),
#         ]
#         reg_models = [
#             DefaultCNN(cuda=True),
#         ]
#         if torch.cuda.is_available():
#             for i, g in enumerate(g_models):
#                 g_models[i] = g.cuda()
#             for i, f in enumerate(f_models):
#                 f_models[i] = f.cuda()
#             for i, reg in enumerate(reg_models):
#                 reg_models[i] = reg.cuda()
#         return [g_models, f_models, reg_models]
#     elif args.dataset in ["mnist_x", "femnist_x"]:
#         g_models = [
#             DefaultCNN(cuda=True),
#         ]
#         f_models = [
#              MLPModel(input_dim=1, layer_widths=[20],
#                      activation=nn.LeakyReLU).double(),
#         ]
#         reg_models = [
#             DefaultCNN(cuda=True),
#         ]
#         if torch.cuda.is_available():
#             for i, g in enumerate(g_models):
#                 g_models[i] = g.cuda()
#             for i, f in enumerate(f_models):
#                 f_models[i] = f.cuda()
#             for i, reg in enumerate(reg_models):
#                 reg_models[i] = reg.cuda()
#         return [g_models, f_models, reg_models]
#     elif args.dataset in ["mnist_z", "femnist_z"]:
#         g_models = [
#             MLPModel(input_dim=1, layer_widths=[20],
#                      activation=nn.LeakyReLU).double(),
#         ]
#         f_models = [
#             DefaultCNN(cuda=True),
#         ]
#         reg_models = [
#            MLPModel(input_dim=1, layer_widths=[20],
#                      activation=nn.LeakyReLU).double(),
#         ]
#         if torch.cuda.is_available():
#             for i, g in enumerate(g_models):
#                 g_models[i] = g.cuda()
#             for i, f in enumerate(f_models):
#                 f_models[i] = f.cuda()
#             for i, reg in enumerate(reg_models):
#                 reg_models[i] = reg.cuda()
#         return [g_models, f_models, reg_models]
#     elif args.dataset in ["cifar10_xz", "cifar_xz"]:
#         g_models = [CIFAR10CNN(cuda=True)]
#         f_models = [CIFAR10CNN(cuda=True)]
#         reg_models = [CIFAR10CNN(cuda=True)]
#         if torch.cuda.is_available():
#             for i in range(len(g_models)): g_models[i] = g_models[i].cuda()
#             for i in range(len(f_models)): f_models[i] = f_models[i].cuda()
#             for i in range(len(reg_models)): reg_models[i] = reg_models[i].cuda()
#         return [g_models, f_models, reg_models]

#     elif args.dataset in ["cifar10_x", "cifar_x"]:
#         g_models = [CIFAR10CNN(cuda=True)]
#         f_models = [MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()]
#         reg_models = [CIFAR10CNN(cuda=True)]
#         if torch.cuda.is_available():
#             for i in range(len(g_models)): g_models[i] = g_models[i].cuda()
#             for i in range(len(f_models)): f_models[i] = f_models[i].cuda()
#             for i in range(len(reg_models)): reg_models[i] = reg_models[i].cuda()
#         return [g_models, f_models, reg_models]

#     elif args.dataset in ["cifar10_z", "cifar_z"]:
#         g_models = [MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()]
#         f_models = [CIFAR10CNN(cuda=True)]
#         reg_models = [MLPModel(input_dim=1, layer_widths=[20], activation=nn.LeakyReLU).double()]
#         if torch.cuda.is_available():
#             for i in range(len(g_models)): g_models[i] = g_models[i].cuda()
#             for i in range(len(f_models)): f_models[i] = f_models[i].cuda()
#             for i in range(len(reg_models)): reg_models[i] = reg_models[i].cuda()
#         return [g_models, f_models, reg_models]

#     elif model_name == "cnn_web" and args.dataset == "cifar10":
#         logging.info("CNN_WEB + CIFAR10")
#         model = CNN_WEB()
#     elif model_name == "lr" and args.dataset == "cifar10":
#         logging.info("LogisticRegression + CIFAR10")
#         model = [
#             [LogisticRegression_Cifar10(32 * 32 * 3, output_dim)],
#             [LogisticRegression_Cifar10(32 * 32 * 3, output_dim)],
#             [LogisticRegression_Cifar10(32 * 32 * 3, output_dim)]
#         ]
#     elif model_name == "cnn" and args.dataset == "mnist":
#         logging.info("CNN + MNIST")
#         model = CNN_DropOut(False)
#     elif model_name == "cnn" and args.dataset == "femnist":
#         logging.info("CNN + FederatedEMNIST")
#         model = CNN_DropOut(False)
#     elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
#         logging.info("ResNet18_GN + Federated_CIFAR100")
#         model = resnet18()
#     elif model_name == "rnn" and args.dataset == "shakespeare":
#         logging.info("RNN + shakespeare")
#         model = RNN_OriginalFedAvg()
#     elif model_name == "rnn" and args.dataset == "fed_shakespeare":
#         logging.info("RNN + fed_shakespeare")
#         model = RNN_FedShakespeare()
#     elif model_name == "lr" and args.dataset == "stackoverflow_lr":
#         logging.info("lr + stackoverflow_lr")
#         model = LogisticRegression(10000, output_dim)
#     elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
#         logging.info("RNN + stackoverflow_nwp")
#         model = RNN_StackOverFlow()
#     elif model_name == "resnet56":
#         if args.federated_optimizer == "FedGKT":
#             client_model = resnet_client.resnet8_56(c=output_dim)
#             server_model = resnet_server.resnet56_server(c=output_dim)
#             model = (client_model, server_model)
#         else:
#             model = resnet56(class_num=output_dim)
#     elif model_name == "mobilenet":
#         model = mobilenet(class_num=output_dim)
#     elif model_name == "mobilenet_v3":
#         """model_mode \in {LARGE: 5.15M, SMpALL: 2.94M}"""
#         model = MobileNetV3(model_mode="LARGE")
#     elif model_name == "efficientnet":
#         model = EfficientNet()
#     elif model_name == "darts" and args.dataset == "cifar10":
#         if args.stage == "search":
#             criterion = nn.CrossEntropyLoss()
#             model = Network(args.init_channels, output_dim, args.layers, criterion)
#         elif args.stage == "train":
#             genotype = genotypes.FedNAS_V1
#             model = NetworkCIFAR(args.init_channels, output_dim, args.layers, args.auxiliary, genotype)
#     elif model_name == "GAN" and args.dataset == "mnist":
#         gen = Generator()
#         disc = Discriminator()
#         model = (gen, disc)
#     elif model_name == "lenet" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
#         from .mobile.mnn_lenet import create_mnn_lenet5_model
        
#         create_mnn_lenet5_model(args.global_model_file_path)
#         model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
#     elif model_name == "resnet20" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
#         from .mobile.mnn_resnet import create_mnn_resnet20_model

#         create_mnn_resnet20_model(args.global_model_file_path)
#         model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
#     else:
#         raise Exception("no such model definition, please check the argument spelling or customize your own model")
#     # Ensure the model is in the [g_models, f_models, reg_models] format for DeepGMM trainers
#     if model is not None and (not isinstance(model, list) or len(model) < 3):
#         # Fallback wrapper: wrap single model or short list into DeepGMM format
#         # Note: This might share weights if we don't have access to the creation logic here,
#         # but it's better than crashing. For specific models, the branches above should be used.
#         model = [[model], [model], [model]]

#     return model
