import json
import random

import numpy as np
import paho.mqtt.client as mqtt_client
import requests
import torch

from .stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from .FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from .ImageNet.data_loader import load_partition_data_ImageNet
from .Landmarks.data_loader import load_partition_data_landmarks
from .MNIST.data_loader import load_partition_data_mnist, download_mnist
from .cifar10.data_loader import load_partition_data_cifar10
from .cifar10.efficient_loader import efficient_load_partition_data_cifar10
from .cifar100.data_loader import load_partition_data_cifar100
from .cinic10.data_loader import load_partition_data_cinic10
from .edge_case_examples.data_loader import load_poisoned_dataset
from .fed_cifar100.data_loader import load_partition_data_federated_cifar100
from .fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from .file_operation import *
from .shakespeare.data_loader import load_partition_data_shakespeare
from .stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from ..core.mlops import MLOpsConfigs


import boto3
from botocore.config import Config


def connect_mqtt(mqtt_config) -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Host!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # generate client ID with pub prefix randomly
    client_id = f"python-mqtt-{random.randint(0, 1000)}"
    client = mqtt_client.Client(client_id, clean_session=False)
    client.username_pw_set(mqtt_config["MQTT_USER"], mqtt_config["MQTT_PWD"])
    client.connect(mqtt_config["BROKER_HOST"], mqtt_config["BROKER_PORT"])
    return client


def subscribe(s3_obj, BUCKET_NAME, client: mqtt_client, args):
    def on_message(client, userdata, msg):
        logging.info(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        if msg.payload.decode():
            disconnect(client)
            make_dir(
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % args.client_id,
                )
            )
            # start download the file
            download_s3_file(
                s3_obj,
                BUCKET_NAME,
                json.loads(msg.payload.decode())["edge_id"],
                json.loads(msg.payload.decode())["dataset"],
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % args.client_id,
                ),
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % args.client_id,
                    "cifar-10-python.tar.gz",
                ),
            )

    topic = "data_svr/dataset/%s" % args.client_id
    client.subscribe(topic)
    client.on_message = on_message


def disconnect(client: mqtt_client):
    client.disconnect()
    logging.info(f"Received message, Mqtt stop listen.")


def setup_s3_service(s3_config):
    _config = Config(
        retries={
            'max_attempts': 4,
            'mode': 'standard'
        }
    )
    # s3 client
    s3 = boto3.client('s3', region_name=s3_config["CN_REGION_NAME"], aws_access_key_id=s3_config["CN_S3_AKI"],
                      aws_secret_access_key=s3_config["CN_S3_SAK"], config=_config)
    BUCKET_NAME = s3_config["BUCKET_NAME"]
    return s3, BUCKET_NAME


def data_server_preprocess(args):
    mqtt_config, s3_config, _, _ = MLOpsConfigs.fetch_all_configs()
    s3_obj, BUCKET_NAME = setup_s3_service(s3_config)

    args.private_local_data = ""
    if args.process_id == 0:
        pass
    else:
        client = connect_mqtt(mqtt_config)
        subscribe(s3_obj, BUCKET_NAME, client, args)
        if args.dataset == "cifar10":
            # Mlops Run
            # check mlops run_status
            private_local_dir, split_status, edgeids, dataset_s3_key = check_rundata(args)
            args.private_local_data = private_local_dir
            # MLOPS Run. User supply the local data dir
            if len(args.private_local_data) != 0:
                logging.info("User has set the private local data dir")
                disconnect(client)
            # MLOPS Run need to Split Data
            elif len(args.synthetic_data_url) != 0:
                if split_status == 0 or split_status == 3:
                    logging.info("Data Server Start Splitting Dataset")
                    split_edge_data(args, edgeids)
                elif split_status == 1:
                    logging.info("Data Server Is Splitting Dataset, Waiting For Mqtt Message")
                elif split_status == 2:
                    logging.info("Data Server Splitted Dataset Complete")
                    query_data_server(args, args.client_id, s3_obj, BUCKET_NAME)
                    disconnect(client)
            elif len(args.data_cache_dir) != 0:
                logging.info("No synthetic data url and private local data dir")
                return
        client.loop_forever()


def split_edge_data(args, edge_list=None):
    try:
        url = "http://127.0.0.1:5000/split_dataset"
        edge_list = json.loads(edge_list)
        json_params = {"runId": args.run_id, "edgeIds": edge_list, "dataset": args.dataset}
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        result = response.json()["errno"]
        return result
    except requests.exceptions.SSLError as err:
        print(err)


def check_rundata(args):
    # local simulation run
    logging.info("Checking Run Data")
    # mlops run
    try:
        url = "http://127.0.0.1:5000/check_rundata"
        json_params = {
            "runId": args.run_id,
        }
        response = requests.post(
            url,
            json=json_params,
            verify=True,
            headers={"content-type": "application/json", "Connection": "keep-alive"},
        )
        return response.json()["private_local_dir"], response.json()["split_status"], response.json()["edgeids"], response.json()["dataset_s3_key"]
    except requests.exceptions.SSLError as err:
        print(err)


def query_data_server(args, edgeId, s3_obj, BUCKET_NAME):
    try:
        url = "http://127.0.0.1:5000/get_edge_dataset"
        json_params = {"runId": args.run_id, "edgeId": edgeId}
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        if response.json()["errno"] == 0:
            if not check_is_download(
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % edgeId,
                    "cifar-10-batches-py",
                )
            ):
                make_dir(
                    os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % edgeId,
                    )
                )
                # start download the file
                download_s3_file(
                    s3_obj,
                    BUCKET_NAME,
                    edgeId,
                    response.json()["dataset_key"],
                    os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % edgeId,
                    ),
                    os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % edgeId,
                        "cifar-10-python.tar.gz",
                    ),
                )
            else:
                logging.info("Edge Data Already Exists. Start Training Now.")
        return response.json()
    except requests.exceptions.SSLError as err:
        print(err)
        return err


def load(args):
    return load_synthetic_data(args)


def _to_tensor(x, float_cast=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        return x
    if float_cast:
        if x.dtype in (torch.float32, torch.float64):
            return x
        return x.float()
    return x


def _normalize_batch_item(item):
    if not isinstance(item, (list, tuple)):
        raise TypeError(f"Unexpected batch item type: {type(item)}")

    if len(item) == 2:
        x, y = item
        z = None
        g = None
        w = None
    elif len(item) == 3:
        x, y, z = item
        g = None
        w = None
    elif len(item) == 4:
        # assume (g, w, x, y) or (x, y, z, extra), but preserve x,y mainly
        g, w, x, y = item
        z = None
    elif len(item) >= 5:
        g, w, x, y, z = item[:5]
    else:
        raise ValueError(f"Unexpected batch tuple length: {len(item)}")

    x = _to_tensor(x, float_cast=True)
    y = _to_tensor(y)
    z = _to_tensor(z, float_cast=True) if z is not None else None
    g = _to_tensor(g, float_cast=True) if g is not None else None
    w = _to_tensor(w, float_cast=True) if w is not None else None

    if z is not None and g is not None and w is not None:
        return (g, w, x, y, z)
    if z is not None:
        return (x, y, z)
    return (x, y)


def combine_batches(batches):
    if len(batches) == 0:
        return []

    item = batches[0]
    if len(item) == 2:
        xs = [batch[0] for batch in batches]
        ys = [batch[1] for batch in batches]
        return [(torch.cat(xs, dim=0), torch.cat(ys, dim=0))]
    if len(item) == 3:
        xs = [batch[0] for batch in batches]
        ys = [batch[1] for batch in batches]
        zs = [batch[2] for batch in batches]
        return [(torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(zs, dim=0))]
    if len(item) == 5:
        gs = [batch[0] for batch in batches]
        ws = [batch[1] for batch in batches]
        xs = [batch[2] for batch in batches]
        ys = [batch[3] for batch in batches]
        zs = [batch[4] for batch in batches]
        return [(
            torch.cat(gs, dim=0),
            torch.cat(ws, dim=0),
            torch.cat(xs, dim=0),
            torch.cat(ys, dim=0),
            torch.cat(zs, dim=0),
        )]
    raise ValueError(f"Unexpected normalized batch tuple length: {len(item)}")


def to_batch_list(data):
    # Handle simple scenario Dataset objects that hold arrays/tensors as attributes
    if hasattr(data, "x") and hasattr(data, "y"):
        x = data.x
        y = data.y
        z = getattr(data, "z", None)
        g = getattr(data, "g", None)
        w = getattr(data, "w", None)

        x = _to_tensor(x, float_cast=True)
        y = _to_tensor(y)
        z = _to_tensor(z, float_cast=True) if z is not None else None
        g = _to_tensor(g, float_cast=True) if g is not None else None
        w = _to_tensor(w, float_cast=True) if w is not None else None

        if z is not None and g is not None and w is not None:
            return [(g, w, x, y, z)]
        if z is not None:
            return [(x, y, z)]
        return [(x, y)]

    # Torch-style Dataset with .tensors
    if isinstance(data, torch.utils.data.Dataset):
        if hasattr(data, "tensors") and len(getattr(data, "tensors", ())) >= 2:
            tensors = tuple(getattr(data, "tensors", ()))
            if len(tensors) >= 5:
                g, w, x, y, z = tensors[:5]
                return [(_to_tensor(g, float_cast=True), _to_tensor(w, float_cast=True), _to_tensor(x, float_cast=True), _to_tensor(y), _to_tensor(z, float_cast=True))]
            if len(tensors) == 3:
                x, y, z = tensors
                return [(_to_tensor(x, float_cast=True), _to_tensor(y), _to_tensor(z, float_cast=True))]
            return [(_to_tensor(tensors[0], float_cast=True), _to_tensor(tensors[1]))]
        if hasattr(data, "__len__") and hasattr(data, "__getitem__"):
            lst = [data[i] for i in range(len(data))]
            converted = []
            for item in lst:
                normalized = _normalize_batch_item(item)
                converted.append(normalized)
            return converted
        raise TypeError(f"Cannot convert Dataset of type {type(data)} to batch list")

    # Generic iterable of batches (list, generator that was realized)
    try:
        lst = list(data)
    except TypeError:
        raise TypeError(f"Cannot convert object of type {type(data)} to batch list")

    converted = []
    for item in lst:
        normalized = _normalize_batch_item(item)
        converted.append(normalized)
    return converted


def load_synthetic_data(args):
    dataset_name = args.dataset
    centralized = False
    val_data_num = 0
    val_data_global = None
    val_data_local_dict = None
    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    # if dataset_name == "zoo":
    # if dataset_name == "zoo" or dataset_name == "mnist":
    zoo_datasets = ['linear', 'abs', 'sin', 'step']
    # eICU scenarios reuse the same .npz contract; the scenario file name selects
    # which g0/seed variant to load (set via scenario_name in the run config).
    is_eicu = str(dataset_name).startswith("eicu")
    if not hasattr(args, "scenario_name") or args.scenario_name is None:
        args.scenario_name = "main"
    if dataset_name == "zoo" or dataset_name == "mnist" or dataset_name in zoo_datasets or is_eicu:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        if dataset_name in zoo_datasets:
            args.scenario_name = dataset_name
        # if not hasattr(args, "scenario_name") or args.scenario_name is None:
        #     args.scenario_name = "main"
        (
            client_num,
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
        ) = load_partition_data_mnist(
            args,
            args.batch_size
        )
        # Convert DataLoaders to lists of batches for zoo datasets to enable combine_batches
        if dataset_name in zoo_datasets or is_eicu:
            train_data_local_dict = {
                cid: list(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
            }
            test_data_local_dict = {
                cid: list(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()
            }
            val_data_local_dict = {
                cid: list(val_data_local_dict[cid]) for cid in val_data_local_dict.keys()
            }
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
    # elif dataset_name == "mnist_xz":
    elif dataset_name in ["mnist_xz", "femnist_xz"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
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
        ) = load_partition_data_mnist(
            args,
            args.batch_size
        )
    # elif dataset_name == "mnist_z":
    elif dataset_name in ["mnist_z", "femnist_z"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
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
        ) = load_partition_data_mnist(
            args,
            args.batch_size
        )
    # elif dataset_name == "mnist_x":
    elif dataset_name in ["mnist_x", "femnist_x"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
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
        ) = load_partition_data_mnist(
            args,
            args.batch_size
        )
        
    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_emnist(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_shakespeare(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_cifar100(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_ImageNet(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_cache_dir, "mini_gld_train_split.csv")
        fed_test_map_file = os.path.join(args.data_cache_dir, "mini_gld_test.csv")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_cache_dir, "federated_train.csv")
        fed_test_map_file = os.path.join(args.data_cache_dir, "test.csv")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    elif dataset_name=="cifar_100_xz":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
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
        ) = efficient_load_partition_data_cifar10(
            args
            
        )

    # elif  dataset_name =="cifar_x":
    # elif dataset_name in ["cifar10_x", "cifar_x", "cifar10_z", "cifar_z", "cifar10_xz", "cifar_xz"]:
    elif dataset_name in ["cifar10_x", "cifar_x", "cifar10_z", "cifar_z", "cifar10_xz", "cifar_xz", "cifar10_y", "cifar_y"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
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
        ) = efficient_load_partition_data_cifar10(
            args
            
        )

    else:
        if dataset_name == "cifar10":
            if hasattr(args, "synthetic_data_url") or hasattr(args, "private_local_data"):
                if hasattr(args, "synthetic_data_url"):
                    args.private_local_data = ""
                else:
                    args.synthetic_data_url = ""
                if args.process_id != 0:
                    args.data_cache_dir = os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % args.client_id,
                    )
                (
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
                    # client_num
                ) = efficient_load_partition_data_cifar10(
                    args,
                    args.dataset,
                    args.data_cache_dir,
                    args.partition_method,
                    args.partition_alpha,
                    args.client_num_in_total,
                    args.batch_size,
                    # args.process_id,
                    # args.synthetic_data_url,
                    # args.private_local_data
                )

                if centralized:
                    train_data_local_num_dict = {
                        0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())
                    }
                    train_data_local_dict = {
                        0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in
                            train_data_local_dict[cid]]
                    }
                    test_data_local_dict = {
                        0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]
                    }
                    args.client_num_in_total = 1

                if full_batch:
                    from types import SimpleNamespace

                    def _wrap_global_inline(data):
                        combined = combine_batches(to_batch_list(data))
                        if not combined:
                            return SimpleNamespace(x=None, y=None, z=None, g=None, w=None)
                        item = combined[0]
                        if len(item) == 2:
                            x_cat, y_cat = item
                            return SimpleNamespace(x=x_cat, y=y_cat, z=None, g=None, w=None)
                        if len(item) == 3:
                            x_cat, y_cat, z_cat = item
                            return SimpleNamespace(x=x_cat, y=y_cat, z=z_cat, g=None, w=None)
                        if len(item) == 5:
                            g_cat, w_cat, x_cat, y_cat, z_cat = item
                            return SimpleNamespace(x=x_cat, y=y_cat, z=z_cat, g=g_cat, w=w_cat)
                        raise ValueError(f"Unexpected combined batch length: {len(item)}")

                    train_data_global = _wrap_global_inline(train_data_global)
                    test_data_global = _wrap_global_inline(test_data_global)
                    train_data_local_dict = {
                        cid: combine_batches(to_batch_list(train_data_local_dict[cid])) for cid in train_data_local_dict.keys()
                    }
                    test_data_local_dict = {
                        cid: combine_batches(to_batch_list(test_data_local_dict[cid])) for cid in test_data_local_dict.keys()
                    }
                    args.batch_size = args_batch_size

                dataset = [
                    train_data_num,
                    test_data_num,
                    train_data_global,
                    test_data_global,
                    train_data_local_num_dict,
                    train_data_local_dict,
                    test_data_local_dict,
                    class_num,
                ]

                return dataset, class_num
            else:
                # data_loader = load_partition_data_cifar10
                data_loader = efficient_load_partition_data_cifar10

        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = data_loader(
            args.dataset,
            args.data_cache_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
        )

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())
        }
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]
        }
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]
        }
        args.client_num_in_total = 1

    if full_batch:
        # combine_batches returns a list with a single normalized tuple.
        # Wrap globals into a simple object with attributes so downstream code
        # can access `.x`, `.y`, `.z`, `.g`, `.w` as expected.
        from types import SimpleNamespace

        def _wrap_global(data):
            combined = combine_batches(to_batch_list(data))
            if not combined:
                return SimpleNamespace(x=None, y=None, z=None, g=None, w=None)
            item = combined[0]
            if len(item) == 2:
                x_cat, y_cat = item
                return SimpleNamespace(x=x_cat, y=y_cat, z=None, g=None, w=None)
            if len(item) == 3:
                x_cat, y_cat, z_cat = item
                return SimpleNamespace(x=x_cat, y=y_cat, z=z_cat, g=None, w=None)
            if len(item) == 5:
                g_cat, w_cat, x_cat, y_cat, z_cat = item
                return SimpleNamespace(x=x_cat, y=y_cat, z=z_cat, g=g_cat, w=w_cat)
            raise ValueError(f"Unexpected combined batch length: {len(item)}")

        train_data_global = _wrap_global(train_data_global)
        test_data_global = _wrap_global(test_data_global)
        val_data_global = _wrap_global(val_data_global)

        # For local dicts we keep the combined list-of-1 tuples; callers that
        # expect iterables over batches will still work. If later code needs
        # per-client `.x` access, we can wrap those as well.
        train_data_local_dict = {
            cid: combine_batches(to_batch_list(train_data_local_dict[cid])) for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {cid: combine_batches(to_batch_list(test_data_local_dict[cid])) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [
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
    ]

    return dataset, class_num


def load_poisoned_dataset_from_edge_case_examples(args):
    return load_poisoned_dataset(args=args)

