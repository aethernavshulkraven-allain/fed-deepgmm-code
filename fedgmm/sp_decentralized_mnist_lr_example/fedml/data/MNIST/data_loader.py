import json
import os
import numpy as np
import wget
import time
from ...ml.engine import ml_engine_adapter
import zipfile
from ...constants import FEDML_DATA_MNIST_URL
import hashlib
import logging
from scenarios.abstract_scenario import AbstractScenario
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from experiment_utils import exact_sample_count

cwd = os.getcwd()


def load_data_natural(args, train, test, dev):
    """Partition by the natural client key carried in the scenario.

    Used for eICU, where a client is a real hospital rather than a Dirichlet draw.
    Two differences from :func:`load_data` matter scientifically:

    * All three splits are keyed off the *same* ``client_id``, so client k's train
      and validation rows describe the same hospital. The Dirichlet path draws the
      splits independently, which is harmless for i.i.d. synthetic data but would
      mean a hospital was validated against other hospitals' patients.
    * Client sizes come from the data, not from ``partition_alpha``. Real hospital
      imbalance is the heterogeneity this study is about, so it must not be resampled.

    Clients absent from a split get an empty loader rather than being dropped, so
    client indices stay aligned across splits.
    """
    splits = {"train": train, "test": test, "dev": dev}
    for name, split in splits.items():
        if getattr(split, "client_id", None) is None:
            raise ValueError(
                f"partition_method='natural' requires client_id on the {name} split; "
                "regenerate the scenario with a client_id array"
            )

    datasets = {
        name: TensorDataset(split.g, split.w, split.x, split.y, split.z)
        for name, split in splits.items()
    }

    # Only clients represented in *every* split are usable: a client with no
    # training rows cannot train, and one with no validation/test rows cannot be
    # evaluated (the trainer indexes each client's first eval batch directly).
    # Dropping them here, loudly, is better than silently mis-indexing.
    per_split_ids = {
        name: set(np.asarray(split.client_id).reshape(-1).tolist())
        for name, split in splits.items()
    }
    usable = set.intersection(*per_split_ids.values())
    dropped = set.union(*per_split_ids.values()) - usable
    if dropped:
        logging.warning(
            "natural partition: dropping %d of %d clients absent from at least one "
            "split (train/dev/test coverage = %d/%d/%d)",
            len(dropped),
            len(usable) + len(dropped),
            len(per_split_ids["train"]),
            len(per_split_ids["dev"]),
            len(per_split_ids["test"]),
        )
    if not usable:
        raise ValueError(
            "natural partition left no client present in all of train/dev/test; "
            "the cohort is too small or too fragmented for a federated run"
        )

    unique_clients = np.array(sorted(usable))
    n_clients = int(len(unique_clients))

    # The natural partition defines the client count; a mismatched config value
    # would silently drop hospitals or index past the end.
    if int(getattr(args, "client_num_in_total", n_clients)) != n_clients:
        logging.warning(
            "overriding client_num_in_total=%s with %d natural clients",
            getattr(args, "client_num_in_total", None),
            n_clients,
        )
    args.client_num_in_total = n_clients
    if int(getattr(args, "client_num_per_round", n_clients)) > n_clients:
        args.client_num_per_round = n_clients

    index_of = {int(c): i for i, c in enumerate(unique_clients)}

    per_split = {}
    for name, split in splits.items():
        ids = np.asarray(split.client_id).reshape(-1)
        buckets = {i: [] for i in range(n_clients)}
        for row, cid in enumerate(ids):
            target = index_of.get(int(cid))
            if target is not None:  # rows of dropped clients are excluded
                buckets[target].append(row)
        per_split[name] = buckets

    def build(name):
        local, counts = {}, {}
        for client, rows in per_split[name].items():
            subset = Subset(datasets[name], rows)
            batch_size = max(min(int(args.batch_size), max(len(rows), 1)), 1)
            local[client] = DataLoader(
                subset, batch_size=batch_size, shuffle=len(rows) > 0
            )
            counts[client] = len(rows)
        return local, counts

    train_local, train_counts = build("train")
    test_local, _ = build("test")
    val_local, _ = build("dev")

    logging.info(
        "natural partition: %d clients, train sizes min=%d median=%d max=%d",
        n_clients,
        min(train_counts.values()),
        int(np.median(list(train_counts.values()))),
        max(train_counts.values()),
    )
    return train_local, test_local, val_local, train_counts, {}, {}


def load_data(args, train, test, dev):
    # Assuming args has attributes 'client_num' and 'batch_size'
    clients_num = range(args.client_num_in_total)

    # Creating TensorDatasets for train, test, and dev sets
    train_dataset = TensorDataset(train.g, train.w, train.x, train.y, train.z)
    test_dataset = TensorDataset(test.g, test.w, test.x, test.y, test.z)
    dev_dataset = TensorDataset(dev.g, dev.w, dev.x, dev.y, dev.z)
    num_train_samples = len(train_dataset)
    num_test_samples = len(test_dataset)
    num_dev_samples = len(dev_dataset)
    min_samples_per_client = 5
    available_train_samples = num_train_samples - args.client_num_in_total * min_samples_per_client
    available_test_samples = num_test_samples - args.client_num_in_total * min_samples_per_client
    available_dev_samples = num_dev_samples - args.client_num_in_total * min_samples_per_client

    # Creating DataLoader for batching
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    # Dictionaries to hold the data for each client
    train_data_local_dict = {user: [] for user in clients_num}
    test_data_local_dict = {user: [] for user in clients_num}
    val_data_local_dict = {user: [] for user in clients_num}
    train_data_local_num_dict = {}
    test_data_local_num_dict = {}
    val_data_local_num_dict = {}
    # np.random.seed(int(time.time()))
    proportions_train = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
    proportions_test = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
    proportions_dev = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)

    # Adjust proportions to consider the minimum samples reserved for each client
    train_samples_per_client = (proportions_train * available_train_samples).astype(int) + min_samples_per_client
    test_samples_per_client = (proportions_test * available_test_samples).astype(int) + min_samples_per_client
    dev_samples_per_client = (proportions_dev * available_dev_samples).astype(int) + min_samples_per_client

    # Re-adjust to ensure the exact number of samples is distributed
    train_samples_per_client[-1] = num_train_samples - sum(train_samples_per_client[:-1])
    test_samples_per_client[-1] = num_test_samples - sum(test_samples_per_client[:-1])
    dev_samples_per_client[-1] = num_dev_samples - sum(dev_samples_per_client[:-1])

    # Creating subsets and data loaders for each client
    indices_train = np.random.permutation(num_train_samples)
    indices_test = np.random.permutation(num_test_samples)
    indices_dev = np.random.permutation(num_dev_samples)

    # train_data_local_dict = {}
    # test_data_local_dict = {}
    # val_data_local_dict = {}

    start = 0
    for i in clients_num:
        end = start + train_samples_per_client[i]
        subset_indices = indices_train[start:end]
        # train_data_local_dict[i] = DataLoader(Subset(train_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        train_data_local_dict[i] = DataLoader(Subset(train_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        train_data_local_num_dict[i] = exact_sample_count(subset_indices)

        start = end

    start = 0
    for i in clients_num:
        end = start + test_samples_per_client[i]
        subset_indices = indices_test[start:end]
        # test_data_local_dict[i] = DataLoader(Subset(test_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        test_data_local_dict[i] = DataLoader(Subset(test_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        test_data_local_num_dict[i] = exact_sample_count(subset_indices)

        start = end

    start = 0
    for i in clients_num:
        end = start + dev_samples_per_client[i]
        subset_indices = indices_dev[start:end]
        # val_data_local_dict[i] = DataLoader(Subset(dev_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        val_data_local_dict[i] = DataLoader(Subset(dev_dataset, subset_indices), batch_size=args.batch_size, shuffle=True)
        val_data_local_num_dict[i] = exact_sample_count(subset_indices)

        start = end
    # Distributing the batches among clients
    # for i, (train_batch, test_batch, dev_batch) in enumerate(zip(train_loader, test_loader, dev_loader)):
    #     user = i % args.client_num_in_total
    #     train_data_local_dict[user].append(train_batch)
    #     test_data_local_dict[user].append(test_batch)
    #     val_data_local_dict[user].append(dev_batch)

    
    # Calculate number of batches directly if dataset sizes and batch_size are known
#     num_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
#     num_test_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
#     num_dev_batches = (len(dev_dataset) + args.batch_size - 1) // args.batch_size

# # Generate Dirichlet distribution proportions
#     proportions_train = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total, 1).flatten()
#     proportions_test = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total, 1).flatten()
#     proportions_dev = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total, 1).flatten()

# # Distribute batches based on calculated proportions and avoid early list conversion
#     for i in clients_num:
#       train_indices = np.random.choice(num_train_batches, int(proportions_train[i] * num_train_batches), replace=False)
#       test_indices = np.random.choice(num_test_batches, int(proportions_test[i] * num_test_batches), replace=False)
#       dev_indices = np.random.choice(num_dev_batches, int(proportions_dev[i] * num_dev_batches), replace=False)

#     # Convert loaders to lists when needed and distribute
#       all_train_batches = list(train_loader) if 'all_train_batches' not in locals() else all_train_batches
#       all_test_batches = list(test_loader) if 'all_test_batches' not in locals() else all_test_batches
#       all_dev_batches = list(dev_loader) if 'all_dev_batches' not in locals() else all_dev_batches

#       for idx in train_indices:
#         train_data_local_dict[i].append(all_train_batches[idx])
#       for idx in test_indices:
#         test_data_local_dict[i].append(all_test_batches[idx])
#       for idx in dev_indices:
#         val_data_local_dict[i].append(all_dev_batches[idx])


    return (
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        train_data_local_num_dict,
        test_data_local_num_dict,
        val_data_local_num_dict
    )


def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir, exist_ok=True)

    file_path = os.path.join(data_cache_dir, "MNIST.zip")
    logging.info(file_path)

    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    file_extracted_path = os.path.join(data_cache_dir, "MNIST")
    if not os.path.exists(file_extracted_path):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_cache_dir)

def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = sorted(cdata["users"])

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size, device_id, train_path="MNIST_mobile", test_path="MNIST_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_mnist(batch_size, train_path, test_path)


# def load_partition_data_mnist(
#     args, batch_size
# ):
#     scenario = AbstractScenario(filename="data/mnist_x/" + args.scenario_name + ".npz") 
#     scenario.info()
#     scenario.to_tensor()
#     # scenario.to_cuda()
#     if torch.cuda.is_available():
#         scenario.to_cuda()
#     train = scenario.get_dataset("train")
#     dev = scenario.get_dataset("dev")
#     test = scenario.get_dataset("test")
    
#     train_data_local_dict, test_data_local_dict,\
#     val_data_local_dict, train_data_local_num_dict,\
#     test_data_local_num_dict, val_data_local_num_dict = load_data(args, train, test, dev)

#     return (
#         args.client_num_in_total,
#         train.y.shape[0],
#         test.y.shape[0],
#         dev.y.shape[0],
#         train,
#         test,
#         dev,
#         train_data_local_num_dict,
#         train_data_local_dict,
#         test_data_local_dict,
#         val_data_local_dict,
#         10,
#     )

def load_partition_data_mnist(
    args, batch_size
):
    import os
    # Try multiple possible paths for the scenario file
    scenario_filename = args.scenario_name + ".npz"
    # Determine primary directory based on dataset name
    zoo_datasets = ['linear', 'abs', 'sin', 'step']
    if args.dataset in zoo_datasets:
        primary_dir = "zoo"
    elif args.dataset == "mnist":
        primary_dir = "mnist_x" # default fallback for plain 'mnist'
    else:
        primary_dir = args.dataset # use dataset name directly (e.g. 'mnist_z', 'mnist_x')

    possible_paths = [
        os.path.join("data", primary_dir, scenario_filename),
        os.path.join(args.data_cache_dir, primary_dir, scenario_filename),
        os.path.join("data", "mnist_x", scenario_filename), # legacy fallback
        os.path.join("data", "zoo", scenario_filename),     # legacy fallback
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
            
    if path is None:
        raise FileNotFoundError(f"Could not find scenario file {scenario_filename} in any of these locations: {possible_paths}. Please check your YAML 'dataset' value and that you have generated the data.")

    # The cwd-relative candidate is searched BEFORE data_cache_dir, and runs are
    # launched from this directory -- whose data/ holds the original archives. A
    # config that points data_cache_dir at a corrected dataset would therefore
    # train on the original one without any visible sign. Refuse the ambiguity
    # rather than silently preferring cwd.
    cwd_candidate, cache_candidate = possible_paths[0], possible_paths[1]
    if (os.path.abspath(cwd_candidate) != os.path.abspath(cache_candidate)
            and os.path.exists(cwd_candidate) and os.path.exists(cache_candidate)):
        raise ValueError(
            "Ambiguous scenario archive: both "
            f"{os.path.abspath(cwd_candidate)} and {os.path.abspath(cache_candidate)} "
            "exist, and the working-directory copy would silently win. Set "
            "scenario_name to an absolute path to state which archive this run means."
        )

    resolved_path = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    resolved_sha256 = digest.hexdigest()

    # scenario_checksum is already an effective-config field but was never
    # verified on this path, so a run could declare one archive and load another.
    declared = str(getattr(args, "scenario_checksum", "") or "")
    if declared and declared != resolved_sha256:
        raise ValueError(
            f"Scenario archive checksum mismatch for {resolved_path}: config declares "
            f"{declared} but the file on disk is {resolved_sha256}. Refusing to train on "
            "an archive the run does not describe."
        )

    setattr(args, "_resolved_scenario_path", resolved_path)
    setattr(args, "_resolved_scenario_sha256", resolved_sha256)
    logging.info(f"Loading scenario from: {path}")
    logging.info(f"Resolved scenario archive: {resolved_path} sha256={resolved_sha256}")
    scenario = AbstractScenario(filename=path) 
    scenario.info()
    scenario.to_tensor()
    # Keep scenario tensors on CPU during loading.
    # The runtime layer moves tensors to FedML's resolved device.

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    
    # 'natural' keys the partition off the scenario's client_id (e.g. eICU
    # hospitalid); anything else keeps the original Dirichlet behaviour.
    partitioner = (
        load_data_natural
        if str(getattr(args, "partition_method", "")).lower() == "natural"
        else load_data
    )
    train_data_local_dict, test_data_local_dict,\
    val_data_local_dict, train_data_local_num_dict,\
    test_data_local_num_dict, val_data_local_num_dict = partitioner(args, train, test, dev)
    zoo_datasets = ['linear', 'abs', 'sin', 'step']
    if args.dataset in zoo_datasets or str(args.dataset).startswith("eicu"):
        class_num = 1
    else:
        class_num = 10 # Default for MNIST

    return (
        args.client_num_in_total,
        train.y.shape[0],
        test.y.shape[0],
        dev.y.shape[0],
        train,
        test,
        dev,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        # None,
        class_num,
    )
