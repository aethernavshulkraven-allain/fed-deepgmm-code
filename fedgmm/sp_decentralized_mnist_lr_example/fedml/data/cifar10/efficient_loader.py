import logging
from contextlib import nullcontext

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from scenarios.abstract_scenario import AbstractScenario
from .without_reload import CIFAR10_truncated, CIFAR10_truncated_WO_reload
from torch.utils.data import DataLoader, TensorDataset, Subset


def _profile_span(args, phase, detail=""):
    profiler = getattr(args, "_fedgmm_runtime_profiler", None)
    if profiler is None:
        return nullcontext()
    return profiler.span(phase, detail=detail)


def _client_loader_kwargs(args, shuffle=True):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
    }
    num_workers = max(0, int(getattr(args, "dataloader_num_workers", 0)))
    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if bool(getattr(args, "dataloader_pin_memory", False)) and bool(getattr(args, "using_gpu", False)):
        kwargs["pin_memory"] = True
    return kwargs


# generate the non-IID distribution for all methods
def read_data_distribution(filename="./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt",):
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(",", ""))
    return distribution


def read_net_dataidx_map(filename="./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt",):
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])

    return train_transform, valid_transform


def load_cifar10_data(datadir, process_id, private_local_data, resize=32, augmentation=True, data_efficient_load=False):
    train_transform, test_transform = _data_transforms_cifar10()

    is_download = True;

    if data_efficient_load:
        cifar10_train_ds = CIFAR10(datadir, train=True, download=True, transform=train_transform)
        cifar10_test_ds = CIFAR10(datadir, train=False, download=True, transform=test_transform)
    else:
        cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=is_download, transform=train_transform)
        cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=is_download, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

    return (X_train, y_train, X_test, y_test, cifar10_train_ds, cifar10_test_ds)


def partition_data(dataset, datadir, partition, n_nets, alpha, process_id, private_local_data):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test, cifar10_train_ds, cifar10_test_ds = load_cifar10_data(datadir, process_id, private_local_data)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = "./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt"
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = "./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt"
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, cifar10_train_ds, cifar10_test_ds


# for centralized training
def get_dataloader(args,train,test,dev,):
    clients_num = range(args.client_num_in_total)
    with _profile_span(args, "dataloader_tensor_dataset_create"):
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

    train_data_local_dict = {user: [] for user in clients_num}
    test_data_local_dict = {user: [] for user in clients_num}
    val_data_local_dict = {user: [] for user in clients_num}
    train_data_local_num_dict = {}
    test_data_local_num_dict = {}
    val_data_local_num_dict = {}
    # np.random.seed(int(time.time()))
    with _profile_span(args, "dataloader_dirichlet_partition"):
        proportions_train = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
        proportions_test = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
        proportions_dev = np.random.dirichlet([args.partition_alpha] * args.client_num_in_total)
        train_samples_per_client = (proportions_train * available_train_samples).astype(int) + min_samples_per_client
        test_samples_per_client = (proportions_test * available_test_samples).astype(int) + min_samples_per_client
        dev_samples_per_client = (proportions_dev * available_dev_samples).astype(int) + min_samples_per_client

        # Re-adjust to ensure the exact number of samples is distributed
        train_samples_per_client[-1] = num_train_samples - sum(train_samples_per_client[:-1])
        test_samples_per_client[-1] = num_test_samples - sum(test_samples_per_client[:-1])
        dev_samples_per_client[-1] = num_dev_samples - sum(dev_samples_per_client[:-1])

    # Creating subsets and data loaders for each client
    train_loader_kwargs = _client_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = _client_loader_kwargs(args, shuffle=True)
    loader_detail = (
        f"clients={args.client_num_in_total}; batch_size={args.batch_size}; "
        f"num_workers={train_loader_kwargs.get('num_workers', 0)}; "
        f"pin_memory={bool(train_loader_kwargs.get('pin_memory', False))}"
    )
    with _profile_span(args, "dataloader_create_per_client_loaders", detail=loader_detail):
        indices_train = np.random.permutation(num_train_samples)
        indices_test = np.random.permutation(num_test_samples)
        indices_dev = np.random.permutation(num_dev_samples)
        start = 0
        for i in clients_num:
            end = start + train_samples_per_client[i]
            subset_indices = indices_train[start:end]
            train_data_local_dict[i] = DataLoader(Subset(train_dataset, subset_indices), **train_loader_kwargs)
            start = end

        start = 0
        for i in clients_num:
            end = start + test_samples_per_client[i]
            subset_indices = indices_test[start:end]
            test_data_local_dict[i] = DataLoader(Subset(test_dataset, subset_indices), **eval_loader_kwargs)
            start = end

        start = 0
        for i in clients_num:
            end = start + dev_samples_per_client[i]
            subset_indices = indices_dev[start:end]
            val_data_local_dict[i] = DataLoader(Subset(dev_dataset, subset_indices), **eval_loader_kwargs)
            start = end
    for user in clients_num:
        # Record real subset cardinalities. ``len(loader) * batch_size`` rounds
        # the final partial batch up and corrupts both aggregation weights and
        # deterministic full-gradient preflight accounting.
        train_data_local_num_dict[user] = int(train_samples_per_client[user])
        test_data_local_num_dict[user] = int(test_samples_per_client[user])
        val_data_local_num_dict[user] = int(dev_samples_per_client[user])

    return (
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        train_data_local_num_dict,
        test_data_local_num_dict,
        val_data_local_num_dict
    )


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR10(
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    transform_train, transform_test = _data_transforms_cifar10()

    if data_efficient_load:
        dl_obj = CIFAR10_truncated_WO_reload
        train_ds = dl_obj(
            datadir, dataidxs=dataidxs, train=True, transform=transform_train, full_dataset=full_train_dataset
        )
        test_ds = dl_obj(datadir, train=False, transform=transform_test, full_dataset=full_test_dataset)
    else:
        dl_obj = CIFAR10_truncated
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True,)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True,)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(
    process_id,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    data_efficient_load=True,
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
        cifar10_train_ds,
        cifar10_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def resolve_cifar10_scenario_archive(args):
    """Resolve and verify the scenario archive a CIFAR-10 run declares.

    The path used to be built by string concatenation, "data/" + dataset + "/" +
    scenario_name, which cannot express an absolute scenario_name: it produced
    "data/cifar10_z//abs/path.npz" and the run died on load. The MNIST/FEMNIST
    loader only accepted absolute paths by accident -- os.path.join discards
    everything before an absolute component -- so this states the rule outright.

    Absolute paths are the only way to reach the corrected CIFAR-X/XZ archives,
    whose originals sit at the relative location under the same name. So the
    declared scenario_checksum is verified here too: a run that names one archive
    must never train on another.
    """
    import hashlib
    import os

    filename = args.scenario_name + ".npz"
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join("data", args.dataset, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scenario archive not found: {os.path.abspath(path)}")

    resolved_path = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    resolved_sha256 = digest.hexdigest()

    declared = str(getattr(args, "scenario_checksum", "") or "")
    if declared and declared != resolved_sha256:
        raise ValueError(
            f"Scenario archive checksum mismatch for {resolved_path}: config declares "
            f"{declared} but the file on disk is {resolved_sha256}. Refusing to train on "
            "an archive the run does not describe."
        )
    setattr(args, "_resolved_scenario_path", resolved_path)
    setattr(args, "_resolved_scenario_sha256", resolved_sha256)
    logging.info(f"Resolved scenario archive: {resolved_path} sha256={resolved_sha256}")
    return resolved_path


def efficient_load_partition_data_cifar10(args):

    with _profile_span(args, "scenario_load", detail=f"dataset={args.dataset}; scenario={args.scenario_name}"):
        scenario = AbstractScenario(filename=resolve_cifar10_scenario_archive(args))

    with _profile_span(args, "scenario_info"):
        scenario.info()
    with _profile_span(args, "scenario_to_tensor"):
        scenario.to_tensor()
    # Keep scenario tensors on CPU during loading.
    # The runtime layer moves batches to FedML's resolved device.

    with _profile_span(args, "scenario_split_extract"):
        train = scenario.get_dataset("train")
        dev = scenario.get_dataset("dev")
        test = scenario.get_dataset("test")
    # (
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     net_dataidx_map,
    #     traindata_cls_counts,
    #     cifar10_train_ds,
    #     cifar10_test_ds,
    # ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha, process_id,)
    # class_num = len(np.unique(y_train))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    with _profile_span(args, "get_dataloader"):
        train_data_local_dict, test_data_local_dict,\
        val_data_local_dict, train_data_local_num_dict,\
        test_data_local_num_dict, val_data_local_num_dict = get_dataloader(args,train,test,dev)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    # test_data_num = len(test_data_global)

    # # get local dataset
    # data_local_num_dict = dict()
    # train_data_local_dict = dict()
    # test_data_local_dict = dict()

    # for client_idx in range(client_number):
    #     dataidxs = net_dataidx_map[client_idx]
    #     local_data_num = len(dataidxs)
    #     data_local_num_dict[client_idx] = local_data_num
    #     logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

    #     # training batch size = 64; algorithms batch size = 32
    #     train_data_local, test_data_local = get_dataloader(
    #         dataset,
    #         data_dir,
    #         batch_size,
    #         batch_size,
    #         dataidxs,
    #         data_efficient_load=True,
    #         full_train_dataset=cifar10_train_ds,
    #         full_test_dataset=cifar10_test_ds,
    #     )
    #     logging.info(
    #         "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
    #         % (client_idx, len(train_data_local), len(test_data_local))
    #     )
    #     train_data_local_dict[client_idx] = train_data_local
    #     test_data_local_dict[client_idx] = test_data_local
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
        None,
    )
