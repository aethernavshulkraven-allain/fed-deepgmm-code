#!/usr/bin/env python3
"""Federated closed-form polynomial OLS and 2SLS baselines for the low-dimensional curves.

For each synthetic function (abs, step, linear, sin) two baselines are fitted
on the training split, which is distributed over the same clients as the
federated DeepGMM runs:

  OLS   ridge regression of y on polynomial features of x. Ignores the
        instrument, so it is biased by the x-y confounding.
  2SLS  sieve two-stage least squares: every polynomial feature of x is
        projected onto polynomial features of the instrument z (first stage),
        and y is regressed on those projections (second stage).

Clients. The partition is the one the federated runs use: ``load_data`` from
``fedgmm/sp_decentralized_mnist_lr_example/fedml/data/MNIST/data_loader.py``
with the RNG seeded as ``fedml.init`` seeds it (1000 clients, Dirichlet alpha
0.5). Train, dev and test rows are each split across the clients.

Protocol. Every client message is a sum over that client's own rows, checked
against a fixed schema whose shapes depend only on public quantities (feature
counts, number of candidates), so no per-row value reaches the server.

  round 1  n, sum x, sum z, sum y                               -> input means, mean y
  round 2  sum (x - mean)^2, sum (z - mean)^2                   -> input stds
  round 3  sums of every monomial of the standardized inputs    -> monomial means
  round 4  centered cross products: x-x, z-z, z-x, x-y, z-y     -> standardized Gram blocks
  server   solves ridge OLS and 2SLS for every grid point from the Gram blocks
  round 5  per-candidate sums of squared errors on the client's dev and test rows
           (against y and against the true g)                   -> selection and readouts

Lower-degree features are the leading columns of the maximum-degree monomial
list (ordered by total degree), so one round 4 serves the whole grid.

Selection (validation split only; test MSE is a post-selection readout):
  2SLS  lowest structural MSE of g-hat against the true g on dev, the rule the
        DeepGMM runs use for checkpoint selection.
  OLS   lowest error predicting y on dev, OLS's own goal.
  Ties go to lower degree, then stronger ridge.

Because the closed form needs only these sums, the result equals the pooled
closed form; tests/test_lowdim_iv_baselines_federated.py checks this on every
grid point.

Usage (from the repository root):
  python scripts/lowdim_iv_baselines_federated.py
Outputs go to results/lowdim_iv_baselines_federated/.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "fedgmm" / "sp_decentralized_mnist_lr_example"
ZOO_DIR = EXPERIMENT_DIR / "data" / "zoo"
OUTPUT_ROOT = ROOT / "results" / "lowdim_iv_baselines_federated"
NAME = "lowdim_iv_baselines_federated"

FUNCTIONS = ("abs", "step", "linear", "sin")
METHODS = ("poly_ols", "poly_2sls")
X_DEGREES = tuple(range(1, 11))
Z_DEGREES = tuple(range(1, 11))
# OLS tuned to predict y keeps gaining (very slightly) with degree, so it gets
# a longer degree range; its grid is tiny.
OLS_X_DEGREES = tuple(range(1, 16))
# Ridge penalty per sample on standardized features. 1e-10 is effectively
# unpenalized while keeping high-degree solves well posed.
RIDGE_ALPHAS = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SELECTION = {"poly_ols": "validation_y_mse", "poly_2sls": "validation_mse"}

CLIENTS = 1000
PARTITION_ALPHA = 0.5
SEEDS = (0, 1, 2)
PRIMARY_SEED = 0


# ---------------------------------------------------------------- shared definitions

def monomial_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    """All exponent tuples of total degree 1..degree (no constant term), by total degree."""
    return [
        exps for total in range(1, degree + 1)
        for exps in itertools.product(range(total + 1), repeat=n_vars)
        if sum(exps) == total
    ]


def complexity(params: dict[str, Any]) -> tuple:
    """Tie-break toward the simpler fit: lower degrees, then stronger penalties."""
    return (params["x_degree"], params.get("z_degree", 0),
            -params.get("alpha", params.get("second_alpha", 0.0)), -params.get("first_alpha", 0.0))


def load(dataset: str) -> dict[str, np.ndarray]:
    path = ZOO_DIR / f"{dataset}.npz"
    with np.load(path, allow_pickle=True) as archive:
        data = {}
        for split in ("train", "dev", "test"):
            data[f"{split}_x"] = np.asarray(archive[f"{split}_x"], dtype=float).reshape(-1, 1)
            data[f"{split}_z"] = np.asarray(archive[f"{split}_z"], dtype=float)
            data[f"{split}_y"] = np.asarray(archive[f"{split}_y"], dtype=float).reshape(-1)
            data[f"{split}_g"] = np.asarray(archive[f"{split}_g"], dtype=float).reshape(-1)
    data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def monomials(scaled: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    return np.prod(scaled[:, None, :] ** exponents[None, :, :], axis=2)


# ---------------------------------------------------------------- messages

class SchemaError(ValueError):
    pass


def check_message(message: dict[str, Any], schema: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Accept a client message only if it has exactly the schema's keys and shapes."""
    if set(message) != set(schema):
        raise SchemaError(f"message keys {sorted(message)} != schema {sorted(schema)}")
    checked = {}
    for key, shape in schema.items():
        value = np.asarray(message[key], dtype=float)
        if value.shape != shape:
            raise SchemaError(f"{key}: shape {value.shape} != schema {shape}")
        if not np.isfinite(value).all():
            raise SchemaError(f"{key}: not finite")
        checked[key] = value
    return checked


# ---------------------------------------------------------------- clients

class Client:
    """Holds one client's rows. Each round returns sums over those rows only."""

    def __init__(self, rows: dict[str, dict[str, np.ndarray]]):
        self._rows = rows  # {"train"|"dev"|"test": {"x","z","y","g"}}

    def round1(self) -> dict[str, Any]:
        tr = self._rows["train"]
        return {"n": [len(tr["y"])], "sum_x": tr["x"].sum(axis=0), "sum_z": tr["z"].sum(axis=0),
                "sum_y": [tr["y"].sum()]}

    def round2(self, public: dict[str, np.ndarray]) -> dict[str, Any]:
        tr = self._rows["train"]
        return {"ss_x": ((tr["x"] - public["mean_x"]) ** 2).sum(axis=0),
                "ss_z": ((tr["z"] - public["mean_z"]) ** 2).sum(axis=0)}

    def _raw(self, split: str, public: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rows = self._rows[split]
        raw_x = monomials((rows["x"] - public["mean_x"]) / public["std_x"], public["exp_x"])
        raw_z = monomials((rows["z"] - public["mean_z"]) / public["std_z"], public["exp_z"])
        return raw_x, raw_z

    def round3(self, public: dict[str, np.ndarray]) -> dict[str, Any]:
        raw_x, raw_z = self._raw("train", public)
        return {"sum_mx": raw_x.sum(axis=0), "sum_mz": raw_z.sum(axis=0)}

    def round4(self, public: dict[str, np.ndarray]) -> dict[str, Any]:
        raw_x, raw_z = self._raw("train", public)
        cx, cz = raw_x - public["mono_mean_x"], raw_z - public["mono_mean_z"]
        cy = self._rows["train"]["y"] - public["mean_y"]
        return {"xx": cx.T @ cx, "zz": cz.T @ cz, "zx": cz.T @ cx, "xy": cx.T @ cy, "zy": cz.T @ cy}

    def round5(self, public: dict[str, np.ndarray], blocks: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
        """Sums of squared errors per candidate. ``blocks``: (number of x features, coefficient matrix)."""
        out = {"n_dev": [len(self._rows["dev"]["y"])], "n_test": [len(self._rows["test"]["y"])]}
        for split in ("dev", "test"):
            raw_x, _ = self._raw(split, public)
            features = (raw_x - public["mono_mean_x"]) / public["mono_std_x"]
            rows = self._rows[split]
            parts_g, parts_y = [], []
            for k, coefficients in blocks:
                pred = public["mean_y"] + features[:, :k] @ coefficients
                parts_g.append(((pred - rows["g"][:, None]) ** 2).sum(axis=0))
                parts_y.append(((pred - rows["y"][:, None]) ** 2).sum(axis=0))
            out[f"sse_{split}_g"] = np.concatenate(parts_g)
            out[f"sse_{split}_y"] = np.concatenate(parts_y)
        return out


# ---------------------------------------------------------------- server

def ridge_solve(gram: np.ndarray, rhs: np.ndarray, alpha: float, n: int) -> np.ndarray:
    """Ridge in Gram form: solve(F'F + alpha n I, F't)."""
    return np.linalg.solve(gram + alpha * n * np.eye(gram.shape[0]), rhs)


def candidates(method: str, kz_of: Callable[[int], int]) -> list[dict[str, Any]]:
    if method == "poly_ols":
        return [{"x_degree": d, "alpha": a} for d in OLS_X_DEGREES for a in RIDGE_ALPHAS]
    return [
        {"x_degree": dx, "z_degree": dz, "first_alpha": a1, "second_alpha": a2}
        for dx, dz in itertools.product(X_DEGREES, Z_DEGREES)
        if kz_of(dz) >= dx  # order condition: at least as many instrument features as x features
        for a1 in RIDGE_ALPHAS for a2 in RIDGE_ALPHAS
    ]


def federated_fit(clients: list[Client], method: str, x_max_degree: int, z_max_degree: int) -> dict[str, Any]:
    """Run the five rounds for one method; return every candidate's scores and the server state."""
    exp_x = np.asarray(monomial_exponents(1, x_max_degree), dtype=float)
    exp_z = np.asarray(monomial_exponents(2, z_max_degree), dtype=float)
    kx_max, kz_max = len(exp_x), len(exp_z)
    kz_of = lambda d: len(monomial_exponents(2, d))  # noqa: E731
    sent = []

    def collect(round_name: str, schema: dict[str, tuple[int, ...]], call) -> dict[str, np.ndarray]:
        total = None
        for client in clients:
            message = check_message(call(client), schema)
            total = message if total is None else {k: total[k] + v for k, v in message.items()}
        sent.append({"round": round_name, "schema": {k: list(v) for k, v in schema.items()}})
        return total

    r1 = collect("1", {"n": (1,), "sum_x": (1,), "sum_z": (2,), "sum_y": (1,)}, lambda c: c.round1())
    n = int(r1["n"][0])
    public: dict[str, np.ndarray] = {"mean_x": r1["sum_x"] / n, "mean_z": r1["sum_z"] / n,
                                     "mean_y": r1["sum_y"][0] / n, "exp_x": exp_x, "exp_z": exp_z}
    r2 = collect("2", {"ss_x": (1,), "ss_z": (2,)}, lambda c: c.round2(public))
    public["std_x"], public["std_z"] = np.sqrt(r2["ss_x"] / n), np.sqrt(r2["ss_z"] / n)
    r3 = collect("3", {"sum_mx": (kx_max,), "sum_mz": (kz_max,)}, lambda c: c.round3(public))
    public["mono_mean_x"], public["mono_mean_z"] = r3["sum_mx"] / n, r3["sum_mz"] / n
    r4 = collect("4", {"xx": (kx_max, kx_max), "zz": (kz_max, kz_max), "zx": (kz_max, kx_max),
                       "xy": (kx_max,), "zy": (kz_max,)}, lambda c: c.round4(public))
    sx, sz = np.sqrt(np.diag(r4["xx"]) / n), np.sqrt(np.diag(r4["zz"]) / n)
    public["mono_std_x"], public["mono_std_z"] = sx, sz
    # Standardized Gram blocks: F'F = D^-1 C'C D^-1, F'(y - ybar) = D^-1 C'(y - ybar).
    g_xx = r4["xx"] / np.outer(sx, sx)
    g_zz = r4["zz"] / np.outer(sz, sz)
    g_zx = r4["zx"] / np.outer(sz, sx)
    g_xy, g_zy = r4["xy"] / sx, r4["zy"] / sz

    grid = candidates(method, kz_of)
    coefs = []
    for params in grid:
        kx = params["x_degree"]
        if method == "poly_ols":
            beta = ridge_solve(g_xx[:kx, :kx], g_xy[:kx], params["alpha"], n)
        else:
            kz = kz_of(params["z_degree"])
            gzz, gzx = g_zz[:kz, :kz], g_zx[:kz, :kx]
            projection = ridge_solve(gzz, gzx, params["first_alpha"], n)
            beta = ridge_solve(projection.T @ gzz @ projection, projection.T @ g_zy[:kz],
                               params["second_alpha"], n)
        coefs.append(beta)
    # Group candidates by number of x features so clients score each group in one product.
    order = sorted(range(len(grid)), key=lambda i: grid[i]["x_degree"])
    blocks, block_index = [], []
    for kx, members in itertools.groupby(order, key=lambda i: grid[i]["x_degree"]):
        members = list(members)
        blocks.append((kx, np.column_stack([coefs[i] for i in members])))
        block_index.extend(members)
    m = len(grid)
    r5 = collect("5", {"n_dev": (1,), "n_test": (1,), "sse_dev_g": (m,), "sse_dev_y": (m,),
                       "sse_test_g": (m,), "sse_test_y": (m,)}, lambda c: c.round5(public, blocks))
    n_dev, n_test = r5["n_dev"][0], r5["n_test"][0]
    rows = [None] * m
    for position, i in enumerate(block_index):
        rows[i] = {**grid[i], "validation_mse": r5["sse_dev_g"][position] / n_dev,
                   "validation_y_mse": r5["sse_dev_y"][position] / n_dev,
                   "test_mse": r5["sse_test_g"][position] / n_test}
    return {"rows": rows, "coefs": coefs, "public": public, "n_train": n, "messages": sent}


def predict(state: dict[str, Any], index: int, x: np.ndarray) -> np.ndarray:
    public, beta = state["public"], state["coefs"][index]
    raw = monomials((x - public["mean_x"]) / public["std_x"], public["exp_x"])
    features = (raw - public["mono_mean_x"]) / public["mono_std_x"]
    return public["mean_y"] + features[:, :len(beta)] @ beta


# ---------------------------------------------------------------- partition

def pipeline_partition(dataset: str, seed: int) -> dict[str, dict[int, np.ndarray]]:
    """Client row indices for train/dev/test, from the federated runs' own load_data."""
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from fedml.data.MNIST.data_loader import load_data
    from scenarios.abstract_scenario import AbstractScenario

    scenario = AbstractScenario(filename=str(ZOO_DIR / f"{dataset}.npz"))
    scenario.to_tensor()
    train, dev, test = (scenario.get_dataset(s) for s in ("train", "dev", "test"))
    # batch_size only shapes the DataLoaders; it draws nothing from the RNG.
    args = types.SimpleNamespace(client_num_in_total=CLIENTS, partition_alpha=PARTITION_ALPHA, batch_size=256)
    np.random.seed(seed)  # what fedml.init does before the federated runs load data
    train_loaders, test_loaders, dev_loaders, *_ = load_data(args, train, test, dev)
    as_idx = lambda loaders: {k: np.asarray(v.dataset.indices) for k, v in loaders.items()}  # noqa: E731
    return {"train": as_idx(train_loaders), "dev": as_idx(dev_loaders), "test": as_idx(test_loaders)}


def build_clients(dataset: str, seed: int) -> tuple[list[Client], dict[str, Any]]:
    data = load(dataset)
    parts = pipeline_partition(dataset, seed)
    audit = {}
    for split in ("train", "dev", "test"):
        total = len(data[f"{split}_y"])
        allidx = np.concatenate([parts[split][k] for k in range(CLIENTS)])
        if len(allidx) != total or len(np.unique(allidx)) != total:
            raise SystemExit(f"{dataset} seed {seed}: {split} partition is not a disjoint cover")
        audit[f"{split}_client_counts"] = [int(len(parts[split][k])) for k in range(CLIENTS)]
    clients = [
        Client({split: {"x": data[f"{split}_x"][parts[split][k]], "z": data[f"{split}_z"][parts[split][k]],
                        "y": data[f"{split}_y"][parts[split][k]], "g": data[f"{split}_g"][parts[split][k]]}
                for split in ("train", "dev", "test")})
        for k in range(CLIENTS)
    ]
    return clients, audit


# ---------------------------------------------------------------- run

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(dataset: str, method: str, seed: int, clients: list[Client], audit: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    state = federated_fit(clients, method, max(OLS_X_DEGREES if method == "poly_ols" else X_DEGREES), max(Z_DEGREES))
    metric = SELECTION[method]
    rows = state["rows"]
    best = min(range(len(rows)), key=lambda i: (rows[i][metric], complexity(rows[i])))
    params = {k: v for k, v in rows[best].items() if k not in ("validation_mse", "validation_y_mse", "test_mse")}
    # The selected model is a known function; the server evaluates it on the
    # plotting grid, as the federated DeepGMM runs evaluate their global model.
    data = load(dataset)
    curve = predict(state, best, data["test_x"])

    run_id = f"{NAME}_{dataset}_{method}_seed{seed}"
    run_dir = OUTPUT_ROOT / dataset / method / f"seed_{seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "predictions.npz", x=data["test_x"].reshape(-1), true_g=data["test_g"],
             best_validation_prediction=curve, final_prediction=curve,
             algorithm=method, variant="federated_closed_form", seed=seed, run_id=run_id)
    write_json(run_dir / "metrics.json", {
        "best_validation_mse": float(rows[best]["validation_mse"]),
        "test_mse_at_best_validation": float(rows[best]["test_mse"]),
        "selection_metric": metric, "selection_score": float(rows[best][metric]),
        "selected_hyperparameters": params, "grid_size": len(rows),
        "runtime_seconds": time.perf_counter() - t0,
    })
    write_json(run_dir / "effective_config.json", {
        "run_id": run_id, "method": method, "dataset": dataset,
        "training_scope": "federated_closed_form", "clients": CLIENTS,
        "partition_alpha": PARTITION_ALPHA, "partition_seed": seed,
        "partition_source": "fedgmm/sp_decentralized_mnist_lr_example/fedml/data/MNIST/data_loader.py::load_data",
        "client_counts": audit, "messages": state["messages"],
        "selection_split": "dev (per-client sums)", "selection_metric": metric,
        "selected_hyperparameters": params, "archive": f"data/zoo/{dataset}.npz", "archive_sha256": data["sha256"],
    })
    write_csv(OUTPUT_ROOT / "grids" / f"seed_{seed}" / f"{dataset}_{method}.csv",
              sorted(rows, key=lambda r: (r[metric], complexity(r))))
    return {"dataset": dataset, "method": method, "seed": seed, "selection_metric": metric,
            **{f"selected_{k}": v for k, v in params.items()},
            "best_validation_mse": float(rows[best]["validation_mse"]),
            "test_mse_at_best_validation": float(rows[best]["test_mse"]),
            "run_dirs": str(run_dir.relative_to(ROOT))}


def main() -> None:
    summary = []
    for dataset in FUNCTIONS:
        for seed in SEEDS:
            clients, audit = build_clients(dataset, seed)
            for method in METHODS:
                row = run_one(dataset, method, seed, clients, audit)
                summary.append(row)
                print(f"{dataset:6s} {method:9s} seed {seed}: test MSE {row['test_mse_at_best_validation']:.5f}",
                      flush=True)
    write_csv(OUTPUT_ROOT / "summary_all_seeds.csv", summary)
    for method in METHODS:
        write_csv(OUTPUT_ROOT / f"final_selected_{method}.csv",
                  [row for row in summary if row["method"] == method and row["seed"] == PRIMARY_SEED])
    # The fit must not depend on how the clients split the data.
    for dataset in FUNCTIONS:
        for method in METHODS:
            picks = {json.dumps({k: v for k, v in row.items() if k.startswith("selected_")}, sort_keys=True)
                     for row in summary if row["dataset"] == dataset and row["method"] == method}
            if len(picks) != 1:
                raise SystemExit(f"{dataset}/{method}: different partitions selected different hyperparameters")


if __name__ == "__main__":
    main()
