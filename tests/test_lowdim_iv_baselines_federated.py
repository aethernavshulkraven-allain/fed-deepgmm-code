"""The federated closed-form OLS/2SLS must equal the pooled closed form exactly.

Small synthetic IV data, a reduced grid, and uneven client splits: every grid
point's federated validation and test errors must match an independent pooled
computation, for two different partitions, and the message schema must reject
per-row payloads.
"""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fed", ROOT / "scripts" / "lowdim_iv_baselines_federated.py")
fed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fed)


@pytest.fixture(autouse=True)
def small_grid(monkeypatch):
    monkeypatch.setattr(fed, "X_DEGREES", (1, 2, 3))
    monkeypatch.setattr(fed, "Z_DEGREES", (1, 2))
    monkeypatch.setattr(fed, "OLS_X_DEGREES", (1, 2, 3, 4))
    monkeypatch.setattr(fed, "RIDGE_ALPHAS", (1e-8, 1e-3, 1.0))


def synthetic(seed=0, n=900):
    rng = np.random.default_rng(seed)
    data = {}
    for split in ("train", "dev", "test"):
        z = rng.normal(size=(n, 2))
        u = rng.normal(size=n)
        x = (z[:, 0] + 0.5 * z[:, 1] + u).reshape(-1, 1)
        g = np.sin(x[:, 0])
        data.update({f"{split}_x": x, f"{split}_z": z, f"{split}_g": g,
                     f"{split}_y": g + u + 0.3 * rng.normal(size=n)})
    return data


def split_clients(data, seed, n_clients=13):
    rng = np.random.default_rng(seed)
    clients_rows = [dict() for _ in range(n_clients)]
    for split in ("train", "dev", "test"):
        n = len(data[f"{split}_y"])
        cuts = np.sort(rng.choice(np.arange(1, n), size=n_clients - 1, replace=False))
        for k, idx in enumerate(np.split(rng.permutation(n), cuts)):
            clients_rows[k][split] = {key: data[f"{split}_{key}"][idx] for key in ("x", "z", "y", "g")}
    return [fed.Client(rows) for rows in clients_rows]


class PooledFeatures:
    """Monomials of train-standardized inputs, each column train-standardized (pooled rows)."""

    def __init__(self, train, degree):
        self.in_mean, self.in_std = train.mean(axis=0), train.std(axis=0)
        self.exponents = np.asarray(fed.monomial_exponents(train.shape[1], degree), dtype=float)
        raw = fed.monomials((train - self.in_mean) / self.in_std, self.exponents)
        self.out_mean, self.out_std = raw.mean(axis=0), raw.std(axis=0)

    def __call__(self, data):
        return (fed.monomials((data - self.in_mean) / self.in_std, self.exponents) - self.out_mean) / self.out_std


def ridge(features, target, alpha):
    n, k = features.shape
    return np.linalg.solve(features.T @ features + alpha * n * np.eye(k), features.T @ target)


def pooled_grid(data, method):
    """Independent pooled closed form for every grid point, computed from all rows at once."""
    y_mean = data["train_y"].mean()
    out = []
    if method == "poly_ols":
        for d in fed.OLS_X_DEGREES:
            phi = PooledFeatures(data["train_x"], d)
            for a in fed.RIDGE_ALPHAS:
                beta = ridge(phi(data["train_x"]), data["train_y"] - y_mean, a)
                out.append(({"x_degree": d, "alpha": a},
                            {s: y_mean + phi(data[f"{s}_x"]) @ beta for s in ("dev", "test")}))
        return out
    for dx, dz in itertools.product(fed.X_DEGREES, fed.Z_DEGREES):
        if len(fed.monomial_exponents(2, dz)) < dx:
            continue
        phi, psi = PooledFeatures(data["train_x"], dx), PooledFeatures(data["train_z"], dz)
        psi_train = psi(data["train_z"])
        for a1 in fed.RIDGE_ALPHAS:
            phi_hat = psi_train @ ridge(psi_train, phi(data["train_x"]), a1)
            for a2 in fed.RIDGE_ALPHAS:
                beta = ridge(phi_hat, data["train_y"] - y_mean, a2)
                out.append(({"x_degree": dx, "z_degree": dz, "first_alpha": a1, "second_alpha": a2},
                            {s: y_mean + phi(data[f"{s}_x"]) @ beta for s in ("dev", "test")}))
    return out


def fit(data, method, partition_seed):
    return fed.federated_fit(split_clients(data, partition_seed), method,
                             max(fed.OLS_X_DEGREES if method == "poly_ols" else fed.X_DEGREES), max(fed.Z_DEGREES))


@pytest.mark.parametrize("method", ["poly_ols", "poly_2sls"])
@pytest.mark.parametrize("partition_seed", [1, 2])
def test_every_grid_point_matches_pooled(method, partition_seed):
    data = synthetic()
    state = fit(data, method, partition_seed)
    pooled = pooled_grid(data, method)
    assert len(state["rows"]) == len(pooled)
    for row, (params, pred) in zip(state["rows"], pooled):
        assert {k: row[k] for k in params} == params
        assert row["validation_mse"] == pytest.approx(np.mean((pred["dev"] - data["dev_g"]) ** 2), rel=1e-9, abs=1e-12)
        assert row["validation_y_mse"] == pytest.approx(np.mean((pred["dev"] - data["dev_y"]) ** 2), rel=1e-9, abs=1e-12)
        assert row["test_mse"] == pytest.approx(np.mean((pred["test"] - data["test_g"]) ** 2), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("method", ["poly_ols", "poly_2sls"])
def test_selected_curve_matches_pooled_curve(method):
    data = synthetic()
    state = fit(data, method, 3)
    metric = fed.SELECTION[method]
    best = min(range(len(state["rows"])), key=lambda i: (state["rows"][i][metric], fed.complexity(state["rows"][i])))
    params = {k: v for k, v in state["rows"][best].items() if k not in ("validation_mse", "validation_y_mse", "test_mse")}
    pooled = next(pred for p, pred in pooled_grid(data, method) if p == params)
    np.testing.assert_allclose(fed.predict(state, best, data["test_x"]), pooled["test"], rtol=1e-9, atol=1e-11)


def test_schema_rejects_per_row_payloads():
    schema = {"sum_x": (1,)}
    fed.check_message({"sum_x": [3.0]}, schema)
    with pytest.raises(fed.SchemaError):
        fed.check_message({"sum_x": np.arange(5.0)}, schema)  # a per-row vector
    with pytest.raises(fed.SchemaError):
        fed.check_message({"sum_x": [3.0], "rows": np.arange(5.0)}, schema)  # an extra key
