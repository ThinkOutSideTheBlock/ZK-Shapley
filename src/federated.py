"""
Federated demand forecasting via exact federated ridge regression.

Design choice (stated explicitly, not hidden): rather than simulating SGD-based
FedAvg over communication rounds -- which introduces stochastic training noise,
learning-rate sensitivity, and non-determinism that would contaminate Shapley
marginal-contribution estimates -- we use the *federated normal equations*
formulation. For ridge regression, the optimal weight vector is

    w* = (X^T X + lambda I)^{-1} X^T y

which is additively separable in per-firm sufficient statistics:

    A_i = X_i^T X_i  (d x d),   b_i = X_i^T y_i  (d,)
    A_S = sum_{i in S} A_i,      b_S = sum_{i in S} b_i
    w_S = (A_S + lambda I)^{-1} b_S

This is mathematically *exact*: the federated model trained on coalition S is
identical to centralizing the raw data of firms in S and running ridge
regression directly, while only ever transmitting (A_i, b_i) -- not raw demand
data D_i. This is a well-established efficient FL technique for linear models
(one-shot / non-iterative federated linear regression) and is the natural
mechanism to pair with the cryptographic commitment layer (firms commit to and
prove properties of (A_i, b_i) rather than an iterative gradient stream).

Two direct benefits for this paper's purposes:
  1. v(S) is deterministic given the data partition -- no stochastic-optimizer
     confound when computing Shapley values.
  2. Coalition values for *all* 2^n subsets can be computed by simple matrix
     summation, making exact Shapley tractable for the validation experiments
     without retraining 2^n separate models from scratch.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .demand import DemandSimulationResult


N_LAGS = 3
N_SEASONAL_FEATURES = 2  # sin, cos
N_FEATURES = N_LAGS + N_SEASONAL_FEATURES + 1  # +1 intercept


@dataclass
class FirmDataset:
    """Featurized, firm-private train/holdout split. X, y never leave this object's owner."""
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray

    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]


def _build_features(D: np.ndarray, t_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct supervised-learning features for one-step-ahead demand forecasting:
    3 autoregressive lags + public seasonal sin/cos features + intercept.
    Returns (X, y) aligned so that row k predicts D[t_index[k]] from D[t_index[k]-1..3]
    and the seasonal phase at t_index[k].
    """
    rows_X, rows_y = [], []
    for t in t_index:
        if t < N_LAGS:
            continue
        lags = D[t - N_LAGS:t][::-1]  # most recent lag first
        phase = 2 * np.pi * t / 7.0
        seasonal_feats = np.array([np.sin(phase), np.cos(phase)])
        feat = np.concatenate([[1.0], lags, seasonal_feats])  # intercept + lags + seasonal
        rows_X.append(feat)
        rows_y.append(D[t])
    return np.asarray(rows_X), np.asarray(rows_y)


def build_firm_datasets(
    sim: DemandSimulationResult,
    holdout_fraction: float = 0.2,
) -> dict[str, FirmDataset]:
    """
    For each firm, build features from its own private trailing window (length
    firm.n_obs) and split chronologically into train/holdout (no shuffling --
    this is a time series; shuffling would leak future information into training,
    a common and serious correctness bug in naive forecasting pipelines).
    """
    datasets: dict[str, FirmDataset] = {}
    for firm in sim.firms:
        D = sim.series[firm.name]
        start = len(D) - firm.n_obs  # firm only "owns" its trailing n_obs window
        t_index = np.arange(max(start, N_LAGS), len(D))
        X, y = _build_features(D, t_index)

        n_holdout = max(int(len(X) * holdout_fraction), 10)
        X_train, y_train = X[:-n_holdout], y[:-n_holdout]
        X_holdout, y_holdout = X[-n_holdout:], y[-n_holdout:]

        datasets[firm.name] = FirmDataset(
            name=firm.name, X_train=X_train, y_train=y_train,
            X_holdout=X_holdout, y_holdout=y_holdout,
        )
    return datasets


def sufficient_statistics(ds: FirmDataset) -> tuple[np.ndarray, np.ndarray]:
    """A_i = X^T X, b_i = X^T y -- the only quantities a firm ever transmits."""
    A = ds.X_train.T @ ds.X_train
    b = ds.X_train.T @ ds.y_train
    return A, b


def fit_ridge_from_statistics(A: np.ndarray, b: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """w* = (A + lambda I)^{-1} b. Uses solve() rather than explicit inversion (numerically stable)."""
    d = A.shape[0]
    return np.linalg.solve(A + lam * np.eye(d), b)


def aggregate_coalition_statistics(
    datasets: dict[str, FirmDataset], coalition: frozenset[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum sufficient statistics over a coalition of firms. Empty coalition -> zero model."""
    d = N_FEATURES
    A_sum = np.zeros((d, d))
    b_sum = np.zeros(d)
    for name in coalition:
        A, b = sufficient_statistics(datasets[name])
        A_sum += A
        b_sum += b
    return A_sum, b_sum


def evaluate_holdout_rmse(
    w: np.ndarray, datasets: dict[str, FirmDataset], eval_firms: frozenset[str]
) -> float:
    """
    Pooled out-of-sample RMSE across eval_firms' private holdout sets, using a
    model fit on (possibly different) coalition statistics.
    """
    sq_errors, n_total = [], 0
    for name in eval_firms:
        ds = datasets[name]
        if ds.X_holdout.shape[0] == 0:
            continue
        preds = ds.X_holdout @ w
        sq_errors.append(np.sum((preds - ds.y_holdout) ** 2))
        n_total += ds.X_holdout.shape[0]
    if n_total == 0:
        return float("nan")
    return float(np.sqrt(np.sum(sq_errors) / n_total))


def fit_and_evaluate_coalition(
    datasets: dict[str, FirmDataset],
    train_coalition: frozenset[str],
    eval_firms: frozenset[str],
    lam: float = 1.0,
) -> float:
    """
    Train a federated ridge model using only train_coalition's sufficient
    statistics, then evaluate holdout RMSE on eval_firms. This is the core
    primitive used by both the Shapley value function and the inventory
    downstream-cost pipeline.
    """
    if len(train_coalition) == 0:
        # Degenerate: no data shared -> predict the per-firm training mean
        # (autarky baseline -- each firm forecasting with only its own data
        # collapsed to a constant, i.e. no model at all, the true "v(emptyset)=worst case").
        preds_list, true_list = [], []
        for name in eval_firms:
            ds = datasets[name]
            if ds.X_holdout.shape[0] == 0:
                continue
            mean_pred = ds.y_train.mean() if ds.y_train.shape[0] > 0 else 0.0
            preds_list.append(np.full(ds.X_holdout.shape[0], mean_pred))
            true_list.append(ds.y_holdout)
        if not preds_list:
            return float("nan")
        all_preds = np.concatenate(preds_list)
        all_true = np.concatenate(true_list)
        return float(np.sqrt(np.mean((all_preds - all_true) ** 2)))

    A, b = aggregate_coalition_statistics(datasets, train_coalition)
    w = fit_ridge_from_statistics(A, b, lam=lam)
    return evaluate_holdout_rmse(w, datasets, eval_firms)
