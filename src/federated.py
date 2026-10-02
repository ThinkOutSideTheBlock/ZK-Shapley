"""
Exact federated ridge aggregation via additive sufficient statistics.

Central algebraic fact (plan Section B.2): for coalition S with member
datasets {(X_i, y_i)}_{i in S}, define

    A_i = X_i^T X_i,   b_i = X_i^T y_i,
    A_S = sum_{i in S} A_i,   b_S = sum_{i in S} b_i,
    beta_hat_S = (A_S + Lambda)^{-1} b_S.

This is ALGEBRAICALLY IDENTICAL, in exact arithmetic, to fitting centralized
ridge regression on the pooled raw data of coalition S: stacking every
member's (X_i, y_i) into (X_S, y_S) gives X_S^T X_S = sum_i X_i^T X_i = A_S
and X_S^T y_S = sum_i X_i^T y_i = b_S by direct block-matrix expansion, so the
two normal-equation systems are literally the same linear system. No raw
observation ever needs to leave a firm's control: each firm need only submit
(A_i, b_i), a d x d matrix and a length-d vector, regardless of how many rows
n_i it has. `verify_exact_aggregation` below is the concrete numerical check
backing the abstract's "bit-identical agreement is additionally verified"
claim -- floating-point agreement is checked directly against a centralized
fit, with the caveat (per plan Section H.8) that bitwise identity is an
implementation-specific result dependent on summation order and linear-
algebra backend, not a claim about exact arithmetic per se.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .demand import DemandSimulationResult

N_FEATURES_DEFAULT = 8  # informational default; actual dimension is read from data at runtime
N_FEATURES = N_FEATURES_DEFAULT


@dataclass
class FirmDataset:
    """
    One firm's chronologically-split dataset: training rows (used to build
    sufficient statistics for coalition estimation), validation rows (reserved
    for hyperparameter selection, e.g. choosing lam), and a holdout evaluation
    horizon (used for both predictive accuracy and downstream inventory-cost
    valuation -- plan Section F.1's three-way partition, using the SAME
    holdout horizon for every coalition within a replication).
    """
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    gamma: float = 0.0
    sigma: float = 0.0

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.X_val.shape[0])

    @property
    def n_holdout(self) -> int:
        return int(self.X_holdout.shape[0])


def build_firm_datasets(
    sim_result: DemandSimulationResult,
    val_fraction: float = 0.15,
    holdout_fraction: float = 0.2,
    min_holdout: int = 20,
) -> dict[str, FirmDataset]:
    """
    Chronologically split each firm's raw series into train / validation /
    holdout, in that temporal order (no shuffling -- these are time series and
    the holdout must represent a genuine out-of-sample future horizon, per
    plan Section F.1). Enforces a floor of `min_holdout` rows to avoid
    degenerate evaluation windows for firms with small n_obs under a "severe"
    sample-size heterogeneity profile.
    """
    datasets: dict[str, FirmDataset] = {}
    for name, raw in sim_result.firms.items():
        T = raw.n_obs
        n_holdout = max(int(round(T * holdout_fraction)), min_holdout)
        n_holdout = min(n_holdout, T - 10)  # leave at least 10 rows for train+val
        n_holdout = max(n_holdout, 0)
        remaining = T - n_holdout
        n_val = max(int(round(remaining * val_fraction)), 0)
        n_train = remaining - n_val

        X_train = raw.X[:n_train]
        y_train = raw.y[:n_train]
        X_val = raw.X[n_train:n_train + n_val]
        y_val = raw.y[n_train:n_train + n_val]
        X_holdout = raw.X[n_train + n_val:]
        y_holdout = raw.y[n_train + n_val:]

        datasets[name] = FirmDataset(
            name=name, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            X_holdout=X_holdout, y_holdout=y_holdout, gamma=raw.gamma, sigma=raw.sigma,
        )
    return datasets


def aggregate_coalition_statistics(
    datasets: dict[str, FirmDataset], coalition: frozenset[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute A_S = sum_{i in S} X_i^T X_i and b_S = sum_{i in S} X_i^T y_i for
    the given coalition, using only each member's TRAINING partition. Members
    outside the coalition contribute nothing (a coalition literally cannot see
    non-member statistics -- this is the privacy boundary the mechanism is
    built around).
    """
    if len(coalition) == 0:
        raise ValueError("Cannot aggregate statistics for the empty coalition.")
    any_name = next(iter(coalition))
    d = datasets[any_name].X_train.shape[1]
    A = np.zeros((d, d))
    b = np.zeros(d)
    for name in coalition:
        X_i = datasets[name].X_train
        y_i = datasets[name].y_train
        A += X_i.T @ X_i
        b += X_i.T @ y_i
    return A, b


def fit_ridge_from_statistics(A: np.ndarray, b: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Solve (A + lam * I) beta = b. This is the exact-aggregation estimator of plan Section B.2."""
    d = A.shape[0]
    lhs = A + lam * np.eye(d)
    return np.linalg.solve(lhs, b)


def fit_ridge_centralized(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Reference centralized ridge fit on pooled raw data, used only by verify_exact_aggregation."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def verify_exact_aggregation(
    datasets: dict[str, FirmDataset], coalition: frozenset[str], lam: float = 1.0
) -> dict[str, float]:
    """
    Numerically verify that the sufficient-statistics route and a centralized
    fit on pooled raw training rows agree, and report the discrepancy. This is
    the concrete check backing the abstract's exact-aggregation claim (plan
    Section B.2); per plan Section H.8, we report a numerical discrepancy
    rather than asserting bitwise identity as a general property.
    """
    A_S, b_S = aggregate_coalition_statistics(datasets, coalition)
    w_suffstat = fit_ridge_from_statistics(A_S, b_S, lam=lam)

    X_pooled = np.concatenate([datasets[i].X_train for i in coalition], axis=0)
    y_pooled = np.concatenate([datasets[i].y_train for i in coalition], axis=0)
    w_centralized = fit_ridge_centralized(X_pooled, y_pooled, lam=lam)

    max_coef_discrepancy = float(np.max(np.abs(w_suffstat - w_centralized)))
    rel_discrepancy = max_coef_discrepancy / (float(np.max(np.abs(w_centralized))) + 1e-12)

    return {
        "max_coefficient_discrepancy": max_coef_discrepancy,
        "relative_discrepancy": rel_discrepancy,
        "coalition_size": len(coalition),
    }


def evaluate_holdout_rmse(w: np.ndarray, datasets: dict[str, FirmDataset], eval_firms: frozenset[str]) -> float:
    """RMSE of predictions X_holdout @ w against y_holdout, pooled across eval_firms."""
    preds_list, true_list = [], []
    for name in eval_firms:
        ds = datasets[name]
        if ds.X_holdout.shape[0] == 0:
            continue
        preds_list.append(ds.X_holdout @ w)
        true_list.append(ds.y_holdout)
    if not true_list:
        return float("nan")
    preds = np.concatenate(preds_list)
    truth = np.concatenate(true_list)
    return float(np.sqrt(np.mean((preds - truth) ** 2)))


def fit_and_evaluate_coalition(
    datasets: dict[str, FirmDataset],
    train_coalition: frozenset[str],
    eval_firms: frozenset[str],
    lam: float = 1.0,
) -> float:
    """
    Train a federated ridge model on train_coalition's sufficient statistics,
    then evaluate holdout RMSE on eval_firms. Core primitive shared by the
    Shapley accuracy-game value function and the inventory-cost pipeline.
    """
    if len(train_coalition) == 0:
        # Autarky baseline: no shared model -> predict each firm's own training mean.
        preds_list, true_list = [], []
        for name in eval_firms:
            ds = datasets[name]
            if ds.X_holdout.shape[0] == 0:
                continue
            mean_pred = ds.y_train.mean() if ds.y_train.shape[0] > 0 else 0.0
            preds_list.append(np.full(ds.X_holdout.shape[0], mean_pred))
            true_list.append(ds.y_holdout)
        if not true_list:
            return float("nan")
        preds = np.concatenate(preds_list)
        truth = np.concatenate(true_list)
        return float(np.sqrt(np.mean((preds - truth) ** 2)))

    A, b = aggregate_coalition_statistics(datasets, train_coalition)
    w = fit_ridge_from_statistics(A, b, lam=lam)
    return evaluate_holdout_rmse(w, datasets, eval_firms)