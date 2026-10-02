"""
Phase-1 helpers for federated.py — firm-specific predictive residual
and size-scaled regularization.

These functions are pure additions; existing symbols keep their signatures
so current tests continue to pass. Call sites in coalition.py / mechanism.py
switch to the new helpers under an explicit Phase-1 flag.
"""
from __future__ import annotations

import numpy as np
from .federated import FirmDataset, aggregate_coalition_statistics, fit_ridge_from_statistics


def coalition_train_size(
    datasets: dict[str, FirmDataset], coalition: frozenset[str]
) -> int:
    """Total number of training rows across members of the coalition."""
    return int(sum(datasets[name].n_train for name in coalition))


def effective_ridge_lambda(
    lam0: float,
    n_S: int,
    mode: str = "size_scaled",
) -> float:
    """
    Regularization strength for a coalition of total training size n_S.

    Modes
    -----
    fixed        : λ = lam0  (legacy Exp-2 behaviour)
    size_scaled  : λ = lam0 * n_S   (constant per-sample penalty; Phase-1 default)
    """
    if mode == "fixed":
        return float(lam0)
    if mode == "size_scaled":
        return float(lam0) * max(int(n_S), 1)
    raise ValueError(f"Unknown lambda mode: {mode!r}")


def fit_ridge_for_coalition(
    datasets: dict[str, FirmDataset],
    coalition: frozenset[str],
    lam0: float = 1.0,
    lambda_mode: str = "size_scaled",
) -> np.ndarray:
    """
    Fit ridge on coalition sufficient statistics with the chosen λ policy.
    """
    if len(coalition) == 0:
        raise ValueError("Cannot fit the empty coalition.")
    A, b = aggregate_coalition_statistics(datasets, coalition)
    n_S = coalition_train_size(datasets, coalition)
    lam = effective_ridge_lambda(lam0, n_S, mode=lambda_mode)
    return fit_ridge_from_statistics(A, b, lam=lam)


def firm_predictive_residual_std(
    beta: np.ndarray,
    ds: FirmDataset,
    min_val_rows: int = 10,
) -> float:
    """
    Firm-specific predictive residual scale for safety-stock.

    Uses RMS about zero (not mean-centered std) so coefficient bias
    from heterogeneous true parameters is retained — required for
    consistency with hold-out RMSE under heterogeneous γ.

    Requires a validation split of at least min_val_rows; no silent
    fallback to in-sample residuals.
    """
    if ds.n_val < min_val_rows:
        raise ValueError(
            f"Firm {ds.name!r}: n_val={ds.n_val} < min_val_rows={min_val_rows}. "
            "Validation residual is required; refusing silent train-residual fallback."
        )
    resid = ds.y_val - ds.X_val @ beta
    # RMS about zero — retains bias term
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return max(rms, 1e-6)


def firm_holdout_rmse(beta: np.ndarray, ds: FirmDataset) -> float:
    """RMSE of β on a single firm's hold-out partition."""
    if ds.n_holdout == 0:
        return float("nan")
    pred = ds.X_holdout @ beta
    return float(np.sqrt(np.mean((pred - ds.y_holdout) ** 2)))
