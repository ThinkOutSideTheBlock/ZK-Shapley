"""
Synthetic multi-firm demand data generation for ZK-Shapley experiments.

Design rationale
-----------------
The paper's central empirical claim (accuracy-based and inventory-cost-based
Shapley allocations need not coincide -- plan Section F.2) can only be studied
if the data-generating process satisfies two properties simultaneously:

  1. Firms genuinely benefit from pooling: a single shared, unknown coefficient
     vector beta_true governs every firm's demand response to its observed
     covariates, so pooling more (noisy, finite-sample) observations of the
     SAME underlying relationship improves everyone's ridge estimate. This is
     the "genuine collaboration gains exist" property required by
     mechanism.run_no_sharing_baseline's contract, verified empirically rather
     than assumed.

  2. Firms differ along experimentally controllable axes that are DECOUPLED
     by design: sample size n_i, signal quality gamma_i (common-factor
     loading), and noise scale sigma_i are drawn independently and
     independently shuffled across firm slots, so "largest firm" and
     "highest signal quality" are never confounded within a replication.
     This is required by plan Section F.2's experimental-factors table.

Demand-generating process (firm i, period t):

    y_{i,t} = x_{i,t}' beta_true + gamma_i * z_t + eps_{i,t},   eps_{i,t} ~ N(0, sigma_i^2)

  - beta_true in R^d: shared, unknown, identical across firms -- what pooling estimates.
  - x_{i,t} in R^d: firm i's own exogenous covariates (intercept, seasonal
    Fourier terms, idiosyncratic AR(1) regressors).
  - z_t: common latent demand-shock factor (AR(1), persistence set by
    `shared_factor_correlation`), with heterogeneous per-firm loadings gamma_i.

This is a linear-Gaussian process by design: the paper's exact-aggregation
claim (Section B.2) is proved specifically for ridge regression, and a
linear-Gaussian generator keeps beta_true identifiable, which the analytic
noise-injection result in theory.py also relies on.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DemandSimulationConfig:
    n_firms: int = 6
    n_periods: int = 400
    n_features: int = 8
    shared_factor_correlation: float = 0.5
    base_n_obs: int = 300
    # "balanced" | "moderate" | "severe"
    n_obs_heterogeneity: str = "moderate"
    # "homogeneous" | "one_high_quality" | "one_noisy"
    signal_quality_profile: str = "homogeneous"
    base_noise_std: float = 1.0
    beta_scale: float = 1.0
    seed: int = 0


@dataclass
class FirmRawSeries:
    """Raw generated (pre-split) time series for one firm."""
    name: str
    X: np.ndarray            # shape (n_obs_i, d)
    y: np.ndarray            # shape (n_obs_i,)
    gamma: float
    sigma: float
    n_obs: int


@dataclass
class DemandSimulationResult:
    firms: dict[str, FirmRawSeries]
    beta_true: np.ndarray
    z: np.ndarray
    config: DemandSimulationConfig


def _draw_n_obs(rng: np.random.Generator, n_firms: int, base_n_obs: int, profile: str) -> np.ndarray:
    """Draw per-firm sample counts under a heterogeneity profile, then shuffle across firm slots."""
    if profile == "balanced":
        counts = np.full(n_firms, base_n_obs, dtype=float)
    elif profile == "moderate":
        counts = base_n_obs * rng.uniform(0.6, 1.6, size=n_firms)
    elif profile == "severe":
        counts = base_n_obs * rng.uniform(0.15, 3.0, size=n_firms)
    else:
        raise ValueError(f"Unknown n_obs_heterogeneity profile '{profile}'.")
    counts = np.clip(np.round(counts).astype(int), 30, None)
    rng.shuffle(counts)
    return counts


def _draw_gammas(rng: np.random.Generator, n_firms: int, profile: str) -> np.ndarray:
    """Draw per-firm common-factor loadings (signal-quality knob), independently shuffled."""
    if profile == "homogeneous":
        gammas = np.full(n_firms, 0.6, dtype=float)
    elif profile == "one_high_quality":
        gammas = np.full(n_firms, 0.4, dtype=float)
        gammas[rng.integers(0, n_firms)] = 1.2
    elif profile == "one_noisy":
        gammas = np.full(n_firms, 0.6, dtype=float)
        gammas[rng.integers(0, n_firms)] = 0.1
    else:
        raise ValueError(f"Unknown signal_quality_profile '{profile}'.")
    rng.shuffle(gammas)
    return gammas


def _draw_sigmas(rng: np.random.Generator, n_firms: int, base_noise_std: float) -> np.ndarray:
    """Idiosyncratic noise stds, independently drawn/shuffled (decoupled from n_obs, gamma)."""
    sigmas = base_noise_std * rng.uniform(0.7, 1.5, size=n_firms)
    rng.shuffle(sigmas)
    return sigmas


def _generate_common_factor(rng: np.random.Generator, n_periods: int, correlation: float) -> np.ndarray:
    """AR(1) common latent demand-shock path, standardized to unit variance."""
    phi = np.clip(correlation, 0.0, 0.98)
    z = np.zeros(n_periods)
    shock_std = np.sqrt(max(1.0 - phi ** 2, 1e-6))
    z[0] = rng.normal(0.0, 1.0)
    for t in range(1, n_periods):
        z[t] = phi * z[t - 1] + rng.normal(0.0, shock_std)
    return (z - z.mean()) / (z.std() + 1e-12)


def _build_feature_matrix(rng: np.random.Generator, n_periods: int, n_features: int) -> np.ndarray:
    """
    Firm i's own exogenous covariate matrix: intercept, seasonal Fourier terms,
    then idiosyncratic AR(1) regressors filling remaining columns. Kept
    exogenous (not autoregressive in own demand) to preserve the exactly-
    identified linear-Gaussian ridge setup.
    """
    d = n_features
    X = np.ones((n_periods, d))
    t_idx = np.arange(n_periods)
    col = 1
    if col < d:
        X[:, col] = np.sin(2 * np.pi * t_idx / 52.0)
        col += 1
    if col < d:
        X[:, col] = np.cos(2 * np.pi * t_idx / 52.0)
        col += 1
    if col < d:
        X[:, col] = np.sin(2 * np.pi * t_idx / 12.0)
        col += 1
    while col < d:
        ar = np.zeros(n_periods)
        ar[0] = rng.normal(0.0, 1.0)
        phi = rng.uniform(0.2, 0.7)
        shock_std = np.sqrt(max(1.0 - phi ** 2, 1e-6))
        for t in range(1, n_periods):
            ar[t] = phi * ar[t - 1] + rng.normal(0.0, shock_std)
        X[:, col] = ar
        col += 1
    return X


def generate_multi_firm_demand(config: DemandSimulationConfig) -> DemandSimulationResult:
    """Generate a full synthetic multi-firm demand dataset per the module-level DGP."""
    rng = np.random.default_rng(config.seed)
    n, d, T = config.n_firms, config.n_features, config.n_periods

    beta_true = config.beta_scale * rng.normal(0.0, 1.0, size=d)
    beta_true[0] = config.beta_scale * 5.0

    z = _generate_common_factor(rng, T, config.shared_factor_correlation)
    n_obs_vec = _draw_n_obs(rng, n, config.base_n_obs,
                            config.n_obs_heterogeneity)
    gamma_vec = _draw_gammas(rng, n, config.signal_quality_profile)
    sigma_vec = _draw_sigmas(rng, n, config.base_noise_std)

    firms: dict[str, FirmRawSeries] = {}
    for i in range(n):
        name = f"firm_{i}"
        n_obs_i = int(min(n_obs_vec[i], T))
        X_full = _build_feature_matrix(rng, T, d)
        X_i = X_full[:n_obs_i]
        z_i = z[:n_obs_i]

        mean_demand = X_i @ beta_true + gamma_vec[i] * z_i
        eps = rng.normal(0.0, sigma_vec[i], size=n_obs_i)
        y_i = np.maximum(mean_demand + eps, 0.0)

        firms[name] = FirmRawSeries(
            name=name, X=X_i, y=y_i, gamma=float(gamma_vec[i]), sigma=float(sigma_vec[i]), n_obs=n_obs_i
        )

    return DemandSimulationResult(firms=firms, beta_true=beta_true, z=z, config=config)
