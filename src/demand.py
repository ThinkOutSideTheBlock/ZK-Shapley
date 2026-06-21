"""
Multi-firm demand generator for federated supply chain forecasting experiments.

Generative model (per firm i, time t):

    L(t)        = shared latent global demand factor (AR(1), common to all firms)
    S(t)        = common deterministic seasonal component
    D_i(t)      = mu_i + rho_i * D_i(t-1) + beta_i * S(t) + gamma_i * L(t) + eps_i(t)
    eps_i(t)    ~ N(0, sigma_i^2)                      (idiosyncratic, firm-private noise)

This construction is the standard justification for federated value creation in
demand forecasting (cf. Giannakas et al. 2022; Zheng et al. 2024 on FL demand
forecasting): firms observe a *noisy, partial* view of a common latent demand
process. No single firm can recover L(t) from its own series alone, but a
federated estimator that pools (without sharing raw data) reduces idiosyncratic
noise via averaging while preserving the shared signal. This gives a
non-degenerate, monotone-in-expectation coalitional value function v(S), which
is required for Shapley value analysis to be meaningful (an FL setup where
firms' data were *independent* would trivially make v(S) additive and the
Shapley value collapse to proportional attribution -- an uninteresting case
explicitly avoided here).

All randomness is seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class FirmConfig:
    """Per-firm generative parameters."""
    name: str
    mu: float          # baseline demand level
    rho: float          # AR(1) persistence on own past demand, in (0, 1)
    beta: float         # loading on common seasonal component
    gamma: float        # loading on shared latent factor (the "collaboration value" channel)
    sigma: float        # idiosyncratic noise std (firm-private, irreducible without pooling)
    n_obs: int           # number of historical periods this firm holds privately


@dataclass
class DemandSimulationResult:
    firms: list[FirmConfig]
    series: dict[str, np.ndarray]      # firm name -> length-T demand array
    latent: np.ndarray                  # the shared latent factor (NOT observable to any firm)
    seasonal: np.ndarray                # common seasonal component (observable, public)
    t: np.ndarray = field(repr=False)


def _seasonal_component(T: int, period: int = 7, amplitude: float = 1.0) -> np.ndarray:
    """Deterministic weekly seasonal pattern, common knowledge to all firms (e.g. day-of-week)."""
    t = np.arange(T)
    return amplitude * np.sin(2 * np.pi * t / period) + 0.3 * amplitude * np.sin(4 * np.pi * t / period)


def _simulate_latent_factor(T: int, rho_L: float, sigma_L: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) latent macro demand factor shared (but unobserved directly) across all firms."""
    L = np.zeros(T)
    innovations = rng.normal(0.0, sigma_L, size=T)
    for t in range(1, T):
        L[t] = rho_L * L[t - 1] + innovations[t]
    return L


def default_firm_configs(n_firms: int = 5, seed: int = 0) -> list[FirmConfig]:
    """
    Construct heterogeneous firm configurations representing a 5-tier supply chain
    (e.g., regional retailers feeding a shared upstream demand process), with
    deliberately unequal data quantity and quality -- this heterogeneity is what
    makes the Shapley attribution problem non-trivial (a homogeneous-firm setup
    would make every firm's marginal contribution identical by symmetry, which
    would not stress-test the fairness mechanism).
    """
    rng = np.random.default_rng(seed)
    names = [f"firm_{i}" for i in range(n_firms)]

    # Deliberately heterogeneous: data quantity, signal loading, and noise level
    # all vary, simulating firms of different size / data maturity / market exposure.
    # IMPORTANT: n_obs, gamma, and sigma are shuffled INDEPENDENTLY of one another
    # (three separate permutations) so that data quantity, signal informativeness,
    # and noise level are not artificially collinear -- a firm can be "small but
    # high-quality" or "large but noisy", which is what makes the Shapley
    # fairness analysis (payment vs. true contribution quality) a non-trivial,
    # genuinely two-dimensional empirical question rather than a tautology.
    n_obs_grid = np.linspace(120, 720, n_firms).astype(int)        # 120 .. 720 daily obs
    gamma_grid = np.linspace(0.35, 1.4, n_firms)                    # exposure to shared latent demand
    sigma_grid = np.linspace(0.9, 2.6, n_firms)                     # idiosyncratic noise (data quality)
    rng.shuffle(n_obs_grid)
    rng.shuffle(gamma_grid)
    rng.shuffle(sigma_grid)

    configs = []
    for i, name in enumerate(names):
        configs.append(
            FirmConfig(
                name=name,
                mu=rng.uniform(8.0, 15.0),
                rho=rng.uniform(0.25, 0.55),
                beta=rng.uniform(0.6, 1.3),
                gamma=gamma_grid[i],
                sigma=sigma_grid[i],
                n_obs=int(n_obs_grid[i]),
            )
        )
    return configs


def simulate_multi_firm_demand(
    firms: list[FirmConfig],
    rho_L: float = 0.85,
    sigma_L: float = 1.0,
    seed: int = 42,
) -> DemandSimulationResult:
    """
    Simulate correlated demand series for all firms over a common time horizon
    T = max(firm.n_obs); each firm only "owns" (i.e. is allowed to observe) its
    own trailing n_obs window, modeling firms with different histories / market tenure.
    """
    rng = np.random.default_rng(seed)
    T = max(f.n_obs for f in firms) + 1  # +1 to allow lag-1 feature construction
    seasonal = _seasonal_component(T)
    latent = _simulate_latent_factor(T, rho_L=rho_L, sigma_L=sigma_L, rng=rng)

    series: dict[str, np.ndarray] = {}
    for f in firms:
        D = np.zeros(T)
        D[0] = f.mu
        eps = rng.normal(0.0, f.sigma, size=T)
        for t in range(1, T):
            D[t] = (
                f.mu * (1 - f.rho)
                + f.rho * D[t - 1]
                + f.beta * seasonal[t]
                + f.gamma * latent[t]
                + eps[t]
            )
        D = np.clip(D, a_min=0.0, a_max=None)  # demand cannot be negative
        series[f.name] = D

    return DemandSimulationResult(firms=firms, series=series, latent=latent, seasonal=seasonal, t=np.arange(T))
