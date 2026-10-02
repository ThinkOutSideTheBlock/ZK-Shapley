"""
A1 diagnostic: homogeneity of C_unc(σ) = C(σ) - C(σ0), σ0 = 1e-6.
Expect ratios ≈ 2 and 4 under μ/σ ≥ 5, M ∈ {1,2,3}.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.demand import DemandSimulationConfig, generate_multi_firm_demand
from src.federated import build_firm_datasets
from src.coalition import CoalitionConfig, CoalitionEvaluator
from src.federated_phase1_patch import firm_predictive_residual_std
from src.inventory import (
    EchelonConfig,
    default_three_echelon_config,
    simulate_serial_supply_chain,
    _base_stock_level,
)


def mean_cost(mu, residual_std, echelons, seed, T=500, K=40):
    z = float(
        norm.ppf(float(np.clip(echelons[0].service_level, 1e-4, 1 - 1e-4))))
    L = max(int(echelons[0].lead_time), 1)
    S0 = _base_stock_level(mu, residual_std, L, z)
    costs = []
    for k in range(K):
        rng = np.random.default_rng([seed, k])
        eps = rng.standard_normal(T)
        forecast = np.full(T, float(mu))
        demand = np.maximum(0.0, mu + residual_std * eps)
        out = simulate_serial_supply_chain(
            demand, forecast, residual_std, echelons, rng=None, initial_on_hand=S0
        )
        costs.append(out.total_cost)
    return float(np.mean(costs))


def probe_M(label, mu, s1, echelons, seed=100, sigma0=1e-6):
    c0 = mean_cost(mu, sigma0, echelons, seed=seed)
    c1 = mean_cost(mu, s1, echelons, seed=seed)
    c2 = mean_cost(mu, 2 * s1, echelons, seed=seed)
    c4 = mean_cost(mu, 4 * s1, echelons, seed=seed)

    u1, u2, u4 = c1 - c0, c2 - c0, c4 - c0
    r2 = u2 / u1 if u1 > 0 else float("nan")
    r4 = u4 / u1 if u1 > 0 else float("nan")
    rt2 = c2 / c1 if c1 > 0 else float("nan")
    rt4 = c4 / c1 if c1 > 0 else float("nan")

    print(f"=== {label} ===")
    print(f"  C(σ0)={c0:.4f}  C(σ)={c1:.4f}  C(2σ)={c2:.4f}  C(4σ)={c4:.4f}")
    print(f"  TOTAL   ratio 2σ/σ={rt2:.4f}  4σ/σ={rt4:.4f}")
    print(
        f"  UNC    C_unc(σ)={u1:.4f}  ratio 2σ={r2:.4f}  4σ={r4:.4f}  (expect ≈2, ≈4)")
    ok = abs(r2 - 2.0) < 0.15 and abs(r4 - 4.0) < 0.30
    print(
        f"  C_unc homogeneity → {'PASS' if ok else 'CHECK'}  (mono {(u2 > u1) and (u4 > u2)})")
    return ok


def main():
    cfg = DemandSimulationConfig(
        n_firms=3, n_periods=400, n_features=4, seed=7,
        shared_factor_correlation=0.6, n_obs_heterogeneity="moderate",
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=40
    )
    paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))
    f0 = firms[0]
    coal = CoalitionConfig(
        firms=firms, lambda_mode="fixed", residual_mode="predictive",
        ridge_lambda=1.0, seed=7,
    )
    ev = CoalitionEvaluator(datasets, paths, coal)
    beta = ev.fit(frozenset({f0}))
    ds = datasets[f0]
    forecast = np.asarray(ds.X_holdout @ beta, dtype=float)
    s1 = firm_predictive_residual_std(beta, ds)
    mu = float(max(np.mean(np.abs(forecast)), 20.0 * s1))

    configs = [
        ("M=1", [EchelonConfig(1.0, 9.0, 1, 0.95)]),
        ("M=2", [
            EchelonConfig(1.0, 9.0, 1, 0.95),
            EchelonConfig(0.6, 0.0, 2, 0.90),
        ]),
        ("M=3", default_three_echelon_config()),
    ]
    print(f"μ={mu:.3f}  σ={s1:.4f}  μ/σ={mu/s1:.2f}\n")
    for label, echs in configs:
        probe_M(label, mu, s1, echs)
        print()


if __name__ == "__main__":
    main()
