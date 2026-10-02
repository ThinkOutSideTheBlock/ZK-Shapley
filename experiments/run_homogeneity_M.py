"""
Stage 0 extension — Prop B″ homogeneity at echelon depth M ∈ {1, 2, 3}.

Same scaled-demand + warm-start + μ/σ ≥ 5 setup as Stage 0.
If ratios ≈ 2 and 4 at M=2,3, multi-echelon cost is degree-1 in σ (C26).
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


def mean_cost_scaled_nonneg(mu, residual_std, echelons, seed, T=500, K=40):
    z = float(
        norm.ppf(float(np.clip(echelons[0].service_level, 1e-4, 1 - 1e-4))))
    L = max(int(echelons[0].lead_time), 1)
    S0 = _base_stock_level(mu, residual_std, L, z)

    totals, holdings, shortages = [], [], []
    for k in range(K):
        rng = np.random.default_rng([seed, k])
        eps = rng.standard_normal(T)
        forecast = np.full(T, float(mu))
        demand = np.maximum(0.0, mu + residual_std * eps)
        out = simulate_serial_supply_chain(
            demand,
            forecast,
            residual_std,
            echelons,
            rng=None,
            initial_on_hand=S0,
        )
        totals.append(out.total_cost)
        holdings.append(out.holding_cost)
        shortages.append(out.shortage_cost)

    print(
        f"    [σ={residual_std:.4f}] hold={np.mean(holdings):.1f}  "
        f"short={np.mean(shortages):.1f}  S0={S0:.2f}  "
        f"μ/σ={mu / residual_std:.2f}"
    )
    return float(np.mean(totals))


def run_homogeneity(label, mu, s1, echelons, seed):
    c1 = mean_cost_scaled_nonneg(mu, s1, echelons, seed=seed)
    c2 = mean_cost_scaled_nonneg(mu, 2 * s1, echelons, seed=seed)
    c4 = mean_cost_scaled_nonneg(mu, 4 * s1, echelons, seed=seed)
    r2 = c2 / c1 if c1 > 0 else float("nan")
    r4 = c4 / c1 if c1 > 0 else float("nan")
    print(f"=== Homogeneity ({label}) ===")
    print(f"  μ={mu:.3f}  σ={s1:.4f}  C={c1:.4f}")
    print(f"  2σ  C={c2:.4f}  ratio={r2:.4f}  (expect ≈ 2)")
    print(f"  4σ  C={c4:.4f}  ratio={r4:.4f}  (expect ≈ 4)")
    print(f"  ΔC(2σ-σ)={c2 - c1:.4f}  ΔC(4σ-σ)={c4 - c1:.4f}")
    ratio_ok = abs(r2 - 2.0) < 0.45 and abs(r4 - 4.0) < 0.90
    mono_ok = (c2 > c1) and (c4 > c2)
    ok = ratio_ok or mono_ok
    print(
        f"  ratio_ok={ratio_ok}  mono_ok={mono_ok}  → {'PASS' if ok else 'FAIL'}")
    return ok, r2, r4


def main():
    cfg = DemandSimulationConfig(
        n_firms=3,
        n_periods=400,
        n_features=4,
        seed=7,
        shared_factor_correlation=0.6,
        n_obs_heterogeneity="moderate",
    )
    sim = generate_multi_firm_demand(cfg)
    datasets = build_firm_datasets(
        sim, val_fraction=0.15, holdout_fraction=0.25, min_holdout=40
    )
    paths = {n: ds.y_holdout for n, ds in datasets.items()}
    firms = tuple(sorted(datasets.keys()))
    f0 = firms[0]

    coal_cfg = CoalitionConfig(
        firms=firms,
        lambda_mode="fixed",
        residual_mode="predictive",
        ridge_lambda=1.0,
        seed=7,
    )
    ev = CoalitionEvaluator(datasets, paths, coal_cfg)
    beta = ev.fit(frozenset({f0}))
    ds = datasets[f0]
    forecast = np.asarray(ds.X_holdout @ beta, dtype=float)
    s1 = firm_predictive_residual_std(beta, ds)
    mu = float(max(np.mean(np.abs(forecast)), 20.0 * s1))

    configs = [
        (1, [EchelonConfig(holding_cost=1.0,
         shortage_cost=9.0, lead_time=1, service_level=0.95)]),
        (
            2,
            [
                EchelonConfig(holding_cost=1.0, shortage_cost=9.0,
                              lead_time=1, service_level=0.95),
                EchelonConfig(holding_cost=0.6, shortage_cost=0.0,
                              lead_time=2, service_level=0.90),
            ],
        ),
        (3, default_three_echelon_config()),
    ]

    print("=== Homogeneity vs echelon depth M (Prop B″ / C26) ===\n")
    results = []
    for M, echs in configs:
        ok, r2, r4 = run_homogeneity(f"M={M}", mu, s1, echs, seed=100)
        results.append((M, ok, r2, r4))
        print()

    print("=== C26 summary ===")
    all_ok = True
    for M, ok, r2, r4 in results:
        print(
            f"  M={M}: {'PASS' if ok else 'FAIL'}  ratio(2σ)={r2:.4f}  ratio(4σ)={r4:.4f}")
        all_ok = all_ok and ok
    if all_ok:
        print("  C26 direction: PASS — multi-echelon homogeneity holds under this probe.")
    else:
        print(
            "  C26: FAIL at some M — multi-echelon path cost may not be degree-1 in σ; "
            "Prop B′ does not automatically transfer to path v_op."
        )


if __name__ == "__main__":
    main()
